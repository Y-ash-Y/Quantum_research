"""Fixed-policy, repeated-seed telemetry benchmark; not a QKD security proof.

Common honest calibration, independent shared test cohorts and row-level audit
outputs. No hyperparameters, thresholds or policy choices are tuned on tests.
The 11% QBER rule is an explicitly uncalibrated historical reference.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bb84.simulator import _count, _probability
from adversarial.eavesdropper_game import (
    AttackPolicy, build_training_set, expected_information, policy_for_target,
    simulate_session, wilson_interval,
)

FEATURES = ['qber', 'basis_asymmetry', 'signal_gain', 'decoy_ratio']
METHODS = [
    'QBER calibrated', 'Basis asymmetry', 'Decoy ratio',
    'Joint max-tail', 'Joint tail-sum', 'Logistic regression',
    'Random forest', 'RBF SVM', 'QBER 11% reference',
]
POLICIES = {
    'IR weak': AttackPolicy(0.1, 0.95, 0),
    'IR medium': AttackPolicy(0.3, 0.95, 0),
    'PNS weak': policy_for_target(0.05),
    'PNS medium': policy_for_target(0.2),
    'Mixed weak': policy_for_target(0.05, 0.05, 0.9),
    'Mixed medium': policy_for_target(0.2, 0.2, 0.9),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple = (11, 22, 33, 44, 55)
    n_pulses: int = 6000
    channel_noise: float = 0.03
    n_train_secure: int = 300
    n_train_attack: int = 600
    n_calibration: int = 1000
    n_honest_test: int = 1000
    n_attack_test: int = 500
    target_fpr: float = 0.05
    n_estimators: int = 300

    def __post_init__(self):
        seeds = tuple(_count(s, 'seed', 0) for s in self.seeds)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError('Use at least two distinct seeds')
        if any(s > 2**32-1 for s in seeds):
            raise ValueError('Seeds must fit the sklearn 32-bit random-state range')
        object.__setattr__(self, 'seeds', seeds)
        for name in ('n_pulses', 'n_train_secure', 'n_train_attack',
                     'n_calibration', 'n_honest_test', 'n_attack_test', 'n_estimators'):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        for name in ('channel_noise', 'target_fpr'):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not 0 < self.target_fpr < 1:
            raise ValueError('target_fpr must lie strictly between 0 and 1')


def _matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4 or not len(X) or not np.isfinite(X).all():
        raise ValueError('Expected a nonempty finite four-feature matrix')
    return X


@dataclass(frozen=True)
class Threshold:
    value: float | None
    rank: int
    calibration_size: int
    target_fpr: float

    def flags(self, scores):
        scores = _score_vector(scores)
        return np.zeros(len(scores), dtype=bool) if self.value is None else scores > self.value


def _score_vector(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Scores must be nonempty, finite and one-dimensional')
    return scores


def calibrate_threshold(scores, target_fpr):
    """Conservative split-rank cutoff; strict > makes ties conservative.

    k=ceil((m+1)(1-alpha)). If k>m there are too few calibration rows and the
    only supported conservative cutoff is no alarms (value=None, not JSON inf).
    """
    scores = _score_vector(scores)
    alpha = _probability(target_fpr, 'target_fpr')
    if not 0 < alpha < 1:
        raise ValueError('target_fpr must lie strictly between 0 and 1')
    m = len(scores)
    # Decimal spelling as a rational avoids an accidental off-by-one at integer k.
    k = math.ceil(Fraction(m+1) * (1-Fraction(str(alpha))))
    cutoff = None if k > m else float(np.partition(scores, k-1)[k-1])
    return Threshold(cutoff, k, m, alpha)


class JointTailScore:
    """Honest-reference marginal tail ranks, combined then jointly calibrated.

    No independent-feature or chi-square null approximation is assumed.
    Signal gain uses a two-sided median deviation; other directions are fixed
    by physics (high QBER/asymmetry, low decoy ratio).
    """
    def __init__(self, reference, combine):
        reference = _matrix(reference)
        if combine not in ('max', 'sum'):
            raise ValueError('combine must be max or sum')
        self.combine = combine
        self.signal_center = float(np.median(reference[:, 2]))
        self.reference = np.sort(self._orient(reference), axis=0)

    def _orient(self, X):
        X = _matrix(X)
        return np.column_stack([X[:, 0], X[:, 1],
                                abs(X[:, 2]-self.signal_center), -X[:, 3]])

    def score(self, X):
        values = self._orient(X)
        m = len(self.reference)
        # >= rather than > retains ties; +1 prevents zero empirical p-values.
        tails = np.column_stack([
            (1+m-np.searchsorted(self.reference[:, j], values[:, j], side='left'))/(m+1)
            for j in range(4)
        ])
        logtails = -np.log(tails)
        return logtails.max(axis=1) if self.combine == 'max' else 2*logtails.sum(axis=1)


class ColumnScore:
    def __init__(self, column, sign=1):
        self.column, self.sign = column, sign

    def score(self, X):
        return self.sign * _matrix(X)[:, self.column]


class LearnedScore:
    def __init__(self, model):
        self.model = model

    def score(self, X):
        X = _matrix(X)
        # Uncalibrated model scores suffice: only their ordering is needed.
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        index = int(np.flatnonzero(self.model.classes_ == 1)[0])
        return self.model.predict_proba(X)[:, index]


def fit_scorers(X, y, seed, n_estimators):
    X = _matrix(X)
    y = np.asarray(y)
    if y.shape != (len(X),) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Training labels must include classes 0 and 1')
    reference = X[y == 0]
    result = {
        'QBER calibrated': ColumnScore(0),
        'Basis asymmetry': ColumnScore(1),
        'Decoy ratio': ColumnScore(3, -1),
        'Joint max-tail': JointTailScore(reference, 'max'),
        'Joint tail-sum': JointTailScore(reference, 'sum'),
    }
    models = {
        'Logistic regression': make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=seed)),
        'Random forest': RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
        'RBF SVM': make_pipeline(StandardScaler(), SVC(C=1, gamma='scale')),
    }
    for name, model in models.items():
        result[name] = LearnedScore(model.fit(X, y))
    result['QBER 11% reference'] = ColumnScore(0)
    return result


@dataclass
class Cohort:
    features: np.ndarray
    known_bits: np.ndarray
    sifted_bits: np.ndarray
    accepted_attempts: np.ndarray
    attempted: int
    rejected: int

    @property
    def pooled_info(self):
        return float(self.known_bits.sum()/self.sifted_bits.sum())


def sample_cohort(policy, attempted, config, rng):
    features, known, sifted, ids = [], [], [], []
    for index in range(_count(attempted, 'attempted')):
        out = simulate_session(*policy.as_tuple(), n_pulses=config.n_pulses,
                               channel_noise=config.channel_noise, rng=rng,
                               return_details=True)
        if out is not None:
            features.append(out.features); known.append(out.known_bits)
            sifted.append(out.sifted_bits); ids.append(index)
    X = np.asarray(features, dtype=float).reshape(-1, 4)
    if not len(X):
        raise ValueError('No usable benchmark sessions; increase pulse count')
    return Cohort(X, np.asarray(known, dtype=int), np.asarray(sifted, dtype=int),
                  np.asarray(ids, dtype=int), attempted, attempted-len(X))


def rate_record(flags, attempted):
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 1 or not len(flags) or attempted < len(flags):
        raise ValueError('Invalid flag counts')
    k, n = int(flags.sum()), len(flags)
    return {'flagged': k, 'accepted': n, 'attempted': int(attempted),
            'rejected': int(attempted-n), 'rate': k/n,
            'wilson95': list(wilson_interval(k, n))}


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cohort_record(cohort, policy):
    return {'policy': asdict(policy), 'expected_info': expected_information(policy),
            'observed_pooled_info': cohort.pooled_info,
            'known_bits': int(cohort.known_bits.sum()),
            'sifted_bits': int(cohort.sifted_bits.sum()),
            'accepted': len(cohort.features), 'attempted': cohort.attempted,
            'rejected': cohort.rejected,
            'sha256': _digest_arrays(cohort.features, cohort.known_bits,
                                     cohort.sifted_bits, cohort.accepted_attempts)}


def _spread(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'seed_sd': float(values.std(ddof=1)),
            'seed_min': float(values.min()), 'seed_max': float(values.max())}


def summarize_runs(runs):
    """Descriptive mean/SD across seeds, not an invented uncertainty interval."""
    if len(runs) < 2:
        raise ValueError('At least two runs are required')
    summary = {}
    for method in METHODS:
        summary[method] = {
            'honest_fpr': _spread([r['methods'][method]['honest_test']['rate'] for r in runs]),
            'attacks': {family: {
                'detection': _spread([r['methods'][method]['attacks'][family]['rate'] for r in runs]),
                'auc': _spread([r['methods'][method]['attacks'][family]['auc'] for r in runs]),
            } for family in POLICIES},
        }
    return summary


def run_benchmark(config=BenchmarkConfig(), progress=None):
    runs, evidence = [], []
    for seed in config.seeds:
        children = np.random.SeedSequence(seed).spawn(3+len(POLICIES))
        training = build_training_set(
            config.n_train_secure, config.n_train_attack, n_pulses=config.n_pulses,
            channel_noise=config.channel_noise, rng=np.random.default_rng(children[0]),
            return_details=True)
        cal = sample_cohort(AttackPolicy(), config.n_calibration, config,
                            np.random.default_rng(children[1]))
        honest = sample_cohort(AttackPolicy(), config.n_honest_test, config,
                               np.random.default_rng(children[2]))
        scorers = fit_scorers(training.features, training.labels, seed, config.n_estimators)
        cutoffs, methods, saved_scores, saved_flags = {}, {}, {}, {}
        for name, scorer in scorers.items():
            cal_scores = _score_vector(scorer.score(cal.features))
            cutoff = (Threshold(0.11, 0, 0, config.target_fpr)
                      if name == 'QBER 11% reference'
                      else calibrate_threshold(cal_scores, config.target_fpr))
            cutoffs[name] = cutoff
            honest_scores = _score_vector(scorer.score(honest.features))
            flags = cutoff.flags(honest_scores)
            methods[name] = {'calibrated': name != 'QBER 11% reference',
                             'cutoff': asdict(cutoff),
                             'calibration': rate_record(cutoff.flags(cal_scores), cal.attempted),
                             'honest_test': rate_record(flags, honest.attempted), 'attacks': {}}
            saved_scores[name] = {'calibration': cal_scores, 'honest_test': honest_scores}
            saved_flags[name] = {'calibration': cutoff.flags(cal_scores), 'honest_test': flags}
        cohorts = {'calibration': cal, 'honest_test': honest}
        cohort_records = {name: _cohort_record(c, AttackPolicy()) for name, c in cohorts.items()}
        for child, (family, policy) in zip(children[3:], POLICIES.items()):
            cohort = sample_cohort(policy, config.n_attack_test, config,
                                   np.random.default_rng(child))
            cohorts[family] = cohort
            cohort_records[family] = _cohort_record(cohort, policy)
            for name, scorer in scorers.items():
                scores = _score_vector(scorer.score(cohort.features))
                flags = cutoffs[name].flags(scores)
                auc = roc_auc_score(np.r_[np.zeros(len(honest.features)),
                                          np.ones(len(cohort.features))],
                                    np.r_[saved_scores[name]['honest_test'], scores])
                methods[name]['attacks'][family] = {**rate_record(flags, cohort.attempted),
                                                    'auc': float(auc)}
                saved_scores[name][family], saved_flags[name][family] = scores, flags
        # Predeclared comparator, not whichever method happens to win on the test.
        contrasts = {}
        for family in POLICIES:
            reference = saved_flags['Joint max-tail'][family]
            contrasts[family] = {}
            for name in ('Logistic regression', 'Random forest', 'RBF SVM'):
                flags = saved_flags[name][family]
                contrasts[family][name] = {
                    'ml_only': int((flags & ~reference).sum()),
                    'stat_only': int((reference & ~flags).sum()),
                    'accepted': len(flags),
                    'detection_difference': float(flags.mean()-reference.mean()),
                    'honest_fpr_difference': float(
                        methods[name]['honest_test']['rate']
                        - methods['Joint max-tail']['honest_test']['rate']),
                }
        run = {'seed': seed, 'spawn_keys': [list(s.spawn_key) for s in children],
               'training': {'accepted': len(training.features), 'attempted': training.attempted,
                            'rejected': training.rejected,
                            'sha256': _digest_arrays(training.features, training.labels,
                                                     training.policies)},
               'cohorts': cohort_records, 'methods': methods, 'paired_vs_joint_max': contrasts}
        runs.append(run)
        evidence.append({'seed': seed, 'training': training, 'cohorts': cohorts,
                         'scores': saved_scores, 'flags': saved_flags})
        if progress is not None:
            progress(seed)
    paired = {family: {name: {
        'detection_difference': _spread([r['paired_vs_joint_max'][family][name]['detection_difference']
                                         for r in runs]),
        'honest_fpr_difference': _spread([r['paired_vs_joint_max'][family][name]['honest_fpr_difference']
                                         for r in runs]),
    } for name in ('Logistic regression', 'Random forest', 'RBF SVM')} for family in POLICIES}
    report = {'schema_version': 1, 'model_version': 2,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'config': asdict(config), 'runs': runs, 'summary': summarize_runs(runs),
              'paired_vs_joint_max': paired,
              'scope': 'Fixed predeclared policies, nominal common FPR target; '
                       'conditional on usable simulated sessions. Not an equal-realized-FPR '
                       'or adaptive/global security guarantee. Seed SD is not a confidence interval.'}
    return report, evidence


def write_results(report, evidence, destination):
    """Create a new run directory; never overwrite an existing artifact."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', *FEATURES,
              'ir_rate', 'basis_bias', 'pns_frac', 'known_bits', 'sifted_bits']
    fields += [f'{name}::{kind}' for name in METHODS for kind in ('score', 'flag')]
    # Deterministic gzip header; no local filename or time leaks into the payload.
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        import io
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=fields)
            writer.writeheader()
            for entry in evidence:
                training = entry['training']
                for index, (X, label, policy) in enumerate(zip(
                        training.features, training.labels, training.policies)):
                    row = {'seed': entry['seed'], 'cohort': 'training',
                           'session_index': index, 'label': int(label), **dict(zip(FEATURES, X))}
                    row.update(dict(zip(('ir_rate', 'basis_bias', 'pns_frac'), policy)))
                    writer.writerow(row)
                for cohort_name, cohort in entry['cohorts'].items():
                    policy = POLICIES.get(cohort_name, AttackPolicy())
                    for i, X in enumerate(cohort.features):
                        row = {'seed': entry['seed'], 'cohort': cohort_name,
                               'session_index': int(cohort.accepted_attempts[i]),
                               'label': int(cohort_name in POLICIES), **dict(zip(FEATURES, X)),
                               **asdict(policy), 'known_bits': int(cohort.known_bits[i]),
                               'sifted_bits': int(cohort.sifted_bits[i])}
                        for name in METHODS:
                            row[f'{name}::score'] = float(entry['scores'][name][cohort_name][i])
                            row[f'{name}::flag'] = int(entry['flags'][name][cohort_name][i])
                        writer.writerow(row)
    source_root = Path(__file__).resolve().parents[1]
    source_files = ['classical_ml/fair_benchmark.py', 'adversarial/eavesdropper_game.py',
                    'bb84/simulator.py']
    report['provenance'] = {
        'python': sys.version.split()[0], 'platform': platform.system(),
        'architecture': platform.machine(),
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((source_root/p).read_bytes()).hexdigest()
                          for p in source_files},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (destination/'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory (no automatic overwrites)')
    config = BenchmarkConfig(seeds=tuple(args.seeds))
    report, evidence = run_benchmark(config, progress=lambda s: print(f'Seed {s} complete', flush=True))
    write_results(report, evidence, args.output)
    print(f'Artifacts saved in {args.output}')


if __name__ == '__main__':
    main()
