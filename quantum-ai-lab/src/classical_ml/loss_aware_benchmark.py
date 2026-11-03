"""Frozen loss-aware screens against condition-matched honest and IR/PNS cohorts."""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from adversarial.eavesdropper_game import AttackPolicy, build_training_set
from bb84.honest_nuisance import HonestSettings, sample_honest
from bb84.loss_attacks import simulate_loss_session
from bb84.telemetry import COUNT_FIELDS
from classical_ml.count_detectors import CountTests, METHODS as EXACT
from classical_ml.fair_benchmark import (BenchmarkConfig, FEATURES, METHODS as LEGACY,
    Threshold, _digest_arrays, _score_vector, _spread, calibrate_threshold, fit_scorers, rate_record)
from classical_ml.loss_aware_detectors import LossAwareTests, METHODS as NEW, BOUND_FIELDS
from classical_ml.robustness_benchmark import POLICIES as ORIGINAL_POLICIES

METHODS = [*LEGACY, *EXACT, *NEW]
POLICIES = {**ORIGINAL_POLICIES, 'IR unbiased': AttackPolicy(.1, .5, 0)}
NULL_PROBABILITY = float(-np.expm1(-.1*.15))


def default_config():
    return BenchmarkConfig(seeds=(1097, 2097, 3097, 4097, 5097))


def conditions_for(config):
    if config.channel_noise != .03 or config.n_pulses < 2:
        raise ValueError('Protocol requires nominal noise .03 and at least two pulses')
    b = HonestSettings(config.n_pulses, .03)
    return {'nominal': b, 'lower transmissivity': replace(b, eta=.1),
            'higher transmissivity': replace(b, eta=.18),
            'shorter window': replace(b, n_pulses=config.n_pulses//2),
            'higher isotropic noise': replace(b, channel_noise=.06),
            'mild gate errors': replace(b, gate_noise=.01),
            'stronger gate errors': replace(b, gate_noise=.02),
            'combined loss/gates': replace(b, eta=.1, gate_noise=.02)}


@dataclass
class Batch:
    features: np.ndarray
    counts: np.ndarray
    accepted_attempts: np.ndarray
    known: np.ndarray
    sifted: np.ndarray
    attempted: int
    rejected: int


def honest_batch(attempted, settings, rng):
    b = sample_honest(attempted, settings, rng)
    return Batch(b.features, b.counts, b.accepted_attempts, np.zeros(len(b.features), dtype=int),
                 b.counts[:, 0]+b.counts[:, 1], attempted, b.rejected)


def attack_batch(policy, attempted, settings, rng):
    sessions, ids = [], []
    for i in range(attempted):
        session = simulate_loss_session(policy, settings, rng)
        if session is not None:
            sessions.append(session)
            ids.append(i)
    if not sessions:
        raise ValueError('No usable attack sessions; no silent resampling')
    return Batch(np.array([s.features for s in sessions]),
                 np.array([[getattr(s.counts, f) for f in COUNT_FIELDS] for s in sessions]),
                 np.array(ids), np.array([s.known_bits for s in sessions]),
                 np.array([s.sifted_bits for s in sessions]), attempted, attempted-len(sessions))


def _score(scorers, exact, screen, batch):
    new, bounds = screen.evaluate(batch.counts)
    return {**{n: _score_vector(s.score(batch.features)) for n, s in scorers.items()},
            **exact.scores(batch.counts), **new}, bounds


def run_benchmark(config=None, progress=None):
    config = default_config() if config is None else config
    conditions = conditions_for(config)
    specs = [('calibration', 'nominal', AttackPolicy(), 0, config.n_calibration)]
    for c, s in conditions.items():
        specs.append((f'{c}::honest', c, AttackPolicy(), 0, config.n_honest_test))
        if s.gate_noise == 0:
            specs.extend((f'{c}::{p}', c, policy, 1, config.n_attack_test) for p, policy in POLICIES.items())
    runs, evidence = [], []
    for seed in config.seeds:
        streams = np.random.SeedSequence(seed).spawn(1+len(specs))
        training = build_training_set(config.n_train_secure, config.n_train_attack,
            n_pulses=config.n_pulses, channel_noise=.03,
            rng=np.random.default_rng(streams[0]), return_details=True)
        scorers = fit_scorers(training.features, training.labels, seed, config.n_estimators)
        exact, screen = CountTests(NULL_PROBABILITY), LossAwareTests(config.target_fpr)
        cohorts, metadata, scores, flags, bounds, methods = {}, {}, {}, {}, {}, {}
        for stream, (key, condition, policy, label, attempted) in zip(streams[1:], specs):
            settings = conditions[condition]
            rng = np.random.default_rng(stream)
            b = attack_batch(policy, attempted, settings, rng) if label else honest_batch(attempted, settings, rng)
            values, limits = _score(scorers, exact, screen, b)
            cohorts[key], bounds[key] = b, limits
            metadata[key] = {'settings': asdict(settings), 'condition': condition, 'label': label,
                'policy': asdict(policy), 'attempted': attempted, 'accepted': len(b.features),
                'rejected': b.rejected, 'count_totals': b.counts.sum(axis=0).tolist(),
                'sha256': _digest_arrays(b.features, b.counts, b.accepted_attempts, b.known, b.sifted)}
            for n in METHODS:
                analytic = n in (*EXACT, *NEW)
                if key == 'calibration':
                    cutoff = (Threshold(screen.cutoffs[n] if n in NEW else -config.target_fpr, 0, 0, config.target_fpr)
                              if analytic else Threshold(.11, 0, 0, config.target_fpr)
                              if n == 'QBER 11% reference' else calibrate_threshold(values[n], config.target_fpr))
                    methods[n] = {'cutoff': {**asdict(cutoff), 'operator': '>=' if analytic else '>'},
                                  'calibrated': not analytic and n != 'QBER 11% reference', 'evaluations': {}}
                    scores[n], flags[n] = {}, {}
                cutoff = methods[n]['cutoff']['value']
                alarm = (np.zeros(len(values[n]), dtype=bool) if cutoff is None else
                         values[n] >= cutoff if analytic else values[n] > cutoff)
                stats = rate_record(alarm, attempted)
                if label:
                    control = f'{condition}::honest'
                    honest = scores[n][control]
                    stats['auc'] = float(roc_auc_score(np.r_[np.zeros(len(honest)), np.ones(len(values[n]))],
                                                     np.r_[honest, values[n]]))
                    stats['honest_control'] = control
                if key == 'calibration':
                    methods[n]['calibration'] = stats
                else:
                    methods[n]['evaluations'][key] = stats
                scores[n][key], flags[n][key] = values[n], alarm
            if progress:
                progress(f'Seed {seed}: {key} complete')
        runs.append({'seed': seed, 'spawn_keys': [list(s.spawn_key) for s in streams],
                     'training': {'accepted': len(training.features), 'attempted': training.attempted,
                         'rejected': training.rejected,
                         'sha256': _digest_arrays(training.features, training.labels, training.policies)},
                     'cohorts': metadata, 'methods': methods})
        evidence.append({'seed': seed, 'training': training, 'cohorts': cohorts,
                         'scores': scores, 'flags': flags, 'bounds': bounds})
    summary = {n: {key: {'alarm_rate': _spread([r['methods'][n]['evaluations'][key]['rate'] for r in runs]),
                        **({'auc': _spread([r['methods'][n]['evaluations'][key]['auc'] for r in runs])}
                           if 'auc' in runs[0]['methods'][n]['evaluations'][key] else {})}
                   for key in runs[0]['methods'][n]['evaluations']} for n in METHODS}
    return {'schema_version': 'loss-aware-1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'config': asdict(config), 'conditions': {k: asdict(v) for k, v in conditions.items()},
            'policies': {k: asdict(v) for k, v in POLICIES.items()}, 'count_fields': list(COUNT_FIELDS),
            'source': {'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5},
            'screen_budget': {'decoy': [config.target_fpr/2]*2, 'joint': [config.target_fpr/3]*3,
                              'intensity_ratio': .2},
            'trusted_count_null': NULL_PROBABILITY, 'runs': runs, 'summary': summary,
            'scope': 'Restricted stationary independent-pulse simulation; known intensities, no dark counts. '
                'Rates conditional on usable sessions; bound is for unselected counts. No general device, '
                'security, equal-realized-FPR, quantum advantage or novelty claim.'}, evidence


def write_results(report, evidence, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', 'n_pulses', 'channel_noise', 'eta', 'gate_noise',
              *FEATURES, *COUNT_FIELDS, 'ir_rate', 'basis_bias', 'pns_frac', 'known_bits', 'sifted_bits', *BOUND_FIELDS]
    fields += [f'{n}::{k}' for n in METHODS for k in ('score', 'flag')]
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for run, e in zip(report['runs'], evidence):
                for i, (X, y, p) in enumerate(zip(e['training'].features, e['training'].labels, e['training'].policies)):
                    writer.writerow({'seed': e['seed'], 'cohort': 'training', 'session_index': i, 'label': int(y),
                        **report['conditions']['nominal'], **dict(zip(FEATURES, X)),
                        **dict(zip(('ir_rate', 'basis_bias', 'pns_frac'), p))})
                for key, b in e['cohorts'].items():
                    m = run['cohorts'][key]
                    for i, X in enumerate(b.features):
                        row = {'seed': e['seed'], 'cohort': key, 'session_index': int(b.accepted_attempts[i]),
                               'label': m['label'], **m['settings'], **m['policy'], **dict(zip(FEATURES, X)),
                               **dict(zip(COUNT_FIELDS, map(int, b.counts[i]))),
                               'known_bits': int(b.known[i]), 'sifted_bits': int(b.sifted[i]),
                               **{f: float(e['bounds'][key][f][i]) for f in BOUND_FIELDS}}
                        for n in METHODS:
                            row[f'{n}::score'], row[f'{n}::flag'] = float(e['scores'][n][key][i]), int(e['flags'][n][key][i])
                        writer.writerow(row)
    root = Path(__file__).resolve().parents[1]
    sources = ['bb84/honest_nuisance.py', 'bb84/loss_attacks.py', 'bb84/simulator.py', 'bb84/telemetry.py',
               'adversarial/eavesdropper_game.py', 'classical_ml/fair_benchmark.py',
               'classical_ml/count_detectors.py', 'classical_ml/count_audit.py', 'classical_ml/nuisance_audit.py',
               'classical_ml/robustness_benchmark.py', 'classical_ml/loss_aware_detectors.py',
               'classical_ml/loss_aware_benchmark.py', 'classical_ml/loss_aware_audit.py']
    report['provenance'] = {'python': sys.version.split()[0],
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination/'loss_aware.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    report, evidence = run_benchmark(progress=lambda s: print(s, flush=True))
    write_results(report, evidence, args.output)
    print(f'Artifacts saved in {args.output}', flush=True)


if __name__ == '__main__':
    main()
