"""Fresh-seed count-aware vs frozen-ratio detector evaluation."""
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
import platform
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from adversarial.eavesdropper_game import (
    AttackPolicy, ETA, MU_D, build_training_set, simulate_session,
)
from bb84.telemetry import COUNT_FIELDS
from classical_ml.count_detectors import CountTests, METHODS as COUNT_METHODS
from classical_ml.fair_benchmark import (
    BenchmarkConfig, Cohort, FEATURES, METHODS as LEGACY_METHODS, Threshold,
    _cohort_record, _digest_arrays, _score_vector, _spread, calibrate_threshold,
    fit_scorers, rate_record,
)
from classical_ml.robustness_benchmark import POLICIES as STRENGTH_POLICIES, conditions_for

METHODS = [*LEGACY_METHODS, *COUNT_METHODS]
POLICIES = {**STRENGTH_POLICIES, 'IR unbiased': AttackPolicy(0.10, 0.5, 0)}
DECOY_NULL = float(-np.expm1(-MU_D*ETA))
MIMIC_NOISE = 0.03+0.10*(1-0.03)/2


def default_config():
    return BenchmarkConfig(seeds=(1001, 2002, 3003, 4004, 5005))


@dataclass
class CountedCohort(Cohort):
    counts: np.ndarray


def sample_cohort(policy, attempted, config, rng):
    X, known, sifted, ids, counts = [], [], [], [], []
    for index in range(attempted):
        result = simulate_session(*policy.as_tuple(), n_pulses=config.n_pulses,
                                  channel_noise=config.channel_noise, rng=rng, return_details=True)
        if result is not None:
            X.append(result.features); known.append(result.known_bits)
            sifted.append(result.sifted_bits); ids.append(index)
            counts.append(result.counts.as_tuple())
    if not X:
        raise ValueError('No usable count-aware sessions')
    return CountedCohort(np.asarray(X, dtype=float), np.asarray(known, dtype=int),
                         np.asarray(sifted, dtype=int), np.asarray(ids, dtype=int),
                         attempted, attempted-len(X), np.asarray(counts, dtype=int))


def _metadata(cohort, policy, condition, settings, label):
    return {**_cohort_record(cohort, policy), **settings, 'condition': condition, 'label': label,
            'counts_sha256': _digest_arrays(cohort.counts),
            'feature_means': cohort.features.mean(axis=0).tolist()}


def _scores(scorers, exact, cohort):
    return {**{n: _score_vector(s.score(cohort.features)) for n, s in scorers.items()},
            **exact.scores(cohort.counts)}


def _auc(attack, honest):
    return float(roc_auc_score(np.r_[np.zeros(len(honest)), np.ones(len(attack))],
                               np.r_[honest, attack]))


def run_benchmark(config=None, progress=None):
    config = default_config() if config is None else config
    if config.channel_noise != 0.03:
        raise ValueError('This protocol fixes nominal noise at 0.03 for its mimic control')
    conditions = conditions_for(config)
    runs, evidence = [], []
    for seed in config.seeds:
        children = np.random.SeedSequence(seed).spawn(3+len(conditions)*(1+len(POLICIES)))
        training = build_training_set(config.n_train_secure, config.n_train_attack,
                                      n_pulses=config.n_pulses, channel_noise=config.channel_noise,
                                      rng=np.random.default_rng(children[0]), return_details=True)
        cal = sample_cohort(AttackPolicy(), config.n_calibration, config,
                            np.random.default_rng(children[1]))
        scorers = fit_scorers(training.features, training.labels, seed, config.n_estimators)
        exact = CountTests(DECOY_NULL)
        values = _scores(scorers, exact, cal)
        methods, scores, flags = {}, {}, {}
        for name in METHODS:
            count_test = name in COUNT_METHODS
            cutoff = (Threshold(-config.target_fpr, 0, 0, config.target_fpr) if count_test
                      else Threshold(0.11, 0, 0, config.target_fpr) if name == 'QBER 11% reference'
                      else calibrate_threshold(values[name], config.target_fpr))
            alarm = (values[name] >= cutoff.value if count_test else cutoff.flags(values[name]))
            methods[name] = {'calibrated': not count_test and name != 'QBER 11% reference',
                             'cutoff': {**asdict(cutoff), 'operator': '>=' if count_test else '>',
                                        'kind': 'analytic' if count_test else 'reference' if
                                        name == 'QBER 11% reference' else 'rank'},
                             'calibration': rate_record(alarm, cal.attempted), 'evaluations': {}}
            scores[name], flags[name] = {'calibration': values[name]}, {'calibration': alarm}
        cohorts = {'calibration': cal}
        nominal_settings = {'channel_noise': config.channel_noise, 'n_pulses': config.n_pulses}
        metadata = {'calibration': _metadata(cal, AttackPolicy(), 'calibration', nominal_settings, 0)}

        def evaluate(key, policy, condition, settings, child, honest_control=None):
            cfg = replace(config, **settings)
            cohort = sample_cohort(policy, config.n_attack_test if honest_control else config.n_honest_test,
                                   cfg, np.random.default_rng(child))
            cohorts[key] = cohort
            metadata[key] = _metadata(cohort, policy, condition, settings, int(honest_control is not None))
            values = _scores(scorers, exact, cohort)
            for name in METHODS:
                cutoff = methods[name]['cutoff']
                alarm = (values[name] >= cutoff['value'] if name in COUNT_METHODS
                         else np.zeros(len(values[name]), dtype=bool) if cutoff['value'] is None
                         else values[name] > cutoff['value'])
                stats = rate_record(alarm, cohort.attempted)
                if honest_control is not None:
                    stats.update(auc=_auc(values[name], scores[name][honest_control]),
                                 honest_control=honest_control)
                methods[name]['evaluations'][key] = stats
                scores[name][key], flags[name][key] = values[name], alarm

        stream = 2
        for condition, settings in conditions.items():
            control = f'{condition}::honest'
            evaluate(control, AttackPolicy(), condition, settings, children[stream])
            stream += 1
            for family, policy in POLICIES.items():
                evaluate(f'{condition}::{family}', policy, condition, settings, children[stream], control)
                stream += 1
            if progress:
                progress(f'Seed {seed}: {condition} complete')
        evaluate('noise mimic::honest', AttackPolicy(), 'noise mimic',
                 {'channel_noise': MIMIC_NOISE, 'n_pulses': config.n_pulses}, children[stream])
        for name in METHODS:
            methods[name]['mimic_unbiased_ir_auc'] = _auc(scores[name]['nominal::IR unbiased'],
                                                         scores[name]['noise mimic::honest'])
        runs.append({'seed': seed, 'spawn_keys': [list(c.spawn_key) for c in children],
                     'training': {'accepted': len(training.features), 'attempted': training.attempted,
                                  'rejected': training.rejected,
                                  'sha256': _digest_arrays(training.features, training.labels,
                                                           training.policies)},
                     'cohorts': metadata, 'methods': methods})
        evidence.append({'seed': seed, 'training': training, 'cohorts': cohorts,
                         'scores': scores, 'flags': flags})
    summary = {n: {key: {
        'alarm_rate': _spread([r['methods'][n]['evaluations'][key]['rate'] for r in runs]),
        **({'auc': _spread([r['methods'][n]['evaluations'][key]['auc'] for r in runs])}
           if runs[0]['cohorts'][key]['label'] else {}),
    } for key in runs[0]['methods'][n]['evaluations']} for n in METHODS}
    report = {'schema_version': 'counts-1', 'model_version': 2, 'telemetry_version': 1,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
              'conditions': {**conditions, 'noise mimic': {'channel_noise': MIMIC_NOISE,
                                                         'n_pulses': config.n_pulses}},
              'policies': {n: asdict(p) for n, p in POLICIES.items()},
              'count_null': {'eta': ETA, 'mu_decoy': MU_D, 'decoy_click_probability': DECOY_NULL},
              'count_fields': list(COUNT_FIELDS), 'runs': runs, 'summary': summary,
              'mimic_unbiased_ir_auc': {n: _spread([r['methods'][n]['mimic_unbiased_ir_auc'] for r in runs])
                                      for n in METHODS},
              'scope': 'Fixed analytic count tests vs frozen nominal ratio/ML scorers. Trusted '
                       'source/loss null, isotropic independent honest errors; conditional on usable '
                       'simulation telemetry. Not equal-realized-FPR, loss robustness, finite-key '
                       'security, hardware or novelty evidence. SD is replicate spread, not a CI.'}
    return report, evidence


def write_results(report, evidence, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', 'condition', 'n_pulses', 'channel_noise',
              *FEATURES, *COUNT_FIELDS, 'ir_rate', 'basis_bias', 'pns_frac', 'known_bits', 'sifted_bits']
    fields += [f'{n}::{k}' for n in METHODS for k in ('score', 'flag')]
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=fields)
            writer.writeheader()
            for run, entry in zip(report['runs'], evidence):
                for i, (X, label, policy) in enumerate(zip(entry['training'].features,
                                                         entry['training'].labels,
                                                         entry['training'].policies)):
                    writer.writerow({'seed': entry['seed'], 'cohort': 'training', 'session_index': i,
                                     'label': int(label), 'condition': 'training',
                                     'n_pulses': report['config']['n_pulses'],
                                     'channel_noise': report['config']['channel_noise'],
                                     **dict(zip(FEATURES, X)),
                                     **dict(zip(('ir_rate', 'basis_bias', 'pns_frac'), policy))})
                for key, cohort in entry['cohorts'].items():
                    meta = run['cohorts'][key]
                    for i, X in enumerate(cohort.features):
                        row = {'seed': entry['seed'], 'cohort': key,
                               'session_index': int(cohort.accepted_attempts[i]),
                               'label': meta['label'], 'condition': meta['condition'],
                               'n_pulses': meta['n_pulses'], 'channel_noise': meta['channel_noise'],
                               **dict(zip(FEATURES, X)), **dict(zip(COUNT_FIELDS, map(int, cohort.counts[i]))),
                               **meta['policy'], 'known_bits': int(cohort.known_bits[i]),
                               'sifted_bits': int(cohort.sifted_bits[i])}
                        for name in METHODS:
                            row[f'{name}::score'] = float(entry['scores'][name][key][i])
                            row[f'{name}::flag'] = int(entry['flags'][name][key][i])
                        writer.writerow(row)
    root = Path(__file__).resolve().parents[1]
    sources = ['classical_ml/count_benchmark.py', 'classical_ml/count_detectors.py',
               'classical_ml/count_audit.py', 'classical_ml/fair_benchmark.py',
               'classical_ml/robustness_benchmark.py', 'bb84/telemetry.py',
               'adversarial/eavesdropper_game.py', 'bb84/simulator.py']
    report['provenance'] = {
        'python': sys.version.split()[0], 'platform': platform.system(), 'architecture': platform.machine(),
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (destination/'counts.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(default_config().seeds))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    report, evidence = run_benchmark(replace(default_config(), seeds=tuple(args.seeds)),
                                      progress=lambda s: print(s, flush=True))
    write_results(report, evidence, args.output)
    print(f'Artifacts saved in {args.output}', flush=True)


if __name__ == '__main__':
    main()
