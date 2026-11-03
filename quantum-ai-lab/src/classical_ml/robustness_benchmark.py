"""Predeclared noise/window and attack-strength shifts with frozen detectors.

No temporal drift or unseen attack families are simulated. Every attack AUC
uses condition-matched honest controls; the nominal calibration stays frozen.
"""
import argparse
import csv
from dataclasses import asdict, replace
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

from adversarial.eavesdropper_game import AttackPolicy, build_training_set, policy_for_target
from classical_ml.fair_benchmark import (
    BenchmarkConfig, FEATURES, METHODS, Threshold, _cohort_record, _digest_arrays,
    _score_vector, _spread, calibrate_threshold, fit_scorers, rate_record, sample_cohort,
)

POLICIES = {
    'IR anchor': AttackPolicy(0.10, 0.95, 0),
    'IR below support': AttackPolicy(0.01, 0.95, 0),
    'IR above support': AttackPolicy(0.80, 0.95, 0),
    'PNS anchor': policy_for_target(0.20),
    'PNS below support': AttackPolicy(0, 0.5, 0.01),
    'PNS above support': AttackPolicy(0, 0.5, 0.85),
}


def default_config():
    return BenchmarkConfig(seeds=(101, 202, 303, 404, 505))


def conditions_for(config):
    if config.n_pulses < 2:
        raise ValueError('Need at least two pulses to define a shorter window')
    return {
        'nominal': {'channel_noise': config.channel_noise, 'n_pulses': config.n_pulses},
        'lower noise': {'channel_noise': 0.01, 'n_pulses': config.n_pulses},
        'higher noise': {'channel_noise': 0.06, 'n_pulses': config.n_pulses},
        'shorter window': {'channel_noise': config.channel_noise,
                           'n_pulses': config.n_pulses // 2},
    }


def summarize(runs):
    return {name: {cohort: {
        'alarm_rate': _spread([r['methods'][name]['evaluations'][cohort]['rate'] for r in runs]),
        **({'auc': _spread([r['methods'][name]['evaluations'][cohort]['auc'] for r in runs])}
           if 'auc' in runs[0]['methods'][name]['evaluations'][cohort] else {}),
    } for cohort in runs[0]['methods'][name]['evaluations']} for name in METHODS}


def run_robustness(config=None, progress=None):
    config = default_config() if config is None else config
    conditions = conditions_for(config)
    runs, evidence = [], []
    # Each evaluation cohort gets its own independent stream; only training and
    # calibration may influence fitted objects or the operating cutoffs.
    for seed in config.seeds:
        children = np.random.SeedSequence(seed).spawn(2 + len(conditions)*(1+len(POLICIES)))
        training = build_training_set(
            config.n_train_secure, config.n_train_attack, n_pulses=config.n_pulses,
            channel_noise=config.channel_noise, rng=np.random.default_rng(children[0]),
            return_details=True)
        cal = sample_cohort(AttackPolicy(), config.n_calibration, config,
                            np.random.default_rng(children[1]))
        scorers = fit_scorers(training.features, training.labels, seed, config.n_estimators)
        methods, cutoffs, scores, flags = {}, {}, {}, {}
        for name, scorer in scorers.items():
            values = _score_vector(scorer.score(cal.features))
            cutoff = (Threshold(0.11, 0, 0, config.target_fpr)
                      if name == 'QBER 11% reference'
                      else calibrate_threshold(values, config.target_fpr))
            cutoffs[name] = cutoff
            methods[name] = {'calibrated': name != 'QBER 11% reference',
                             'cutoff': asdict(cutoff),
                             'calibration': rate_record(cutoff.flags(values), cal.attempted),
                             'evaluations': {}}
            scores[name] = {'calibration': values}
            flags[name] = {'calibration': cutoff.flags(values)}
        cohorts = {'calibration': cal}
        metadata = {'calibration': {**_cohort_record(cal, AttackPolicy()),
                                   'condition': 'calibration', 'label': 0,
                                   'channel_noise': config.channel_noise,
                                   'n_pulses': config.n_pulses}}
        stream = 2
        for condition, settings in conditions.items():
            cfg = replace(config, **settings)
            honest_key = f'{condition}::honest'
            for family, policy in [('honest', AttackPolicy()), *POLICIES.items()]:
                key = f'{condition}::{family}'
                attempted = config.n_honest_test if family == 'honest' else config.n_attack_test
                cohort = sample_cohort(policy, attempted, cfg, np.random.default_rng(children[stream]))
                stream += 1
                cohorts[key] = cohort
                metadata[key] = {**_cohort_record(cohort, policy), **settings,
                                 'condition': condition, 'label': int(family != 'honest'),
                                 'feature_means': cohort.features.mean(axis=0).tolist()}
                for name, scorer in scorers.items():
                    values = _score_vector(scorer.score(cohort.features))
                    alarm = cutoffs[name].flags(values)
                    stats = rate_record(alarm, cohort.attempted)
                    if family != 'honest':
                        honest = scores[name][honest_key]
                        stats['auc'] = float(roc_auc_score(
                            np.r_[np.zeros(len(honest)), np.ones(len(values))],
                            np.r_[honest, values]))
                        stats['honest_control'] = honest_key
                    methods[name]['evaluations'][key] = stats
                    scores[name][key], flags[name][key] = values, alarm
            if progress is not None:
                progress(f'Seed {seed}: {condition} complete')
        runs.append({'seed': seed, 'spawn_keys': [list(s.spawn_key) for s in children],
                     'training': {'accepted': len(training.features), 'attempted': training.attempted,
                                  'rejected': training.rejected,
                                  'sha256': _digest_arrays(training.features, training.labels,
                                                           training.policies)},
                     'cohorts': metadata, 'methods': methods})
        evidence.append({'seed': seed, 'training': training, 'cohorts': cohorts,
                         'scores': scores, 'flags': flags})
    report = {'schema_version': 'robustness-1', 'model_version': 2,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'config': asdict(config), 'conditions': conditions,
              'policies': {k: asdict(v) for k, v in POLICIES.items()},
              'runs': runs, 'summary': summarize(runs),
              'scope': 'Frozen nominal training/calibration, condition-matched honest AUC controls; '
                       'conditional on usable stationary simulated sessions. No temporal drift, '
                       'loss drift, unseen families, security proof or novelty claim. Seed SD '
                       'is not a confidence interval.'}
    return report, evidence


def write_results(report, evidence, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', 'condition',
              'n_pulses', 'channel_noise', *FEATURES,
              'ir_rate', 'basis_bias', 'pns_frac', 'known_bits', 'sifted_bits']
    fields += [f'{name}::{kind}' for name in METHODS for kind in ('score', 'flag')]
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=fields)
            writer.writeheader()
            for run, entry in zip(report['runs'], evidence):
                training = entry['training']
                for i, (X, label, policy) in enumerate(zip(
                        training.features, training.labels, training.policies)):
                    row = {'seed': entry['seed'], 'cohort': 'training', 'session_index': i,
                           'label': int(label), 'condition': 'training',
                           'n_pulses': report['config']['n_pulses'],
                           'channel_noise': report['config']['channel_noise'],
                           **dict(zip(FEATURES, X)),
                           **dict(zip(('ir_rate', 'basis_bias', 'pns_frac'), policy))}
                    writer.writerow(row)
                for key, cohort in entry['cohorts'].items():
                    meta = run['cohorts'][key]
                    for i, X in enumerate(cohort.features):
                        row = {'seed': entry['seed'], 'cohort': key,
                               'session_index': int(cohort.accepted_attempts[i]),
                               'label': meta['label'], 'condition': meta['condition'],
                               'n_pulses': meta['n_pulses'], 'channel_noise': meta['channel_noise'],
                               **dict(zip(FEATURES, X)), **meta['policy'],
                               'known_bits': int(cohort.known_bits[i]),
                               'sifted_bits': int(cohort.sifted_bits[i])}
                        for name in METHODS:
                            row[f'{name}::score'] = float(entry['scores'][name][key][i])
                            row[f'{name}::flag'] = int(entry['flags'][name][key][i])
                        writer.writerow(row)
    root = Path(__file__).resolve().parents[1]
    sources = ['classical_ml/robustness_benchmark.py', 'classical_ml/robustness_audit.py',
               'classical_ml/fair_benchmark.py', 'adversarial/eavesdropper_game.py', 'bb84/simulator.py']
    report['provenance'] = {
        'python': sys.version.split()[0], 'platform': platform.system(),
        'architecture': platform.machine(),
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (destination/'robustness.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(default_config().seeds))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new directory (no automatic overwrites)')
    config = replace(default_config(), seeds=tuple(args.seeds))
    report, evidence = run_robustness(config, progress=lambda s: print(s, flush=True))
    write_results(report, evidence, args.output)
    print(f'Artifacts saved in {args.output}', flush=True)


if __name__ == '__main__':
    main()
