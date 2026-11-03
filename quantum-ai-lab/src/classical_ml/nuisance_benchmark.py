"""Predeclared honest-only loss/device stress test of frozen nominal detectors."""
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

from adversarial.eavesdropper_game import build_training_set
from bb84.honest_nuisance import HonestSettings, probabilities, sample_honest
from bb84.telemetry import COUNT_FIELDS
from classical_ml.count_detectors import CountTests, METHODS as EXACT
from classical_ml.fair_benchmark import (
    BenchmarkConfig, FEATURES, METHODS as LEGACY, Threshold, _digest_arrays,
    _score_vector, _spread, calibrate_threshold, fit_scorers, rate_record,
)

METHODS = [*LEGACY, *EXACT]
NULL_PROBABILITY = float(-np.expm1(-0.1*0.15))


def default_config():
    return BenchmarkConfig(seeds=(1069, 2069, 3069, 4069, 5069))


def conditions_for(config):
    base = HonestSettings(config.n_pulses, config.channel_noise)
    return {'nominal': base, 'lower transmissivity': replace(base, eta=.10),
            'higher transmissivity': replace(base, eta=.18),
            'mild gate errors': replace(base, gate_noise=.01),
            'stronger gate errors': replace(base, gate_noise=.02),
            'combined loss/gates': replace(base, eta=.10, gate_noise=.02),
            'combined shorter window': replace(base, eta=.10, gate_noise=.02,
                                                n_pulses=max(1, config.n_pulses//2))}


def _score(scorers, exact, batch):
    return {**{n: _score_vector(s.score(batch.features)) for n, s in scorers.items()},
            **exact.scores(batch.counts)}


def _cohort_metadata(batch, settings):
    return {'settings': asdict(settings), 'accepted': len(batch.features),
            'attempted': batch.attempted, 'rejected': batch.rejected,
            'sha256': _digest_arrays(batch.features, batch.counts, batch.accepted_attempts),
            'count_totals': batch.counts.sum(axis=0).tolist(),
            'physical_reference': probabilities(settings)}


def run_benchmark(config=None, progress=None):
    config = default_config() if config is None else config
    if config.channel_noise != .03:
        raise ValueError('Protocol fixes nominal channel_noise at .03')
    conditions = conditions_for(config)
    runs, evidence = [], []
    for seed in config.seeds:
        children = np.random.SeedSequence(seed).spawn(2+len(conditions))
        training = build_training_set(config.n_train_secure, config.n_train_attack,
                                      n_pulses=config.n_pulses, channel_noise=config.channel_noise,
                                      rng=np.random.default_rng(children[0]), return_details=True)
        scorers = fit_scorers(training.features, training.labels, seed, config.n_estimators)
        exact = CountTests(NULL_PROBABILITY)
        cal = sample_honest(config.n_calibration, conditions['nominal'],
                            np.random.default_rng(children[1]))
        values = _score(scorers, exact, cal)
        methods, scores, flags = {}, {}, {}
        for name in METHODS:
            is_exact = name in EXACT
            cutoff = (Threshold(-config.target_fpr, 0, 0, config.target_fpr) if is_exact
                      else Threshold(.11, 0, 0, config.target_fpr) if name == 'QBER 11% reference'
                      else calibrate_threshold(values[name], config.target_fpr))
            alarm = values[name] >= cutoff.value if is_exact else cutoff.flags(values[name])
            methods[name] = {'calibrated': not is_exact and name != 'QBER 11% reference',
                             'cutoff': {**asdict(cutoff), 'operator': '>=' if is_exact else '>'},
                             'calibration': rate_record(alarm, cal.attempted), 'evaluations': {}}
            scores[name], flags[name] = {'calibration': values[name]}, {'calibration': alarm}
        cohorts = {'calibration': cal}
        metadata = {'calibration': _cohort_metadata(cal, conditions['nominal'])}
        for index, (condition, settings) in enumerate(conditions.items(), start=2):
            batch = sample_honest(config.n_honest_test, settings,
                                  np.random.default_rng(children[index]))
            cohorts[condition] = batch
            metadata[condition] = _cohort_metadata(batch, settings)
            values = _score(scorers, exact, batch)
            for name in METHODS:
                cutoff = methods[name]['cutoff']['value']
                alarm = (values[name] >= cutoff if name in EXACT else
                         np.zeros(len(values[name]), dtype=bool) if cutoff is None else
                         values[name] > cutoff)
                methods[name]['evaluations'][condition] = rate_record(alarm, batch.attempted)
                scores[name][condition], flags[name][condition] = values[name], alarm
            if progress:
                progress(f'Seed {seed}: {condition} complete')
        runs.append({'seed': seed, 'spawn_keys': [list(c.spawn_key) for c in children],
                     'training': {'accepted': len(training.features), 'attempted': training.attempted,
                                  'rejected': training.rejected,
                                  'sha256': _digest_arrays(training.features, training.labels,
                                                           training.policies)},
                     'cohorts': metadata, 'methods': methods})
        evidence.append({'seed': seed, 'training': training, 'cohorts': cohorts,
                         'scores': scores, 'flags': flags})
    report = {'schema_version': 'honest-nuisance-1', 'model_version': 2,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
              'conditions': {k: asdict(v) for k, v in conditions.items()},
              'count_fields': list(COUNT_FIELDS),
              'trusted_count_null': {'eta': .15, 'mu_decoy': .1, 'decoy_click_probability': NULL_PROBABILITY},
              'source': {'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5},
              'runs': runs,
              'summary': {n: {c: _spread([r['methods'][n]['evaluations'][c]['rate'] for r in runs])
                              for c in conditions} for n in METHODS},
              'scope': 'Honest-only aggregate-count loss/device stress test. No test-time null '
                       'correction, attack power, AUC, hardware/security or novelty claim. '
                       'Rates conditional on usable sessions; SD is replicate spread, not CI.'}
    return report, evidence


def write_results(report, evidence, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', 'n_pulses', 'channel_noise', 'eta', 'gate_noise',
              *FEATURES, *COUNT_FIELDS, 'ir_rate', 'basis_bias', 'pns_frac']
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
                                     'label': int(label), **report['conditions']['nominal'],
                                     **dict(zip(FEATURES, X)),
                                     **dict(zip(('ir_rate', 'basis_bias', 'pns_frac'), policy))})
                for key, batch in entry['cohorts'].items():
                    for i, X in enumerate(batch.features):
                        row = {'seed': entry['seed'], 'cohort': key,
                               'session_index': int(batch.accepted_attempts[i]), 'label': 0,
                               **run['cohorts'][key]['settings'], **dict(zip(FEATURES, X)),
                               **dict(zip(COUNT_FIELDS, map(int, batch.counts[i]))),
                               'ir_rate': 0, 'basis_bias': .5, 'pns_frac': 0}
                        for name in METHODS:
                            row[f'{name}::score'] = float(entry['scores'][name][key][i])
                            row[f'{name}::flag'] = int(entry['flags'][name][key][i])
                        writer.writerow(row)
    root = Path(__file__).resolve().parents[1]
    sources = ['bb84/honest_nuisance.py', 'bb84/simulator.py', 'bb84/telemetry.py',
               'adversarial/eavesdropper_game.py', 'classical_ml/fair_benchmark.py',
               'classical_ml/count_detectors.py', 'classical_ml/count_audit.py',
               'classical_ml/nuisance_benchmark.py', 'classical_ml/nuisance_audit.py']
    report['provenance'] = {
        'python': sys.version.split()[0], 'platform': platform.system(), 'architecture': platform.machine(),
        'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (destination/'nuisance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
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
