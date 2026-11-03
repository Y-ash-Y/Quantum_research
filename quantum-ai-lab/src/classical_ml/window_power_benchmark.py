"""Fixed-window count-screen power study; no fitted model or sequential stopping."""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adversarial.eavesdropper_game import AttackPolicy, policy_for_target
from bb84.honest_nuisance import HonestSettings, probabilities
from bb84.simulator import _count, _probability
from bb84.telemetry import COUNT_FIELDS
from classical_ml.count_detectors import CountTests, METHODS as EXACT
from classical_ml.fair_benchmark import FEATURES, _digest_arrays, _spread, rate_record
from classical_ml.loss_aware_benchmark import honest_batch, attack_batch
from classical_ml.loss_aware_detectors import LossAwareTests, METHODS as NEW, BOUND_FIELDS

METHODS = (*EXACT, *NEW)
POLICIES = {'IR biased': AttackPolicy(.1, .95, 0), 'IR unbiased': AttackPolicy(.1, .5, 0),
            'PNS moderate': policy_for_target(.2), 'PNS weak': AttackPolicy(0, .5, .01)}


@dataclass(frozen=True)
class WindowConfig:
    seeds: tuple = (1117, 2117, 3117, 4117, 5117)
    windows: tuple = (6000, 20000, 60000, 120000)
    honest_attempts: int = 500
    attack_attempts: int = 250
    alpha: float = .05

    def __post_init__(self):
        seeds = tuple(_count(v, 'seed', 0) for v in self.seeds)
        windows = tuple(_count(v, 'window', 2) for v in self.windows)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s > 2**32-1 for s in seeds):
            raise ValueError('Need at least two unique 32-bit seeds')
        if not windows or windows != tuple(sorted(set(windows))):
            raise ValueError('Windows must be nonempty, increasing and distinct')
        object.__setattr__(self, 'seeds', seeds)
        object.__setattr__(self, 'windows', windows)
        for f in ('honest_attempts', 'attack_attempts'):
            object.__setattr__(self, f, _count(getattr(self, f), f))
        object.__setattr__(self, 'alpha', _probability(self.alpha, 'alpha'))
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must lie strictly between zero and one')


def conditions_for(n):
    return {'nominal': HonestSettings(n), 'lower transmission': HonestSettings(n, eta=.1),
            'honest gate stress': HonestSettings(n, gate_noise=.02)}


def cohort_key(n, condition, family):
    return f'{n}::{condition}::{family}'


def run_benchmark(config=None, progress=None):
    config = WindowConfig() if config is None else config
    specs = []
    for n in config.windows:
        for c, settings in conditions_for(n).items():
            specs.append((cohort_key(n, c, 'honest'), c, settings, AttackPolicy(), 0, config.honest_attempts))
            if settings.gate_noise == 0:
                specs.extend((cohort_key(n, c, family), c, settings, policy, 1, config.attack_attempts)
                             for family, policy in POLICIES.items())
    runs, evidence = [], []
    screen = LossAwareTests(config.alpha)
    exact = CountTests(float(-np.expm1(-.1*.15)))
    cutoffs = {**{n: -config.alpha for n in EXACT}, **screen.cutoffs}
    for seed in config.seeds:
        streams = np.random.SeedSequence(seed).spawn(len(specs))
        cohorts, metadata, scores, flags, bounds, methods = {}, {}, {}, {}, {}, {}
        for n in METHODS:
            methods[n] = {'cutoff': {'value': cutoffs[n], 'operator': '>='}, 'evaluations': {}}
            scores[n], flags[n] = {}, {}
        for stream, (key, condition, settings, policy, label, attempted) in zip(streams, specs):
            rng = np.random.default_rng(stream)
            b = attack_batch(policy, attempted, settings, rng) if label else honest_batch(attempted, settings, rng)
            new, limits = screen.evaluate(b.counts)
            values = {**exact.scores(b.counts), **new}
            cohorts[key], bounds[key] = b, limits
            metadata[key] = {'condition': condition, 'settings': asdict(settings), 'label': label,
                'policy': asdict(policy), 'attempted': attempted, 'accepted': len(b.counts), 'rejected': b.rejected,
                'sha256': _digest_arrays(b.features, b.counts, b.accepted_attempts, b.known, b.sifted),
                'count_totals': b.counts.sum(axis=0).tolist(),
                **({'honest_physical_reference': probabilities(settings)} if not label else {})}
            for n in METHODS:
                alarm = values[n] >= cutoffs[n]
                stat = rate_record(alarm, attempted)
                if label:
                    control = cohort_key(settings.n_pulses, condition, 'honest')
                    h = scores[n][control]
                    stat['auc'] = float(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(values[n]))],
                                                     np.r_[h, values[n]]))
                    stat['honest_control'] = control
                methods[n]['evaluations'][key] = stat
                scores[n][key], flags[n][key] = values[n], alarm
            if progress:
                progress(f'Seed {seed}: {key} complete')
        runs.append({'seed': seed, 'spawn_keys': [list(s.spawn_key) for s in streams],
                     'cohorts': metadata, 'methods': methods})
        evidence.append({'seed': seed, 'cohorts': cohorts, 'bounds': bounds, 'scores': scores, 'flags': flags})
    summary = {n: {key: {'alarm_rate': _spread([r['methods'][n]['evaluations'][key]['rate'] for r in runs]),
                         **({'auc': _spread([r['methods'][n]['evaluations'][key]['auc'] for r in runs])}
                            if 'auc' in runs[0]['methods'][n]['evaluations'][key] else {})}
                   for key in metadata} for n in METHODS}
    report = {'schema_version': 'window-power-1', 'model_version': 2,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
              'conditions': {str(n): {c: asdict(s) for c, s in conditions_for(n).items()} for n in config.windows},
              'policies': {p: asdict(s) for p, s in POLICIES.items()}, 'count_fields': list(COUNT_FIELDS),
              'source': {'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5},
              'budget': {'decoy': [config.alpha/2]*2, 'joint': [config.alpha/3]*3, 'intensity_ratio': .2},
              'nominal_decoy_null': float(-np.expm1(-.1*.15)), 'runs': runs, 'summary': summary,
              'scope': 'Independent fixed-window power/resource study with unchanged count screens. '
                       'No ML training, sequential stopping, general-device bound, final-key security or novelty claim. '
                       'Rates conditional on usable sessions; signal/decoy bound is for unselected independent counts.'}
    return report, evidence


def write_results(report, evidence, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'session_audit.csv.gz'
    fields = ['seed', 'cohort', 'session_index', 'label', 'n_pulses', 'channel_noise', 'eta', 'gate_noise',
              *FEATURES, *COUNT_FIELDS, 'ir_rate', 'basis_bias', 'pns_frac', 'known_bits', 'sifted_bits', *BOUND_FIELDS]
    fields += [f'{n}::{field}' for n in METHODS for field in ('score', 'flag')]
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for r, e in zip(report['runs'], evidence):
                for key, b in e['cohorts'].items():
                    m = r['cohorts'][key]
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
               'classical_ml/loss_aware_benchmark.py', 'classical_ml/window_power_benchmark.py',
               'classical_ml/power_numerics.py', 'classical_ml/window_power_audit.py']
    report['provenance'] = {'packages': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'qiskit', 'qiskit-aer')},
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'audit_file': path.name, 'audit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination/'window_power.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
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
