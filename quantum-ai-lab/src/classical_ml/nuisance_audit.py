"""Independent audit of honest-only nuisance counts, frozen alarms and physics."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Independent combinatorial/CDF and digest helpers, never production scoring.
from classical_ml.count_audit import (
    COUNT_FIELDS, EXACT_NAMES, FEATURES, _close, _digest, _spread,
    explicit_binomial_cdf, integer_fisher,
)

METHODS = {'QBER calibrated', 'Basis asymmetry', 'Decoy ratio', 'Joint max-tail',
           'Joint tail-sum', 'Logistic regression', 'Random forest', 'RBF SVM',
           'QBER 11% reference', *EXACT_NAMES}


def physical_reference(settings, source):
    """Closed forms independent of production Aer lookups and gain helpers."""
    a, contraction = 1-settings['gate_noise'], 1-settings['channel_noise']
    return {'q_z': (2-contraction*(1+a))/4,
            'q_x': (2-contraction*(a*a+a*a*a))/4,
            'signal_click': -math.expm1(-source['mu_signal']*settings['eta']),
            'decoy_click': -math.expm1(-source['mu_decoy']*settings['eta'])}


def _rate_check(stats, flags, attempted, rejected):
    n, k = len(flags), int(sum(flags))
    if (stats['accepted'] != n or stats['flagged'] != k or stats['attempted'] != attempted
            or stats['rejected'] != rejected or n+rejected != attempted):
        raise ValueError('Rate/count denominator mismatch')
    _close(stats['rate'], k/n, 'Rate mismatch')
    z = 1.959963984540054
    p, den = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    bounds = (0 if k == 0 else max(0, center-radius),
              1 if k == n else min(1, center+radius))
    if not np.allclose(bounds, stats['wilson95'], rtol=0, atol=1e-12):
        raise ValueError('Wilson interval mismatch')


def audit_results(destination):
    destination = Path(destination)
    report = json.loads((destination/'nuisance.json').read_text())
    if report['schema_version'] != 'honest-nuisance-1':
        raise ValueError('Unsupported nuisance schema')
    provenance = report['provenance']
    if provenance['audit_file'] != 'session_audit.csv.gz':
        raise ValueError('Unexpected audit filename')
    path = destination/provenance['audit_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['audit_sha256']:
        raise ValueError('Audit file hash mismatch')
    root = Path(__file__).resolve().parents[1]
    for source, digest in provenance['source_sha256'].items():
        file = (root/source).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Current source fingerprint mismatch')
    with gzip.open(path, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    ids = set()
    for row in rows:
        identity = (int(row['seed']), row['cohort'], int(row['session_index']))
        if identity in ids:
            raise ValueError('Duplicate session identity')
        ids.add(identity)
        groups.setdefault(identity[:2], []).append(row)
    config, conditions = report['config'], report['conditions']
    base = {'n_pulses': config['n_pulses'], 'channel_noise': .03, 'eta': .15, 'gate_noise': 0.0}
    expected_conditions = {
        'nominal': base, 'lower transmissivity': {**base, 'eta': .10},
        'higher transmissivity': {**base, 'eta': .18},
        'mild gate errors': {**base, 'gate_noise': .01},
        'stronger gate errors': {**base, 'gate_noise': .02},
        'combined loss/gates': {**base, 'eta': .10, 'gate_noise': .02},
        'combined shorter window': {**base, 'eta': .10, 'gate_noise': .02,
                                   'n_pulses': max(1, config['n_pulses']//2)},
    }
    runs = report['runs']
    if (conditions != expected_conditions or report['source'] != {
            'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5}
            or [r['seed'] for r in runs] != config['seeds'] or len(runs) < 2
            or report['count_fields'] != list(COUNT_FIELDS)
            or set(report['summary']) != METHODS):
        raise ValueError('Protocol/seed/schema inventory mismatch')
    null = report['trusted_count_null']
    if null['eta'] != .15 or null['mu_decoy'] != .1:
        raise ValueError('Trusted null was corrected using test conditions')
    probability = -math.expm1(-.1*.15)
    _close(null['decoy_click_probability'], probability, 'Trusted decoy null mismatch')
    keys = ('calibration', *conditions)
    if set(groups) != {(r['seed'], k) for r in runs for k in ('training', *keys)}:
        raise ValueError('Missing/unexpected raw cohorts')
    rate_checks = exact_checks = physical_checks = 0
    physics_z, rejection_cohorts = [], 0
    for run in runs:
        seed = run['seed']
        if set(run['cohorts']) != set(keys) or set(run['methods']) != METHODS:
            raise ValueError('Run cohort/method inventory mismatch')
        training = groups[(seed, 'training')]
        X = np.array([[float(r[f]) for f in FEATURES] for r in training])
        y = np.array([int(r['label']) for r in training])
        policies = np.array([[float(r[f]) for f in ('ir_rate', 'basis_bias', 'pns_frac')]
                             for r in training])
        t = run['training']
        if (len(training) != t['accepted'] or t['accepted']+t['rejected'] != t['attempted']
                or len(training) != config['n_train_secure']+config['n_train_attack']
                or _digest(X, y, policies) != t['sha256'] or set(y) != {0, 1}
                or not np.isfinite(X).all()
                or any(any(r[f] for f in COUNT_FIELDS) for r in training)
                or np.any((y == 1) & (policies[:, 0] == 0) & (policies[:, 2] == 0))):
            raise ValueError('Training provenance mismatch')
        for row in training:
            if any(float(row[k]) != value for k, value in base.items()):
                raise ValueError('Training is not nominal')
        for key in keys:
            meta, cohort = run['cohorts'][key], groups[(seed, key)]
            settings = base if key == 'calibration' else conditions[key]
            expected_attempted = config['n_calibration'] if key == 'calibration' else config['n_honest_test']
            C = np.array([[int(r[f]) for f in COUNT_FIELDS] for r in cohort])
            X = np.array([[float(r[f]) for f in FEATURES] for r in cohort])
            attempt_ids = np.array([int(r['session_index']) for r in cohort])
            if (meta['settings'] != settings or meta['attempted'] != expected_attempted
                    or len(cohort) != meta['accepted']
                    or meta['accepted']+meta['rejected'] != meta['attempted']
                    or np.any(attempt_ids < 0) or np.any(attempt_ids >= meta['attempted'])
                    or not np.isfinite(X).all()
                    or _digest(X, C, attempt_ids) != meta['sha256']
                    or C.sum(axis=0).tolist() != meta['count_totals']):
                raise ValueError('Cohort provenance mismatch')
            ref = physical_reference(settings, report['source'])
            for name, value in ref.items():
                _close(meta['physical_reference'][name], value, 'Aer/closed-form physics mismatch')
                physical_checks += 1
            nz, nx, ez, ex, ns, nd, ds, dd = map(int, C.sum(axis=0))
            # Descriptive only. If sessions were rejected, conditioning changes
            # the count law and these cannot be called unconditional null tests.
            if meta['rejected']:
                rejection_cohorts += 1
            else:
                for total, observed, p in [(nz, ez, ref['q_z']), (nx, ex, ref['q_x']),
                                           (ns, ds, ref['signal_click']), (nd, dd, ref['decoy_click'])]:
                    variance = total*p*(1-p)
                    if variance:
                        physics_z.append((observed-total*p)/math.sqrt(variance))
            scores_exact = {name: [] for name in EXACT_NAMES}
            for row, feature, counts in zip(cohort, X, C):
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, counts)
                if (min(counts) < 0 or nz == 0 or nx == 0 or ds == 0 or nd == 0
                        or ez > nz or ex > nx or ds > ns or dd > nd
                        or nz+nx < 20 or nz+nx > ds+dd or ns+nd != settings['n_pulses']
                        or int(row['label']) != 0 or float(row['ir_rate']) != 0
                        or float(row['pns_frac']) != 0 or float(row['basis_bias']) != .5
                        or any(float(row[k]) != v for k, v in settings.items())):
                    raise ValueError('Impossible counts or non-honest evaluation')
                features = [(ez+ex)/(nz+nx), abs(ez/nz-ex/nx), ds/ns, (dd/nd)/(ds/ns)]
                if not np.allclose(feature, features, rtol=0, atol=1e-12):
                    raise ValueError('Count-derived feature mismatch')
                basis = integer_fisher(nz, nx, ez, ex)
                decoy = explicit_binomial_cdf(nd, dd, probability)
                for n, value in zip(EXACT_NAMES, (-basis, -decoy, -min(1, 2*min(basis, decoy)))):
                    scores_exact[n].append(value)
                exact_checks += 3
            for name, method in run['methods'].items():
                scores = np.array([float(r[f'{name}::score']) for r in cohort])
                flags = np.array([int(r[f'{name}::flag']) for r in cohort])
                cutoff = method['cutoff']['value']
                is_exact = name in EXACT_NAMES
                if (method['cutoff']['operator'] != ('>=' if is_exact else '>')
                        or method['calibrated'] != (not is_exact and name != 'QBER 11% reference')
                        or method['cutoff']['target_fpr'] != config['target_fpr']):
                    raise ValueError('Cutoff/calibration metadata mismatch')
                if is_exact:
                    if cutoff != -config['target_fpr'] or not np.allclose(
                            scores, scores_exact[name], rtol=0, atol=1e-10):
                        raise ValueError('Independent p-value/cutoff mismatch')
                elif name == 'QBER 11% reference' and cutoff != .11:
                    raise ValueError('Reference cutoff mismatch')
                expected = (np.zeros(len(scores)) if cutoff is None else
                            scores >= cutoff if is_exact else scores > cutoff)
                if not np.isfinite(scores).all() or not np.array_equal(flags, expected):
                    raise ValueError('Frozen score/flag mismatch')
                stats = method['calibration'] if key == 'calibration' else method['evaluations'][key]
                if 'auc' in stats:
                    raise ValueError('Honest-only cohorts cannot establish attack AUC')
                _rate_check(stats, flags, meta['attempted'], meta['rejected'])
                rate_checks += 1
                if key == 'calibration' and method['calibrated']:
                    rank = math.ceil((len(scores)+1)*(1-Fraction(str(config['target_fpr']))))
                    expected = None if rank > len(scores) else sorted(scores)[rank-1]
                    if (cutoff != expected or method['cutoff']['rank'] != rank
                            or method['cutoff']['calibration_size'] != len(scores)):
                        raise ValueError('Rank calibration mismatch')
        for name in METHODS:
            if set(run['methods'][name]['evaluations']) != set(conditions):
                raise ValueError('Method evaluation inventory mismatch')
    for name, summaries in report['summary'].items():
        if set(summaries) != set(conditions):
            raise ValueError('Summary condition mismatch')
        for condition, summary in summaries.items():
            _spread([r['methods'][name]['evaluations'][condition]['rate'] for r in runs], summary)
    return {'audit_rows': len(rows), 'rate_checks': rate_checks,
            'independent_exact_pvalue_checks': exact_checks,
            'independent_physical_reference_checks': physical_checks,
            'unrejected_cohort_physics_max_abs_z': max(map(abs, physics_z), default=None),
            'rejected_cohorts_excluded_from_unconditional_physics_z': rejection_cohorts,
            'honest_only_labels': 'passed', 'frozen_cutoffs': 'passed',
            'count_derived_features': 'passed', 'seed_summaries': 'passed',
            'wilson_intervals': 'passed', 'current_source_fingerprints': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(audit_results(parser.parse_args().destination), indent=2))
