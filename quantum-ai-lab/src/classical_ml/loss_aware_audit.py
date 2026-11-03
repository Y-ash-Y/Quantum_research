"""Independent finite-sum audit; no production screen or beta quantile calls."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from classical_ml.count_audit import (COUNT_FIELDS, EXACT_NAMES, FEATURES, _close,
    _digest, _manual_auc, _spread, explicit_binomial_cdf, integer_fisher)
from classical_ml.nuisance_audit import _rate_check, physical_reference

NEW = ('Loss-aware decoy', 'Loss-aware joint', 'Loss plug-in decoy')
METHODS = {'QBER calibrated', 'Basis asymmetry', 'Decoy ratio', 'Joint max-tail',
           'Joint tail-sum', 'Logistic regression', 'Random forest', 'RBF SVM',
           'QBER 11% reference', *EXACT_NAMES, *NEW}


def _mapped(p):
    return 1 if p == 1 else -math.expm1(.2*math.log1p(-p))


def audit_results(destination):
    destination = Path(destination)
    r = json.loads((destination/'loss_aware.json').read_text())
    p = r['provenance']
    if r['schema_version'] != 'loss-aware-1' or p['audit_file'] != 'session_audit.csv.gz':
        raise ValueError('Schema/filename mismatch')
    path = destination/p['audit_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != p['audit_sha256']:
        raise ValueError('Audit file hash mismatch')
    root = Path(__file__).resolve().parents[1]
    required = {'bb84/loss_attacks.py', 'classical_ml/loss_aware_detectors.py',
                'classical_ml/loss_aware_benchmark.py', 'classical_ml/loss_aware_audit.py'}
    if not required <= set(p['source_sha256']):
        raise ValueError('Missing source fingerprints')
    for source, digest in p['source_sha256'].items():
        file = (root/source).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Current source fingerprint mismatch')
    config, alpha = r['config'], r['config']['target_fpr']
    b = {'n_pulses': config['n_pulses'], 'channel_noise': .03, 'eta': .15, 'gate_noise': 0.0}
    conditions = {'nominal': b, 'lower transmissivity': {**b, 'eta': .1},
        'higher transmissivity': {**b, 'eta': .18},
        'shorter window': {**b, 'n_pulses': config['n_pulses']//2},
        'higher isotropic noise': {**b, 'channel_noise': .06},
        'mild gate errors': {**b, 'gate_noise': .01},
        'stronger gate errors': {**b, 'gate_noise': .02},
        'combined loss/gates': {**b, 'eta': .1, 'gate_noise': .02}}
    policies = {'IR anchor': {'ir_rate': .1, 'basis_bias': .95, 'pns_frac': 0},
        'IR below support': {'ir_rate': .01, 'basis_bias': .95, 'pns_frac': 0},
        'IR above support': {'ir_rate': .8, 'basis_bias': .95, 'pns_frac': 0},
        'IR unbiased': {'ir_rate': .1, 'basis_bias': .5, 'pns_frac': 0},
        'PNS below support': {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': .01},
        'PNS above support': {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': .85}}
    # Independently solve the old nominal 20% raw-knowledge anchor.
    signal, decoy = -math.expm1(-.5*.15), -math.expm1(-.1*.15)
    multi_s, multi_d = 1-math.exp(-.5)*1.5, 1-math.exp(-.1)*1.1
    A, B = (signal+decoy)/2, (signal+signal/multi_s*multi_d)/2
    rho = .2*A/(.8*B+.2*A)
    policies['PNS anchor'] = {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': rho}
    if (r['conditions'] != conditions or config['channel_noise'] != .03
            or r['source'] != {'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5}
            or r['screen_budget'] != {'decoy': [alpha/2]*2, 'joint': [alpha/3]*3, 'intensity_ratio': .2}
            or r['count_fields'] != list(COUNT_FIELDS) or set(r['summary']) != METHODS
            or [run['seed'] for run in r['runs']] != config['seeds'] or len(r['runs']) < 2
            or len(set(config['seeds'])) != len(config['seeds'])
            or set(r['policies']) != set(policies)):
        raise ValueError('Protocol inventory mismatch')
    for key, policy in policies.items():
        for field, value in policy.items():
            _close(r['policies'][key][field], value, 'Frozen policy mismatch')
    _close(r['trusted_count_null'], decoy, 'Nominal trusted null mismatch')
    zero = {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': 0}
    specs = {'calibration': ('nominal', zero, 0, config['n_calibration'])}
    for c, s in conditions.items():
        specs[f'{c}::honest'] = (c, zero, 0, config['n_honest_test'])
        if not s['gate_noise']:
            specs.update({f'{c}::{name}': (c, policy, 1, config['n_attack_test']) for name, policy in policies.items()})
    with gzip.open(path, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    groups, ids = {}, set()
    for row in rows:
        identity = (int(row['seed']), row['cohort'], int(row['session_index']))
        if identity in ids:
            raise ValueError('Duplicate session identity')
        ids.add(identity)
        groups.setdefault(identity[:2], []).append(row)
    if set(groups) != {(seed, key) for seed in config['seeds'] for key in ('training', *specs)}:
        raise ValueError('Raw cohort inventory mismatch')
    rates = checks = intervals = aucs = physics = rejected_physics = 0
    physical_z = []
    for run in r['runs']:
        seed = run['seed']
        if set(run['cohorts']) != set(specs) or set(run['methods']) != METHODS:
            raise ValueError('Run inventory mismatch')
        if run['spawn_keys'] != [[i] for i in range(1+len(specs))]:
            raise ValueError('Independent stream inventory mismatch')
        training = groups[seed, 'training']
        X = np.array([[float(row[f]) for f in FEATURES] for row in training])
        y = np.array([int(row['label']) for row in training])
        ps = np.array([[float(row[f]) for f in zero] for row in training])
        t = run['training']
        if (len(training) != config['n_train_secure']+config['n_train_attack']
                or len(training) != t['accepted'] or t['accepted']+t['rejected'] != t['attempted']
                or _digest(X, y, ps) != t['sha256']
                or int(sum(y == 0)) != config['n_train_secure'] or int(sum(y == 1)) != config['n_train_attack']
                or not np.isfinite(X).all() or any(any(row[f] for f in COUNT_FIELDS) for row in training)):
            raise ValueError('Training provenance mismatch')
        for row in training:
            if any(float(row[f]) != value for f, value in b.items()):
                raise ValueError('Non-nominal training')
        for key, (c, policy, label, attempted) in specs.items():
            settings, m, cohort = conditions[c], run['cohorts'][key], groups[seed, key]
            C = np.array([[int(row[f]) for f in COUNT_FIELDS] for row in cohort])
            X = np.array([[float(row[f]) for f in FEATURES] for row in cohort])
            attempt_ids = np.array([int(row['session_index']) for row in cohort])
            known = np.array([int(row['known_bits']) for row in cohort])
            sifted = np.array([int(row['sifted_bits']) for row in cohort])
            if (m['settings'] != settings or m['condition'] != c or m['label'] != label
                    or any(abs(m['policy'][f]-policy[f]) > 1e-12 for f in policy)
                    or m['attempted'] != attempted or m['accepted'] != len(cohort)
                    or m['accepted']+m['rejected'] != attempted
                    or np.any(attempt_ids < 0) or np.any(attempt_ids >= attempted)
                    or np.any(np.diff(attempt_ids) <= 0)
                    or m['count_totals'] != C.sum(axis=0).tolist()
                    or _digest(X, C, attempt_ids, known, sifted) != m['sha256']):
                raise ValueError('Cohort provenance mismatch')
            expected = {n: [] for n in (*EXACT_NAMES, *NEW)}
            for row, feature, counts, k, s in zip(cohort, X, C, known, sifted):
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, counts)
                if (min(counts) < 0 or not nz or not nx or not ds or not nd
                        or ez > nz or ex > nx or ds > ns or dd > nd or nz+nx < 20
                        or nz+nx > ds+dd or ns+nd != settings['n_pulses']
                        or s != nz+nx or k < 0 or k > s or (not label and k != 0)
                        or int(row['label']) != label
                        or any(float(row[f]) != value for f, value in settings.items())
                        or any(abs(float(row[f])-value) > 1e-12 for f, value in policy.items())):
                    raise ValueError('Impossible telemetry or policy/label mismatch')
                if not np.allclose(feature, [(ez+ex)/(nz+nx), abs(ez/nz-ex/nx), ds/ns, (dd/nd)/(ds/ns)], atol=1e-12, rtol=0):
                    raise ValueError('Count-derived feature mismatch')
                basis = integer_fisher(nz, nx, ez, ex)
                old = explicit_binomial_cdf(nd, dd, decoy)
                tails = []
                for field, delta in [('signal_lower_decoy', alpha/2), ('signal_lower_joint', alpha/3)]:
                    lower = float(row[field])
                    if not math.isfinite(lower) or not 0 <= lower <= ds/ns:
                        raise ValueError('Signal confidence bound outside range')
                    if ds == 0:
                        if lower != 0:
                            raise ValueError('Zero signal endpoint mismatch')
                    elif abs(1-explicit_binomial_cdf(ns, ds-1, lower)-delta) > 1e-9:
                        raise ValueError('Independent confidence-bound inversion mismatch')
                    tails.append(explicit_binomial_cdf(nd, dd, _mapped(lower)))
                    intervals += 1
                plugin = explicit_binomial_cdf(nd, dd, _mapped(ds/ns))
                vals = (-basis, -old, -min(1, 2*min(basis, old)), -tails[0], -min(tails[1], basis), -plugin)
                for n, value in zip((*EXACT_NAMES, *NEW), vals):
                    expected[n].append(value)
                checks += 6
            # Independent marginal references. z scores are descriptive, not
            # automatic multiple-testing decisions or checks after rejection.
            ref = physical_reference(settings, r['source'])
            if label:
                qs = ref['signal_click']
                qforward = qs/multi_s
                ref['decoy_click'] = ((1-policy['pns_frac'])*ref['decoy_click']
                                      +policy['pns_frac']*qforward*multi_d)
                ref['q_z'] += policy['ir_rate']*(1-policy['basis_bias'])*(1-settings['channel_noise'])/2
                ref['q_x'] += policy['ir_rate']*policy['basis_bias']*(1-settings['channel_noise'])/2
            if not all(0 <= q <= 1 for q in ref.values()):
                raise ValueError('Physical reference out of range')
            physics += 4
            if m['rejected']:
                rejected_physics += 1
            else:
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, C.sum(axis=0))
                for total, observed, prob in [(nz, ez, ref['q_z']), (nx, ex, ref['q_x']),
                                             (ns, ds, ref['signal_click']), (nd, dd, ref['decoy_click'])]:
                    variance = total*prob*(1-prob)
                    if variance:
                        physical_z.append((observed-total*prob)/math.sqrt(variance))
            for n, method in run['methods'].items():
                scores = np.array([float(row[f'{n}::score']) for row in cohort])
                flags = np.array([int(row[f'{n}::flag']) for row in cohort])
                analytic = n in (*EXACT_NAMES, *NEW)
                calibrated = not analytic and n != 'QBER 11% reference'
                cutoff = method['cutoff']['value']
                if (method['cutoff']['operator'] != ('>=' if analytic else '>')
                        or method['calibrated'] != calibrated or method['cutoff']['target_fpr'] != alpha):
                    raise ValueError('Cutoff metadata mismatch')
                if analytic:
                    target = (-alpha/2, -alpha/3, -alpha)[NEW.index(n)] if n in NEW else -alpha
                    if cutoff != target or not np.allclose(scores, expected[n], rtol=0, atol=1e-9):
                        raise ValueError('Independent screening score/cutoff mismatch')
                elif n == 'QBER 11% reference' and cutoff != .11:
                    raise ValueError('Reference cutoff mismatch')
                if key == 'calibration' and calibrated:
                    rank = math.ceil((len(scores)+1)*(1-Fraction(str(alpha))))
                    target = sorted(scores)[rank-1] if rank <= len(scores) else None
                    if cutoff != target or method['cutoff']['rank'] != rank or method['cutoff']['calibration_size'] != len(scores):
                        raise ValueError('Frozen calibration mismatch')
                alarm = np.zeros(len(scores)) if cutoff is None else scores >= cutoff if analytic else scores > cutoff
                if not np.isfinite(scores).all() or not np.array_equal(flags, alarm):
                    raise ValueError('Frozen flag mismatch')
                stat = method['calibration'] if key == 'calibration' else method['evaluations'][key]
                _rate_check(stat, flags, attempted, m['rejected'])
                rates += 1
                if label:
                    control = f'{c}::honest'
                    hs = [float(row[f'{n}::score']) for row in groups[seed, control]]
                    if stat['honest_control'] != control:
                        raise ValueError('Condition-matched control mismatch')
                    _close(stat['auc'], _manual_auc(scores, hs), 'Independent AUC mismatch')
                    aucs += 1
                elif 'auc' in stat:
                    raise ValueError('Honest-only cohort cannot establish AUC')
        for n in METHODS:
            if set(run['methods'][n]['evaluations']) != set(specs)-{'calibration'}:
                raise ValueError('Evaluation inventory mismatch')
    for n in METHODS:
        if set(r['summary'][n]) != set(specs)-{'calibration'}:
            raise ValueError('Summary inventory mismatch')
        for key, summary in r['summary'][n].items():
            _spread([run['methods'][n]['evaluations'][key]['rate'] for run in r['runs']], summary['alarm_rate'])
            if 'auc' in summary:
                _spread([run['methods'][n]['evaluations'][key]['auc'] for run in r['runs']], summary['auc'])
    return {'audit_rows': len(rows), 'rate_checks': rates, 'independent_screening_score_checks': checks,
            'independent_confidence_bound_inversions': intervals, 'independent_auc_checks': aucs,
            'independent_physical_reference_checks': physics,
            'unrejected_cohort_physics_max_abs_z': max(map(abs, physical_z), default=None),
            'rejected_cohorts_excluded_from_unconditional_physics_z': rejected_physics,
            'current_source_fingerprints': 'passed',
            'condition_matched_controls': 'passed', 'frozen_cutoffs': 'passed', 'count_derived_features': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(audit_results(parser.parse_args().destination), indent=2))
