"""Independent audit of frozen fixed-window power, counts and physical marginals."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from classical_ml.count_audit import COUNT_FIELDS, FEATURES, _close, _digest, _manual_auc, _spread
from classical_ml.nuisance_audit import _rate_check, physical_reference
from classical_ml.power_numerics import recurrence_cdf, recurrence_survival, recurrence_fisher

EXACT = ('Count Fisher basis', 'Count binomial decoy', 'Count Bonferroni joint')
NEW = ('Loss-aware decoy', 'Loss-aware joint', 'Loss plug-in decoy')
METHODS = (*EXACT, *NEW)
SOURCES = {'bb84/honest_nuisance.py', 'bb84/loss_attacks.py', 'bb84/simulator.py', 'bb84/telemetry.py',
           'adversarial/eavesdropper_game.py', 'classical_ml/fair_benchmark.py',
           'classical_ml/count_detectors.py', 'classical_ml/count_audit.py', 'classical_ml/nuisance_audit.py',
           'classical_ml/robustness_benchmark.py', 'classical_ml/loss_aware_detectors.py',
           'classical_ml/loss_aware_benchmark.py', 'classical_ml/window_power_benchmark.py',
           'classical_ml/power_numerics.py', 'classical_ml/window_power_audit.py'}


def mapped(p):
    return 1 if p == 1 else -math.expm1(.2*math.log1p(-p))


def audit_results(destination):
    destination = Path(destination)
    r = json.loads((destination/'window_power.json').read_text())
    p, cfg = r['provenance'], r['config']
    if r['schema_version'] != 'window-power-1' or r['model_version'] != 2 or p['audit_file'] != 'session_audit.csv.gz':
        raise ValueError('Schema/file mismatch')
    if set(p['source_sha256']) != SOURCES:
        raise ValueError('Source inventory mismatch')
    root = Path(__file__).resolve().parents[1]
    for source, digest in p['source_sha256'].items():
        file = (root/source).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Current source fingerprint mismatch')
    path = destination/p['audit_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != p['audit_sha256']:
        raise ValueError('Raw checksum mismatch')
    alpha = cfg['alpha']
    if (not 0 < alpha < 1 or cfg['windows'] != sorted(set(cfg['windows']))
            or not cfg['windows'] or min(cfg['windows']) < 2 or len(cfg['seeds']) < 2
            or len(set(cfg['seeds'])) != len(cfg['seeds']) or cfg['honest_attempts'] <= 0 or cfg['attack_attempts'] <= 0
            or [run['seed'] for run in r['runs']] != cfg['seeds']
            or r['count_fields'] != list(COUNT_FIELDS) or set(r['summary']) != set(METHODS)
            or r['source'] != {'mu_signal': .5, 'mu_decoy': .1, 'decoy_fraction': .5}
            or r['budget'] != {'decoy': [alpha/2]*2, 'joint': [alpha/3]*3, 'intensity_ratio': .2}):
        raise ValueError('Protocol inventory mismatch')
    signal, nominal_decoy = -math.expm1(-.5*.15), -math.expm1(-.1*.15)
    multi_s, multi_d = 1-math.exp(-.5)*1.5, 1-math.exp(-.1)*1.1
    A, B = (signal+nominal_decoy)/2, (signal+signal/multi_s*multi_d)/2
    policies = {'IR biased': {'ir_rate': .1, 'basis_bias': .95, 'pns_frac': 0},
                'IR unbiased': {'ir_rate': .1, 'basis_bias': .5, 'pns_frac': 0},
                'PNS moderate': {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': .2*A/(.8*B+.2*A)},
                'PNS weak': {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': .01}}
    if set(r['policies']) != set(policies):
        raise ValueError('Policy inventory mismatch')
    for name, policy in policies.items():
        if set(r['policies'][name]) != set(policy):
            raise ValueError('Policy field mismatch')
        for field, value in policy.items():
            _close(r['policies'][name][field], value, 'Frozen policy mismatch')
    _close(r['nominal_decoy_null'], nominal_decoy, 'Nominal decoy null changed')
    conditions, specs = {}, {}
    zero = {'ir_rate': 0, 'basis_bias': .5, 'pns_frac': 0}
    for n in cfg['windows']:
        base = {'n_pulses': n, 'channel_noise': .03, 'eta': .15, 'gate_noise': 0.0}
        conditions[str(n)] = {'nominal': base, 'lower transmission': {**base, 'eta': .1},
                              'honest gate stress': {**base, 'gate_noise': .02}}
        for c, settings in conditions[str(n)].items():
            specs[f'{n}::{c}::honest'] = (c, settings, zero, 0, cfg['honest_attempts'])
            if not settings['gate_noise']:
                for name, policy in policies.items():
                    specs[f'{n}::{c}::{name}'] = (c, settings, policy, 1, cfg['attack_attempts'])
    if r['conditions'] != conditions:
        raise ValueError('Condition inventory mismatch')
    with gzip.open(path, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    groups, ids = {}, set()
    for row in rows:
        identity = (int(row['seed']), row['cohort'], int(row['session_index']))
        if identity in ids:
            raise ValueError('Duplicate session identity')
        ids.add(identity)
        groups.setdefault(identity[:2], []).append(row)
    if set(groups) != {(seed, key) for seed in cfg['seeds'] for key in specs}:
        raise ValueError('Raw cohort inventory mismatch')
    rates = scores_checked = intervals = aucs = physical = rejected_physics = 0
    physical_z = []
    for run in r['runs']:
        seed = run['seed']
        if (set(run['cohorts']) != set(specs) or set(run['methods']) != set(METHODS)
                or run['spawn_keys'] != [[i] for i in range(len(specs))]):
            raise ValueError('Run/stream inventory mismatch')
        for key, (condition, settings, policy, label, attempted) in specs.items():
            meta, cohort = run['cohorts'][key], groups[seed, key]
            C = np.array([[int(row[f]) for f in COUNT_FIELDS] for row in cohort])
            X = np.array([[float(row[f]) for f in FEATURES] for row in cohort])
            indexes = np.array([int(row['session_index']) for row in cohort])
            known = np.array([int(row['known_bits']) for row in cohort])
            sifted = np.array([int(row['sifted_bits']) for row in cohort])
            if (meta['settings'] != settings or meta['condition'] != condition or meta['label'] != label
                    or any(abs(meta['policy'][f]-value) > 1e-12 for f, value in policy.items())
                    or meta['attempted'] != attempted or meta['accepted'] != len(cohort)
                    or meta['accepted']+meta['rejected'] != attempted
                    or meta['count_totals'] != C.sum(axis=0).tolist()
                    or not np.isfinite(X).all() or np.any(indexes < 0) or np.any(indexes >= attempted)
                    or np.any(np.diff(indexes) <= 0)
                    or _digest(X, C, indexes, known, sifted) != meta['sha256']):
                raise ValueError('Cohort provenance mismatch')
            expected = {n: [] for n in METHODS}
            for row, feature, counts, k, s in zip(cohort, X, C, known, sifted):
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, counts)
                if (min(counts) < 0 or not nz or not nx or not ds or not nd or ez > nz or ex > nx
                        or ds > ns or dd > nd or nz+nx < 20 or nz+nx > ds+dd
                        or ns+nd != settings['n_pulses'] or s != nz+nx or not 0 <= k <= s
                        or (not label and k != 0) or int(row['label']) != label
                        or any(float(row[f]) != value for f, value in settings.items())
                        or any(abs(float(row[f])-value) > 1e-12 for f, value in policy.items())):
                    raise ValueError('Impossible counts/label/policy/settings')
                if not np.allclose(feature, [(ez+ex)/(nz+nx), abs(ez/nz-ex/nx), ds/ns, (dd/nd)/(ds/ns)], rtol=0, atol=1e-12):
                    raise ValueError('Count-derived feature mismatch')
                fisher = recurrence_fisher(nz, nx, ez, ex)
                old = recurrence_cdf(nd, dd, nominal_decoy)
                tails = []
                for field, delta in [('signal_lower_decoy', alpha/2), ('signal_lower_joint', alpha/3)]:
                    lower = float(row[field])
                    if not math.isfinite(lower) or not 0 <= lower <= ds/ns:
                        raise ValueError('Signal bound range mismatch')
                    if abs(recurrence_survival(ns, ds, lower)-delta) > 1e-8:
                        raise ValueError('Independent signal-bound inversion mismatch')
                    tails.append(recurrence_cdf(nd, dd, mapped(lower)))
                    intervals += 1
                plugin = recurrence_cdf(nd, dd, mapped(ds/ns))
                for n, value in zip(METHODS, (-fisher, -old, -min(1, 2*min(fisher, old)),
                                             -tails[0], -min(tails[1], fisher), -plugin)):
                    expected[n].append(value)
                scores_checked += 6
            ref = physical_reference(settings, r['source'])
            if label:
                ref['decoy_click'] = ((1-policy['pns_frac'])*ref['decoy_click']
                    +policy['pns_frac']*ref['signal_click']/multi_s*multi_d)
                ref['q_z'] += policy['ir_rate']*(1-policy['basis_bias'])*(1-settings['channel_noise'])/2
                ref['q_x'] += policy['ir_rate']*policy['basis_bias']*(1-settings['channel_noise'])/2
            else:
                for name, value in ref.items():
                    _close(meta['honest_physical_reference'][name], value, 'Aer/closed-form reference mismatch')
            physical += 4
            if meta['rejected']:
                rejected_physics += 1
            else:
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, C.sum(axis=0))
                for total, observed, prob in [(nz, ez, ref['q_z']), (nx, ex, ref['q_x']),
                                             (ns, ds, ref['signal_click']), (nd, dd, ref['decoy_click'])]:
                    var = total*prob*(1-prob)
                    if var:
                        physical_z.append((observed-total*prob)/math.sqrt(var))
            for n, method in run['methods'].items():
                cutoff = (-alpha/2, -alpha/3, -alpha)[NEW.index(n)] if n in NEW else -alpha
                if method['cutoff'] != {'value': cutoff, 'operator': '>='}:
                    raise ValueError('Frozen cutoff mismatch')
                values = np.array([float(row[f'{n}::score']) for row in cohort])
                flags = np.array([int(row[f'{n}::flag']) for row in cohort])
                if (not np.isfinite(values).all() or not np.allclose(values, expected[n], rtol=0, atol=1e-8)
                        or not np.array_equal(flags, values >= cutoff)):
                    raise ValueError('Independent score/frozen flag mismatch')
                stat = method['evaluations'][key]
                _rate_check(stat, flags, attempted, meta['rejected'])
                rates += 1
                if label:
                    control = f'{settings["n_pulses"]}::{condition}::honest'
                    h = [float(row[f'{n}::score']) for row in groups[seed, control]]
                    if stat['honest_control'] != control:
                        raise ValueError('Condition/window-matched control mismatch')
                    _close(stat['auc'], _manual_auc(values, h), 'Independent AUC mismatch')
                    aucs += 1
                elif 'auc' in stat:
                    raise ValueError('Honest-only cohort cannot establish attack AUC')
        for n in METHODS:
            if set(run['methods'][n]['evaluations']) != set(specs):
                raise ValueError('Method inventory mismatch')
    for n, summaries in r['summary'].items():
        if set(summaries) != set(specs):
            raise ValueError('Summary inventory mismatch')
        for key, summary in summaries.items():
            _spread([run['methods'][n]['evaluations'][key]['rate'] for run in r['runs']], summary['alarm_rate'])
            if bool(specs[key][3]) != ('auc' in summary):
                raise ValueError('Summary AUC inventory mismatch')
            if 'auc' in summary:
                _spread([run['methods'][n]['evaluations'][key]['auc'] for run in r['runs']], summary['auc'])
    return {'audit_rows': len(rows), 'rate_checks': rates, 'independent_score_checks': scores_checked,
            'independent_confidence_bound_inversions': intervals, 'independent_auc_checks': aucs,
            'independent_physical_reference_checks': physical,
            'unrejected_cohort_physics_max_abs_z': max(map(abs, physical_z), default=None),
            'rejected_cohorts_excluded_from_unconditional_physics_z': rejected_physics,
            'current_source_fingerprints': 'passed', 'fixed_windows_and_cutoffs': 'passed',
            'matched_controls': 'passed', 'count_derived_features': 'passed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(audit_results(parser.parse_args().destination), indent=2))
