"""Replay fresh public decisions and independently reconstruct KL covers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import rel_entr
from scipy.stats import binomtest, hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import PublicBatch
from research.exact_comparison import central_exact_pvalues
from audit_exact_comparison import independent_decisions


def close(a, b):
    if not np.allclose(a, b, rtol=1e-9, atol=1e-9):
        raise ValueError('Numerical reconciliation mismatch')


def ci(k, n, failure):
    result = binomtest(k, n).proportion_ci(1-failure, method='exact')
    return [result.low, result.high]


def audit(directory):
    root = Path(__file__).resolve().parents[1]
    d = json.loads((directory/'geometry.json').read_text())
    frozen = json.loads((directory/'frozen_protocol.json').read_text())
    if d['seed'] != 2026100506 or d['frozen_protocol'] != frozen:
        raise ValueError('Frozen summary disagreement')
    for group, base in [('source_sha256', root), ('input_sha256', root), ('output_sha256', directory.resolve())]:
        for name, digest in d[group].items():
            path = (base/name).resolve()
            if base.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'{group} mismatch')
    # Reconstruct design independently, not by importing the runner's case helper.
    expected_cases = {}
    for phase in ('stress', 'confirmation'):
        multipliers = [1, 1.0625, 1.1875, 1.375, 1.75, 2] if phase == 'stress' else [1, 1.0625]
        biases = [0, .0625, .1875, .3125, .4375, .5] if phase == 'stress' else [.4375, .5, .5625]
        rows = []
        for fid, floor in enumerate([.0025, .005, .01, .02]):
            for m in multipliers:
                for b in biases:
                    rows.append({'case_id': len(rows), 'floor_id': fid, 'floor': floor,
                                 'rate_multiplier': m, 'rate': round(2*floor*m, 12), 'beta': b})
        expected_cases[phase] = rows
    expected_frozen = {'seed': 2026100506, 'plan_ids': [9, 10], 'trials': {'stress': 512, 'confirmation': 4096},
                       'cases': expected_cases, 'detector_alpha': .05, 'fresh_interval_failure_budget': .05,
                       'confirmation_contrasts': 16}
    if frozen != expected_frozen or len(d['cells']) != 336:
        raise ValueError('Predeclared cases changed')
    old = json.loads((root/next(iter(d['input_sha256']))).read_text())
    plans = {p['plan_id']: p for p in old['plans']}
    cells = {(c['phase'], c['plan_id'], c['case_id']): c for c in d['cells']}
    if len(cells) != 336 or len(d['archives']) != 4:
        raise ValueError('Cell/archive duplication or omission')
    seen, files, total, sampled = set(), {'frozen_protocol.json'}, 0, 0
    for entry in d['archives']:
        phase, pid = entry['phase'], entry['plan_id']
        if phase not in expected_cases or pid not in (9, 10):
            raise ValueError('Unexpected phase/plan')
        files.add(entry['file'])
        profile = plans[pid]; trials = frozen['trials'][phase]
        with np.load(directory/entry['file'], allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        if set(arrays) != {'case_ids', 'y_cal', 'n_cal', 's_cal', 'y_mon', 'n_mon', 's_mon', 'alarm'}:
            raise ValueError('Unexpected/private public fields')
        size = len(expected_cases[phase])
        if not np.array_equal(arrays['case_ids'], np.arange(size)):
            raise ValueError('Unexpected case axis')
        for name, values in arrays.items():
            shape = (size,) if name == 'case_ids' else ((size, trials) if name == 'alarm'
                     else (size, trials, 4 if name.startswith('s_') else 6))
            if values.shape != shape or values.dtype.kind not in 'ibu':
                raise ValueError('Archive shape/type mismatch')
        for case in expected_cases[phase]:
            i = case['case_id']; key = phase, pid, i
            if key in seen or key not in cells:
                raise ValueError('Unexpected/repeated cell')
            seen.add(key); cell = cells[key]
            if any(cell[k] != v for k, v in case.items()) or cell['trials'] != trials:
                raise ValueError('Case metadata mismatch')
            cal = PublicBatch(profile['calibration_emitted'], arrays['y_cal'][i], arrays['n_cal'][i], arrays['s_cal'][i])
            mon = PublicBatch(profile['monitoring_emitted'], arrays['y_mon'][i], arrays['n_mon'][i], arrays['s_mon'][i])
            if not np.array_equal(cal.denominators[:, :4], cal.sifted):
                raise ValueError('Calibration disclosure mismatch')
            pv = central_exact_pvalues(cal, mon)
            swap = [2, 3, 0, 1, 4, 5]; ss = [2, 3, 0, 1]
            rc = PublicBatch(cal.emitted, cal.numerators[:, swap], cal.denominators[:, swap], cal.sifted[:, ss])
            rm = PublicBatch(mon.emitted, mon.numerators[:, swap], mon.denominators[:, swap], mon.sifted[:, ss])
            if not np.array_equal(central_exact_pvalues(rc, rm), pv[:, swap]):
                raise ValueError('Reflection replay mismatch')
            alarm = np.any(pv <= .05/6, axis=1)
            if not np.array_equal(alarm, arrays['alarm'][i]) or cell['alarms'] != int(alarm.sum()):
                raise ValueError('Decision count mismatch')
            close(cell['alarm_rate'], int(alarm.sum())/trials)
            close(cell['marginal_ci95'], ci(int(alarm.sum()), trials, .05))
            close(cell['simultaneous_ci95'], ci(int(alarm.sum()), trials, .05/336))
            if cell['reflection_pvalue_rows_checked'] != trials:
                raise ValueError('Reflection coverage count mismatch')
            for row in (0, trials//2, trials-1):
                _, independent = independent_decisions(cal, mon, row, .05)
                if bool(alarm[row]) != bool(independent):
                    raise ValueError('Independent scalar decision mismatch')
                sampled += 1
            total += trials
        print(f'audit {phase} plan {pid} complete', flush=True)
    if seen != set(cells) or files != set(d['output_sha256']) or d['public_pairs'] != total or total != 344064:
        raise ValueError('Output/trial coverage mismatch')
    # Independent one-sided conditional hypergeometric formula for 16 contrasts.
    contrast_index = {(c['plan_id'], c['floor_id'], c['contrast']): c for c in d['contrasts']}
    if len(contrast_index) != 16:
        raise ValueError('Missing contrast')
    for pid in (9, 10):
        for fid in range(4):
            baseline = next(c for c in d['cells'] if c['phase'] == 'confirmation' and c['plan_id'] == pid
                            and c['floor_id'] == fid and c['rate_multiplier'] == 1 and c['beta'] == .5)
            for name, rate, bias in [('off_grid_bias', 1, .4375), ('higher_rate', 1.0625, .5)]:
                candidate = next(c for c in d['cells'] if c['phase'] == 'confirmation' and c['plan_id'] == pid
                                 and c['floor_id'] == fid and c['rate_multiplier'] == rate and c['beta'] == bias)
                a, b = candidate['alarms'], baseline['alarms']; n = 4096
                p = float(hypergeom.cdf(a, 2*n, a+b, n))
                result = contrast_index[pid, fid, name]
                if result['first_case_id'] != candidate['case_id'] or result['baseline_case_id'] != baseline['case_id']:
                    raise ValueError('Contrast identity mismatch')
                close(result['difference_first_minus_balanced_floor'], (a-b)/n)
                close(result['one_sided_lower_power_p'], p)
                close(result['bonferroni_lower_power_p'], min(1., p*16))
                la, ua = candidate['simultaneous_ci95']; lb, ub = baseline['simultaneous_ci95']
                close(result['simultaneous_difference_ci95'], [la-ub, ua-lb])
    # Reconstruct all 112 covers from old probabilities and independent rel_entr.
    if len(d['continuous_covers']) != 112:
        raise ValueError('Continuous-domain coverage count mismatch')
    cover_index = {(c['plan_id'], c['floor_id'], c['region']): c for c in d['continuous_covers']}
    if len(cover_index) != 112:
        raise ValueError('Repeated continuous cover')
    pooled = {(c['plan_id'], c['policy_id']): c for c in old['pooled_search']}
    count_corners = 0
    for pid, plan in plans.items():
        for fid, floor in enumerate([.0025, .005, .01, .02]):
            rates = [round(2*floor*m, 12) for m in (1, 1.125, 1.25, 1.5, 2)]
            biases = [i/8 for i in range(9)]
            for region, end in [('near_floor', 4*floor), ('full_eligible', 1.)]:
                result = cover_index[pid, fid, region]
                if result['rate_domain'] != [2*floor, end] or result['bias_domain'] != [0., 1.] or len(result['cells']) != 45:
                    raise ValueError('Rectangle domain mismatch')
                r_edges = [2*floor]+[(a+b)/2 for a, b in zip(rates, rates[1:])]+[end]
                b_edges = [0.]+[(a+b)/2 for a, b in zip(biases, biases[1:])]+[1.]
                ids = set(); lowers = []
                gain = (-math.expm1(-.075)-math.expm1(-.015))/2
                exposure = plan['monitoring_emitted']*plan['monitoring_disclosure']
                for c in result['cells']:
                    i = c['anchor_policy_id']; policy = old['policies'][i]
                    if i in ids or policy['rate'] not in rates or policy['z_basis_probability'] not in biases:
                        raise ValueError('Anchor coverage mismatch')
                    ids.add(i); ri = rates.index(policy['rate']); bi = biases.index(policy['z_basis_probability'])
                    rb, bb = r_edges[ri:ri+2], b_edges[bi:bi+2]
                    if c['rate_bounds'] != rb or c['bias_bounds'] != bb:
                        raise ValueError('Cell edge mismatch')
                    r, beta = policy['rate'], policy['z_basis_probability']
                    p = np.array([.015+.97*r*(1-beta)/2, .015+.97*r*beta/2])
                    maxima = []
                    for rr in rb:
                        for bbeta in bb:
                            q = np.array([.015+.97*rr*(1-bbeta)/2, .015+.97*rr*bbeta/2])
                            divergence = exposure*gain/4*float((rel_entr(p, q)+rel_entr(1-p, 1-q)).sum())
                            maxima.append(max(0., divergence)); count_corners += 1
                    maximum = max(maxima); radius = min(1., math.sqrt(maximum/2))
                    close(c['maximum_corner_kl'], maximum); close(c['tv_radius'], radius)
                    lower = pooled[pid, i]['simultaneous_search_ci975'][0]
                    close(c['anchor_lower'], lower); lowers.append(max(0., lower-radius))
                lo = min(lowers)
                up = next(e['grid_minimum_joint95'][1] for e in old['envelopes'] if e['plan_id'] == pid and e['floor_id'] == fid)
                close(result['lower'], lo); close(result['upper'], up)
                if result['consistent'] != (lo <= up):
                    raise ValueError('Continuous bracket conclusion mismatch')
    return {'status': 'passed', 'public_pairs_reconciled': total, 'original_decision_replays': total,
            'reflection_pvalue_rows_replayed': total, 'independent_scalar_decisions': sampled,
            'fresh_cells_checked': 336, 'contrasts_checked': 16, 'continuous_covers_checked': 112,
            'corner_KLs_checked': count_corners, 'source_hashes_checked': len(d['source_sha256']),
            'input_hashes_checked': len(d['input_sha256']), 'output_hashes_checked': len(d['output_sha256']),
            'limits': 'Full replay shares scorers; sampled scalar APIs and independent rel_entr share SciPy. Floating bounds are not outward-rounded certificates.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('--save', action='store_true')
    args = parser.parse_args(); result = audit(args.directory)
    if args.save:
        with (args.directory/'audit.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
