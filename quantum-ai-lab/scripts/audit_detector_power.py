"""Replay archives and independently reconstruct conditional power calculations.

Incomplete-beta/log-gamma and sampled scalar Fisher APIs provide independent
calculation paths, but share SciPy numerics; not an independent physics engine.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import betainc, gammaln
from scipy.stats import binom, binomtest, fisher_exact, hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import PublicBatch
from research.exact_comparison import exact_change_alarm, central_exact_pvalues
from run_allocation_frontier import plans


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, name, atol=2e-10):
    require(np.allclose(actual, expected, rtol=2e-10, atol=atol), name)


def independent_endpoint(values, delta, side):
    n = len(values)
    mean = math.fsum(values)/n
    variance = math.fsum((float(v)-mean)**2 for v in values)/(n-1)
    radius = math.sqrt(2*variance*math.log(2/delta)/n)+7*math.log(2/delta)/(3*(n-1))
    endpoint = max(0., mean-radius) if side == 'lower' else min(1., mean+radius)
    return mean, variance, radius, endpoint


def independent_components(k, q, weights):
    """Use regularized incomplete-beta identities, not binom.cdf/sf helpers."""
    n = k['n_mon'][:, None]
    u, l = k['high_cutoff'], k['low_cutoff']
    q = np.tile(q, len(n)//4)[:, None]
    high = np.zeros(u.shape)
    mask = (u >= 1) & (u <= n)
    qq, nn = np.broadcast_to(q, u.shape), np.broadcast_to(n, u.shape)
    high[mask] = betainc(u[mask], nn[mask]-u[mask]+1, qq[mask])
    high[u <= 0] = 1.
    low = np.zeros(l.shape)
    mask = (l >= 0) & (l < n)
    low[mask] = betainc(nn[mask]-l[mask], l[mask]+1, 1-qq[mask])
    low[l >= n] = 1.
    return (np.sum(weights*high, axis=1).reshape(-1, 4),
            np.sum(weights*low, axis=1).reshape(-1, 4))


def joint(coordinates, gains):
    return np.where(gains, 1., 1-np.prod(1-np.minimum(coordinates, 1.), axis=1))


def audit(path):
    root = Path(__file__).resolve().parents[1]
    s = json.loads((path/'power.json').read_text())
    require(not (path/'audit.json').exists(), 'Audit output exists; preserve it')
    for group, base in [('source_sha256', root), ('output_sha256', path)]:
        for name, digest in s[group].items():
            require(hashlib.sha256((base/name).read_bytes()).hexdigest() == digest, f'Hash mismatch: {name}')
    p = s['protocol']
    require(json.loads((path/'frozen_protocol.json').read_text()) == p, 'Frozen protocol mismatch')
    require(p['seed'] == 2026100907 and p['plan_ids'] == [9, 10] and p['bias_cells'] == 16,
            'Unexpected frozen design')
    require(p['mean_functions'] == 144 and p['direct_cells'] == 10, 'Confidence multiplicity')
    require(p['mean_failure_budget']+p['direct_failure_budget'] == .05, 'Joint confidence budget')
    profiles = {v['plan_id']: v['plan'] for v in plans()}
    checks = {'ancillary_pairs': 0, 'boundary_entries': 0, 'scalar_fisher_boundaries': 0,
              'conditional_function_rows': 0, 'mean_endpoints': 0, 'envelopes': 0,
              'direct_public_pairs': 0, 'direct_decision_replays': 0,
              'sampled_scalar_direct_decisions': 0}
    for pid in p['plan_ids']:
        plan = profiles[pid]
        a = dict(np.load(path/f'ancillary_plan_{pid}.npz', allow_pickle=False))
        k = dict(np.load(path/f'kernel_plan_{pid}.npz', allow_pickle=False))
        functions = dict(np.load(path/f'conditional_plan_{pid}.npz', allow_pickle=False))
        t = len(a['gain_alarm'])
        require(t == p['ancillary_trials'], 'Ancillary count')
        # Validate full bookkeeping through the public count type, with errors
        # explicitly absent rather than claiming the zero placeholders were drawn.
        batches = []
        for stage, emitted in [('cal', plan.calibration_emitted), ('mon', plan.monitoring_emitted)]:
            y = np.column_stack((np.zeros((t, 4), dtype=np.int64), a[f'detected_{stage}']))
            batches.append(PublicBatch(emitted, y, a[f'n_{stage}'], a[f's_{stage}']))
        gain = np.any(central_exact_pvalues(*batches)[:, 4:] <= p['alpha']/6, axis=1)
        require(np.array_equal(gain, a['gain_alarm']), 'Ancillary gain decisions')
        require(np.array_equal(k['n_cal'], a['n_cal'][:, :4].ravel()), 'Calibration denominator binding')
        require(np.array_equal(k['n_mon'], a['n_mon'][:, :4].ravel()), 'Monitoring denominator binding')
        nc, nm = k['n_cal'][:, None], k['n_mon'][:, None]
        values = k['calibration_values'][None, :]
        weights = np.exp(gammaln(nc+1)-gammaln(values+1)-gammaln(nc-values+1)
                         +values*math.log(.015)+(nc-values)*math.log1p(-.015))
        weights = np.where(values <= nc, weights, 0.)
        near(k['weights'], weights, 'Log-gamma calibration weights', atol=2e-11)
        missing = binom.cdf(int(values[0, 0])-1, nc[:, 0], .015)+binom.sf(int(values[0, -1]), nc[:, 0], .015)
        near(k['missing_mass'], missing, 'Omitted calibration mass', atol=1e-16)
        require(np.max(missing) <= p['calibration_truncation']*1.01, 'Truncation budget')
        u, l = k['high_cutoff'], k['low_cutoff']
        m, n, av = np.broadcast_arrays(nc, nm, values)
        valid = (m > 0) & (n > 0) & (av <= m)
        require(np.all(l < u), 'Tail overlap')
        for cutoff, high_side in [(u, True), (l, False)]:
            inside = valid & (cutoff >= 0) & (cutoff <= n)
            neighbors = cutoff-1 if high_side else cutoff+1
            interior = valid & (neighbors >= 0) & (neighbors <= n)
            cdf = (lambda b, mask: hypergeom.cdf(av[mask], m[mask]+n[mask], av[mask]+b[mask], m[mask])) if high_side else (
                   lambda b, mask: hypergeom.sf(av[mask]-1, m[mask]+n[mask], av[mask]+b[mask], m[mask]))
            require(np.all(cdf(cutoff, inside) <= p['alpha']/12), 'Stored rejection boundary')
            require(np.all(cdf(neighbors, interior) > p['alpha']/12), 'Adjacent nonrejection boundary')
            flat = np.flatnonzero(inside)
            for index in flat[np.linspace(0, len(flat)-1, 64, dtype=int)]:
                row, col = np.unravel_index(index, cutoff.shape)
                aa, bb = int(av[row, col]), int(cutoff[row, col])
                mm, nn = int(m[row, col]), int(n[row, col])
                tail = fisher_exact([[aa, mm-aa], [bb, nn-bb]],
                                    alternative='less' if high_side else 'greater').pvalue
                require(tail <= p['alpha']/12+1e-14, 'Independent scalar Fisher boundary')
                checks['scalar_fisher_boundaries'] += 1
        checks['boundary_entries'] += int(u.size+l.size)
        checks['ancillary_pairs'] += t
        for e in (r for r in s['envelopes'] if r['plan_id'] == pid):
            fid, floor = e['floor_id'], e['floor']
            require(e['rate_domain'] == [2*floor, 1.] and e['bias_domain'] == [0., 1.], 'Full continuous domain')
            endpoints = []
            for cell in e['cells']:
                cid = cell['cell_id']
                blo, bhi = cell['bias_bounds']
                require([blo, bhi] == [cid/32, (cid+1)/32], 'Frozen bias partition')
                qz, qx = .015+.485*2*floor*(1-bhi), .015+.485*2*floor*blo
                high, _ = independent_components(k, [qz, qz, qx, qx], weights)
                expected = joint(high, gain)
                actual = functions['cell_lower'][fid*16+cid]
                near(actual, expected, 'Incomplete-beta continuous lower function')
                for key, value in zip(('mean', 'sample_variance', 'radius', 'endpoint'),
                                      independent_endpoint(actual, .04/144, 'lower')):
                    near(cell[key], value, f'Independent cell {key}')
                endpoints.append(cell['endpoint']); checks['mean_endpoints'] += 1
                checks['conditional_function_rows'] += t
            q = .015+.485*floor
            high, low = independent_components(k, [q]*4, weights)
            for side, actual, expected in [
                    ('lower', functions['balanced_lower'][fid], joint(high+low, gain)),
                    ('upper', functions['balanced_upper'][fid], joint(high+low+missing.reshape(-1, 4), gain))]:
                near(actual, expected, 'Incomplete-beta balanced full power')
                for key, value in zip(('mean', 'sample_variance', 'radius', 'endpoint'),
                                      independent_endpoint(actual, .04/144, side)):
                    near(e[f'balanced_{side}'][key], value, f'Independent point {key}')
                checks['mean_endpoints'] += 1; checks['conditional_function_rows'] += t
            lower, upper = min(endpoints), e['balanced_upper']['endpoint']
            near(e['continuous_minimum_joint95'], [lower, upper], 'Continuous envelope')
            near(e['balanced_regret_upper_joint95'], upper-lower, 'Regret upper bound')
            require(e['balanced_within_five_points'] == (upper-lower <= .05), 'Regret gate')
            checks['envelopes'] += 1
        print(f'plan {pid}: all conditional calculations audited', flush=True)
    for r in s['direct_validation']:
        pid, case = r['plan_id'], r['case_id']
        plan = profiles[pid]
        data = dict(np.load(path/f'direct_plan_{pid}_case_{case}.npz', allow_pickle=False))
        c = PublicBatch(plan.calibration_emitted, data['y_cal'], data['n_cal'], data['s_cal'])
        m = PublicBatch(plan.monitoring_emitted, data['y_mon'], data['n_mon'], data['s_mon'])
        decisions = exact_change_alarm(c, m)
        require(np.array_equal(decisions, data['alarm']), 'Direct full-decision replay')
        count = int(decisions.sum()); n = len(decisions)
        require(count == r['alarms'] and n == r['trials'] == p['direct_trials'], 'Direct rate counts')
        interval = binomtest(count, n).proportion_ci(confidence_level=1-.01/10, method='exact')
        near(r['simultaneous_ci99'], [interval.low, interval.high], 'Independent direct CI API')
        if r['floor_id'] is not None:
            e = next(e for e in s['envelopes'] if e['plan_id'] == pid and e['floor_id'] == r['floor_id'])
            lo, hi = e['balanced_point_joint95']
            overlap = max(lo, interval.low) <= min(hi, interval.high)
            require(r['analytic_interval_overlap'] == overlap, 'Direct/analytic consistency flag')
        for index in np.linspace(0, n-1, 12, dtype=int):
            pvalues = []
            for j in range(6):
                aa, mm, bb, nn = (int(x) for x in (c.numerators[index, j], c.denominators[index, j],
                                                   m.numerators[index, j], m.denominators[index, j]))
                if not mm or not nn:
                    pvalues.append(1.)
                else:
                    table = [[aa, mm-aa], [bb, nn-bb]]
                    pvalues.append(min(1., 2*min(fisher_exact(table, alternative='less').pvalue,
                                                 fisher_exact(table, alternative='greater').pvalue)))
            require(decisions[index] == (min(pvalues) <= .05/6), 'Sampled scalar direct decision')
            checks['sampled_scalar_direct_decisions'] += 1
        checks['direct_public_pairs'] += n; checks['direct_decision_replays'] += n
    require(checks['mean_endpoints'] == 144 and checks['envelopes'] == 8, 'Coverage totals')
    result = {'passed': True, 'checks': checks,
              'all_direct_analytic_intervals_overlap': all(r.get('analytic_interval_overlap', True)
                                                          for r in s['direct_validation']),
              'limits': ['Boundary and power checks share SciPy numerical infrastructure.',
                         'No independent optical/hardware model or outward-rounded certificate.']}
    (path/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
