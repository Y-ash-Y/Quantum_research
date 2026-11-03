"""Stream saved public/private marginals; audit selection and held-out summaries."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import COORDINATES, PublicBatch, change_alarm


def close(a, b, tolerance=1e-10):
    if not np.allclose(a, b, rtol=0, atol=tolerance):
        raise ValueError('Numerical reconciliation mismatch')


def categorical_bound(plan, rate, bias):
    """Independent categorical likelihood calculation; no research KL helpers."""
    m, n, f = plan['calibration_emitted'], plan['monitoring_emitted'], plan['monitoring_disclosure']
    gain = (1-math.exp(-.075)+1-math.exp(-.015))/2
    e = n*f
    qa = (.015+.97*rate*(1-bias)/2, .015+.97*rate*bias/2)
    h = [(m*.015+e*q)/(m+e) for q in qa] if m+e else [.015, .015]
    def probabilities(qs, frac):
        p = [1-gain, gain/2, gain*(1-frac)/2]
        for q in (qs[0], qs[0], qs[1], qs[1]):
            p += [gain*frac*(1-q)/8, gain*frac*q/8]
        return p
    def kl(ps, qs):
        return math.fsum(p*math.log(p/q) for p, q in zip(ps, qs) if p)
    divergence = max(0., m*kl(probabilities((.015, .015), 1), probabilities(h, 1))+
                     n*kl(probabilities(qa, f), probabilities(h, f)))
    return divergence, min(1., .05+math.sqrt(divergence/2))


def audit(directory):
    summary = json.loads((directory/'frontier.json').read_text())
    selections = json.loads((directory/'frozen_selections.json').read_text())
    if selections != summary['frozen_selections'] or summary['budget'] != 4000000:
        raise ValueError('Frozen choices or budget changed')
    for name, expected in summary['output_sha256'].items():
        path = (directory/name).resolve()
        if path.parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Output hash mismatch')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in summary['source_sha256'].items():
        path = (root/relative).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source version hash mismatch')
    plans = {p['plan_id']: p for p in summary['plans']}
    policies = {p['policy_id']: p for p in summary['policies']}
    if len(plans) != 14 or len(policies) != 45:
        raise ValueError('Frozen plan/policy grid changed')
    for plan in plans.values():
        m, n, f = plan['calibration_emitted'], plan['monitoring_emitted'], plan['monitoring_disclosure']
        if m+n != summary['budget'] or not 0 <= f <= 1:
            raise ValueError('Emission budget mismatch')
        if plan['regime'] == 'fixed_sacrifice':
            close(m+n*f, 2000000, 1e-7)
        elif plan['regime'] == 'fixed_fraction':
            close(f, .25)
        else:
            raise ValueError('Unknown regime')
        gain = (-math.expm1(-.075)-math.expm1(-.015))/2
        expected = {'calibration_sacrificed': m*gain/2, 'monitoring_disclosed': n*f*gain/2,
                    'total_sacrificed': (m+n*f)*gain/2, 'remaining_candidates': n*(1-f)*gain/2}
        for name in expected:
            close(plan['expected_resources'][name], expected[name])
    cell_index = {}
    for phase, cells in summary['cells'].items():
        for cell in cells:
            key = (phase, cell['plan_id'], cell['case_id'], cell['policy_id'])
            if key in cell_index:
                raise ValueError('Duplicate summary cell')
            cell_index[key] = cell
    decisions = public_rows = private_rows = independent = 0
    audited_keys = set()
    phase_rows = defaultdict(int)
    with gzip.open(directory/'public_counts.csv.gz', 'rt', newline='') as pub_stream, \
         gzip.open(directory/'private_knowledge.csv.gz', 'rt', newline='') as private_stream:
        pub = csv.DictReader(pub_stream)
        private = iter(csv.DictReader(private_stream))
        if any('certified' in field for field in pub.fieldnames):
            raise ValueError('Private knowledge leaked into public export')
        def key_for(row):
            return row['phase'], int(row['plan_id']), int(row['case_id']), int(row['policy_id'])
        for key, group in groupby(pub, key_for):
            if key in audited_keys or key not in cell_index:
                raise ValueError('Repeated or unrecognized public group')
            audited_keys.add(key)
            rows = list(group)
            phase, plan_id, case_id, policy_id = key
            plan, cell = plans[plan_id], cell_index[key]
            if [int(r['trial']) for r in rows] != list(range(cell['trials'])):
                raise ValueError('Trial IDs missing or reordered')
            batches = []
            for stage, emitted in [('cal', plan['calibration_emitted']), ('mon', plan['monitoring_emitted'])]:
                ys, ns, ss = [], [], []
                for row in rows:
                    y = [int(row[f'{stage}_{j}_numerator']) for j in COORDINATES]
                    ns_row = [int(row[f'{stage}_{j}_denominator']) for j in COORDINATES]
                    s = [int(row[f'{stage}_{j}_sifted']) for j in COORDINATES[:4]]
                    if (any(not 0 <= a <= b for a, b in zip(y, ns_row))
                            or any(not 0 <= a <= b for a, b in zip(ns_row[:4], s))
                            or sum(ns_row[4:]) != emitted or sum(s) > sum(y[4:])
                            or (stage == 'cal' and ns_row[:4] != s)):
                        raise ValueError('Public-count conservation failed')
                    ys.append(y); ns.append(ns_row); ss.append(s)
                batches.append(PublicBatch(emitted, np.array(ys), np.array(ns), np.array(ss)))
            cal, mon = batches
            recorded = np.array([int(row['alarm']) for row in rows])
            replay = change_alarm(cal, mon, summary['alpha'], summary['method'])
            if not np.array_equal(replay, recorded):
                raise ValueError('Alarm replay mismatch')
            trials, count = len(rows), int(recorded.sum())
            if trials != cell['trials'] or count != cell['alarms']:
                raise ValueError('Summary count mismatch')
            close(cell['alarm_rate'], count/trials)
            ci = binomtest(count, trials).proportion_ci(.95, method='exact')
            close(cell['alarm_rate_ci95'], [ci.low, ci.high])
            for name, value in [('mean_calibration_sacrificed', cal.disclosed.mean()),
                                ('mean_monitoring_disclosed', mon.disclosed.mean()),
                                ('mean_remaining_candidates', mon.remaining_candidates.mean())]:
                close(cell[name], value)
            if phase != 'honest_control':
                known = []
                for trial in range(trials):
                    row = next(private, None)
                    if row is None or key_for(row) != key or int(row['trial']) != trial:
                        raise ValueError('Private/public identity mismatch')
                    k = [int(row[f'certified_{j}']) for j in COORDINATES[:4]]
                    retained = mon.sifted[trial]-mon.denominators[trial, :4]
                    if any(not 0 <= a <= b for a, b in zip(k, retained)):
                        raise ValueError('Certified labels exceed retained-only bits')
                    policy = policies[policy_id]
                    if ((policy['z_basis_probability'] == 0 and any(k[:2]))
                            or (policy['z_basis_probability'] == 1 and any(k[2:]))):
                        raise ValueError('Certified basis label mismatch')
                    known.append(k)
                known = np.array(known)
                retained_total = int(mon.remaining_candidates.sum())
                if (int(known.sum()) != cell['certified_retained_bits']
                        or retained_total != cell['retained_candidate_bits']
                        or int(known[~replay].sum()) != cell['undetected_certified_bits']):
                    raise ValueError('Private summary reconciliation failed')
                if retained_total:
                    close(cell['pooled_certified_fraction'], known.sum()/retained_total)
                elif cell['pooled_certified_fraction'] is not None:
                    raise ValueError('Undefined information fraction was filled')
                private_rows += trials
            if phase == 'evaluation':
                simultaneous = binomtest(count, trials).proportion_ci(1-.05/56, method='exact')
                close(cell['simultaneous_ci95'], [simultaneous.low, simultaneous.high])
                floor = cell['minimum_certified_fraction']
                if policies[policy_id]['rate']/2 < floor:
                    raise ValueError('Ineligible Eve selected')
                for name, rate, bias in [('selected_policy_bound', policies[policy_id]['rate'], policies[policy_id]['z_basis_probability']),
                                          ('balanced_floor_bound', 2*floor, .5)]:
                    div, ceiling = categorical_bound(plan, rate, bias)
                    close(cell[name]['joint_kl_nats'], div, 1e-8)
                    close(cell[name]['power_ceiling'], ceiling, 1e-8)
                # Independent interface on three predeclared rows per evaluation
                # cell; full replay is explicitly not independent of its scorer.
                for row_id in sorted({0, trials//2, trials-1}):
                    flag = False
                    for j in range(6):
                        bounds = []
                        for batch in batches:
                            n, k = int(batch.denominators[row_id, j]), int(batch.numerators[row_id, j])
                            if n:
                                b = binomtest(k, n).proportion_ci(1-summary['alpha']/12, method='exact')
                                bounds.append((b.low, b.high))
                            else:
                                bounds.append((0., 1.))
                        (clo, chi), (mlo, mhi) = bounds
                        flag |= clo > mhi or mlo > chi
                    if flag != bool(recorded[row_id]):
                        raise ValueError('Independent sampled decision mismatch')
                    independent += 1
            public_rows += trials
            decisions += trials
            phase_rows[phase] += trials
        if next(private, None) is not None:
            raise ValueError('Unmatched private rows')
    if audited_keys != set(cell_index) or public_rows != summary['totals']['public_pairs'] or private_rows != summary['totals']['private_rows']:
        raise ValueError('Rows or summary cells missing')
    for phase, size, expected_groups in [('search', summary['search_trials'], 630),
                                         ('evaluation', summary['evaluation_trials'], 56),
                                         ('honest_control', summary['control_trials'], 56)]:
        if phase_rows[phase] != size*expected_groups:
            raise ValueError('Predeclared phase size mismatch')
    search_index = {(c['plan_id'], c['policy_id']): c for c in summary['cells']['search']}
    independent_responses = []
    for plan in summary['plans']:
        for floor_id, floor in enumerate(summary['floors']):
            rates = {round(2*floor*v, 12) for v in (1, 1.5, 2)}
            candidates = [p for p in summary['policies'] if p['rate'] in rates and p['rate']/2 >= floor]
            selected = min(candidates, key=lambda p: (search_index[plan['plan_id'], p['policy_id']]['alarm_rate'],
                p['rate'], abs(p['z_basis_probability']-.5), p['z_basis_probability']))
            response = {'plan_id': plan['plan_id'], 'floor_id': floor_id, 'minimum_certified_fraction': floor,
                        'policy_id': selected['policy_id'],
                        'search_alarm_rate': search_index[plan['plan_id'], selected['policy_id']]['alarm_rate']}
            independent_responses.append(response)
            cell = cell_index['evaluation', plan['plan_id'], floor_id, selected['policy_id']]
            if cell['minimum_certified_fraction'] != floor:
                raise ValueError('Evaluation floor mismatch')
    if independent_responses != selections['eve_responses']:
        raise ValueError('Search selection provenance mismatch')
    independent_winners = []
    for regime in ('fixed_fraction', 'fixed_sacrifice'):
        for floor_id in range(4):
            candidates = [r for r in independent_responses if plans[r['plan_id']]['regime'] == regime and r['floor_id'] == floor_id]
            winner = min(candidates, key=lambda r: (-r['search_alarm_rate'],
                abs(plans[r['plan_id']]['calibration_emitted']/4000000-1/3),
                plans[r['plan_id']]['calibration_emitted']))
            independent_winners.append({'regime': regime, **winner})
    if independent_winners != selections['defender_plans']:
        raise ValueError('Defender selection provenance mismatch')
    return {'status': 'passed', 'public_pairs_reconciled': public_rows,
            'private_rows_reconciled': private_rows, 'decision_replays': decisions,
            'independent_sampled_decisions': independent, 'summary_cells_reconciled': len(cell_index),
            'selection_checks': len(independent_responses)+len(independent_winners),
            'independent_categorical_bound_checks': 112,
            'source_hashes_checked': len(summary['source_sha256']),
            'limits': 'Full replay shares the scorer; independent decisions cover sampled evaluation rows. No global best-response proof.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    result = audit(args.directory)
    if args.save:
        with (args.directory/'audit.json').open('x') as stream:
            stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
