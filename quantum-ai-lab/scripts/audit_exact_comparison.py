"""Audit saved public/private pairs, frozen responses and detector differences."""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import hashlib
from itertools import groupby
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest, fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import COORDINATES, PublicBatch, change_alarm
from research.exact_comparison import exact_change_alarm

RULES = ('cp', 'central_exact')


def close(a, b):
    if not np.allclose(a, b, rtol=0, atol=1e-10):
        raise ValueError('Numerical summary mismatch')


def independent_decisions(cal, mon, row_id, alpha):
    cp, exact = False, False
    for j in range(6):
        values, bounds = [], []
        for batch in (cal, mon):
            y, n = int(batch.numerators[row_id, j]), int(batch.denominators[row_id, j])
            values.append((y, n))
            b = binomtest(y, n).proportion_ci(1-alpha/12, method='exact') if n else None
            bounds.append((b.low, b.high) if b else (0., 1.))
        (clo, chi), (mlo, mhi) = bounds
        cp |= clo > mhi or mlo > chi
        (yc, nc), (ym, nm) = values
        if nc and nm:
            table = [[yc, nc-yc], [ym, nm-ym]]
            p = min(1., 2*min(fisher_exact(table, alternative='less').pvalue,
                              fisher_exact(table, alternative='greater').pvalue))
            exact |= p <= alpha/6
    return cp, exact


def audit(directory):
    summary = json.loads((directory/'comparison.json').read_text())
    frozen = json.loads((directory/'frozen_selections.json').read_text())
    if (summary['budget'] != 4000000 or summary['alpha_per_rule'] != .05
            or summary['seed'] != 2026100504 or summary['frozen_selections'] != frozen
            or tuple(summary['rules']) != RULES or summary['floors'] != [.0025, .005, .01, .02]):
        raise ValueError('Frozen protocol mismatch')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in summary['source_sha256'].items():
        path = (root/relative).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source hash mismatch')
    for name, expected in summary['output_sha256'].items():
        path = (directory/name).resolve()
        if path.parent != directory.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Output hash mismatch')
    plans = {p['plan_id']: p for p in summary['plans']}
    policies = {p['policy_id']: p for p in summary['policies']}
    if len(plans) != 14 or len(policies) != 45:
        raise ValueError('Grid size changed')
    for plan in plans.values():
        m, n, f = plan['calibration_emitted'], plan['monitoring_emitted'], plan['monitoring_disclosure']
        if m+n != 4000000 or not 0 <= f <= 1:
            raise ValueError('Budget conservation failed')
        close(m+f*n, 2000000) if plan['regime'] == 'fixed_sacrifice' else close(f, .25)
    cells = {}
    for phase, groups in summary['cells'].items():
        for cell in groups:
            key = (phase, cell['plan_id'], cell['case_id'], cell['policy_id'])
            if key in cells:
                raise ValueError('Duplicate summary cell')
            cells[key] = cell
    totals = defaultdict(int)
    visited = set()
    with gzip.open(directory/'public_counts.csv.gz', 'rt', newline='') as pub_stream, \
         gzip.open(directory/'private_knowledge.csv.gz', 'rt', newline='') as pri_stream:
        public, private = csv.DictReader(pub_stream), iter(csv.DictReader(pri_stream))
        if any('certified' in f for f in public.fieldnames):
            raise ValueError('Private fields present in public export')
        def row_key(row):
            return row['phase'], int(row['plan_id']), int(row['case_id']), int(row['policy_id'])
        for key, group in groupby(public, row_key):
            if key in visited or key not in cells:
                raise ValueError('Unknown or repeated transcript group')
            visited.add(key)
            phase, plan_id, case_id, policy_id = key
            rows, cell, plan = list(group), cells[key], plans[plan_id]
            trials = len(rows)
            if (trials != cell['trials'] or [int(r['trial']) for r in rows] != list(range(trials))
                    or any(r['selection_rule'] != cell['selection_rule'] for r in rows)):
                raise ValueError('Trial identity mismatch')
            batches = []
            for stage, emitted in [('cal', plan['calibration_emitted']), ('mon', plan['monitoring_emitted'])]:
                ys = np.array([[int(r[f'{stage}_{c}_numerator']) for c in COORDINATES] for r in rows])
                ns = np.array([[int(r[f'{stage}_{c}_denominator']) for c in COORDINATES] for r in rows])
                ss = np.array([[int(r[f'{stage}_{c}_sifted']) for c in COORDINATES[:4]] for r in rows])
                batch = PublicBatch(emitted, ys, ns, ss)
                if ((stage == 'cal' and not np.array_equal(ns[:, :4], ss))
                        or (stage == 'mon' and plan['monitoring_disclosure'] == 0 and np.any(ns[:, :4]))):
                    raise ValueError('Disclosure bookkeeping mismatch')
                batches.append(batch)
            cal, mon = batches
            scores = {'cp': change_alarm(cal, mon, .05, 'clopper-pearson'),
                      'central_exact': exact_change_alarm(cal, mon, .05)}
            for rule in RULES:
                recorded = np.array([int(r[f'alarm_{rule}']) for r in rows])
                if not np.array_equal(scores[rule], recorded):
                    raise ValueError('Decision replay mismatch')
                count = int(recorded.sum())
                if count != cell['rules'][rule]['alarms']:
                    raise ValueError('Alarm count mismatch')
                close(cell['rules'][rule]['alarm_rate'], count/trials)
                ci = binomtest(count, trials).proportion_ci(.95, method='exact')
                close(cell['rules'][rule]['alarm_rate_ci95'], [ci.low, ci.high])
                totals['decision_replays'] += trials
            cp_only = int(np.sum(scores['cp'] & ~scores['central_exact']))
            exact_only = int(np.sum(scores['central_exact'] & ~scores['cp']))
            both = int(np.sum(scores['cp'] & scores['central_exact']))
            paired = cell['paired']
            if (cp_only != paired['cp_only'] or exact_only != paired['exact_only']
                    or both != paired['both'] or trials-both-cp_only-exact_only != paired['neither']):
                raise ValueError('Paired discordances mismatch')
            close(paired['difference_exact_minus_cp'], (exact_only-cp_only)/trials)
            discordant = cp_only+exact_only
            # Exact integer combinatorial sum, not the runner's binomtest helper.
            p = float(min(Fraction(1), Fraction(2*sum(comb(discordant, j) for j in
                range(min(cp_only, exact_only)+1)), 2**discordant))) if discordant else 1.
            close(paired['mcnemar_exact_p'], p)
            for name, value in [('mean_calibration_sacrificed', cal.disclosed.mean()),
                                ('mean_monitoring_disclosed', mon.disclosed.mean()),
                                ('mean_remaining_candidates', mon.remaining_candidates.mean())]:
                close(cell[name], value)
            if phase != 'honest_control':
                known = []
                policy = policies[policy_id]
                for trial in range(trials):
                    row = next(private, None)
                    if row is None or row_key(row) != key or int(row['trial']) != trial:
                        raise ValueError('Private/public identity mismatch')
                    k = [int(row[f'certified_{c}']) for c in COORDINATES[:4]]
                    retained = mon.sifted[trial]-mon.denominators[trial, :4]
                    if (any(not 0 <= a <= b for a, b in zip(k, retained))
                            or (policy['z_basis_probability'] == 0 and any(k[:2]))
                            or (policy['z_basis_probability'] == 1 and any(k[2:]))):
                        raise ValueError('Retained knowledge labels inconsistent')
                    known.append(k)
                known = np.array(known)
                retained_total = int(mon.remaining_candidates.sum())
                if int(known.sum()) != cell['certified_retained_bits'] or retained_total != cell['retained_candidate_bits']:
                    raise ValueError('Private summary mismatch')
                if retained_total:
                    close(cell['pooled_certified_fraction'], known.sum()/retained_total)
                elif cell['pooled_certified_fraction'] is not None:
                    raise ValueError('Undefined fraction filled with a value')
                for rule in RULES:
                    if int(known[~scores[rule]].sum()) != cell['undetected_certified_bits'][rule]:
                        raise ValueError('Undetected knowledge count mismatch')
                totals['private_rows'] += trials
            if phase == 'evaluation':
                marginal, simultaneous = {}, {}
                for rule in RULES:
                    c = cell['rules'][rule]
                    ci = binomtest(c['alarms'], trials).proportion_ci(.975, method='exact')
                    marginal[rule] = [ci.low, ci.high]
                    ci = binomtest(c['alarms'], trials).proportion_ci(1-.05/224, method='exact')
                    simultaneous[rule] = [ci.low, ci.high]
                    close(c['simultaneous_ci95'], simultaneous[rule])
                for field, intervals in [('difference_ci95_conservative', marginal),
                                         ('difference_simultaneous_ci95_conservative', simultaneous)]:
                    close(paired[field], [intervals['central_exact'][0]-intervals['cp'][1],
                                          intervals['central_exact'][1]-intervals['cp'][0]])
                close(paired['mcnemar_bonferroni_p112'], min(1., p*112))
                for row_id in sorted({0, trials//2, trials-1}):
                    c, e = independent_decisions(cal, mon, row_id, .05)
                    if c != scores['cp'][row_id] or e != scores['central_exact'][row_id]:
                        raise ValueError('Independent sampled decision mismatch')
                    totals['independent_sampled_rule_decisions'] += 2
            totals['public_pairs'] += trials
            totals[phase] += trials
        if next(private, None) is not None:
            raise ValueError('Extra unmatched private rows')
    if visited != set(cells) or dict((k, totals[k]) for k in ('public_pairs', 'private_rows')) != summary['totals']:
        raise ValueError('Transcript or cell coverage mismatch')
    for phase, count, size in [('search', 630, summary['search_trials']),
                               ('evaluation', 112, summary['evaluation_trials']),
                               ('honest_control', 56, summary['control_trials'])]:
        if totals[phase] != count*size:
            raise ValueError('Predeclared phase size mismatch')
    search = {(c['plan_id'], c['policy_id']): c for c in summary['cells']['search']}
    for rule_id, rule in enumerate(RULES):
        responses = []
        for plan in summary['plans']:
            for floor_id, floor in enumerate(summary['floors']):
                allowed = {round(2*floor*m, 12) for m in (1, 1.5, 2)}
                eligible = [p for p in policies.values() if p['rate'] in allowed and p['rate']/2 >= floor]
                selected = min(eligible, key=lambda p: (search[plan['plan_id'], p['policy_id']]['rules'][rule]['alarm_rate'],
                    p['rate'], abs(p['z_basis_probability']-.5), p['z_basis_probability']))
                response = {'plan_id': plan['plan_id'], 'floor_id': floor_id, 'minimum_certified_fraction': floor,
                            'policy_id': selected['policy_id'],
                            'search_alarm_rate': search[plan['plan_id'], selected['policy_id']]['rules'][rule]['alarm_rate']}
                responses.append(response)
                cell = cells['evaluation', plan['plan_id'], rule_id*4+floor_id, selected['policy_id']]
                if cell['selection_rule'] != rule or cell['minimum_certified_fraction'] != floor:
                    raise ValueError('Selected evaluation identity mismatch')
        winners = []
        for regime in ('fixed_fraction', 'fixed_sacrifice'):
            for floor_id in range(4):
                choices = [r for r in responses if plans[r['plan_id']]['regime'] == regime and r['floor_id'] == floor_id]
                w = min(choices, key=lambda r: (-r['search_alarm_rate'],
                    abs(plans[r['plan_id']]['calibration_emitted']/4000000-1/3), plans[r['plan_id']]['calibration_emitted']))
                winners.append({'regime': regime, **w})
        if {'eve_responses': responses, 'defender_plans': winners} != frozen[rule]:
            raise ValueError('Independent selection provenance mismatch')
        totals['selection_checks'] += len(responses)+len(winners)
    return {'status': 'passed', 'public_pairs_reconciled': totals['public_pairs'],
            'private_rows_reconciled': totals['private_rows'], 'decision_replays': totals['decision_replays'],
            'independent_sampled_rule_decisions': totals['independent_sampled_rule_decisions'],
            'paired_summary_checks': len(cells), 'summary_cells_reconciled': len(cells),
            'selection_checks': totals['selection_checks'], 'source_hashes_checked': len(summary['source_sha256']),
            'limits': 'Full replay shares scorers. Independent scalar decisions cover sampled evaluation rows; conditional small-table tests are exhaustive only over their declared small domain.'}


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
