"""Reconcile numeric archives, independent searches and finite-grid envelopes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import PublicBatch, change_alarm
from research.exact_comparison import exact_change_alarm
from audit_exact_comparison import independent_decisions


def close(a, b):
    if not np.allclose(a, b, rtol=0, atol=1e-10):
        raise ValueError('Numerical reconciliation mismatch')


def interval(count, trials, alpha):
    b = binomtest(count, trials).proportion_ci(1-alpha, method='exact')
    return [b.low, b.high]


def audit(directory):
    d = json.loads((directory/'stability.json').read_text())
    frozen = json.loads((directory/'frozen_selections.json').read_text())
    root = Path(__file__).resolve().parents[1]
    if (d['budget'] != 4000000 or d['seed'] != 2026100505 or d['alpha_per_rule'] != .05
            or d['floors'] != [.0025, .005, .01, .02] or d['frozen_selections'] != frozen):
        raise ValueError('Frozen protocol mismatch')
    for kind in ('source_sha256', 'input_sha256', 'output_sha256'):
        base = directory.resolve() if kind == 'output_sha256' else root.resolve()
        for relative, expected in d[kind].items():
            path = (base/relative).resolve()
            if base not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{kind} mismatch')
    plans = {p['plan_id']: p for p in d['plans']}
    policies = {p['policy_id']: p for p in d['policies']}
    rates = sorted({round(2*k*m, 12) for k in d['floors'] for m in (1, 1.125, 1.25, 1.5, 2)})
    expected_policies = [(r, b/8) for r in rates for b in range(9)]
    if len(plans) != 14 or [(p['rate'], p['z_basis_probability']) for p in d['policies']] != expected_policies:
        raise ValueError('Expanded grid changed')
    for plan in plans.values():
        m, n, f = plan['calibration_emitted'], plan['monitoring_emitted'], plan['monitoring_disclosure']
        if m+n != 4000000 or not 0 <= f <= 1:
            raise ValueError('Budget mismatch')
        close(m+f*n, 2000000) if plan['regime'] == 'fixed_sacrifice' else close(f, .25)
    cells = {}
    for phase, rows in d['cells'].items():
        for cell in rows:
            key = phase, cell['plan_id'], cell['case_id']
            if key in cells:
                raise ValueError('Duplicate cell')
            cells[key] = cell
    observed, totals = set(), defaultdict(int)
    eval_count = len(d['cells']['evaluation'])
    files = {'frozen_selections.json'}
    for archive in d['archives']:
        phase, plan_id = archive['phase'], archive['plan_id']
        plan = plans[plan_id]
        files.add(archive['public_file'])
        private = None
        if phase == 'evaluation':
            files.add(archive['private_file'])
            private = np.load(directory/archive['private_file'], allow_pickle=False)
            if set(private.files) != {'case_ids', 'policy_ids', 'certified'}:
                raise ValueError('Unexpected private archive fields')
        with np.load(directory/archive['public_file'], allow_pickle=False) as public:
            expected_fields = {'case_ids', 'policy_ids', 'y_cal', 'n_cal', 's_cal', 'y_mon', 'n_mon', 's_mon', 'alarm_central_exact'}
            if not phase.startswith('search'):
                expected_fields.add('alarm_cp')
            if set(public.files) != expected_fields:
                raise ValueError('Unexpected or private public-archive fields')
            case_ids = public['case_ids'].tolist()
            if case_ids != archive['case_ids'] or case_ids != list(range(len(case_ids))):
                raise ValueError('Archive case mapping mismatch')
            if private is not None and (not np.array_equal(public['case_ids'], private['case_ids'])
                    or not np.array_equal(public['policy_ids'], private['policy_ids'])):
                raise ValueError('Private/public case mapping mismatch')
            arrays = {name: public[name] for name in public.files}
            if any(a.dtype.kind not in 'ibu' for a in arrays.values()):
                raise ValueError('Nonintegral public array')
            trials = (d['search_trials_per_replica'] if phase.startswith('search') else
                      d['evaluation_trials'] if phase == 'evaluation' else d['control_trials'])
            for name, values in arrays.items():
                shape = ((len(case_ids),) if name in ('case_ids', 'policy_ids') else
                         (len(case_ids), trials) if name.startswith('alarm_') else
                         (len(case_ids), trials, 4 if name.startswith('s_') else 6))
                if values.shape != shape:
                    raise ValueError('Unexpected archive axis or unaccounted rows')
            if private is not None and private['certified'].shape != (len(case_ids), trials, 4):
                raise ValueError('Unexpected private archive shape')
            for axis, case_id in enumerate(case_ids):
                key = phase, plan_id, case_id
                if key in observed or key not in cells:
                    raise ValueError('Repeated or unknown archive cell')
                observed.add(key)
                cell = cells[key]
                if int(arrays['policy_ids'][axis]) != cell['policy_id']:
                    raise ValueError('Policy axis mismatch')
                batches = []
                for stage, emitted in [('cal', plan['calibration_emitted']), ('mon', plan['monitoring_emitted'])]:
                    batch = PublicBatch(emitted, arrays[f'y_{stage}'][axis], arrays[f'n_{stage}'][axis], arrays[f's_{stage}'][axis])
                    if (len(batch.numerators) != cell['trials'] or
                            (stage == 'cal' and not np.array_equal(batch.denominators[:, :4], batch.sifted)) or
                            (stage == 'mon' and plan['monitoring_disclosure'] == 0 and np.any(batch.denominators[:, :4]))):
                        raise ValueError('Disclosure/count conservation mismatch')
                    batches.append(batch)
                cal, mon = batches
                scores = {'central_exact': exact_change_alarm(cal, mon)}
                if not phase.startswith('search'):
                    scores['cp'] = change_alarm(cal, mon, .05, 'clopper-pearson')
                for rule, replay in scores.items():
                    if not np.array_equal(replay, arrays[f'alarm_{rule}'][axis]):
                        raise ValueError('Full decision replay mismatch')
                    count = int(replay.sum())
                    result = cell if phase.startswith('search') else cell['rules'][rule]
                    if count != result['alarms']:
                        raise ValueError('Alarm count mismatch')
                    close(result['alarm_rate'], count/cell['trials'])
                    if not phase.startswith('search'):
                        close(result['marginal_ci95'], interval(count, cell['trials'], .05))
                    if phase == 'evaluation':
                        close(result['simultaneous_evaluation_ci975'], interval(count, cell['trials'], .025/(2*eval_count)))
                    totals['decision_replays'] += cell['trials']
                if not phase.startswith('search'):
                    cp_only = int(np.sum(scores['cp'] & ~scores['central_exact']))
                    exact_only = int(np.sum(scores['central_exact'] & ~scores['cp']))
                    both = int(np.sum(scores['cp'] & scores['central_exact']))
                    pair = cell['paired']
                    if (pair['cp_only'] != cp_only or pair['exact_only'] != exact_only or pair['both'] != both
                            or pair['neither'] != cell['trials']-both-cp_only-exact_only):
                        raise ValueError('Paired counts mismatch')
                    close(pair['difference_exact_minus_cp'], (exact_only-cp_only)/cell['trials'])
                    discordant = cp_only+exact_only
                    # Independent central symmetric binomial-tail formula.
                    p = min(1., 2*float(binomtest(min(cp_only, exact_only), discordant, .5,
                                                alternative='less').pvalue)) if discordant else 1.
                    close(pair['mcnemar_exact_p'], p)
                    if phase == 'evaluation':
                        close(pair['mcnemar_bonferroni_p'], min(1., p*eval_count))
                for name, value in [('mean_remaining_candidates', mon.remaining_candidates.mean()),
                                    ('mean_calibration_sacrificed', cal.disclosed.mean()),
                                    ('mean_monitoring_disclosed', mon.disclosed.mean())]:
                    close(cell[name], value)
                if private is not None:
                    known = private['certified'][axis]
                    retained = mon.sifted-mon.denominators[:, :4]
                    policy = policies[cell['policy_id']]
                    if (known.shape != retained.shape or known.dtype.kind not in 'iu'
                            or np.any(known < 0) or np.any(known > retained)
                            or (policy['z_basis_probability'] == 0 and np.any(known[:, :2]))
                            or (policy['z_basis_probability'] == 1 and np.any(known[:, 2:]))):
                        raise ValueError('Retained-only knowledge labels mismatch')
                    if int(known.sum()) != cell['certified_retained_bits'] or int(retained.sum()) != cell['retained_candidate_bits']:
                        raise ValueError('Private count mismatch')
                    if retained.sum():
                        close(cell['pooled_certified_fraction'], known.sum()/retained.sum())
                    elif cell['pooled_certified_fraction'] is not None:
                        raise ValueError('Undefined proxy replaced')
                    totals['private_rows'] += cell['trials']
                    for row in sorted({0, cell['trials']//2, cell['trials']-1}):
                        cp, exact = independent_decisions(cal, mon, row, .05)
                        if cp != scores['cp'][row] or exact != scores['central_exact'][row]:
                            raise ValueError('Independent sampled decision mismatch')
                        totals['independent_sampled_rule_decisions'] += 2
                totals['public_pairs'] += cell['trials']
                totals[phase] += cell['trials']
        if private is not None:
            private.close()
        print(f"audit {phase} plan {plan_id} complete", flush=True)
    if files != set(d['output_sha256']) or observed != set(cells):
        raise ValueError('Missing archive or cell coverage')
    if {k: totals[k] for k in ('public_pairs', 'private_rows')} != d['totals']:
        raise ValueError('Trial totals mismatch')
    for phase, cases, trials in [('search_a', 2142, d['search_trials_per_replica']),
                                 ('search_b', 2142, d['search_trials_per_replica']),
                                 ('evaluation', eval_count, d['evaluation_trials']),
                                 ('honest_control', 56, d['control_trials'])]:
        if totals[phase] != cases*trials:
            raise ValueError('Predeclared phase size mismatch')
    search = {phase: {(c['plan_id'], c['policy_id']): c for c in d['cells'][phase]}
              for phase in ('search_a', 'search_b')}
    pooled = {(c['plan_id'], c['policy_id']): c for c in d['pooled_search']}
    if len(pooled) != 2142:
        raise ValueError('Missing pooled search cells')
    for key, cell in pooled.items():
        count = sum(search[phase][key]['alarms'] for phase in search)
        trials = 2*d['search_trials_per_replica']
        if cell['alarms'] != count or cell['trials'] != trials:
            raise ValueError('Pooled search counts mismatch')
        close(cell['alarm_rate'], count/trials)
        close(cell['simultaneous_search_ci975'], interval(count, trials, .025/2142))
    old = json.loads((root/'results/benchmarks/exact-comparison-v1/comparison.json').read_text())
    old_grid = {p['policy_id']: p for p in old['policies']}
    old_responses = {(r['plan_id'], r['floor_id']): old_grid[r['policy_id']]
                     for r in old['frozen_selections']['central_exact']['eve_responses']}
    reconstructed = []
    for plan in d['plans']:
        for fid, floor in enumerate(d['floors']):
            rates = {round(2*floor*m, 12) for m in (1, 1.125, 1.25, 1.5, 2)}
            ids = [i for i, p in policies.items() if p['rate'] in rates and p['rate']/2 >= floor]
            if len(ids) != 45:
                raise ValueError('Candidate eligibility mismatch')
            pid = plan['plan_id']
            def ties(i):
                p = policies[i]
                return p['rate'], abs(p['z_basis_probability']-.5), p['z_basis_probability']
            a = sorted(ids, key=lambda i: (search['search_a'][pid, i]['alarms'], ties(i)))
            b = sorted(ids, key=lambda i: (search['search_b'][pid, i]['alarms'], ties(i)))
            selected = min(ids, key=lambda i: (pooled[pid, i]['alarms'], ties(i)))
            previous = next(i for i in ids if policies[i]['rate'] == old_responses[pid, fid]['rate']
                            and policies[i]['z_basis_probability'] == old_responses[pid, fid]['z_basis_probability'])
            probes = [i for i in ids if policies[i]['rate'] == round(2*floor, 12)
                      and policies[i]['z_basis_probability'] in (0, .5, 1)]
            shortlist = sorted(set(a[:2]+b[:2]+[selected, previous]+probes))
            response = {'floor': floor, 'winner_a': a[0], 'winner_b': b[0], 'winner_pooled': selected,
                        'previous_policy': previous, 'shortlist': shortlist,
                        'independent_searches_agree': a[0] == b[0], 'plan_id': pid, 'floor_id': fid,
                        'pooled_search_alarms': pooled[pid, selected]['alarms']}
            reconstructed.append(response)
            evaluated = [c for c in d['cells']['evaluation'] if c['plan_id'] == pid and c['floor_id'] == fid]
            if [c['policy_id'] for c in evaluated] != shortlist or any(c['floor'] != floor for c in evaluated):
                raise ValueError('Frozen shortlist evaluation mismatch')
            lower = min(pooled[pid, i]['simultaneous_search_ci975'][0] for i in ids)
            upper = min(c['rules']['central_exact']['simultaneous_evaluation_ci975'][1] for c in evaluated)
            selected_cell = next(c for c in evaluated if c['policy_id'] == selected)
            consistent = lower <= upper
            regret = max(0., selected_cell['rules']['central_exact']['simultaneous_evaluation_ci975'][1]-lower) if consistent else None
            env = next(e for e in d['envelopes'] if e['plan_id'] == pid and e['floor_id'] == fid)
            close(env['grid_minimum_joint95'], [lower, upper])
            if (env['intervals_consistent'] != consistent or env['selected_policy'] != selected
                    or env['tolerance'] != .05 or env['floor'] != floor):
                raise ValueError('Envelope identity mismatch')
            close(env['selected_regret_ceiling'], regret) if consistent else None
            if env['selected_within_tolerance_certified_on_joint_event'] != (consistent and regret <= .05):
                raise ValueError('Regret conclusion mismatch')
    if reconstructed != frozen['responses'] or eval_count != frozen['evaluation_cells']:
        raise ValueError('Independent shortlist selection mismatch')
    defender = []
    for regime in ('fixed_fraction', 'fixed_sacrifice'):
        for fid in range(4):
            choices = [r for r in reconstructed if plans[r['plan_id']]['regime'] == regime and r['floor_id'] == fid]
            w = min(choices, key=lambda r: (-r['pooled_search_alarms'],
                abs(plans[r['plan_id']]['calibration_emitted']/4000000-1/3), plans[r['plan_id']]['calibration_emitted']))
            defender.append({'regime': regime, 'floor_id': fid, 'plan_id': w['plan_id'], 'policy_id': w['winner_pooled']})
    if defender != frozen['defender_plans']:
        raise ValueError('Frozen defender selection mismatch')
    return {'status': 'passed', 'public_pairs_reconciled': totals['public_pairs'],
            'private_rows_reconciled': totals['private_rows'], 'decision_replays': totals['decision_replays'],
            'summary_cells_reconciled': len(cells), 'pooled_search_cells_checked': len(pooled),
            'shortlist_checks': len(reconstructed), 'defender_checks': len(defender), 'envelope_checks': len(d['envelopes']),
            'independent_sampled_rule_decisions': totals['independent_sampled_rule_decisions'],
            'source_hashes_checked': len(d['source_sha256']), 'input_hashes_checked': len(d['input_sha256']),
            'limits': 'Full replay shares scorers. Independent scalar decisions are sampled and share SciPy distribution machinery; the grid envelope does not certify continuous attacks.'}


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
