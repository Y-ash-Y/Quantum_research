"""Frozen fixed-budget IR grid study, with independent search and evaluation."""
import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.allocation_frontier import (
    BudgetPlan, IRPolicy, broad_pauli_information_bound, constant_sacrifice_plan,
    policy_sort_key, sample_retained_certified_counts,
)
from research.calibration_protocol import COORDINATES, change_alarm, gate_law, pauli_law, sample_public_batch

BUDGET, SEED, ALPHA = 4000000, 2026100503, .05
FLOORS = (.0025, .005, .01, .02)
BIASES = (0., .25, .5, .75, 1.)
MULTIPLIERS = (1., 1.5, 2.)
PHASES = {'search': 0, 'evaluation': 1, 'honest_control': 2}


def plans():
    result = []
    regimes = [('fixed_fraction', (.05, .1, .2, 1/3, .5, .7, .9)),
               ('fixed_sacrifice', (.05, .1, .2, 1/3, .4, .45, .5))]
    for regime, fractions in regimes:
        for fraction in fractions:
            m = int(BUDGET*fraction)
            plan = (BudgetPlan(BUDGET, m, .25) if regime == 'fixed_fraction'
                    else constant_sacrifice_plan(BUDGET, m, BUDGET//2))
            result.append({'plan_id': len(result), 'regime': regime, 'plan': plan})
    return result


def policies():
    rates = sorted({round(2*k*scale, 12) for k in FLOORS for scale in MULTIPLIERS})
    return [IRPolicy(r, bias) for r in rates for bias in BIASES]


def candidate_ids(grid, floor):
    allowed = {round(2*floor*scale, 12) for scale in MULTIPLIERS}
    return [i for i, policy in enumerate(grid) if policy.rate in allowed and policy.eligible(floor)]


def select_responses(search, profiles, grid):
    index = {(c['plan_id'], c['policy_id']): c for c in search}
    responses = []
    for profile in profiles:
        for floor_id, floor in enumerate(FLOORS):
            selected = min(candidate_ids(grid, floor), key=lambda i: (
                index[profile['plan_id'], i]['alarm_rate'], policy_sort_key(grid[i])))
            responses.append({'plan_id': profile['plan_id'], 'floor_id': floor_id,
                              'minimum_certified_fraction': floor, 'policy_id': selected,
                              'search_alarm_rate': index[profile['plan_id'], selected]['alarm_rate']})
    winners = []
    for regime in ('fixed_fraction', 'fixed_sacrifice'):
        eligible_plans = {p['plan_id']: p['plan'] for p in profiles if p['regime'] == regime}
        for floor_id in range(len(FLOORS)):
            candidates = [r for r in responses if r['plan_id'] in eligible_plans and r['floor_id'] == floor_id]
            winner = min(candidates, key=lambda r: (-r['search_alarm_rate'],
                abs(eligible_plans[r['plan_id']].calibration_emitted/BUDGET-1/3),
                eligible_plans[r['plan_id']].calibration_emitted))
            winners.append({'regime': regime, **winner})
    return responses, winners


def _writer(stack, path, fields):
    raw = stack.enter_context(path.open('xb'))
    compressed = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0))
    stream = stack.enter_context(io.TextIOWrapper(compressed, encoding='utf-8', newline=''))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-trials', type=int, default=192)
    parser.add_argument('--evaluation-trials', type=int, default=600)
    parser.add_argument('--control-trials', type=int, default=400)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve it and choose a new directory')
    if any(not 1 <= n <= 2000 for n in (args.search_trials, args.evaluation_trials, args.control_trials)):
        parser.error('Each phase permits 1–2000 repetitions per cell')
    profiles, grid = plans(), policies()
    nominal = gate_law()
    controls = [('nominal', nominal), ('gate_error', gate_law(.03, .02)),
                ('high_error_pauli', pauli_law((0., .2, .7, .1))), ('zero_transmission', gate_law(eta=0))]
    public_fields = ['phase', 'plan_id', 'case_id', 'policy_id', 'trial', 'alarm']
    for stage in ('cal', 'mon'):
        public_fields += [f'{stage}_{coord}_{count}' for coord in COORDINATES
                          for count in ('numerator', 'denominator')]
        public_fields += [f'{stage}_{coord}_sifted' for coord in COORDINATES[:4]]
    private_fields = ['phase', 'plan_id', 'case_id', 'policy_id', 'trial']+[
        f'certified_{coord}' for coord in COORDINATES[:4]]
    args.output.mkdir(parents=True, exist_ok=False)
    cells = {'search': [], 'evaluation': [], 'honest_control': []}
    totals = {'public_pairs': 0, 'private_rows': 0}
    with ExitStack() as stack:
        public_writer = _writer(stack, args.output/'public_counts.csv.gz', public_fields)
        private_writer = _writer(stack, args.output/'private_knowledge.csv.gz', private_fields)

        def run_cell(profile, phase, case_id, trials, policy_id=-1, control=None):
            plan, plan_id = profile['plan'], profile['plan_id']
            seeds = np.random.SeedSequence([SEED, PHASES[phase], plan_id, case_id]).spawn(3)
            cal_law = control if control is not None else nominal
            mon_law = control if control is not None else grid[policy_id].law()
            cal = sample_public_batch(trials, plan.calibration_emitted, cal_law, 1., np.random.default_rng(seeds[0]))
            mon = sample_public_batch(trials, plan.monitoring_emitted, mon_law,
                                      plan.monitoring_disclosure, np.random.default_rng(seeds[1]))
            alarm = change_alarm(cal, mon, ALPHA, 'clopper-pearson')
            known = (sample_retained_certified_counts(mon, grid[policy_id], np.random.default_rng(seeds[2]))
                     if control is None else None)
            count = int(alarm.sum())
            ci = binomtest(count, trials).proportion_ci(.95, method='exact')
            cell = {'phase': phase, 'plan_id': plan_id, 'case_id': case_id, 'policy_id': policy_id,
                    'alarms': count, 'trials': trials, 'alarm_rate': count/trials,
                    'alarm_rate_ci95': [ci.low, ci.high],
                    'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                    'mean_monitoring_disclosed': float(mon.disclosed.mean()),
                    'mean_remaining_candidates': float(mon.remaining_candidates.mean())}
            if known is not None:
                retained_total = int(mon.remaining_candidates.sum())
                cell.update({'certified_retained_bits': int(known.sum()),
                             'retained_candidate_bits': retained_total,
                             'pooled_certified_fraction': int(known.sum())/retained_total if retained_total else None,
                             'undetected_certified_bits': int(known[~alarm].sum())})
            for trial in range(trials):
                key = {'phase': phase, 'plan_id': plan_id, 'case_id': case_id,
                       'policy_id': policy_id, 'trial': trial}
                row = {**key, 'alarm': int(alarm[trial])}
                for stage, batch in [('cal', cal), ('mon', mon)]:
                    for j, coord in enumerate(COORDINATES):
                        row[f'{stage}_{coord}_numerator'] = int(batch.numerators[trial, j])
                        row[f'{stage}_{coord}_denominator'] = int(batch.denominators[trial, j])
                    for j, coord in enumerate(COORDINATES[:4]):
                        row[f'{stage}_{coord}_sifted'] = int(batch.sifted[trial, j])
                public_writer.writerow(row)
                if known is not None:
                    private_writer.writerow({**key, **{f'certified_{coord}': int(known[trial, j])
                                            for j, coord in enumerate(COORDINATES[:4])}})
            totals['public_pairs'] += trials
            totals['private_rows'] += trials if known is not None else 0
            return cell

        for profile in profiles:
            for policy_id in range(len(grid)):
                cells['search'].append(run_cell(profile, 'search', policy_id, args.search_trials, policy_id))
        responses, winners = select_responses(cells['search'], profiles, grid)
        # No evaluation results exist yet: freeze choices here, before fresh RNG streams.
        selections = {'eve_responses': responses, 'defender_plans': winners}
        (args.output/'frozen_selections.json').write_text(json.dumps(selections, indent=2)+'\n')
        simultaneous_level = 1-.05/len(responses)
        for response in responses:
            profile = profiles[response['plan_id']]
            cell = run_cell(profile, 'evaluation', response['floor_id'], args.evaluation_trials,
                            response['policy_id'])
            joint_ci = binomtest(cell['alarms'], cell['trials']).proportion_ci(simultaneous_level, method='exact')
            cell.update({'minimum_certified_fraction': response['minimum_certified_fraction'],
                         'simultaneous_ci95': [joint_ci.low, joint_ci.high],
                         'selected_policy_bound': broad_pauli_information_bound(
                             profile['plan'], grid[response['policy_id']]),
                         'balanced_floor_bound': broad_pauli_information_bound(
                             profile['plan'], IRPolicy(2*response['minimum_certified_fraction'], .5))})
            cells['evaluation'].append(cell)
        for profile in profiles:
            for control_id, (name, law) in enumerate(controls):
                cell = run_cell(profile, 'honest_control', control_id, args.control_trials, control=law)
                cell['control'] = name
                cells['honest_control'].append(cell)
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), root/'src/research/allocation_frontier.py',
                    root/'src/research/calibration_protocol.py', root/'src/research/identifiability.py']
    output_paths = ['public_counts.csv.gz', 'private_knowledge.csv.gz', 'frozen_selections.json']
    summary = {'schema_version': 1, 'kind': 'heldout_grid_selected_resource_study_not_global_frontier',
               'budget': BUDGET, 'seed': SEED, 'alpha': ALPHA, 'method': 'clopper-pearson',
               'floors': FLOORS, 'search_trials': args.search_trials, 'evaluation_trials': args.evaluation_trials,
               'control_trials': args.control_trials, 'totals': totals,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'plans': [{'plan_id': p['plan_id'], 'regime': p['regime'], **asdict(p['plan']),
                          'monitoring_emitted': p['plan'].monitoring_emitted,
                          'expected_resources': p['plan'].expected_resources()} for p in profiles],
               'policies': [{'policy_id': i, **asdict(p), 'expected_certified_fraction': p.expected_certified_fraction}
                            for i, p in enumerate(grid)],
               'frozen_selections': selections, 'cells': cells,
               'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
               'output_sha256': {p: hashlib.sha256((args.output/p).read_bytes()).hexdigest() for p in output_paths},
               'limits': ['Expected interception-certified raw fraction, not total/final-key information.',
                          'Grid-selected responses evaluated fresh; no global or continuous optimum.',
                          'Stable attack-free calibration, iid loss and error-blind public sampling assumed.',
                          'Same emissions; sacrifice/candidate expectations equal only in fixed-sacrifice regime.',
                          'Simultaneous 95% intervals cover the 56 frozen evaluation rates, not unknown worst-case power.',
                          'Floating KL diagnostics and standard inequalities do not establish novelty.']}
    (args.output/'frontier.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), **totals,
                      'search_cells': len(cells['search']), 'evaluation_cells': len(cells['evaluation']),
                      'control_cells': len(cells['honest_control'])}, indent=2))


if __name__ == '__main__':
    main()
