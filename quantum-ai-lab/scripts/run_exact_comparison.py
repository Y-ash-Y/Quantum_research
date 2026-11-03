"""Frozen paired comparison: central exact conditional alarm versus CP."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import COORDINATES, change_alarm, gate_law, pauli_law, sample_public_batch
from research.allocation_frontier import sample_retained_certified_counts
from research.exact_comparison import exact_change_alarm
from run_allocation_frontier import (BUDGET, ALPHA, FLOORS, plans, policies, select_responses, _writer)

SEED = 2026100504
RULES = ('cp', 'central_exact')
PHASES = {'search': 0, 'evaluation': 1, 'honest_control': 2}


def score_pair(calibration, monitoring):
    return {'cp': change_alarm(calibration, monitoring, ALPHA, 'clopper-pearson'),
            'central_exact': exact_change_alarm(calibration, monitoring, ALPHA)}


def paired_summary(scores):
    cp_only = int(np.sum(scores['cp'] & ~scores['central_exact']))
    exact_only = int(np.sum(scores['central_exact'] & ~scores['cp']))
    discordant = cp_only+exact_only
    return {'cp_only': cp_only, 'exact_only': exact_only,
            'both': int(np.sum(scores['cp'] & scores['central_exact'])),
            'neither': int(np.sum(~scores['cp'] & ~scores['central_exact'])),
            'difference_exact_minus_cp': (exact_only-cp_only)/len(scores['cp']),
            'mcnemar_exact_p': float(binomtest(exact_only, discordant).pvalue) if discordant else 1.}


def freeze_choices(search, profiles, grid):
    frozen = {}
    for rule in RULES:
        projected = [{**c, 'alarm_rate': c['rules'][rule]['alarm_rate']} for c in search]
        responses, winners = select_responses(projected, profiles, grid)
        frozen[rule] = {'eve_responses': responses, 'defender_plans': winners}
    return frozen


def add_evaluation_uncertainty(cell, number_of_cells):
    intervals = {}
    for rule in RULES:
        r = cell['rules'][rule]
        simultaneous = binomtest(r['alarms'], cell['trials']).proportion_ci(
            1-.05/(2*number_of_cells), method='exact')
        r['simultaneous_ci95'] = [simultaneous.low, simultaneous.high]
        ci = binomtest(r['alarms'], cell['trials']).proportion_ci(.975, method='exact')
        intervals[rule] = [ci.low, ci.high]
    cell['paired']['difference_ci95_conservative'] = [
        intervals['central_exact'][0]-intervals['cp'][1],
        intervals['central_exact'][1]-intervals['cp'][0]]
    cell['paired']['difference_simultaneous_ci95_conservative'] = [
        cell['rules']['central_exact']['simultaneous_ci95'][0]-cell['rules']['cp']['simultaneous_ci95'][1],
        cell['rules']['central_exact']['simultaneous_ci95'][1]-cell['rules']['cp']['simultaneous_ci95'][0]]
    cell['paired']['mcnemar_bonferroni_p112'] = min(1., number_of_cells*cell['paired']['mcnemar_exact_p'])


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
    public_fields = ['phase', 'plan_id', 'case_id', 'policy_id', 'trial', 'selection_rule']+[
        f'alarm_{r}' for r in RULES]
    for stage in ('cal', 'mon'):
        public_fields += [f'{stage}_{c}_{count}' for c in COORDINATES for count in ('numerator', 'denominator')]
        public_fields += [f'{stage}_{c}_sifted' for c in COORDINATES[:4]]
    private_fields = ['phase', 'plan_id', 'case_id', 'policy_id', 'trial']+[
        f'certified_{c}' for c in COORDINATES[:4]]
    args.output.mkdir(parents=True, exist_ok=False)
    cells = {'search': [], 'evaluation': [], 'honest_control': []}
    totals = {'public_pairs': 0, 'private_rows': 0}
    with ExitStack() as stack:
        pub_writer = _writer(stack, args.output/'public_counts.csv.gz', public_fields)
        pri_writer = _writer(stack, args.output/'private_knowledge.csv.gz', private_fields)

        def run_cell(profile, phase, case_id, trials, policy_id=-1, control=None, selection_rule='none'):
            plan, plan_id = profile['plan'], profile['plan_id']
            seeds = np.random.SeedSequence([SEED, PHASES[phase], plan_id, case_id]).spawn(3)
            cal = sample_public_batch(trials, plan.calibration_emitted, control or nominal,
                                      1., np.random.default_rng(seeds[0]))
            mon = sample_public_batch(trials, plan.monitoring_emitted,
                                      control if control is not None else grid[policy_id].law(),
                                      plan.monitoring_disclosure, np.random.default_rng(seeds[1]))
            scores = score_pair(cal, mon)
            known = (sample_retained_certified_counts(mon, grid[policy_id], np.random.default_rng(seeds[2]))
                     if control is None else None)
            cell = {'phase': phase, 'plan_id': plan_id, 'case_id': case_id, 'policy_id': policy_id,
                    'selection_rule': selection_rule, 'trials': trials, 'rules': {}, 'paired': paired_summary(scores),
                    'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                    'mean_monitoring_disclosed': float(mon.disclosed.mean()),
                    'mean_remaining_candidates': float(mon.remaining_candidates.mean())}
            for rule in RULES:
                count = int(scores[rule].sum())
                ci = binomtest(count, trials).proportion_ci(.95, method='exact')
                cell['rules'][rule] = {'alarms': count, 'alarm_rate': count/trials,
                                      'alarm_rate_ci95': [ci.low, ci.high]}
            if known is not None:
                retained_total = int(mon.remaining_candidates.sum())
                cell.update({'certified_retained_bits': int(known.sum()),
                             'retained_candidate_bits': retained_total,
                             'pooled_certified_fraction': int(known.sum())/retained_total if retained_total else None,
                             'undetected_certified_bits': {r: int(known[~scores[r]].sum()) for r in RULES}})
            for trial in range(trials):
                key = {'phase': phase, 'plan_id': plan_id, 'case_id': case_id,
                       'policy_id': policy_id, 'trial': trial}
                row = {**key, 'selection_rule': selection_rule,
                       **{f'alarm_{r}': int(scores[r][trial]) for r in RULES}}
                for stage, batch in [('cal', cal), ('mon', mon)]:
                    for j, coord in enumerate(COORDINATES):
                        row[f'{stage}_{coord}_numerator'] = int(batch.numerators[trial, j])
                        row[f'{stage}_{coord}_denominator'] = int(batch.denominators[trial, j])
                    for j, coord in enumerate(COORDINATES[:4]):
                        row[f'{stage}_{coord}_sifted'] = int(batch.sifted[trial, j])
                pub_writer.writerow(row)
                if known is not None:
                    pri_writer.writerow({**key, **{f'certified_{c}': int(known[trial, j])
                                                  for j, c in enumerate(COORDINATES[:4])}})
            totals['public_pairs'] += trials
            totals['private_rows'] += trials if known is not None else 0
            return cell

        for profile in profiles:
            for policy_id in range(len(grid)):
                cells['search'].append(run_cell(profile, 'search', policy_id, args.search_trials, policy_id))
            print(f"search plan {profile['plan_id']} complete", flush=True)
        frozen = freeze_choices(cells['search'], profiles, grid)
        (args.output/'frozen_selections.json').write_text(json.dumps(frozen, indent=2)+'\n')
        evaluation_cells = 2*len(profiles)*len(FLOORS)
        for rule_id, rule in enumerate(RULES):
            for response in frozen[rule]['eve_responses']:
                cell = run_cell(profiles[response['plan_id']], 'evaluation',
                                rule_id*len(FLOORS)+response['floor_id'], args.evaluation_trials,
                                response['policy_id'], selection_rule=rule)
                cell['minimum_certified_fraction'] = response['minimum_certified_fraction']
                add_evaluation_uncertainty(cell, evaluation_cells)
                cells['evaluation'].append(cell)
            print(f'evaluation for {rule} frozen responses complete', flush=True)
        for profile in profiles:
            for control_id, (name, law) in enumerate(controls):
                cell = run_cell(profile, 'honest_control', control_id, args.control_trials, control=law)
                cell['control'] = name
                cells['honest_control'].append(cell)
    root = Path(__file__).resolve().parents[1]
    source_paths = ['scripts/run_exact_comparison.py', 'scripts/run_allocation_frontier.py',
                    'src/research/exact_comparison.py', 'src/research/calibration_protocol.py',
                    'src/research/allocation_frontier.py', 'src/research/identifiability.py']
    outputs = ['public_counts.csv.gz', 'private_knowledge.csv.gz', 'frozen_selections.json']
    summary = {'schema_version': 1, 'kind': 'paired_frozen_grid_responses_not_global_detector_robustness',
               'budget': BUDGET, 'seed': SEED, 'alpha_per_rule': ALPHA, 'rules': RULES,
               'floors': FLOORS, 'search_trials': args.search_trials, 'evaluation_trials': args.evaluation_trials,
               'control_trials': args.control_trials, 'totals': totals,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'plans': [{'plan_id': p['plan_id'], 'regime': p['regime'], **asdict(p['plan']),
                          'monitoring_emitted': p['plan'].monitoring_emitted,
                          'expected_resources': p['plan'].expected_resources()} for p in profiles],
               'policies': [{'policy_id': i, 'rate': p.rate, 'z_basis_probability': p.z_basis_probability,
                             'expected_certified_fraction': p.expected_certified_fraction} for i, p in enumerate(grid)],
               'frozen_selections': frozen, 'cells': cells,
               'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
               'output_sha256': {p: hashlib.sha256((args.output/p).read_bytes()).hexdigest() for p in outputs},
               'limits': ['Central doubled-tail exact test, not probability-ordered Fisher two-sided test.',
                          'Each rule separately has alpha=.05; their OR is not a 5% rule.',
                          'Expected retained raw-bit proxy, not total or final-key information.',
                          'Policies/plans selected on search only; no global or continuous best-response guarantee.',
                          'Paired comparisons use the same public counts under the same attack within each cell.',
                          'Stable attack-free calibration and iid error-blind disclosure assumed.',
                          'Exactness in real arithmetic; floating tails are not outward-rounded certificates.']}
    (args.output/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), **totals,
                      'search_cells': len(cells['search']), 'evaluation_cells': len(cells['evaluation']),
                      'control_cells': len(cells['honest_control'])}, indent=2))


if __name__ == '__main__':
    main()
