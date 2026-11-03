"""Two independent enlarged Eve searches and untouched shortlist evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import gate_law, pauli_law, sample_public_batch
from research.allocation_frontier import sample_retained_certified_counts
from research.exact_comparison import exact_change_alarm
from research.best_response_stability import (
    FLOORS, expanded_grid, eligible_ids, make_shortlist, exact_interval, grid_envelope,
)
from run_allocation_frontier import plans, BUDGET
from run_exact_comparison import score_pair, paired_summary

SEED = 2026100505
PHASES = {'search_a': 0, 'search_b': 1, 'evaluation': 2, 'honest_control': 3}


def save_block(directory, relative, records):
    """Bound memory to one plan/phase; never overwrite a saved archive."""
    arrays = {'case_ids': np.array([r[0]['case_id'] for r in records], dtype=np.int32),
              'policy_ids': np.array([r[0]['policy_id'] for r in records], dtype=np.int32)}
    for stage, pos in [('cal', 1), ('mon', 2)]:
        for short, name in [('y', 'numerators'), ('n', 'denominators'), ('s', 'sifted')]:
            arrays[f'{short}_{stage}'] = np.stack([getattr(r[pos], name) for r in records]).astype(np.int32)
    for rule in records[0][3]:
        arrays[f'alarm_{rule}'] = np.stack([r[3][rule] for r in records])
    with (directory/relative).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def freeze_shortlists(search_cells, profiles, grid, previous):
    indices = {phase: {(c['plan_id'], c['policy_id']): c['alarms'] for c in search_cells[phase]}
               for phase in ('search_a', 'search_b')}
    responses = []
    for profile in profiles:
        plan_id = profile['plan_id']
        a = {i: indices['search_a'][plan_id, i] for i in range(len(grid))}
        b = {i: indices['search_b'][plan_id, i] for i in range(len(grid))}
        for floor_id, floor in enumerate(FLOORS):
            response = make_shortlist(a, b, grid, floor, previous[plan_id, floor_id])
            response.update({'plan_id': plan_id, 'floor_id': floor_id,
                             'pooled_search_alarms': a[response['winner_pooled']]+b[response['winner_pooled']]})
            responses.append(response)
    defender = []
    for regime in ('fixed_fraction', 'fixed_sacrifice'):
        for floor_id in range(len(FLOORS)):
            choices = [r for r in responses if profiles[r['plan_id']]['regime'] == regime and r['floor_id'] == floor_id]
            selected = min(choices, key=lambda r: (-r['pooled_search_alarms'],
                abs(profiles[r['plan_id']]['plan'].calibration_emitted/BUDGET-1/3),
                profiles[r['plan_id']]['plan'].calibration_emitted))
            defender.append({'regime': regime, 'floor_id': floor_id, 'plan_id': selected['plan_id'],
                             'policy_id': selected['winner_pooled']})
    return {'responses': responses, 'defender_plans': defender,
            'evaluation_cells': sum(len(r['shortlist']) for r in responses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--search-trials', type=int, default=512)
    parser.add_argument('--evaluation-trials', type=int, default=2048)
    parser.add_argument('--control-trials', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve it and choose a fresh directory')
    if any(not 1 <= n <= 4096 for n in (args.search_trials, args.evaluation_trials, args.control_trials)):
        parser.error('Use 1–4096 trials per cell')
    root = Path(__file__).resolve().parents[1]
    input_names = ['results/benchmarks/exact-comparison-v1/comparison.json',
                   'results/benchmarks/exact-comparison-v1/frozen_selections.json']
    old_summary = json.loads((root/input_names[0]).read_text())
    old_frozen = json.loads((root/input_names[1]).read_text())
    if old_summary['frozen_selections'] != old_frozen:
        raise ValueError('Step 4 frozen choices disagree')
    profiles, grid = plans(), expanded_grid()
    grid_map = {(p.rate, p.z_basis_probability): i for i, p in enumerate(grid)}
    old_policies = {p['policy_id']: p for p in old_summary['policies']}
    previous = {(r['plan_id'], r['floor_id']): grid_map[(old_policies[r['policy_id']]['rate'],
                 old_policies[r['policy_id']]['z_basis_probability'])]
                for r in old_frozen['central_exact']['eve_responses']}
    if len(previous) != 56:
        raise ValueError('Missing previous frozen responses')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'public').mkdir(); (args.output/'private').mkdir()
    cells = {phase: [] for phase in PHASES}
    outputs, archive_index = [], []
    nominal = gate_law()

    def simulate(profile, phase, case_id, policy_id, trials, control=None):
        plan, plan_id = profile['plan'], profile['plan_id']
        seeds = np.random.SeedSequence([SEED, PHASES[phase], plan_id, case_id]).spawn(3)
        cal = sample_public_batch(trials, plan.calibration_emitted,
                                  control if control is not None else nominal, 1., np.random.default_rng(seeds[0]))
        mon = sample_public_batch(trials, plan.monitoring_emitted,
                                  control if control is not None else grid[policy_id].law(),
                                  plan.monitoring_disclosure, np.random.default_rng(seeds[1]))
        scores = ({'central_exact': exact_change_alarm(cal, mon)} if phase.startswith('search') else score_pair(cal, mon))
        cell = {'phase': phase, 'plan_id': plan_id, 'case_id': case_id, 'policy_id': policy_id,
                'trials': trials, 'mean_remaining_candidates': float(mon.remaining_candidates.mean()),
                'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                'mean_monitoring_disclosed': float(mon.disclosed.mean())}
        if phase.startswith('search'):
            cell['alarms'] = int(scores['central_exact'].sum())
            cell['alarm_rate'] = cell['alarms']/trials
        else:
            cell['rules'] = {}
            for rule, alarm in scores.items():
                count = int(alarm.sum())
                cell['rules'][rule] = {'alarms': count, 'alarm_rate': count/trials,
                                      'marginal_ci95': exact_interval(count, trials, .05)}
            cell['paired'] = paired_summary(scores)
        known = None
        if phase == 'evaluation':
            known = sample_retained_certified_counts(mon, grid[policy_id], np.random.default_rng(seeds[2]))
            total = int(mon.remaining_candidates.sum())
            cell.update({'certified_retained_bits': int(known.sum()), 'retained_candidate_bits': total,
                         'pooled_certified_fraction': int(known.sum())/total if total else None})
        return cell, cal, mon, scores, known

    def persist(profile, phase, records):
        relative = f"public/{phase}_plan_{profile['plan_id']}.npz"
        save_block(args.output, relative, records)
        outputs.append(relative)
        entry = {'phase': phase, 'plan_id': profile['plan_id'], 'public_file': relative,
                 'case_ids': [r[0]['case_id'] for r in records]}
        if phase == 'evaluation':
            private_name = f"private/evaluation_plan_{profile['plan_id']}.npz"
            with (args.output/private_name).open('xb') as stream:
                np.savez_compressed(stream, case_ids=np.array(entry['case_ids'], dtype=np.int32),
                                    policy_ids=np.array([r[0]['policy_id'] for r in records], dtype=np.int32),
                                    certified=np.stack([r[4] for r in records]).astype(np.int32))
            entry['private_file'] = private_name
            outputs.append(private_name)
        archive_index.append(entry)

    for phase in ('search_a', 'search_b'):
        for profile in profiles:
            records = [simulate(profile, phase, i, i, args.search_trials) for i in range(len(grid))]
            cells[phase].extend(r[0] for r in records)
            persist(profile, phase, records)
            print(f"{phase} plan {profile['plan_id']} complete", flush=True)
    frozen = freeze_shortlists(cells, profiles, grid, previous)
    with (args.output/'frozen_selections.json').open('x') as stream:
        stream.write(json.dumps(frozen, indent=2)+'\n')
    outputs.append('frozen_selections.json')
    pooled = []
    index_b = {(c['plan_id'], c['policy_id']): c for c in cells['search_b']}
    for a in cells['search_a']:
        b = index_b[a['plan_id'], a['policy_id']]
        count, trials = a['alarms']+b['alarms'], a['trials']+b['trials']
        pooled.append({'plan_id': a['plan_id'], 'policy_id': a['policy_id'], 'alarms': count,
                       'trials': trials, 'alarm_rate': count/trials,
                       'simultaneous_search_ci975': exact_interval(count, trials, .025/len(cells['search_a']))})
    evaluation_count = frozen['evaluation_cells']
    for profile in profiles:
        records = []
        for response in (r for r in frozen['responses'] if r['plan_id'] == profile['plan_id']):
            for policy_id in response['shortlist']:
                r = simulate(profile, 'evaluation', len(records), policy_id, args.evaluation_trials)
                r[0].update({'floor_id': response['floor_id'], 'floor': response['floor']})
                for result in r[0]['rules'].values():
                    result['simultaneous_evaluation_ci975'] = exact_interval(
                        result['alarms'], args.evaluation_trials, .025/(2*evaluation_count))
                r[0]['paired']['mcnemar_bonferroni_p'] = min(1., evaluation_count*r[0]['paired']['mcnemar_exact_p'])
                records.append(r)
        cells['evaluation'].extend(r[0] for r in records)
        persist(profile, 'evaluation', records)
        print(f"evaluation plan {profile['plan_id']} complete", flush=True)
    controls = [('nominal', nominal), ('gate_error', gate_law(.03, .02)),
                ('high_error_pauli', pauli_law((0., .2, .7, .1))), ('zero_transmission', gate_law(eta=0))]
    for profile in profiles:
        records = []
        for case_id, (name, law) in enumerate(controls):
            r = simulate(profile, 'honest_control', case_id, -1, args.control_trials, control=law)
            r[0]['control'] = name
            records.append(r)
        cells['honest_control'].extend(r[0] for r in records)
        persist(profile, 'honest_control', records)
    pooled_index = {(c['plan_id'], c['policy_id']): c for c in pooled}
    envelopes = []
    for response in frozen['responses']:
        pid, fid = response['plan_id'], response['floor_id']
        eligible = eligible_ids(grid, response['floor'])
        evaluated = [c for c in cells['evaluation'] if c['plan_id'] == pid and c['floor_id'] == fid]
        interval_map = {c['policy_id']: c['rules']['central_exact']['simultaneous_evaluation_ci975'] for c in evaluated}
        envelope = grid_envelope({i: pooled_index[pid, i]['simultaneous_search_ci975'] for i in eligible},
                                 interval_map, eligible, response['winner_pooled'])
        envelopes.append({'plan_id': pid, 'floor_id': fid, 'floor': response['floor'],
                          'selected_policy': response['winner_pooled'], **envelope})
    source_paths = ['scripts/run_best_response_stability.py', 'src/research/best_response_stability.py',
                    'scripts/run_exact_comparison.py', 'scripts/run_allocation_frontier.py',
                    'src/research/calibration_protocol.py', 'src/research/exact_comparison.py',
                    'src/research/allocation_frontier.py', 'src/research/identifiability.py']
    summary = {'schema_version': 1, 'kind': 'independent_search_and_finite_grid_envelope_not_continuous_optimum',
               'seed': SEED, 'budget': BUDGET, 'alpha_per_rule': .05, 'floors': FLOORS,
               'search_trials_per_replica': args.search_trials, 'evaluation_trials': args.evaluation_trials,
               'control_trials': args.control_trials,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'plans': [{'plan_id': p['plan_id'], 'regime': p['regime'], **asdict(p['plan']),
                          'monitoring_emitted': p['plan'].monitoring_emitted,
                          'expected_resources': p['plan'].expected_resources()} for p in profiles],
               'policies': [{'policy_id': i, 'rate': p.rate, 'z_basis_probability': p.z_basis_probability,
                             'expected_certified_fraction': p.expected_certified_fraction} for i, p in enumerate(grid)],
               'frozen_selections': frozen, 'pooled_search': pooled, 'envelopes': envelopes,
               'cells': cells, 'archives': archive_index,
               'totals': {'public_pairs': sum(c['trials'] for groups in cells.values() for c in groups),
                          'private_rows': sum(c['trials'] for c in cells['evaluation'])},
               'input_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in input_names},
               'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
               'output_sha256': {p: hashlib.sha256((args.output/p).read_bytes()).hexdigest() for p in outputs},
               'limits': ['Joint 95% Monte Carlo bracket covers the declared finite grid, not continuous attacks.',
                          'Search winner agreement is descriptive; policies/plans remain frozen before evaluation.',
                          'Stable attack-free calibration, iid loss and error-blind disclosure assumed.',
                          'Raw-knowledge proxy is not total information or final-key leakage.',
                          'False-alarm alpha and Monte Carlo confidence budgets are distinct.',
                          'Numerical tails/intervals are floating point, not rounding certificates or novelty proofs.']}
    encoded = json.dumps(summary, indent=2, allow_nan=False)+'\n'
    with (args.output/'stability.json').open('x') as stream:
        stream.write(encoded)
    print(json.dumps({'output': str(args.output), **summary['totals'],
                      'evaluation_cells': evaluation_count, 'public_archives': len(archive_index)}, indent=2))


if __name__ == '__main__':
    main()
