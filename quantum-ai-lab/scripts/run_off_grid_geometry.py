"""Frozen off-grid probes and a deterministic continuous-law bound extension."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.allocation_frontier import IRPolicy
from research.calibration_protocol import gate_law, sample_public_batch
from research.exact_comparison import central_exact_pvalues
from research.best_response_stability import FLOORS, exact_interval, eligible_ids, expanded_grid
from research.off_grid_geometry import ERROR_SWAP, reflect_public, axis_cells, cell_radius, continuous_envelope
from run_allocation_frontier import plans, BUDGET

SEED = 2026100506
TRIALS = {'stress': 512, 'confirmation': 4096}
PHASES = {'stress': 0, 'confirmation': 1}
PLAN_IDS = (9, 10)


def frozen_cases(phase):
    if phase not in PHASES:
        raise ValueError('Unknown phase')
    multipliers = (1., 1.0625, 1.1875, 1.375, 1.75, 2.) if phase == 'stress' else (1., 1.0625)
    biases = (0., .0625, .1875, .3125, .4375, .5) if phase == 'stress' else (.4375, .5, .5625)
    return [{'case_id': i, 'floor_id': fid, 'floor': floor, 'rate_multiplier': m,
             'rate': round(2*floor*m, 12), 'beta': b}
            for i, (fid, floor, m, b) in enumerate(
                (fid, floor, m, b) for fid, floor in enumerate(FLOORS) for m in multipliers for b in biases)]


def cover_extension(previous):
    grid = expanded_grid()
    pooled = {(c['plan_id'], c['policy_id']): c for c in previous['pooled_search']}
    results = []
    for profile in plans():
        pid, plan = profile['plan_id'], profile['plan']
        for fid, floor in enumerate(FLOORS):
            ids = eligible_ids(grid, floor)
            rates = sorted({grid[i].rate for i in ids})
            biases = sorted({grid[i].z_basis_probability for i in ids})
            for region, upper_rate in [('near_floor', 4*floor), ('full_eligible', 1.)]:
                rc, bc = dict(axis_cells(rates, 2*floor, upper_rate)), dict(axis_cells(biases, 0., 1.))
                cells = []
                for i in ids:
                    g = grid[i]
                    bounds = cell_radius(plan, g, rc[g.rate], bc[g.z_basis_probability])
                    cells.append({'anchor_policy_id': i, 'rate_bounds': rc[g.rate],
                                  'bias_bounds': bc[g.z_basis_probability],
                                  'anchor_lower': pooled[pid, i]['simultaneous_search_ci975'][0], **bounds})
                upper = next(e['grid_minimum_joint95'][1] for e in previous['envelopes']
                             if e['plan_id'] == pid and e['floor_id'] == fid)
                results.append({'plan_id': pid, 'floor_id': fid, 'floor': floor, 'region': region,
                                'rate_domain': [2*floor, upper_rate], 'bias_domain': [0., 1.],
                                **continuous_envelope(cells, upper), 'cells': cells})
    return results


def confirmation_contrasts(cells):
    lookup = {(c['plan_id'], c['floor_id'], c['rate_multiplier'], c['beta']): c
              for c in cells if c['phase'] == 'confirmation'}
    result = []
    for pid in PLAN_IDS:
        for fid in range(4):
            base = lookup[pid, fid, 1., .5]
            for name, candidate in [('off_grid_bias', lookup[pid, fid, 1., .4375]),
                                    ('higher_rate', lookup[pid, fid, 1.0625, .5])]:
                a, b = candidate['alarms'], base['alarms']
                n, m = candidate['trials'], base['trials']
                p = float(fisher_exact([[a, n-a], [b, m-b]], alternative='less').pvalue)
                la, ua = candidate['simultaneous_ci95']; lb, ub = base['simultaneous_ci95']
                result.append({'plan_id': pid, 'floor_id': fid, 'contrast': name,
                               'first_case_id': candidate['case_id'], 'baseline_case_id': base['case_id'],
                               'difference_first_minus_balanced_floor': a/n-b/m,
                               'simultaneous_difference_ci95': [la-ub, ua-lb],
                               'one_sided_lower_power_p': p,
                               'bonferroni_lower_power_p': min(1., 16*p)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a fresh name to preserve evidence')
    root = Path(__file__).resolve().parents[1]
    input_name = 'results/benchmarks/best-response-stability-v2/stability.json'
    previous = json.loads((root/input_name).read_text())
    # Bind the inherited study to the current unchanged scorers and inputs.
    for group, base in [('source_sha256', root), ('input_sha256', root),
                        ('output_sha256', root/Path(input_name).parent)]:
        for name, digest in previous[group].items():
            if hashlib.sha256((base/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Inherited Step 5 evidence/source hash mismatch')
    cases = {phase: frozen_cases(phase) for phase in PHASES}
    frozen = {'seed': SEED, 'plan_ids': PLAN_IDS, 'trials': TRIALS, 'cases': cases,
              'detector_alpha': .05, 'fresh_interval_failure_budget': .05,
              'confirmation_contrasts': 16}
    args.output.mkdir(parents=True)
    frozen_text = json.dumps(frozen, indent=2, allow_nan=False)+'\n'
    (args.output/'frozen_protocol.json').write_text(frozen_text)
    covers = cover_extension(previous)
    profiles = {p['plan_id']: p for p in plans()}
    total_cells = len(PLAN_IDS)*sum(len(c) for c in cases.values())
    cells, archives = [], []
    for phase in PHASES:
        for pid in PLAN_IDS:
            plan = profiles[pid]['plan']
            arrays = {name: [] for name in ('case_ids', 'y_cal', 'n_cal', 's_cal', 'y_mon', 'n_mon', 's_mon', 'alarm')}
            for case in cases[phase]:
                seeds = np.random.SeedSequence([SEED, PHASES[phase], pid, case['case_id']]).spawn(2)
                cal = sample_public_batch(TRIALS[phase], plan.calibration_emitted, gate_law(), 1.,
                                          np.random.default_rng(seeds[0]))
                mon = sample_public_batch(TRIALS[phase], plan.monitoring_emitted,
                                          IRPolicy(case['rate'], case['beta']).law(), plan.monitoring_disclosure,
                                          np.random.default_rng(seeds[1]))
                pvalues = central_exact_pvalues(cal, mon)
                reflected = central_exact_pvalues(reflect_public(cal), reflect_public(mon))
                if not np.array_equal(reflected, pvalues[:, ERROR_SWAP]):
                    raise ValueError('Exact reflection invariance failed')
                alarms = np.any(pvalues <= .05/6, axis=1)
                count = int(alarms.sum())
                cells.append({'phase': phase, 'plan_id': pid, **case, 'trials': TRIALS[phase],
                              'alarms': count, 'alarm_rate': count/TRIALS[phase],
                              'marginal_ci95': exact_interval(count, TRIALS[phase], .05),
                              'simultaneous_ci95': exact_interval(count, TRIALS[phase], .05/total_cells),
                              'reflection_pvalue_rows_checked': TRIALS[phase]})
                arrays['case_ids'].append(case['case_id']); arrays['alarm'].append(alarms)
                for stage, batch in [('cal', cal), ('mon', mon)]:
                    for short, attr in [('y', 'numerators'), ('n', 'denominators'), ('s', 'sifted')]:
                        arrays[f'{short}_{stage}'].append(getattr(batch, attr))
            relative = f'{phase}_plan_{pid}.npz'
            with (args.output/relative).open('xb') as stream:
                np.savez_compressed(stream, **{k: np.array(v, dtype=bool if k == 'alarm' else np.int32)
                                               for k, v in arrays.items()})
            archives.append({'phase': phase, 'plan_id': pid, 'file': relative})
            print(f'{phase} plan {pid} complete', flush=True)
    source_names = ['scripts/run_off_grid_geometry.py', 'src/research/off_grid_geometry.py',
                    'docs/off-grid-protocol.md', 'scripts/run_allocation_frontier.py',
                    'src/research/calibration_protocol.py', 'src/research/allocation_frontier.py',
                    'src/research/exact_comparison.py', 'src/research/identifiability.py',
                    'src/research/best_response_stability.py']
    outputs = ['frozen_protocol.json']+[a['file'] for a in archives]
    summary = {'schema_version': 1, 'seed': SEED, 'kind': 'reflection_and_public_KL_cover_not_detector_optimality',
               'frozen_protocol': frozen, 'cells': cells, 'archives': archives,
               'continuous_covers': covers, 'contrasts': confirmation_contrasts(cells),
               'public_pairs': sum(c['trials'] for c in cells),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'input_sha256': {input_name: hashlib.sha256((root/input_name).read_bytes()).hexdigest()},
               'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_names},
               'output_sha256': {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in outputs},
               'limits': ['Symmetry and balanced KL proxy do not prove balanced detector-optimal Eve.',
                          'Inherited continuous bounds and fresh-rate intervals have separate 95% coverage events.',
                          'Floating KL radii are not outward-rounded certificates.',
                          'No final-key security, drift attribution or publication novelty claim.']}
    text = json.dumps(summary, indent=2, allow_nan=False)+'\n'
    with (args.output/'geometry.json').open('x') as stream:
        stream.write(text)
    print(json.dumps({'output': str(args.output), 'public_pairs': summary['public_pairs'], 'cells': total_cells,
                      'continuous_regions': len(covers)}, indent=2))


if __name__ == '__main__':
    main()
