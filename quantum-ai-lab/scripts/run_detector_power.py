"""Frozen detector-specific continuous IR lower envelope and fresh validation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.allocation_frontier import IRPolicy
from research.calibration_protocol import gate_law, sample_public_batch
from research.exact_comparison import central_exact_pvalues, exact_change_alarm
from research.best_response_stability import exact_interval
from research.detector_power import (build_kernel, conditional_point,
                                    continuous_lower_function, empirical_endpoint)
from run_allocation_frontier import plans, BUDGET, FLOORS

SEED, ANCILLARY_TRIALS, DIRECT_TRIALS = 2026100907, 1024, 8192
PLAN_IDS, BIAS_CELLS, ALPHA = (9, 10), 16, .05
FUNCTIONS, MEAN_FAILURE, DIRECT_FAILURE = 144, .04, .01
TRUNCATION, REGRET_TOLERANCE = 1e-12, .05


def frozen_protocol():
    return {'seed': SEED, 'ancillary_trials': ANCILLARY_TRIALS, 'direct_trials': DIRECT_TRIALS,
            'plan_ids': PLAN_IDS, 'floors': FLOORS, 'bias_cells': BIAS_CELLS,
            'alpha': ALPHA, 'mean_functions': FUNCTIONS, 'mean_failure_budget': MEAN_FAILURE,
            'direct_failure_budget': DIRECT_FAILURE, 'direct_cells': 10,
            'calibration_truncation': TRUNCATION, 'regret_tolerance': REGRET_TOLERANCE,
            'noise': .03, 'eta': .15, 'total_emissions': BUDGET}


def _save(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _paired(trials, plan, law, phase, pid, case=0):
    seeds = np.random.SeedSequence([SEED, phase, pid, case]).spawn(2)
    cal = sample_public_batch(trials, plan.calibration_emitted, gate_law(), 1.,
                              np.random.default_rng(seeds[0]))
    mon = sample_public_batch(trials, plan.monitoring_emitted, law, plan.monitoring_disclosure,
                              np.random.default_rng(seeds[1]))
    return cal, mon


def continuous_envelope(cell_bounds, balanced_lower, balanced_upper):
    lower = min(c['endpoint'] for c in cell_bounds)
    upper = balanced_upper['endpoint']
    if lower > upper:
        raise ArithmeticError('Continuous lower envelope exceeds balanced point upper bound')
    regret = upper-lower
    return {'continuous_minimum_joint95': [lower, upper],
            'balanced_point_joint95': [balanced_lower['endpoint'], upper],
            'balanced_regret_upper_joint95': regret,
            'balanced_within_five_points': bool(regret <= REGRET_TOLERANCE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a fresh name to preserve evidence')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    protocol = frozen_protocol()
    (args.output/'frozen_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    profiles = {p['plan_id']: p for p in plans()}
    edges = np.linspace(0, .5, BIAS_CELLS+1)
    results, direct, outputs = [], [], ['frozen_protocol.json']
    began = time.monotonic()
    for pid in PLAN_IDS:
        plan = profiles[pid]['plan']
        cal, mon = _paired(ANCILLARY_TRIALS, plan, gate_law(), 0, pid)
        gain = np.any(central_exact_pvalues(cal, mon)[:, 4:] <= ALPHA/6, axis=1)
        name = f'ancillary_plan_{pid}.npz'
        # Error draws were discarded; this archive is ancillary counts, not a
        # fabricated observed public transcript or a second direct experiment.
        _save(args.output/name, n_cal=cal.denominators, n_mon=mon.denominators,
              s_cal=cal.sifted, s_mon=mon.sifted, detected_cal=cal.numerators[:, 4:],
              detected_mon=mon.numerators[:, 4:], gain_alarm=gain)
        outputs.append(name)
        print(f'plan {pid}: constructing conditional rejection boundaries', flush=True)
        kernel = build_kernel(cal.denominators[:, :4], mon.denominators[:, :4], truncation=TRUNCATION)
        name = f'kernel_plan_{pid}.npz'
        _save(args.output/name, **{field: getattr(kernel, field) for field in
              ('n_cal', 'n_mon', 'calibration_values', 'weights', 'high_cutoff', 'low_cutoff', 'missing_mass')})
        outputs.append(name)
        cell_functions, point_lo, point_hi = [], [], []
        for fid, floor in enumerate(FLOORS):
            cells = []
            for cid, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                values = continuous_lower_function(kernel, floor, (lo, hi), gain)
                bound = empirical_endpoint(values, MEAN_FAILURE/FUNCTIONS, 'lower')
                cells.append({'cell_id': cid, 'bias_bounds': [float(lo), float(hi)], **bound})
                cell_functions.append(values)
            policy = IRPolicy(2*floor, .5)
            lower, upper = conditional_point(kernel, policy.law().bit_errors, gain)
            point_lo.append(lower); point_hi.append(upper)
            lb = empirical_endpoint(lower, MEAN_FAILURE/FUNCTIONS, 'lower')
            ub = empirical_endpoint(upper, MEAN_FAILURE/FUNCTIONS, 'upper')
            results.append({'plan_id': pid, 'plan': asdict(plan), 'floor_id': fid, 'floor': floor,
                            'rate_domain': [2*floor, 1.], 'bias_domain': [0., 1.],
                            'cells': cells, 'balanced_lower': lb, 'balanced_upper': ub,
                            **continuous_envelope(cells, lb, ub)})
            print(f'plan {pid}, floor {floor}: continuous bounds complete', flush=True)
        name = f'conditional_plan_{pid}.npz'
        _save(args.output/name, cell_lower=np.array(cell_functions),
              balanced_lower=np.array(point_lo), balanced_upper=np.array(point_hi))
        outputs.append(name)
        for case in range(5):
            fid = case if case < 4 else None
            law = IRPolicy(2*FLOORS[case], .5).law() if fid is not None else gate_law()
            dc, dm = _paired(DIRECT_TRIALS, plan, law, 1 if fid is not None else 2, pid, case)
            alarms = exact_change_alarm(dc, dm)
            k = int(alarms.sum())
            interval = exact_interval(k, DIRECT_TRIALS, DIRECT_FAILURE/10)
            record = {'plan_id': pid, 'case_id': case, 'floor_id': fid,
                      'kind': 'balanced_floor' if fid is not None else 'stable_honest',
                      'alarms': k, 'trials': DIRECT_TRIALS, 'alarm_rate': k/DIRECT_TRIALS,
                      'simultaneous_ci99': interval}
            if fid is not None:
                analytic = next(r['balanced_point_joint95'] for r in results
                                if r['plan_id'] == pid and r['floor_id'] == fid)
                record['analytic_interval_overlap'] = bool(max(analytic[0], interval[0]) <=
                                                          min(analytic[1], interval[1]))
            name = f'direct_plan_{pid}_case_{case}.npz'
            _save(args.output/name, y_cal=dc.numerators, n_cal=dc.denominators, s_cal=dc.sifted,
                  y_mon=dm.numerators, n_mon=dm.denominators, s_mon=dm.sifted, alarm=alarms)
            outputs.append(name); direct.append(record)
        print(f'plan {pid}: fresh direct validation complete', flush=True)
    names = ['docs/detector-power-protocol.md', 'scripts/run_detector_power.py',
             'src/research/detector_power.py', 'src/research/calibration_protocol.py',
             'src/research/exact_comparison.py', 'src/research/identifiability.py',
             'src/research/allocation_frontier.py', 'scripts/run_allocation_frontier.py',
             'src/research/best_response_stability.py']
    summary = {'schema_version': 1, 'kind': 'monotone_subevent_continuous_lower_not_exact_optimality',
               'protocol': protocol, 'envelopes': results, 'direct_validation': direct,
               'ancillary_pairs': 2*ANCILLARY_TRIALS, 'direct_public_pairs': 10*DIRECT_TRIALS,
               'calculation_seconds': time.monotonic()-began,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
               'output_sha256': {n: hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in outputs},
               'limits': ['Only fixed nominal stable iid IR model and two predeclared allocations.',
                          'Subevent monotonicity is not full-detector power monotonicity.',
                          'Conditional integration removes sampled errors, not random denominators.',
                          'Floating arithmetic is not an outward-rounded security certificate.',
                          'No exact balanced optimum, final-key security or publication novelty claim.']}
    with (args.output/'power.json').open('x') as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'envelopes': len(results), 'direct_public_pairs': summary['direct_public_pairs'],
                      'seconds': summary['calculation_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
