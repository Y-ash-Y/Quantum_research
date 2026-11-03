"""Run the frozen bounded-drift pilot into a new directory only."""
import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from research.allocation_frontier import IRPolicy, constant_sacrifice_plan
from research.calibration_protocol import COORDINATES, gate_law, sample_public_batch
from research.drift_detector import (
    DriftEnvelope, drift_change_alarm, honest_drift_laws,
    hoeffding_family_power_lower,
)

SEED, ALPHA = 2026100911, .05
FLOORS, MULTIPLIERS = (.005, .01, .02), (0., .5, 1., 1.25)
METHODS = ('hoeffding', 'clopper-pearson')


def plans():
    return [(i, constant_sacrifice_plan(4000000, m, 2000000))
            for i, m in ((9, 800000), (10, 1333333))]


def frozen_protocol(trials=512):
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 1024:
        raise ValueError('This pilot allows 1..1024 trials per cohort')
    return {'schema_version': 1, 'seed': SEED, 'alpha': ALPHA, 'trials': trials,
            'channel_noise': .03, 'eta': .15, 'floors': list(FLOORS),
            'drift_multipliers': list(MULTIPLIERS), 'methods': list(METHODS),
            'settings': 24, 'cohorts': 216, 'rate_cells': 432,
            'rate_failure_budget': .04, 'mean_failure_budget': .01,
            'mean_cells': 24, 'calibration_disclosure': 1.,
            'plans': [{'id': i, **asdict(p), **p.expected_resources()}
                      for i, p in plans()],
            'scope': 'Fixed-noise iid IR versus bounded honest between-block error drift; stable gains.'}


def edge(q0, d, sign):
    value = q0+sign*d
    # Stay strictly within the supplied floating envelope, without widening it.
    if abs(value-q0) > d:
        value = math.nextafter(value, q0)
    if abs(value-q0) > d:
        raise ArithmeticError('Unable to construct an in-envelope numerical edge')
    return value


def scenarios(floor, multiplier):
    d = .97*floor*multiplier/2
    envelope = DriftEnvelope(d, d)
    q0, nominal = .015, gate_law(.03)
    upper, lower = edge(q0, d, 1), edge(q0, d, -1)
    cases = []
    for name, z, x, eta in [('honest_stable', q0, q0, .15),
                            ('honest_upper_edge', upper, upper, .15),
                            ('honest_opposite_edges', upper, lower, .15),
                            ('honest_lower_edge', lower, lower, .15),
                            ('honest_zero_transmission', q0, q0, 0.)]:
        cal, mon = honest_drift_laws(q0, q0, z, x, envelope, eta=eta)
        cases.append({'name': name, 'category': 'allowed_honest',
                      'calibration': cal, 'monitoring': mon, 'policy': None})
    for name, r, beta in [('floor_beta_0', 2*floor, 0.),
                           ('floor_beta_025', 2*floor, .25),
                           ('floor_balanced', 2*floor, .5),
                           ('double_rate_balanced', 4*floor, .5)]:
        policy = IRPolicy(r, beta)
        cases.append({'name': name, 'category': 'fixed_attack_probe',
                      'calibration': nominal, 'monitoring': policy.law(),
                      'policy': asdict(policy)})
    return envelope, cases


def interval(count, total, failure):
    ci = binomtest(count, total).proportion_ci(1-failure, method='exact')
    return [ci.low, ci.high]


def draw(trials, plan, case, ids):
    children = np.random.SeedSequence([SEED, *ids]).spawn(2)
    cal = sample_public_batch(trials, plan.calibration_emitted, case['calibration'],
                              1., np.random.default_rng(children[0]))
    mon = sample_public_batch(trials, plan.monitoring_emitted, case['monitoring'],
                              plan.monitoring_disclosure, np.random.default_rng(children[1]))
    return cal, mon


def source_hashes():
    names = ['docs/drift-pilot-protocol.md', 'scripts/run_drift_pilot.py',
             'scripts/audit_drift_pilot.py', 'scripts/audit_calibration_pilot.py',
             'src/research/drift_detector.py', 'src/research/calibration_protocol.py',
             'src/research/allocation_frontier.py', 'src/research/identifiability.py']
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def run(output, trials=512):
    output = Path(output)
    protocol = frozen_protocol(trials)
    hashes = source_hashes()  # Verify sources exist before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    cells, cohorts, means = [], [], []
    archive = output/'public_counts.csv.gz'
    fieldnames = ['plan_id', 'floor_id', 'allowance_id', 'cohort_id', 'trial']
    for stage in ('cal', 'mon'):
        for coord in COORDINATES:
            fieldnames += [f'{stage}_{coord}_numerator', f'{stage}_{coord}_denominator']
        fieldnames += [f'{stage}_{coord}_sifted' for coord in COORDINATES[:4]]
    fieldnames += list(METHODS)+['hoeffding_conditional_lower']
    pairs = 0
    with gzip.open(archive, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for plan_id, plan in plans():
            for fi, floor in enumerate(FLOORS):
                for ai, multiplier in enumerate(MULTIPLIERS):
                    envelope, cases = scenarios(floor, multiplier)
                    for ci, case in enumerate(cases):
                        ids = (plan_id, fi, ai, ci)
                        cal, mon = draw(trials, plan, case, ids)
                        decisions = {m: drift_change_alarm(cal, mon, envelope, alpha=ALPHA, method=m)
                                     for m in METHODS}
                        key = {'plan_id': plan_id, 'floor_id': fi, 'allowance_id': ai, 'cohort_id': ci}
                        cohorts.append({**key, 'scenario': case['name'], 'category': case['category'],
                                        'floor': floor, 'drift_multiplier': multiplier,
                                        'envelope': asdict(envelope),
                                        'calibration': asdict(case['calibration']),
                                        'monitoring': asdict(case['monitoring']), 'policy': case['policy']})
                        lower = None
                        if ci == 0:
                            lower = hoeffding_family_power_lower(cal.denominators[:, :4], mon.denominators[:, :4],
                                                                 envelope, floor, channel_noise=.03, alpha=ALPHA)
                            radius = math.sqrt(math.log(protocol['mean_cells']/protocol['mean_failure_budget'])/(2*trials))
                            means.append({**key, 'mean_conditional_lower': float(lower.mean()),
                                          'one_sided_radius': radius,
                                          'unconditional_power_lower_joint95': max(0., float(lower.mean())-radius),
                                          'nonzero_rows': int(np.count_nonzero(lower)), 'trials': trials})
                        for method, alarm in decisions.items():
                            count = int(alarm.sum())
                            cells.append({**key, 'method': method, 'alarms': count, 'trials': trials,
                                          'alarm_rate': count/trials, 'ci95_marginal': interval(count, trials, .05),
                                          'ci_joint95': interval(count, trials, protocol['rate_failure_budget']/protocol['rate_cells']),
                                          'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                                          'mean_monitoring_disclosed': float(mon.disclosed.mean()),
                                          'mean_remaining_candidates': float(mon.remaining_candidates.mean())})
                        for row_id in range(trials):
                            row = {**key, 'trial': row_id,
                                   'hoeffding_conditional_lower': '' if lower is None else float(lower[row_id])}
                            for stage, batch in (('cal', cal), ('mon', mon)):
                                for j, coord in enumerate(COORDINATES):
                                    row[f'{stage}_{coord}_numerator'] = int(batch.numerators[row_id, j])
                                    row[f'{stage}_{coord}_denominator'] = int(batch.denominators[row_id, j])
                                for j, coord in enumerate(COORDINATES[:4]):
                                    row[f'{stage}_{coord}_sifted'] = int(batch.sifted[row_id, j])
                            row.update({method: int(decisions[method][row_id]) for method in METHODS})
                            writer.writerow(row)
                        pairs += trials
    if len(cells) != protocol['rate_cells'] or len(means) != protocol['mean_cells']:
        raise ArithmeticError('Frozen result cell counts did not reconcile')
    payload = {'protocol': protocol, 'public_pairs': pairs, 'cohorts': cohorts,
               'cells': cells, 'mean_bounds': means, 'source_sha256': hashes,
               'output_sha256': {'public_counts.csv.gz': hashlib.sha256(archive.read_bytes()).hexdigest()},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    (output/'pilot.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--trials', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new directory to preserve all results')
    result = run(args.output, args.trials)
    print(json.dumps({'status': 'completed', 'public_pairs': result['public_pairs'],
                      'rate_cells': len(result['cells']), 'mean_bounds': len(result['mean_bounds'])}))
