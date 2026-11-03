"""Run the frozen Step 12 direct-test comparison, preserving all prior evidence."""
import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from run_drift_pilot import ROOT, ALPHA, FLOORS, MULTIPLIERS, interval, plans, scenarios
from research.calibration_protocol import COORDINATES, sample_public_batch
from research.drift_detector import drift_change_alarm
from research.direct_drift_detector import direct_drift_alarm

SEED = 2026100912
METHODS = ('hoeffding', 'clopper-pearson', 'direct-bernstein')


def frozen_protocol(trials=512):
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 1024:
        raise ValueError('This comparison allows 1..1024 trials per cohort')
    return {'schema_version': 1, 'seed': SEED, 'alpha': ALPHA, 'trials': trials,
            'floors': list(FLOORS), 'drift_multipliers': list(MULTIPLIERS),
            'methods': list(METHODS), 'channel_noise': .03, 'eta': .15,
            'cohorts': 216, 'rate_cells': 648, 'effect_cells': 216,
            'rate_failure_budget': .04, 'effect_failure_budget': .01,
            'plans': [{'id': i, **asdict(p), **p.expected_resources()} for i, p in plans()],
            'scope': 'Fixed-noise iid IR probes versus bounded honest between-block error drift; stable gains.'}


def draw(trials, plan, case, ids):
    children = np.random.SeedSequence([SEED, *ids]).spawn(2)
    return (sample_public_batch(trials, plan.calibration_emitted, case['calibration'], 1.,
                                np.random.default_rng(children[0])),
            sample_public_batch(trials, plan.monitoring_emitted, case['monitoring'],
                                plan.monitoring_disclosure, np.random.default_rng(children[1])))


def score(cal, mon, envelope):
    return {m: (direct_drift_alarm(cal, mon, envelope, alpha=ALPHA) if m=='direct-bernstein'
                else drift_change_alarm(cal, mon, envelope, alpha=ALPHA, method=m)) for m in METHODS}


def paired_effect(direct, reference, failure):
    if (direct.ndim != 1 or direct.shape != reference.shape or not len(direct)
            or direct.dtype != bool or reference.dtype != bool):
        raise ValueError('Expected paired boolean decisions')
    total = len(direct)
    win = int(np.sum(direct & ~reference))
    loss = int(np.sum(~direct & reference))
    both = int(np.sum(direct & reference))
    neither = int(np.sum(~direct & ~reference))
    cw, cl = interval(win, total, failure/2), interval(loss, total, failure/2)
    return {'wins_direct': win, 'losses_direct': loss, 'both_alarm': both,
            'neither_alarm': neither, 'trials': total,
            'net_alarm_rate_difference': (win-loss)/total,
            'ci_joint95': [cw[0]-cl[1], cw[1]-cl[0]]}


def source_hashes():
    names = ['docs/direct-drift-test.md', 'docs/direct-drift-protocol.md',
             'scripts/run_direct_drift_pilot.py', 'scripts/audit_direct_drift_pilot.py',
             'scripts/run_drift_pilot.py', 'scripts/audit_calibration_pilot.py',
             'src/research/direct_drift_detector.py', 'src/research/drift_detector.py',
             'src/research/calibration_protocol.py', 'src/research/allocation_frontier.py',
             'src/research/identifiability.py']
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def run(output, trials=512):
    protocol, hashes = frozen_protocol(trials), source_hashes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = output/'public_counts.csv.gz'
    columns = ['plan_id', 'floor_id', 'allowance_id', 'cohort_id', 'trial']
    for stage in ('cal', 'mon'):
        for coord in COORDINATES:
            columns += [f'{stage}_{coord}_numerator', f'{stage}_{coord}_denominator']
        columns += [f'{stage}_{j}_sifted' for j in COORDINATES[:4]]
    columns += list(METHODS)
    cohorts, cells, effects = [], [], []
    with gzip.open(archive, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for plan_id, plan in plans():
            for fi, floor in enumerate(FLOORS):
                for ai, multiplier in enumerate(MULTIPLIERS):
                    envelope, cases = scenarios(floor, multiplier)
                    for ci, case in enumerate(cases):
                        ids = (plan_id, fi, ai, ci)
                        key = dict(zip(columns[:4], ids))
                        cal, mon = draw(trials, plan, case, ids)
                        decisions = score(cal, mon, envelope)
                        cohorts.append({**key, 'scenario': case['name'], 'category': case['category'],
                                        'floor': floor, 'drift_multiplier': multiplier, 'envelope': asdict(envelope),
                                        'calibration': asdict(case['calibration']), 'monitoring': asdict(case['monitoring']),
                                        'policy': case['policy']})
                        for method, alarm in decisions.items():
                            count = int(alarm.sum())
                            cells.append({**key, 'method': method, 'alarms': count, 'trials': trials,
                                          'alarm_rate': count/trials, 'ci95_marginal': interval(count, trials, .05),
                                          'ci_joint95': interval(count, trials, .04/648),
                                          'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                                          'mean_monitoring_disclosed': float(mon.disclosed.mean()),
                                          'mean_remaining_candidates': float(mon.remaining_candidates.mean())})
                        effects.append({**key, **paired_effect(decisions['direct-bernstein'],
                                                               decisions['clopper-pearson'], .01/216)})
                        for row_id in range(trials):
                            row = {**key, 'trial': row_id}
                            for stage, batch in (('cal', cal), ('mon', mon)):
                                for j, coord in enumerate(COORDINATES):
                                    row[f'{stage}_{coord}_numerator'] = int(batch.numerators[row_id, j])
                                    row[f'{stage}_{coord}_denominator'] = int(batch.denominators[row_id, j])
                                for j, coord in enumerate(COORDINATES[:4]):
                                    row[f'{stage}_{coord}_sifted'] = int(batch.sifted[row_id, j])
                            row.update({m: int(decisions[m][row_id]) for m in METHODS})
                            writer.writerow(row)
    if (len(cohorts), len(cells), len(effects)) != (216, 648, 216):
        raise ArithmeticError('Frozen comparison dimensions failed')
    result = {'protocol': protocol, 'public_pairs': trials*216, 'cohorts': cohorts,
              'cells': cells, 'effects': effects, 'source_sha256': hashes,
              'output_sha256': {'public_counts.csv.gz': hashlib.sha256(archive.read_bytes()).hexdigest()},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}}
    (output/'pilot.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    result = run(args.output, args.trials)
    print(json.dumps({'status': 'completed', 'public_pairs': result['public_pairs'],
                      'rate_cells': len(result['cells']), 'effect_cells': len(result['effects'])}))
