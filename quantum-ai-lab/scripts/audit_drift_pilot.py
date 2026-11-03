"""Replay the bounded-drift archive and independently check fixed scalar rows."""
import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from run_drift_pilot import (
    ALPHA, METHODS, ROOT, draw, frozen_protocol, interval, plans, scenarios,
)
from audit_calibration_pilot import independent_interval
from research.calibration_protocol import COORDINATES, PublicBatch
from research.drift_detector import DriftEnvelope, drift_change_alarm, hoeffding_family_power_lower


def row_key(row):
    return tuple(int(row[k]) for k in ('plan_id', 'floor_id', 'allowance_id', 'cohort_id'))


def close(got, expected, label):
    if not np.allclose(got, expected, rtol=0, atol=1e-12):
        raise ValueError(f'{label} mismatch')


def scalar_lower(nc, nm, envelope, floor, noise, alpha):
    if min(*nc, *nm) == 0:
        return 0.
    gamma = (1-noise)*floor-envelope.z_error-envelope.x_error
    if gamma <= 0:
        return 0.
    h = max(math.sqrt(math.log(24/alpha)/(2*m))+math.sqrt(math.log(24/alpha)/(2*n))
            for m, n in zip(nc, nm))
    s = max(1/m+1/n for m, n in zip(nc, nm))
    t = max(0., gamma/2-h)
    return -math.expm1(-2*t*t/s)


def audit(directory):
    directory = Path(directory)
    saved = json.loads((directory/'pilot.json').read_text())
    protocol = saved['protocol']
    if protocol != frozen_protocol(protocol['trials']):
        raise ValueError('Frozen protocol mismatch')
    for relative, digest in saved['source_sha256'].items():
        p = (ROOT/relative).resolve()
        if ROOT not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Source fingerprint mismatch')
    archive = directory/'public_counts.csv.gz'
    if saved['output_sha256'] != {'public_counts.csv.gz': hashlib.sha256(archive.read_bytes()).hexdigest()}:
        raise ValueError('Archive fingerprint mismatch')
    cohort_map = {row_key(c): c for c in saved['cohorts']}
    cells = {row_key(c)+(c['method'],): c for c in saved['cells']}
    means = {row_key(m)[:3]: m for m in saved['mean_bounds']}
    if (len(cohort_map) != 216 or len(saved['cohorts']) != 216
            or len(cells) != 432 or len(saved['cells']) != 432
            or len(means) != 24 or len(saved['mean_bounds']) != 24):
        raise ValueError('Duplicate or missing summary cells')
    expected = []
    plan_map = dict(plans())
    for plan_id, _ in plans():
        for fi, floor in enumerate(protocol['floors']):
            for ai, multiplier in enumerate(protocol['drift_multipliers']):
                env, cases = scenarios(floor, multiplier)
                for ci, case in enumerate(cases):
                    key = (plan_id, fi, ai, ci)
                    meta = {'plan_id': plan_id, 'floor_id': fi, 'allowance_id': ai, 'cohort_id': ci,
                            'scenario': case['name'], 'category': case['category'],
                            'floor': floor, 'drift_multiplier': multiplier, 'envelope': asdict(env),
                            'calibration': asdict(case['calibration']), 'monitoring': asdict(case['monitoring']),
                            'policy': case['policy']}
                    if cohort_map.get(key) != json.loads(json.dumps(meta)):
                        raise ValueError('Scenario/law/envelope metadata mismatch')
                    expected.append((key, env, case))
    total = replays = sampled = scalar_bounds = 0
    seen = []
    with gzip.open(archive, 'rt', newline='') as stream:
        groups = itertools.groupby(csv.DictReader(stream), key=row_key)
        for group_no, (key, group) in enumerate(groups):
            if group_no >= len(expected) or key != expected[group_no][0]:
                raise ValueError('Duplicate, missing or reordered cohort')
            _, envelope, case = expected[group_no]
            seen.append(key)
            rows = list(group)
            if [int(r['trial']) for r in rows] != list(range(protocol['trials'])):
                raise ValueError('Duplicate, missing or reordered trial ID')
            plan = plan_map[key[0]]
            batches = []
            for stage, emitted in (('cal', plan.calibration_emitted), ('mon', plan.monitoring_emitted)):
                y = np.array([[int(r[f'{stage}_{j}_numerator']) for j in COORDINATES] for r in rows])
                n = np.array([[int(r[f'{stage}_{j}_denominator']) for j in COORDINATES] for r in rows])
                s = np.array([[int(r[f'{stage}_{j}_sifted']) for j in COORDINATES[:4]] for r in rows])
                batch = PublicBatch(emitted, y, n, s)
                if stage == 'cal' and not np.array_equal(n[:, :4], s):
                    raise ValueError('Calibration sacrifice mismatch')
                batches.append(batch)
            cal, mon = batches
            regenerated = draw(protocol['trials'], plan, case, key)
            for a, b in zip(batches, regenerated):
                for name in ('numerators', 'denominators', 'sifted'):
                    if not np.array_equal(getattr(a, name), getattr(b, name)):
                        raise ValueError('Seed/law reproduction mismatch')
            for method in METHODS:
                alarms = drift_change_alarm(cal, mon, envelope, alpha=ALPHA, method=method)
                recorded = np.array([int(r[method]) for r in rows])
                if not np.array_equal(alarms, recorded):
                    raise ValueError('Decision replay mismatch')
                cell = cells[key+(method,)]
                count, trials = int(recorded.sum()), len(rows)
                if count != cell['alarms'] or trials != cell['trials'] or count/trials != cell['alarm_rate']:
                    raise ValueError('Rate count mismatch')
                close(cell['ci95_marginal'], interval(count, trials, .05), 'Marginal interval')
                close(cell['ci_joint95'], interval(count, trials, .04/432), 'Simultaneous interval')
                close([cell['mean_calibration_sacrificed'], cell['mean_monitoring_disclosed'],
                       cell['mean_remaining_candidates']],
                      [cal.disclosed.mean(), mon.disclosed.mean(), mon.remaining_candidates.mean()], 'Resource')
                for row_id in sorted({0, trials//2, trials-1}):
                    flag = False
                    for j, d in enumerate(envelope.coordinate_allowances):
                        (clo, chi), (mlo, mhi) = [independent_interval(int(b.numerators[row_id, j]),
                                               int(b.denominators[row_id, j]), ALPHA, method) for b in batches]
                        flag |= mlo-chi > d or clo-mhi > d
                    if bool(alarms[row_id]) != flag:
                        raise ValueError('Independent scalar decision mismatch')
                    sampled += 1
                replays += trials
            if key[3] == 0:
                floor = protocol['floors'][key[1]]
                values = np.array([float(r['hoeffding_conditional_lower']) for r in rows])
                replay = hoeffding_family_power_lower(cal.denominators[:, :4], mon.denominators[:, :4],
                                                      envelope, floor, channel_noise=.03, alpha=ALPHA)
                close(values, replay, 'Conditional lower replay')
                for row_id, value in enumerate(values):
                    close(value, scalar_lower(cal.denominators[row_id, :4], mon.denominators[row_id, :4],
                                               envelope, floor, .03, ALPHA), 'Scalar conditional lower')
                    scalar_bounds += 1
                mean = math.fsum(values)/trials
                radius = math.sqrt(math.log(24/.01)/(2*trials))
                record = means[key[:3]]
                if record['trials'] != trials or record['nonzero_rows'] != np.count_nonzero(values):
                    raise ValueError('Mean bound count mismatch')
                close([record['mean_conditional_lower'], record['one_sided_radius'], record['unconditional_power_lower_joint95']],
                      [mean, radius, max(0., mean-radius)], 'Mean bound')
            elif any(r['hoeffding_conditional_lower'] != '' for r in rows):
                raise ValueError('Unexpected conditional-bound rows')
            total += len(rows)
    if len(seen) != 216 or total != saved['public_pairs'] or total != 216*protocol['trials']:
        raise ValueError('Public pair/cohort reconciliation mismatch')
    return {'status': 'passed', 'public_pairs_reconciled': total, 'decision_replays': replays,
            'rate_cells': len(cells), 'independent_sampled_decisions': sampled,
            'independent_scalar_lower_rows': scalar_bounds, 'seed_reproduced_pairs': total,
            'source_fingerprints': len(saved['source_sha256']),
            'limits': 'Scalar decisions cover first/middle/last only; CP references share SciPy. No hardware/rounding/priority proof.'}


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
