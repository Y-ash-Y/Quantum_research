"""Audit every fresh count/decision and fixed independent scalar checks."""
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
from scipy.stats import binomtest

from run_direct_drift_pilot import ROOT, ALPHA, draw, frozen_protocol, interval, plans, scenarios, score, source_hashes
from audit_calibration_pilot import independent_interval
from research.calibration_protocol import COORDINATES, PublicBatch


def row_key(row):
    return tuple(int(row[k]) for k in ('plan_id', 'floor_id', 'allowance_id', 'cohort_id'))


def close(a, b, name):
    if not np.allclose(a, b, rtol=0, atol=1e-12):
        raise ValueError(f'{name} mismatch')


def scalar_direct(cal, mon, row, envelope):
    flag = False
    groups = ((0, 1), (2, 3), (4,), (5,))
    for j, (indices, drift) in enumerate(zip(groups, (envelope.z_error, envelope.x_error, 0., 0.))):
        counts = [(sum(int(b.numerators[row, k]) for k in indices),
                   sum(int(b.denominators[row, k]) for k in indices)) for b in (cal, mon)]
        if min(n for _, n in counts) == 0:
            continue
        diff = counts[1][0]/counts[1][1]-counts[0][0]/counts[0][1]
        for direction in (0, 1):
            y, n = counts[direction]
            _, other_n = counts[1-direction]
            if j < 2:
                u = min(.5, binomtest(y, n, alternative='less').proportion_ci(1-ALPHA/16).high)
                other_u = min(.5, u+drift)
                v = u*(1-u)/n + other_u*(1-other_u)/other_n
                b, l = max(1/n, 1/other_n), math.log(16/ALPHA)
                radius = (2*b*l/3+math.sqrt((2*b*l/3)**2+8*v*l))/2
            else:
                radius = math.sqrt(math.log(8/ALPHA)*(1/n+1/other_n)/2)
            flag |= (diff if direction == 0 else -diff) > drift+radius
    return flag


def audit(directory):
    directory = Path(directory)
    saved = json.loads((directory/'pilot.json').read_text())
    if saved['protocol'] != frozen_protocol(saved['protocol']['trials']):
        raise ValueError('Frozen protocol mismatch')
    if set(saved['source_sha256']) != set(source_hashes()):
        raise ValueError('Missing or unexpected source fingerprints')
    for name, digest in saved['source_sha256'].items():
        p = (ROOT/name).resolve()
        if ROOT not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Source fingerprint mismatch')
    archive = directory/'public_counts.csv.gz'
    if saved['output_sha256'] != {'public_counts.csv.gz': hashlib.sha256(archive.read_bytes()).hexdigest()}:
        raise ValueError('Archive fingerprint mismatch')
    cohorts = {row_key(c): c for c in saved['cohorts']}
    cells = {row_key(c)+(c['method'],): c for c in saved['cells']}
    effects = {row_key(e): e for e in saved['effects']}
    if any(len(saved[n]) != size or len(mapping) != size for n, mapping, size in
           [('cohorts', cohorts, 216), ('cells', cells, 648), ('effects', effects, 216)]):
        raise ValueError('Duplicate or missing summary cells')
    expected = []
    for plan_id, plan in plans():
        for fi, floor in enumerate(saved['protocol']['floors']):
            for ai, multiplier in enumerate(saved['protocol']['drift_multipliers']):
                env, cases = scenarios(floor, multiplier)
                for ci, case in enumerate(cases):
                    key = (plan_id, fi, ai, ci)
                    meta = dict(zip(('plan_id', 'floor_id', 'allowance_id', 'cohort_id'), key))
                    meta.update(scenario=case['name'], category=case['category'], floor=floor,
                                drift_multiplier=multiplier, envelope=asdict(env),
                                calibration=asdict(case['calibration']), monitoring=asdict(case['monitoring']),
                                policy=case['policy'])
                    if cohorts.get(key) != json.loads(json.dumps(meta)):
                        raise ValueError('Scenario metadata mismatch')
                    expected.append((key, plan, env, case))
    total = replays = scalar_checks = group_count = 0
    with gzip.open(archive, 'rt', newline='') as stream:
        for group_id, (key, group) in enumerate(itertools.groupby(csv.DictReader(stream), key=row_key)):
            if group_id >= len(expected) or expected[group_id][0] != key:
                raise ValueError('Cohort order/identity mismatch')
            _, plan, env, case = expected[group_id]
            rows = list(group)
            trials = saved['protocol']['trials']
            if [int(r['trial']) for r in rows] != list(range(trials)):
                raise ValueError('Trial order/identity mismatch')
            batches = []
            for stage, emitted in (('cal', plan.calibration_emitted), ('mon', plan.monitoring_emitted)):
                y = np.array([[int(r[f'{stage}_{j}_numerator']) for j in COORDINATES] for r in rows])
                n = np.array([[int(r[f'{stage}_{j}_denominator']) for j in COORDINATES] for r in rows])
                s = np.array([[int(r[f'{stage}_{j}_sifted']) for j in COORDINATES[:4]] for r in rows])
                batches.append(PublicBatch(emitted, y, n, s))
            cal, mon = batches
            for a, b in zip(batches, draw(trials, plan, case, key)):
                for name in ('numerators', 'denominators', 'sifted'):
                    if not np.array_equal(getattr(a, name), getattr(b, name)):
                        raise ValueError('Seed/law reproduction mismatch')
            decisions = score(cal, mon, env)
            for method, alarm in decisions.items():
                recorded = np.array([int(r[method]) for r in rows])
                if not np.array_equal(recorded, alarm.astype(int)):
                    raise ValueError('Decision replay mismatch')
                c = cells[key+(method,)]
                count = int(alarm.sum())
                if c['alarms'] != count or c['trials'] != trials or c['alarm_rate'] != count/trials:
                    raise ValueError('Rate count mismatch')
                close(c['ci95_marginal'], interval(count, trials, .05), 'Marginal interval')
                close(c['ci_joint95'], interval(count, trials, .04/648), 'Joint interval')
                close([c['mean_calibration_sacrificed'], c['mean_monitoring_disclosed'], c['mean_remaining_candidates']],
                      [cal.disclosed.mean(), mon.disclosed.mean(), mon.remaining_candidates.mean()], 'Resource')
                for row in sorted({0, trials//2, trials-1}):
                    if method == 'direct-bernstein':
                        flag = scalar_direct(cal, mon, row, env)
                    else:
                        flag = False
                        for j, d in enumerate(env.coordinate_allowances):
                            (cl, cu), (ml, mu) = [independent_interval(int(b.numerators[row,j]),
                                                  int(b.denominators[row,j]), ALPHA, method) for b in batches]
                            flag |= ml-cu>d or cl-mu>d
                    if bool(alarm[row]) != flag:
                        raise ValueError('Independent scalar decision mismatch')
                    scalar_checks += 1
                replays += trials
            direct, cp = decisions['direct-bernstein'], decisions['clopper-pearson']
            # Four cells reconstructed separately, not through the runner's effect helper.
            a = int(np.count_nonzero(direct & cp))
            b = int(np.count_nonzero(direct & ~cp))
            c = int(np.count_nonzero(~direct & cp))
            d = trials-a-b-c
            e = effects[key]
            if [e['both_alarm'],e['wins_direct'],e['losses_direct'],e['neither_alarm'],e['trials']] != [a,b,c,d,trials]:
                raise ValueError('Paired effect counts mismatch')
            cw, cl = interval(b, trials, .01/(216*2)), interval(c, trials, .01/(216*2))
            close(e['net_alarm_rate_difference'], (b-c)/trials, 'Paired difference')
            close(e['ci_joint95'], [cw[0]-cl[1], cw[1]-cl[0]], 'Paired interval')
            total += trials
            group_count += 1
    if group_count != 216 or total != saved['public_pairs'] or total != 216*saved['protocol']['trials']:
        raise ValueError('Public pair reconciliation mismatch')
    return {'status':'passed', 'public_pairs_reconciled':total, 'decision_replays':replays,
            'rate_cells':648, 'effect_cells':216, 'independent_sampled_decisions':scalar_checks,
            'seed_reproduced_pairs':total, 'source_fingerprints':len(saved['source_sha256']),
            'limits':'Scalar checks cover first/middle/last rows; CP references share SciPy. No rounding/hardware/continuous-power/priority certificate.'}


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
