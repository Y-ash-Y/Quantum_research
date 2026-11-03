"""Reconcile every saved public row; independently check sampled alarm decisions."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import COORDINATES, PublicBatch, change_alarm


def independent_interval(k, n, alpha, method):
    if n == 0:
        return 0., 1.
    if method == 'hoeffding':
        radius = math.sqrt(math.log(24/alpha)/(2*n))
        return max(0., k/n-radius), min(1., k/n+radius)
    bounds = binomtest(k, n).proportion_ci(1-alpha/12, method='exact')
    return bounds.low, bounds.high


def audit(directory):
    summary = json.loads((directory/'pilot.json').read_text())
    raw_path = directory/'public_counts.csv'
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != summary['public_counts_sha256']:
        raise ValueError('Public-count file hash mismatch')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in summary['source_sha256'].items():
        path = (root/relative).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source hash mismatch; preserve the matching source version')
    groups = defaultdict(list)
    with raw_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[(int(row['calibration_emitted']), int(row['monitoring_emitted']),
                    row['scenario'])].append(row)
    total = sum(len(rows) for rows in groups.values())
    if total != summary['paired_transcripts']:
        raise ValueError('Public-row count mismatch')
    cells = {(c['calibration_emitted'], c['monitoring_emitted'], c['scenario'], c['method']): c
             for c in summary['cells']}
    if len(cells) != len(summary['cells']) or len(cells) != 2*len(groups):
        raise ValueError('Duplicate or missing result cells')
    replayed = independent = 0
    for key, rows in groups.items():
        if sorted(int(r['trial']) for r in rows) != list(range(summary['trials_per_cell'])):
            raise ValueError('Duplicate or missing trial IDs')
        batches = []
        for stage, emitted in [('cal', key[0]), ('mon', key[1])]:
            ys, ns, ss = [], [], []
            for row in rows:
                y = [int(row[f'{stage}_{j}_numerator']) for j in COORDINATES]
                n = [int(row[f'{stage}_{j}_denominator']) for j in COORDINATES]
                s = [int(row[f'{stage}_{j}_sifted']) for j in COORDINATES[:4]]
                if (any(not 0 <= a <= b for a, b in zip(y, n))
                        or any(not 0 <= a <= b for a, b in zip(n[:4], s))
                        or sum(n[4:]) != emitted or sum(s) > sum(y[4:])
                        or (stage == 'cal' and n[:4] != s)):
                    raise ValueError('Public-resource conservation failed')
                ys.append(y); ns.append(n); ss.append(s)
            batches.append(PublicBatch(emitted, np.array(ys), np.array(ns), np.array(ss)))
        cal, mon = batches
        for method in ('hoeffding', 'clopper-pearson'):
            alarms = change_alarm(cal, mon, summary['alpha_per_rule_per_pair'], method)
            recorded = np.array([int(row[method]) for row in rows])
            if not np.array_equal(alarms, recorded):
                raise ValueError('Alarm replay mismatch')
            cell = cells[key+(method,)]
            count, trials = int(recorded.sum()), len(rows)
            ci = binomtest(count, trials).proportion_ci(.95, method='exact')
            if (count != cell['alarms'] or trials != cell['trials']
                    or count/trials != cell['alarm_rate']
                    or not np.allclose([ci.low, ci.high], cell['alarm_rate_ci95'], rtol=0, atol=1e-12)
                    or cal.disclosed.mean() != cell['mean_calibration_sacrificed']
                    or mon.disclosed.mean() != cell['mean_monitoring_disclosed']
                    or mon.remaining_candidates.mean() != cell['mean_remaining_candidates']):
                raise ValueError('Summary/resource reconciliation failed')
            replayed += trials
            # Independent scalar Hoeffding / binomial CI implementations, three
            # predeclared rows per cell, not a claim of independent full coverage.
            for row_id in sorted({0, len(rows)//2, len(rows)-1}):
                flag = False
                for j in range(6):
                    intervals = [independent_interval(int(b.numerators[row_id, j]),
                                 int(b.denominators[row_id, j]),
                                 summary['alpha_per_rule_per_pair'], method) for b in batches]
                    (clo, chi), (mlo, mhi) = intervals
                    flag |= clo > mhi or mlo > chi
                if bool(alarms[row_id]) != flag:
                    raise ValueError('Independent sampled decision mismatch')
                independent += 1
    return {'status': 'passed', 'public_pairs_reconciled': total,
            'decision_replays': replayed, 'independent_sampled_decisions': independent,
            'summary_cells_reconciled': len(cells),
            'source_hashes_checked': len(summary['source_sha256']),
            'limits': 'Replay shares the scorer; independent scalar decisions cover sampled rows only.'}


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
