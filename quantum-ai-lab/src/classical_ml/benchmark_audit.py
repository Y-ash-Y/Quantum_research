"""Independent arithmetic audit of serialized benchmark outputs, no model fits."""
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def audit_results(destination):
    destination = Path(destination)
    report = json.loads((destination/'benchmark.json').read_text())
    audit_file = destination/report['provenance']['audit_file']
    if hashlib.sha256(audit_file.read_bytes()).hexdigest() != report['provenance']['audit_sha256']:
        raise ValueError('Audit file hash differs from recorded provenance')
    with gzip.open(audit_file, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    identities = [(r['seed'], r['cohort'], r['session_index']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate session audit identities')
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row['seed']), row['cohort']), []).append(row)
    checks, auc_checks = 0, 0
    for run in report['runs']:
        seed = run['seed']
        train = grouped[(seed, 'training')]
        if len(train) != run['training']['accepted']:
            raise ValueError('Training row count mismatch')
        for row in train:
            if int(row['label']) == 1 and float(row['ir_rate']) == float(row['pns_frac']) == 0:
                raise ValueError('Null positive training policy')
        for cohort_name, metadata in run['cohorts'].items():
            cohort = grouped[(seed, cohort_name)]
            known = sum(int(row['known_bits']) for row in cohort)
            sifted = sum(int(row['sifted_bits']) for row in cohort)
            if (len(cohort) != metadata['accepted'] or known != metadata['known_bits']
                    or sifted != metadata['sifted_bits']
                    or not math.isclose(known/sifted, metadata['observed_pooled_info'], abs_tol=1e-12)):
                raise ValueError('Cohort count/information mismatch')
            for name, method in run['methods'].items():
                scores = np.array([float(row[f'{name}::score']) for row in cohort])
                flags = np.array([int(row[f'{name}::flag']) for row in cohort])
                cutoff = method['cutoff']['value']
                if not np.array_equal(flags, np.zeros(len(scores)) if cutoff is None else scores > cutoff):
                    raise ValueError('Serialized flag does not follow frozen cutoff')
                stats = (method[cohort_name] if cohort_name in ('calibration', 'honest_test')
                         else method['attacks'][cohort_name])
                if (sum(flags) != stats['flagged'] or len(flags) != stats['accepted']
                        or stats['accepted']+stats['rejected'] != stats['attempted']
                        or not math.isclose(sum(flags)/len(flags), stats['rate'], abs_tol=1e-12)):
                    raise ValueError('Metric numerator/denominator mismatch')
                # Independent closed-form recomputation, not the shared helper.
                n, k, z = len(flags), int(sum(flags)), 1.959963984540054
                p = k/n
                den = 1+z*z/n
                center = (p+z*z/(2*n))/den
                radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                bounds = [0.0 if k == 0 else max(0, center-radius),
                          1.0 if k == n else min(1, center+radius)]
                if not np.allclose(bounds, stats['wilson95'], atol=1e-12, rtol=0):
                    raise ValueError('Independent Wilson interval mismatch')
                checks += 1
                if cohort_name == 'calibration' and method['calibrated']:
                    alpha = Fraction(str(report['config']['target_fpr']))
                    rank = math.ceil((len(scores)+1)*(1-alpha))
                    expected = None if rank > len(scores) else sorted(scores)[rank-1]
                    if expected != cutoff or rank != method['cutoff']['rank']:
                        raise ValueError('Conservative rank cutoff mismatch')
                if cohort_name not in ('calibration', 'honest_test'):
                    # Independent Mann-Whitney pair enumeration, not sklearn AUC.
                    h = np.array([float(row[f'{name}::score'])
                                  for row in grouped[(seed, 'honest_test')]])
                    pairs = scores[:, None]-h[None, :]
                    auc = float(((pairs > 0)+0.5*(pairs == 0)).mean())
                    if not math.isclose(auc, stats['auc'], abs_tol=1e-12):
                        raise ValueError('Independent AUC mismatch')
                    auc_checks += 1
        for family, comparisons in run['paired_vs_joint_max'].items():
            cohort = grouped[(seed, family)]
            ref = np.array([int(r['Joint max-tail::flag']) for r in cohort], dtype=bool)
            for name, paired in comparisons.items():
                ml = np.array([int(r[f'{name}::flag']) for r in cohort], dtype=bool)
                only_ml, only_ref = int((ml & ~ref).sum()), int((ref & ~ml).sum())
                if (only_ml != paired['ml_only'] or only_ref != paired['stat_only']
                        or not math.isclose((only_ml-only_ref)/len(ref),
                                            paired['detection_difference'], abs_tol=1e-12)):
                    raise ValueError('Paired detection arithmetic mismatch')
    # Reconcile aggregate rates/AUCs via raw per-run metrics, without benchmark helper.
    for name, method in report['summary'].items():
        series = [('honest_fpr', [r['methods'][name]['honest_test']['rate'] for r in report['runs']])]
        series += [(f'{family}::{metric}', [r['methods'][name]['attacks'][family][key]
                                           for r in report['runs']])
                   for family in method['attacks']
                   for metric, key in (('detection', 'rate'), ('auc', 'auc'))]
        for key, values in series:
            stats = (method[key] if key == 'honest_fpr'
                     else method['attacks'][key.split('::')[0]][key.split('::')[1]])
            mean = sum(values)/len(values)
            sd = math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1))
            if not (math.isclose(mean, stats['mean'], abs_tol=1e-12)
                    and math.isclose(sd, stats['seed_sd'], abs_tol=1e-12)
                    and min(values) == stats['seed_min'] and max(values) == stats['seed_max']):
                raise ValueError('Seed summary mismatch')
    return {'audit_rows': len(rows), 'unique_session_keys': len(identities),
            'rate_checks': checks, 'independent_auc_checks': auc_checks,
            'calibration_cutoffs': 'passed', 'pooled_information': 'passed',
            'paired_differences': 'passed', 'seed_summaries': 'passed',
            'wilson_intervals': 'passed', 'file_hash': 'passed'}
