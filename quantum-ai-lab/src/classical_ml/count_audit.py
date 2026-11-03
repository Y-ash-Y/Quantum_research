"""Independent count/metric audit: integer Fisher enumeration and binomial sums."""
import argparse
import csv
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FEATURES = ('qber', 'basis_asymmetry', 'signal_gain', 'decoy_ratio')


def _close(actual, expected, message):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(message)


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _spread(values, recorded):
    mean = sum(values)/len(values)
    sd = math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1))
    for key, expected in [('mean', mean), ('seed_sd', sd),
                          ('seed_min', min(values)), ('seed_max', max(values))]:
        _close(recorded[key], expected, 'Seed summary mismatch')


COUNT_FIELDS = ('z_sifted', 'x_sifted', 'z_errors', 'x_errors', 'signal_sent',
                'decoy_sent', 'signal_detected', 'decoy_detected')
EXACT_NAMES = ('Count Fisher basis', 'Count binomial decoy', 'Count Bonferroni joint')


@lru_cache(maxsize=65536)
def integer_fisher(nz, nx, ez, ex):
    total, errors = nz+nx, ez+ex
    observed_weight = math.comb(nz, ez)*math.comb(nx, ex)
    numerator = 0
    for k in range(max(0, errors-nx), min(nz, errors)+1):
        weight = math.comb(nz, k)*math.comb(nx, errors-k)
        if weight <= observed_weight:
            numerator += weight
    return numerator/math.comb(total, errors)


@lru_cache(maxsize=65536)
def explicit_binomial_cdf(n, observed, probability):
    if probability == 0 or observed == n:
        return 1.0
    if probability == 1:
        return 0.0
    # Sum in log space: no SciPy binomial or production-score helper.
    logs = [math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
            + k*math.log(probability)+(n-k)*math.log1p(-probability)
            for k in range(observed+1)]
    largest = max(logs)
    return min(1.0, math.exp(largest)*sum(math.exp(v-largest) for v in logs))


def _manual_auc(attack, honest):
    ordered = np.sort(honest)
    lower = np.searchsorted(ordered, attack, side='left')
    upper = np.searchsorted(ordered, attack, side='right')
    return float(sum(lower+0.5*(upper-lower))/(len(attack)*len(honest)))


def audit_results(destination, *, check_current_sources=True):
    destination = Path(destination)
    report = json.loads((destination/'counts.json').read_text())
    if report['schema_version'] != 'counts-1':
        raise ValueError('Unsupported counts schema')
    provenance = report['provenance']
    if provenance['audit_file'] != 'session_audit.csv.gz':
        raise ValueError('Unexpected audit filename')
    audit_file = destination/provenance['audit_file']
    if hashlib.sha256(audit_file.read_bytes()).hexdigest() != provenance['audit_sha256']:
        raise ValueError('Audit file hash mismatch')
    if check_current_sources:
        root = Path(__file__).resolve().parents[1]
        for source, digest in provenance['source_sha256'].items():
            path = (root/source).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Current source fingerprint mismatch')
    with gzip.open(audit_file, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    identities = [(r['seed'], r['cohort'], r['session_index']) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('Duplicate session identities')
    groups = {}
    for row in rows:
        groups.setdefault((int(row['seed']), row['cohort']), []).append(row)
    runs = report['runs']
    if [r['seed'] for r in runs] != report['config']['seeds']:
        raise ValueError('Seed inventory mismatch')
    expected_keys = {(r['seed'], key) for r in runs for key in ('training', *r['cohorts'])}
    if set(groups) != expected_keys:
        raise ValueError('Unexpected or missing evidence cohorts')
    rate_checks = auc_checks = exact_checks = 0
    if report['count_fields'] != list(COUNT_FIELDS):
        raise ValueError('Count schema mismatch')
    expected_probability = -math.expm1(-report['count_null']['mu_decoy']*report['count_null']['eta'])
    _close(report['count_null']['decoy_click_probability'], expected_probability, 'Physical null mismatch')
    methods = set(runs[0]['methods'])
    if len(runs) < 2 or set(report['summary']) != methods or not set(EXACT_NAMES) <= methods:
        raise ValueError('Method/summary inventory mismatch')
    for run in runs:
        seed = run['seed']
        training = groups[(seed, 'training')]
        X = np.array([[float(r[f]) for f in FEATURES] for r in training])
        y = np.array([int(r['label']) for r in training])
        policies = np.array([[float(r[f]) for f in ('ir_rate', 'basis_bias', 'pns_frac')]
                             for r in training])
        if (len(training) != run['training']['accepted']
                or _digest(X, y, policies) != run['training']['sha256']
                or set(y) != {0, 1} or not np.isfinite(X).all()
                or np.any((y == 1) & (policies[:, 0] == 0) & (policies[:, 2] == 0))):
            raise ValueError('Training provenance mismatch')
        if set(run['methods']) != methods:
            raise ValueError('Method inventory mismatch')
        for key, meta in run['cohorts'].items():
            cohort = groups[(seed, key)]
            X = np.array([[float(r[f]) for f in FEATURES] for r in cohort])
            known = np.array([int(r['known_bits']) for r in cohort])
            sifted = np.array([int(r['sifted_bits']) for r in cohort])
            ids = np.array([int(r['session_index']) for r in cohort])
            if (not np.isfinite(X).all() or np.any(known < 0) or np.any(known > sifted)
                    or np.any(sifted < 20) or np.any(ids < 0) or np.any(ids >= meta['attempted'])
                    or len(cohort) != meta['accepted']
                    or meta['accepted']+meta['rejected'] != meta['attempted']
                    or int(known.sum()) != meta['known_bits']
                    or int(sifted.sum()) != meta['sifted_bits']
                    or _digest(X, known, sifted, ids) != meta['sha256']):
                raise ValueError('Cohort provenance mismatch')
            _close(known.sum()/sifted.sum(), meta['observed_pooled_info'], 'Pooled information mismatch')
            if 'feature_means' in meta and not np.allclose(
                    X.mean(axis=0), meta['feature_means'], rtol=0, atol=1e-12):
                raise ValueError('Feature mean mismatch')
            settings = (report['config'] if key == 'calibration'
                        else report['conditions'][meta['condition']])
            for row in cohort:
                if (int(row['label']) != meta['label'] or row['condition'] != meta['condition']
                        or int(row['n_pulses']) != settings['n_pulses']
                        or float(row['channel_noise']) != settings['channel_noise']
                        or any(float(row[p]) != v for p, v in meta['policy'].items())):
                    raise ValueError('Condition or policy metadata mismatch')
            C = np.array([[int(row[f]) for f in COUNT_FIELDS] for row in cohort])
            if _digest(C) != meta['counts_sha256']:
                raise ValueError('Count fingerprint mismatch')
            expected_scores = {name: [] for name in EXACT_NAMES}
            for row, features, counts in zip(cohort, X, C):
                nz, nx, ez, ex, ns, nd, ds, dd = map(int, counts)
                if (min(counts) < 0 or nz == 0 or nx == 0 or nd == 0 or ds == 0
                        or ez > nz or ex > nx or ds > ns or dd > nd
                        or nz+nx > ds+dd or ns+nd != settings['n_pulses']
                        or nz+nx != int(row['sifted_bits'])):
                    raise ValueError('Impossible count telemetry')
                expected_features = [(ez+ex)/(nz+nx), abs(ez/nz-ex/nx), ds/ns, (dd/nd)/(ds/ns)]
                if not np.allclose(features, expected_features, atol=1e-12, rtol=0):
                    raise ValueError('Count-derived feature mismatch')
                basis, decoy = integer_fisher(nz, nx, ez, ex), explicit_binomial_cdf(nd, dd, expected_probability)
                for name, value in zip(EXACT_NAMES, (-basis, -decoy, -min(1.0, 2*min(basis, decoy)))):
                    expected_scores[name].append(value)
                exact_checks += 3
            for name, method in run['methods'].items():
                scores = np.array([float(r[f'{name}::score']) for r in cohort])
                flags = np.array([int(r[f'{name}::flag']) for r in cohort])
                cutoff = method['cutoff']['value']
                operator = method['cutoff']['operator']
                if operator != ('>=' if name in EXACT_NAMES else '>'):
                    raise ValueError('Incorrect cutoff operator')
                expected_flags = (np.zeros(len(scores)) if cutoff is None else
                                  scores >= cutoff if operator == '>=' else scores > cutoff)
                if name in EXACT_NAMES:
                    if cutoff != -report['config']['target_fpr'] or method['calibrated']:
                        raise ValueError('Exact test was incorrectly calibrated')
                    if not np.allclose(scores, expected_scores[name], rtol=0, atol=1e-10):
                        raise ValueError('Independent exact p-value mismatch')
                if not np.isfinite(scores).all() or not np.array_equal(flags, expected_flags):
                    raise ValueError('Frozen cutoff/flag mismatch')
                stats = (method['calibration'] if key == 'calibration'
                         else method['evaluations'][key])
                n, k = len(scores), int(sum(flags))
                if (n != stats['accepted'] or k != stats['flagged']
                        or stats['attempted'] != meta['attempted']
                        or stats['rejected'] != meta['rejected']):
                    raise ValueError('Rate denominator mismatch')
                _close(stats['rate'], k/n, 'Rate mismatch')
                z = 1.959963984540054
                p, den = k/n, 1+z*z/n
                center = (p+z*z/(2*n))/den
                radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                bounds = (0 if k == 0 else max(0, center-radius),
                          1 if k == n else min(1, center+radius))
                if not np.allclose(bounds, stats['wilson95'], atol=1e-12, rtol=0):
                    raise ValueError('Wilson interval mismatch')
                rate_checks += 1
                if key == 'calibration' and method['calibrated']:
                    rank = math.ceil((n+1)*(1-Fraction(str(report['config']['target_fpr']))))
                    expected = None if rank > n else sorted(scores)[rank-1]
                    if expected != cutoff or rank != method['cutoff']['rank']:
                        raise ValueError('Calibration cutoff mismatch')
                if meta['label']:
                    control = f"{meta['condition']}::honest"
                    if stats['honest_control'] != control:
                        raise ValueError('AUC uses an unmatched honest condition')
                    h = np.sort([float(r[f'{name}::score']) for r in groups[(seed, control)]])
                    # Mann-Whitney counts with half weight for ties, independently
                    # of sklearn and without allocating the full pair matrix.
                    left = np.searchsorted(h, scores, side='left')
                    right = np.searchsorted(h, scores, side='right')
                    auc = float(sum(left + 0.5*(right-left))/(len(h)*n))
                    _close(stats['auc'], auc, 'Condition-matched AUC mismatch')
                    auc_checks += 1
        for name in methods:
            attack = [float(row[f'{name}::score']) for row in groups[(seed, 'nominal::IR unbiased')]]
            mimic = [float(row[f'{name}::score']) for row in groups[(seed, 'noise mimic::honest')]]
            _close(run['methods'][name]['mimic_unbiased_ir_auc'], _manual_auc(attack, mimic),
                   'Noise-mimic AUC mismatch')
            auc_checks += 1
    for name, recorded in report['mimic_unbiased_ir_auc'].items():
        _spread([r['methods'][name]['mimic_unbiased_ir_auc'] for r in runs], recorded)
    for name, cohorts in report['summary'].items():
        if set(cohorts) != set(runs[0]['methods'][name]['evaluations']):
            raise ValueError('Summary cohort inventory mismatch')
        for key, recorded in cohorts.items():
            _spread([r['methods'][name]['evaluations'][key]['rate'] for r in runs], recorded['alarm_rate'])
            if 'auc' in recorded:
                _spread([r['methods'][name]['evaluations'][key]['auc'] for r in runs], recorded['auc'])
    return {'audit_rows': len(rows), 'rate_checks': rate_checks,
            'independent_auc_checks': auc_checks, 'independent_exact_pvalue_checks': exact_checks,
            'frozen_cutoffs': 'passed', 'count_derived_features': 'passed',
            'condition_matching': 'passed', 'seed_summaries': 'passed',
            'wilson_intervals': 'passed', 'row_provenance': 'passed',
            'current_source_fingerprints': 'passed' if check_current_sources else 'not checked'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(audit_results(parser.parse_args().destination), indent=2))
