"""Fixed-window protocol, independent recurrences and artifact-integrity tests."""
from copy import deepcopy
import math

import numpy as np
import pytest
from scipy.stats import binom, fisher_exact

from classical_ml.count_audit import explicit_binomial_cdf, integer_fisher
from classical_ml.power_numerics import recurrence_cdf, recurrence_survival, recurrence_fisher
from classical_ml.window_power_benchmark import WindowConfig, run_benchmark, write_results
from classical_ml.window_power_audit import audit_results


@pytest.mark.parametrize('p', [0, .001, .1, .5, .9, 1])
@pytest.mark.parametrize('n', [0, 1, 5, 20, 100])
def test_binomial_recurrence_exhaustive_small_counts(n, p):
    for k in range(n+1):
        expected = explicit_binomial_cdf(n, k, p)
        assert recurrence_cdf(n, k, p) == pytest.approx(expected, abs=1e-12)
        expected_upper = sum(math.comb(n, j)*p**j*(1-p)**(n-j) for j in range(k, n+1))
        assert recurrence_survival(n, k, p) == pytest.approx(expected_upper, abs=1e-12)


@pytest.mark.parametrize('n,p', [(60000, .072), (60000, .015), (100000, .5), (60000, .999)])
def test_large_count_recurrence_vs_independent_library(n, p):
    sd = math.sqrt(n*p*(1-p))
    for z in (-8, -3, -2, 0, 2, 3, 8):
        k = min(n, max(0, int(n*p+z*sd)))
        assert recurrence_cdf(n, k, p) == pytest.approx(binom.cdf(k, n, p), abs=3e-10)
        assert recurrence_survival(n, k, p) == pytest.approx(binom.sf(k-1, n, p), abs=3e-10)


def test_integer_fisher_recurrence_exactly_matches_all_small_tables():
    for nz in range(7):
        for nx in range(7):
            for ez in range(nz+1):
                for ex in range(nx+1):
                    assert recurrence_fisher(nz, nx, ez, ex) == integer_fisher(nz, nx, ez, ex)


@pytest.mark.parametrize('nz,nx,ez,ex', [(2200, 2100, 30, 150), (2000, 2000, 90, 90),
                                     (3000, 3100, 75, 117), (0, 3000, 0, 12)])
def test_large_fisher_tables_vs_independent_library(nz, nx, ez, ex):
    expected = fisher_exact([[ez, nz-ez], [ex, nx-ex]]).pvalue
    assert recurrence_fisher(nz, nx, ez, ex) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('n,k,p', [(-1, 0, .5), (1.5, 0, .5), (True, 0, .5),
    (10, 1.5, .5), (10, 1, math.nan), (10, 1, -1), (10, 1, True)])
def test_invalid_binomial_numerics_rejected(n, k, p):
    for f in (recurrence_cdf, recurrence_survival):
        with pytest.raises(ValueError):
            f(n, k, p)


@pytest.mark.parametrize('table', [(-1, 2, 0, 0), (2, 2, 3, 0), (2.5, 2, 0, 0), (True, 2, 0, 0)])
def test_invalid_fisher_numerics_rejected(table):
    # True == 1: typed cache keys must not reuse valid integer calls here.
    with pytest.raises(ValueError):
        recurrence_fisher(*table)


@pytest.mark.parametrize('kwargs', [{'seeds': (1,)}, {'seeds': (1, 1)}, {'seeds': (-1, 2)},
    {'windows': ()}, {'windows': (6000, 6000)}, {'windows': (20000, 6000)},
    {'windows': (True, 6000)}, {'windows': (1, 6000)}, {'honest_attempts': 0},
    {'attack_attempts': -1}, {'alpha': 0}, {'alpha': 1}, {'alpha': True}])
def test_invalid_protocol_configuration(kwargs):
    with pytest.raises(ValueError):
        WindowConfig(**kwargs)


@pytest.fixture(scope='module')
def small_run():
    return run_benchmark(WindowConfig(seeds=(1117, 2117), windows=(6000, 20000),
                                    honest_attempts=20, attack_attempts=10))


def test_protocol_and_independent_stream_inventory(small_run):
    r, e = small_run
    assert WindowConfig().windows == (6000, 20000, 60000, 120000)
    assert WindowConfig().seeds == (1117, 2117, 3117, 4117, 5117)
    assert len(r['summary']) == 6
    for run in r['runs']:
        assert len(run['cohorts']) == 22 and len(run['spawn_keys']) == 22
        assert len({tuple(k) for k in run['spawn_keys']}) == 22
        for method in run['methods'].values():
            assert 'calibration' not in method
            for key, stat in method['evaluations'].items():
                if 'auc' in stat:
                    assert stat['honest_control'] == '::'.join(key.split('::')[:2])+'::honest'
        assert all(m['label'] == 0 for k, m in run['cohorts'].items() if 'honest gate stress' in k)
    assert all(b.rejected == 0 for run in e for b in run['cohorts'].values())


def test_independent_audit_and_no_overwrite(small_run, tmp_path):
    r, e = small_run
    d = tmp_path/'small'
    write_results(deepcopy(r), e, d)
    a = audit_results(d)
    assert a['audit_rows'] == 560
    assert a['rate_checks'] == 264
    assert a['independent_score_checks'] == 3360
    assert a['independent_confidence_bound_inversions'] == 1120
    assert a['independent_auc_checks'] == 192
    assert a['independent_physical_reference_checks'] == 176
    with pytest.raises(FileExistsError):
        write_results(deepcopy(r), e, d)


@pytest.mark.parametrize('corruption', ['budget', 'window', 'policy', 'control', 'flags', 'bounds', 'physics', 'summary', 'known'])
def test_auditor_rejects_corruptions(small_run, tmp_path, corruption):
    r, e = deepcopy(small_run)
    key = '6000::nominal::PNS moderate'
    if corruption == 'budget':
        r['budget']['decoy'][0] = .01
    elif corruption == 'window':
        r['conditions']['6000']['nominal']['n_pulses'] = 120000
    elif corruption == 'policy':
        r['policies']['PNS moderate']['pns_frac'] = .2
    elif corruption == 'control':
        r['runs'][0]['methods']['Loss-aware joint']['evaluations'][key]['honest_control'] = '20000::nominal::honest'
    elif corruption == 'flags':
        e[0]['flags']['Loss-aware joint'][key][0] = not e[0]['flags']['Loss-aware joint'][key][0]
    elif corruption == 'bounds':
        e[0]['bounds'][key]['signal_lower_joint'][0] += .01
    elif corruption == 'physics':
        r['runs'][0]['cohorts']['6000::nominal::honest']['honest_physical_reference']['q_z'] += .1
    elif corruption == 'summary':
        r['summary']['Loss-aware joint'][key]['alarm_rate']['mean'] += .1
    elif corruption == 'known':
        e[0]['cohorts']['6000::nominal::honest'].known[0] = 1
    d = tmp_path/corruption
    write_results(r, e, d)
    with pytest.raises(ValueError):
        audit_results(d)
