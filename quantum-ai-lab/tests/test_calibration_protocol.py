"""Public transcript integrity, physical controls and finite-sample coverage."""
import itertools
import math

import numpy as np
import pytest
from scipy.stats import binomtest

from research.calibration_protocol import (
    ObservationLaw, PublicBatch, change_alarm, gate_law, pauli_law,
    sample_public_batch, simultaneous_intervals,
)
from research.identifiability import ir_basis_errors, ir_pauli_weights


def block(errors, sizes, emitted=None):
    y = np.atleast_2d(errors).astype(np.int64)
    n = np.atleast_2d(sizes).astype(np.int64)
    emitted = int(n[0, 4:].sum()) if emitted is None else emitted
    return PublicBatch(emitted, y, n, n[:, :4])


@pytest.mark.parametrize('fraction', [0, .25, 1])
def test_sampling_budget_conservation_and_no_hidden_errors(fraction):
    batch = sample_public_batch(200, 20000, gate_law(.03, .02), fraction,
                                np.random.default_rng(92))
    assert np.all(batch.denominators[:, 4:].sum(axis=1) == 20000)
    assert np.all(batch.disclosed+batch.remaining_candidates == batch.sifted.sum(axis=1))
    assert np.all(batch.numerators[:, :4] <= batch.denominators[:, :4])
    assert not hasattr(batch, 'bit_errors') and not hasattr(batch, 'undisclosed_errors')
    assert not hasattr(batch, 'eve_known')
    if fraction == 0:
        assert not batch.disclosed.any() and not batch.numerators[:, :4].any()
    if fraction == 1:
        assert np.array_equal(batch.denominators[:, :4], batch.sifted)
    with pytest.raises(ValueError):
        batch.numerators[0, 0] = 7


def test_sampling_expected_probabilities_errorblind_selection():
    law = ObservationLaw((.1, .3, .6, .9), .15)
    batch = sample_public_batch(4000, 20000, law, .25, np.random.default_rng(73))
    y, n = batch.numerators.sum(axis=0), batch.denominators.sum(axis=0)
    assert y[:4]/n[:4] == pytest.approx(law.bit_errors, abs=.004)
    assert y[4:]/n[4:] == pytest.approx(-np.expm1(-np.array([.5, .1])*.15), abs=.001)
    assert batch.disclosed.sum()/batch.sifted.sum() == pytest.approx(.25, abs=.002)
    # Uniform bit/basis matching is independent of the unequal error rates.
    assert batch.sifted.sum(axis=0)/batch.sifted.sum() == pytest.approx([.25]*4, abs=.003)


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_empty_and_no_loss_rows_are_not_dropped(method):
    batch = sample_public_batch(20, 1000, gate_law(eta=0), 1, np.random.default_rng(3))
    lo, hi = simultaneous_intervals(batch, method=method)
    assert np.all(lo[:, :4] == 0) and np.all(hi[:, :4] == 1)
    assert not change_alarm(batch, batch, method=method).any()
    zero = sample_public_batch(3, 0, gate_law(), .25, np.random.default_rng(1))
    lo, hi = simultaneous_intervals(zero, method=method)
    assert np.all(lo == 0) and np.all(hi == 1)
    assert not change_alarm(zero, zero, method=method).any()


@pytest.mark.parametrize('n,p,method', list(itertools.product(
    [1, 10, 100], [0, .03, .4, 1], ['hoeffding', 'clopper-pearson'])))
def test_interval_coverage_by_independent_exact_binomial_enumeration(n, p, method):
    errors = np.array([[k]*4+[4*n, 4*n] for k in range(n+1)])
    sizes = np.array([[n]*4+[4*n, 4*n]]*(n+1))
    batch = block(errors, sizes)
    lo, hi = simultaneous_intervals(batch, .05, method)
    missed = math.fsum(math.comb(n, k)*p**k*(1-p)**(n-k)
                      for k in range(n+1) if not lo[k, 0] <= p <= hi[k, 0])
    assert missed <= .05/12+1e-12


@pytest.mark.parametrize('n,k', [(1, 0), (1, 1), (100, 3), (1000, 800), (1000000, 50000)])
def test_cp_matches_separate_binomial_ci_interface(n, k):
    batch = block([k]*4+[4*n, 4*n], [n]*4+[4*n, 4*n])
    lo, hi = simultaneous_intervals(batch, .05, 'clopper-pearson')
    reference = binomtest(k, n).proportion_ci(1-.05/12, method='exact')
    assert (lo[0, 0], hi[0, 0]) == pytest.approx((reference.low, reference.high), abs=1e-10)


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_strict_disjoint_alarm_and_count_only_api(method):
    cal = block([0]*4+[2000, 2000], [1000]*4+[2000, 2000])
    mon = block([1000]*4+[2000, 2000], [1000]*4+[2000, 2000])
    assert change_alarm(cal, mon, method=method).tolist() == [True]
    assert not change_alarm(cal, cal, method=method).any()
    with pytest.raises(ValueError):
        change_alarm(gate_law(), mon, method=method)


@pytest.mark.parametrize('noise,rate,bias', [(0, 1, 1), (.03, .0396, 1), (.2, .7, .2)])
def test_broad_honest_pauli_has_ir_observation_probabilities(noise, rate, bias):
    weights = ir_pauli_weights(noise, rate, bias)
    law = pauli_law(tuple(weights[k] for k in ('I', 'X', 'Y', 'Z')))
    qz, qx = ir_basis_errors(noise, rate, bias)
    assert law.bit_errors == pytest.approx((qz, qz, qx, qx))


def test_stable_pauli_outside_gate_family_remains_a_valid_null():
    law = pauli_law((0, .2, .7, .1))
    assert law.bit_errors == pytest.approx((.9, .9, .8, .8))
    rng = np.random.default_rng(219)
    cal = sample_public_batch(400, 200000, law, 1, rng)
    mon = sample_public_batch(400, 200000, law, .25, rng)
    for method in ('hoeffding', 'clopper-pearson'):
        assert change_alarm(cal, mon, method=method).mean() <= .05


@pytest.mark.parametrize('bad', [True, -1, 1.1, math.nan, math.inf, '0.1'])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        ObservationLaw((bad, 0, 0, 0))
    with pytest.raises(ValueError):
        sample_public_batch(1, 100, gate_law(), bad, np.random.default_rng(2))


@pytest.mark.parametrize('bad', [True, -1, 1.5, 2**53, '100'])
def test_invalid_emitted_counts_rejected(bad):
    with pytest.raises(ValueError):
        sample_public_batch(1, bad, gate_law(), .25, np.random.default_rng(2))


@pytest.mark.parametrize('weights', [(1, 1, 0, 0), (0, 0, 0, 0), (1, 0, 0)])
def test_invalid_pauli_laws_rejected(weights):
    with pytest.raises(ValueError):
        pauli_law(weights)


@pytest.mark.parametrize('kind', ['errors', 'revealed', 'sifted', 'sent', 'dtype'])
def test_inconsistent_public_transcripts_rejected(kind):
    y, n, s = np.array([[1]*4+[20, 20]]), np.array([[10]*4+[100, 100]]), np.array([[10]*4])
    if kind == 'errors':
        y[0, 0] = 11
    elif kind == 'revealed':
        n[0, 0] = 11
    elif kind == 'sifted':
        s[0, 0] = 20
    elif kind == 'sent':
        n[0, 4] = 99
    else:
        y = y.astype(float)
    with pytest.raises(ValueError):
        PublicBatch(200, y, n, s)


@pytest.mark.parametrize('alpha,method', [(0, 'hoeffding'), (1, 'hoeffding'),
                                        (1e-20, 'hoeffding'), (.05, 'unknown')])
def test_bad_interval_settings_rejected(alpha, method):
    batch = block([0]*4+[20, 20], [10]*4+[100, 100])
    with pytest.raises(ValueError):
        simultaneous_intervals(batch, alpha, method)
