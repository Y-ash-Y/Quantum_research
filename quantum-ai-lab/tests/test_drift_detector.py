"""Drift-gap decision, binomial coverage and independent conditional power checks."""
from dataclasses import FrozenInstanceError
from functools import lru_cache
from math import comb, fsum, log, sqrt

import numpy as np
import pytest
from scipy.stats import binom

from research.calibration_protocol import (
    ObservationLaw, PublicBatch, change_alarm, pauli_law,
    sample_public_batch, simultaneous_intervals,
)
from research.drift_detector import (
    DriftEnvelope, drift_change_alarm, honest_drift_laws,
    hoeffding_family_power_lower,
)
from research.identifiability import ir_basis_errors


def block(errors, sizes):
    """Valid count fixture; gain intervals all contain one and cannot reject."""
    y = np.atleast_2d(np.asarray(errors, dtype=np.int64))
    n = np.broadcast_to(np.asarray(sizes, dtype=np.int64), y.shape).copy()
    gain_n = np.full((len(y), 2), 4*int(n.max()))
    return PublicBatch(int(gain_n[0].sum()), np.column_stack((y, gain_n)),
                       np.column_stack((n, gain_n)), n)


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_zero_envelope_matches_existing_interval_rule_and_no_mutation(method):
    rng = np.random.default_rng(901)
    c = sample_public_batch(100, 40000, ObservationLaw((.01,)*4), 1, rng)
    m = sample_public_batch(100, 50000, ObservationLaw((.03,)*4), .4, rng)
    before = [x.copy() for x in (c.numerators, c.denominators,
                                 m.numerators, m.denominators)]
    got = drift_change_alarm(c, m, DriftEnvelope(), method=method)
    assert np.array_equal(got, change_alarm(c, m, method=method))
    assert got.dtype == bool and not got.flags.writeable
    for a, b in zip(before, (c.numerators, c.denominators,
                            m.numerators, m.denominators)):
        assert np.array_equal(a, b)


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_strict_boundary_and_both_directions(method):
    c, m = block([[0]*4], 1000), block([[400]*4], 1000)
    _, chi = simultaneous_intervals(c, method=method)
    mlo, _ = simultaneous_intervals(m, method=method)
    gap = float(mlo[0, 0]-chi[0, 0])
    assert 0 < gap < .5
    equal = DriftEnvelope(gap, gap)
    below = DriftEnvelope(np.nextafter(gap, 0), np.nextafter(gap, 0))
    above = DriftEnvelope(np.nextafter(gap, 1), np.nextafter(gap, 1))
    for left, right in ((c, m), (m, c)):
        assert not drift_change_alarm(left, right, equal, method=method)[0]
        assert drift_change_alarm(left, right, below, method=method)[0]
        assert not drift_change_alarm(left, right, above, method=method)[0]


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_basis_allowance_mapping_monotonicity_and_stage_symmetry(method):
    c = block([[0]*4, [0]*4, [0]*4], 1000)
    m = block([[400, 400, 0, 0], [0, 0, 400, 400], [400]*4], 1000)
    assert drift_change_alarm(c, m, DriftEnvelope(.5, 0), method=method).tolist() == [False, True, True]
    assert drift_change_alarm(c, m, DriftEnvelope(0, .5), method=method).tolist() == [True, False, True]
    last = np.ones(3, dtype=bool)
    for d in (0., .05, .2, .4, .5):
        current = drift_change_alarm(c, m, DriftEnvelope(d, d), method=method)
        assert np.all(~current | last)
        assert np.array_equal(current, drift_change_alarm(m, c, DriftEnvelope(d, d), method=method))
        last = current


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
def test_empty_rows_no_disclosure_and_stable_gain_allowance(method):
    rng = np.random.default_rng(3)
    empty = sample_public_batch(4, 0, ObservationLaw((0.,)*4), 1, rng)
    assert not drift_change_alarm(empty, empty, DriftEnvelope(.1, .1), method=method).any()
    zero = sample_public_batch(4, 20000, ObservationLaw((.1,)*4, 0), 1, rng)
    assert not drift_change_alarm(zero, zero, DriftEnvelope(.1, .1), method=method).any()
    # No disclosed monitoring errors can alarm, but gain changes still can.
    c = block([[0]*4], 1000)
    n = np.array([[0]*4+[4000, 4000]])
    unchanged = PublicBatch(8000, np.array([[0]*4+[4000, 4000]]), n, np.zeros((1, 4), dtype=int))
    shifted = PublicBatch(8000, np.array([[0]*6]), n, np.zeros((1, 4), dtype=int))
    assert not drift_change_alarm(c, unchanged, DriftEnvelope(.5, .5), method=method)[0]
    assert drift_change_alarm(c, shifted, DriftEnvelope(.5, .5), method=method)[0]
    assert not drift_change_alarm(empty, zero, DriftEnvelope(), method=method).any()


@pytest.mark.parametrize('cal_z,cal_x,mon_z,mon_x,env', [
    (0, .5, .125, .375, DriftEnvelope(.125, .125)),
    (.25, .125, .25, .125, DriftEnvelope()),
    (.5, 0, 0, .5, DriftEnvelope(.5, .5)),
])
def test_honest_laws_are_validated_and_have_physical_representatives(cal_z, cal_x, mon_z, mon_x, env):
    pair = honest_drift_laws(cal_z, cal_x, mon_z, mon_x, env, eta=.2)
    for law, z, x in zip(pair, (cal_z, mon_z), (cal_x, mon_x)):
        assert law.bit_errors == (z, z, x, x) and law.eta == .2
        assert pauli_law((1-z-x, z, 0, x), .2).bit_errors == law.bit_errors
    with pytest.raises(FrozenInstanceError):
        env.z_error = 0
    assert env.coordinate_allowances == (env.z_error,)*2+(env.x_error,)*2+(0., 0.)


@pytest.mark.parametrize('values', [(.1, .1, .2, .1), (.1, .1, .1, 0),
                                   (.6, .1, .1, .1), (.1, float('nan'), .1, .1)])
def test_invalid_or_out_of_envelope_honest_pair_is_not_clipped(values):
    with pytest.raises(ValueError):
        honest_drift_laws(*values, DriftEnvelope(.02, .02))


def test_overlap_attack_and_honest_pair_have_identical_seeded_public_laws():
    # Exact dyadic probabilities avoid treating rounded boundary arithmetic as proof.
    q0, rate, bias = .125, .5, .5
    qz, qx = ir_basis_errors(2*q0, rate, bias)
    c, honest_m = honest_drift_laws(q0, q0, qz, qx, DriftEnvelope(.125, .125))
    attack_m = ObservationLaw((qz, qz, qx, qx), .15)
    h = sample_public_batch(80, 40000, honest_m, .25, np.random.default_rng(8))
    a = sample_public_batch(80, 40000, attack_m, .25, np.random.default_rng(8))
    for name in ('numerators', 'denominators', 'sifted'):
        assert np.array_equal(getattr(h, name), getattr(a, name))
    reference = sample_public_batch(80, 20000, c, 1, np.random.default_rng(9))
    for method in ('hoeffding', 'clopper-pearson'):
        assert np.array_equal(drift_change_alarm(reference, h, DriftEnvelope(.125, .125), method=method),
                              drift_change_alarm(reference, a, DriftEnvelope(.125, .125), method=method))


@pytest.mark.parametrize('method', ['hoeffding', 'clopper-pearson'])
@pytest.mark.parametrize('m,n', [(0, 0), (1, 3), (32, 45), (80, 100)])
@pytest.mark.parametrize('p,q,d', [(0, 0, 0), (.015, .015, 0),
                                 (.125, .25, .125), (.25, .125, .125),
                                 (.5, .375, .125), (.5, .5, 0)])
def test_single_coordinate_null_size_by_exact_binomial_enumeration(method, m, n, p, q, d):
    # Isolate one coordinate; other error counts are zero and gains equal one.
    a, b = np.meshgrid(np.arange(m+1), np.arange(n+1), indexing='ij')
    cy, my = np.zeros((a.size, 4), dtype=int), np.zeros((b.size, 4), dtype=int)
    cy[:, 0], my[:, 0] = a.ravel(), b.ravel()
    reject = drift_change_alarm(block(cy, m), block(my, n), DriftEnvelope(d, d), method=method)
    # Independent combinatorial masses, not a Monte Carlo FAR check.
    pc = [comb(m, k)*p**k*(1-p)**(m-k) for k in range(m+1)]
    pm = [comb(n, k)*q**k*(1-q)**(n-k) for k in range(n+1)]
    size = fsum(pc[i]*pm[j] for i, j, alarm in
                zip(a.ravel(), b.ravel(), reject) if alarm)
    assert size <= .05/6+1e-12


@lru_cache(maxsize=16)
def scalar_hoeffding_mask(m, n, d, alpha):
    """Independent scalar-formula interval gaps over all integer outcomes."""
    c = np.arange(m+1)/m
    b = np.arange(n+1)/n
    hc, hm = sqrt(log(24/alpha)/(2*m)), sqrt(log(24/alpha)/(2*n))
    clo, chi = np.maximum(0, c-hc), np.minimum(1, c+hc)
    mlo, mhi = np.maximum(0, b-hm), np.minimum(1, b+hm)
    return (mlo[None, :]-chi[:, None] > d) | (clo[:, None]-mhi[None, :] > d)


@pytest.mark.parametrize('r,beta', [(r, b) for r in (.8, 1.) for b in (0, .25, .5, .75, 1)])
def test_uniform_lower_is_below_independently_enumerated_conditional_power(r, beta):
    nc, nm = np.array([[480, 510, 500, 520]]), np.array([[520, 490, 510, 500]])
    env, alpha, floor = DriftEnvelope(.01, .02), .05, .4
    lower = hoeffding_family_power_lower(nc, nm, env, floor, alpha=alpha)[0]
    assert lower > 0
    q = (.015+.485*r*(1-beta),)*2+(.015+.485*r*beta,)*2
    probabilities = []
    for m, n, d, prob in zip(nc[0], nm[0], env.coordinate_allowances, q):
        mask = scalar_hoeffding_mask(int(m), int(n), d, alpha)
        joint = binom.pmf(np.arange(m+1), m, .015)[:, None]*binom.pmf(np.arange(n+1), n, prob)[None, :]
        probabilities.append(float(joint[mask].sum()))
    # Conditional independent error draws. Ignoring gain alarms is a subevent.
    actual_error_union = 1-np.prod(1-np.array(probabilities))
    assert lower <= actual_error_union+1e-12


def test_power_formula_manual_reflection_and_no_probability_leakage_into_alarm():
    nc, nm = np.array([[500, 520, 490, 510]]), np.array([[480, 530, 500, 520]])
    before = nc.copy(), nm.copy()
    env, floor, noise, alpha = DriftEnvelope(.01, .02), .4, .03, .05
    got = hoeffding_family_power_lower(nc, nm, env, floor, channel_noise=noise, alpha=alpha)
    hmax = max(sqrt(log(24/alpha)/(2*m))+sqrt(log(24/alpha)/(2*n))
               for m, n in zip(nc[0], nm[0]))
    smax = max(1/m+1/n for m, n in zip(nc[0], nm[0]))
    residual = ((1-noise)*floor-env.z_error-env.x_error)/2-hmax
    assert got[0] == pytest.approx(1-np.exp(-2*residual**2/smax))
    swap = [2, 3, 0, 1]
    assert got == pytest.approx(hoeffding_family_power_lower(nc[:, swap], nm[:, swap],
                                                           DriftEnvelope(.02, .01), floor))
    assert np.array_equal(nc, before[0]) and np.array_equal(nm, before[1])
    assert not got.flags.writeable
    c = block([[0]*4], 1000)
    with pytest.raises(ValueError):
        drift_change_alarm(ObservationLaw((0.,)*4), c, env)
    with pytest.raises(TypeError):
        drift_change_alarm(c, c, env, channel_noise=noise)
    with pytest.raises(TypeError):
        hoeffding_family_power_lower(nc, nm, env, floor, method='clopper-pearson')


def test_power_zero_counts_overlap_noise_endpoint_and_large_counts():
    full = np.full((1, 4), 10**7, dtype=np.int64)
    env = DriftEnvelope(.125, .125)
    # Exact binary overlap boundary: (1-.5)*.5 == .125+.125.
    assert hoeffding_family_power_lower(full, full, env, .5, channel_noise=.5)[0] == 0
    assert hoeffding_family_power_lower(full, full, DriftEnvelope(), .1, channel_noise=1)[0] == 0
    rows = np.array([[100000]*4, [0, 100000, 100000, 100000], [1]*4])
    lower = hoeffding_family_power_lower(rows, rows, DriftEnvelope(), .1)
    assert lower[0] > .99 and lower[1] == lower[2] == 0
    huge = np.full((1, 4), 2**53-1, dtype=np.int64)
    assert np.isfinite(hoeffding_family_power_lower(huge, huge, DriftEnvelope(), .1)).all()
    assert not np.signbit(hoeffding_family_power_lower(full, full, env, .01)[0])


@pytest.mark.parametrize('bad', [True, -.01, .50001, 1, float('nan'), float('inf'), '0.1'])
def test_bad_envelope_rejected(bad):
    for values in ((bad, 0), (0, bad)):
        with pytest.raises(ValueError):
            DriftEnvelope(*values)


@pytest.mark.parametrize('bad', [True, -1, 0, 1e-20, 1, float('nan'), float('inf'), '.05'])
def test_bad_alpha_rejected_by_alarm_and_bound(bad):
    c = block([[0]*4], 10)
    n = np.full((1, 4), 10, dtype=int)
    with pytest.raises(ValueError):
        drift_change_alarm(c, c, DriftEnvelope(), alpha=bad)
    with pytest.raises(ValueError):
        hoeffding_family_power_lower(n, n, DriftEnvelope(), .1, alpha=bad)


@pytest.mark.parametrize('bad', [np.array([[1., 2., 3., 4.]]), np.array([[-1]*4]),
                                 np.array([[True]*4]), np.empty((0, 4), dtype=int),
                                 np.ones((2, 3), dtype=int), np.full((1, 4), 2**53)])
def test_bad_power_denominators_rejected(bad):
    good = np.ones((1, 4), dtype=int)
    for cal, mon in ((bad, good), (good, bad)):
        with pytest.raises(ValueError):
            hoeffding_family_power_lower(cal, mon, DriftEnvelope(), .1)


@pytest.mark.parametrize('bad', [True, 0, -.1, .6, float('nan'), float('inf'), '0.1'])
def test_bad_knowledge_floor_rejected(bad):
    n = np.ones((1, 4), dtype=int)
    with pytest.raises(ValueError):
        hoeffding_family_power_lower(n, n, DriftEnvelope(), bad)


def test_bad_method_envelope_shapes_eta_and_noise_rejected():
    c, two = block([[0]*4], 10), block([[0]*4, [0]*4], 10)
    with pytest.raises(ValueError):
        drift_change_alarm(c, two, DriftEnvelope())
    with pytest.raises(ValueError):
        drift_change_alarm(c, c, DriftEnvelope(), method='exact')
    with pytest.raises(ValueError):
        drift_change_alarm(c, c, (.01, .01))
    with pytest.raises(ValueError):
        honest_drift_laws(0, 0, 0, 0, DriftEnvelope(), eta=True)
    n = np.ones((1, 4), dtype=int)
    with pytest.raises(ValueError):
        hoeffding_family_power_lower(n, np.ones((2, 4), dtype=int), DriftEnvelope(), .1)
    with pytest.raises(ValueError):
        hoeffding_family_power_lower(n, n, DriftEnvelope(), .1, channel_noise=True)
    with pytest.raises(ValueError):
        hoeffding_family_power_lower(n, n, None, .1)
