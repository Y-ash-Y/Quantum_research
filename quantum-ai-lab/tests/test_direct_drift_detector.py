"""Implementation/proof checks; finite grids are not model-wide size proofs."""
import inspect
import math

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from research.calibration_protocol import ObservationLaw, PublicBatch, sample_public_batch
from research.drift_detector import DriftEnvelope
from research.direct_drift_detector import (
    _error_radius, _half_upper, direct_drift_alarm, direct_drift_directions, pooled_counts,
)


def batches(qc=(.015,)*4, qm=(.025,)*4, rows=11):
    return [sample_public_batch(rows, 1000000, ObservationLaw(q), .5,
                                np.random.default_rng(s)) for q, s in ((qc, 73), (qm, 74))]


def test_count_pooling_inputs_and_outputs_are_readonly():
    cal, mon = batches()
    old = cal.numerators.copy()
    y, n = pooled_counts(cal)
    assert np.array_equal(y[:, 0], cal.numerators[:, :2].sum(axis=1))
    assert np.array_equal(n[:, 1], cal.denominators[:, 2:4].sum(axis=1))
    flags = direct_drift_directions(cal, mon, DriftEnvelope())
    alarm = direct_drift_alarm(cal, mon, DriftEnvelope())
    assert flags.shape == (11, 4, 2) and np.array_equal(alarm, flags.any(axis=(1, 2)))
    assert all(not a.flags.writeable for a in (y, n, flags, alarm))
    assert np.array_equal(cal.numerators, old)
    assert list(inspect.signature(direct_drift_alarm).parameters) == ['calibration', 'monitoring', 'envelope', 'alpha']


def test_preparation_bit_permutation_and_stage_symmetry():
    cal, mon = batches()
    order = [1, 0, 3, 2, 4, 5]
    swapped = [PublicBatch(b.emitted, b.numerators[:, order], b.denominators[:, order],
                            b.sifted[:, [1, 0, 3, 2]]) for b in (cal, mon)]
    env = DriftEnvelope(.002, .004)
    a = direct_drift_directions(cal, mon, env)
    assert np.array_equal(a, direct_drift_directions(*swapped, env))
    assert np.array_equal(a[:, :, ::-1], direct_drift_directions(mon, cal, env))


@pytest.mark.parametrize('theta', [0., .001, .02, .25, .5])
@pytest.mark.parametrize('n', [1, 12, 61])
def test_one_sided_cp_variance_envelope_exact_coverage(theta, n):
    y = np.arange(n+1)
    u = _half_upper(y, np.full(len(y), n), .05/16)
    failure = float(binom.pmf(y, n, theta)[u < theta].sum())
    assert failure <= .05/16+1e-13


@pytest.mark.parametrize('y,n,nm,d', [(0, 9, 15, 0.), (3, 55, 91, .005),
                                     (61, 61, 100, .1), (27, 4000, 8000, .003)])
def test_independent_scalar_radius(y, n, nm, d):
    alpha = .05
    raw_upper = binomtest(y, n, alternative='less').proportion_ci(1-alpha/16).high
    upper = min(.5, raw_upper)
    other = min(.5, upper+d)
    variance = upper*(1-upper)/n + other*(1-other)/nm
    b, log = max(1/n, 1/nm), math.log(16/alpha)
    expected = (2*b*log/3+math.sqrt((2*b*log/3)**2+8*variance*log))/2
    radius = _error_radius(np.array([y]), np.array([float(n)]), np.array([float(nm)]), d, alpha)[0]
    assert radius == pytest.approx(expected, abs=1e-12)
    assert expected**2/(2*(variance+b*expected/3)) == pytest.approx(log)


@pytest.mark.parametrize('theta,mu,d', [(0., 0., 0.), (0., .03, .03),
                                     (.02, .02, 0.), (.02, .06, .04),
                                     (.25, .23, .02), (.5, .5, 0.),
                                     (.49, .5, .02), (.5, .4, .1)])
def test_exhaustive_error_coordinate_size_on_finite_null_grid(theta, mu, d):
    nc, nm, alpha = 32, 49, .2
    yc, ym = np.meshgrid(np.arange(nc+1), np.arange(nm+1), indexing='ij')
    y, z = yc.ravel(), ym.ravel()
    c, m = np.full(y.shape, float(nc)), np.full(y.shape, float(nm))
    diff = z/nm-y/nc
    up = diff > d+_error_radius(y, c, m, d, alpha)
    down = -diff > d+_error_radius(z, m, c, d, alpha)
    weights = binom.pmf(y, nc, theta)*binom.pmf(z, nm, mu)
    assert weights[up].sum() <= alpha/8+1e-13
    assert weights[down].sum() <= alpha/8+1e-13
    assert weights[up | down].sum() <= alpha/4+1e-13


def test_zero_counts_and_coordinate_isolation():
    empty = PublicBatch(0, np.zeros((1, 6), dtype=int), np.zeros((1, 6), dtype=int),
                        np.zeros((1, 4), dtype=int))
    assert not direct_drift_alarm(empty, empty, DriftEnvelope())[0]
    cal = PublicBatch(100000, np.array([[0, 0, 0, 0, 10000, 10000]]),
                      np.array([[0, 0, 0, 0, 50000, 50000]]), np.zeros((1, 4), dtype=int))
    mon = PublicBatch(100000, np.array([[0, 0, 0, 0, 20000, 10000]]),
                      np.array([[0, 0, 0, 0, 50000, 50000]]), np.zeros((1, 4), dtype=int))
    flags = direct_drift_directions(cal, mon, DriftEnvelope(.5, .5))
    assert np.array_equal(np.argwhere(flags), [[0, 2, 0]])


def test_increasing_allowance_cannot_create_error_alarm():
    cal, mon = batches(rows=200)
    narrow = direct_drift_directions(cal, mon, DriftEnvelope())
    wide = direct_drift_directions(cal, mon, DriftEnvelope(.04, .04))
    assert not np.any(wide[:, :2] & ~narrow[:, :2])
    assert np.array_equal(wide[:, 2:], narrow[:, 2:])


@pytest.mark.parametrize('alpha', [True, 0., -1., 1., float('nan'), 1e-13, '0.05'])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        direct_drift_alarm(*batches(), DriftEnvelope(), alpha=alpha)


def test_bad_types_and_unpaired_rows():
    cal, mon = batches()
    with pytest.raises(ValueError):
        pooled_counts(cal.numerators)
    with pytest.raises(ValueError):
        direct_drift_alarm(cal, mon, (.01, .01))
    with pytest.raises(ValueError):
        direct_drift_alarm(cal, batches(rows=3)[1], DriftEnvelope())
