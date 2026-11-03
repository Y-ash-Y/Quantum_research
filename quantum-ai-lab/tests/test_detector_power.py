"""Independent integer enumeration for detector-specific power and its bounds."""
from fractions import Fraction
from math import comb, log, sqrt
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from research.detector_power import (build_kernel, coordinate_components,
                                    conditional_point, continuous_lower_function,
                                    empirical_endpoint)
from research.allocation_frontier import IRPolicy


def enumerated_tails(m, n, a, b):
    if not m or not n:
        return False, False
    total = a+b
    denominator = comb(m+n, total)
    masses = {k: Fraction(comb(m, k)*comb(n, total-k), denominator)
              for k in range(max(0, total-n), min(m, total)+1)}
    high = sum((v for k, v in masses.items() if k <= a), Fraction()) <= Fraction(1, 240)
    low = sum((v for k, v in masses.items() if k >= a), Fraction()) <= Fraction(1, 240)
    return high, low


@pytest.mark.parametrize('m,n', [(0, 0), (0, 8), (8, 0), (3, 5), (7, 8), (12, 10), (20, 25)])
def test_small_integer_boundaries_and_integrated_probability(m, n):
    nc, nm = np.full((1, 4), m), np.full((1, 4), n)
    kernel = build_kernel(nc, nm, q0=.2)
    for i, a in enumerate(kernel.calibration_values):
        for b in range(n+1):
            hi, lo = enumerated_tails(m, n, int(a), b)
            assert (b >= kernel.high_cutoff[0, i]) == hi
            assert (b <= kernel.low_cutoff[0, i]) == lo
    for q in (0., .2, .4, .8, 1.):
        expected_hi = expected_lo = 0.
        for a in range(m+1):
            for b in range(n+1):
                hi, lo = enumerated_tails(m, n, a, b)
                mass = binom.pmf(a, m, .2)*binom.pmf(b, n, q)
                expected_hi += mass*hi; expected_lo += mass*lo
        h, l = coordinate_components(kernel, [q]*4)
        assert h[0, 0] == pytest.approx(expected_hi, abs=2e-12)
        assert l[0, 0] == pytest.approx(expected_lo, abs=2e-12)
        lower, upper = conditional_point(kernel, [q]*4, np.array([False]))
        exact = 1-(1-expected_hi-expected_lo)**4
        assert lower[0] <= exact+2e-12 <= upper[0]+4e-12


@pytest.mark.parametrize('q0', [0., 1., .015, .5])
def test_support_tail_mass_and_inputs_immutable(q0):
    nc, nm = np.array([[0, 20, 50, 1000]]), np.array([[0, 10, 30, 500]])
    before = nc.copy()
    k = build_kernel(nc, nm, q0=q0)
    assert np.all(k.missing_mass <= 1.01e-12)
    assert np.allclose(k.weights.sum(axis=1)+k.missing_mass, 1., atol=1e-12)
    assert not k.weights.flags.writeable
    assert np.array_equal(nc, before)
    lower, upper = conditional_point(k, [0., 1., q0, .7], np.array([False]))
    assert 0 <= lower[0] <= upper[0] <= 1


def test_conditional_union_and_gain_alarm():
    k = build_kernel(np.array([[100]*4, [90]*4]), np.array([[70]*4, [60]*4]))
    p = (.1, .1, .2, .2)
    h, l = coordinate_components(k, p)
    lower, upper = conditional_point(k, p, np.array([False, True]))
    assert lower[0] == pytest.approx(1-np.prod(1-h[0]-l[0]))
    assert lower[1] == upper[1] == 1.


def test_subevent_continuous_bound_and_high_tail_monotonicity():
    k = build_kernel(np.array([[600, 610, 590, 580], [550]*4]),
                     np.array([[300, 310, 320, 330], [250]*4]))
    gain = np.array([False, True])
    floor, interval = .05, (.2, .45)
    bound = continuous_lower_function(k, floor, interval, gain)
    for r in (.1, .3, .8, 1.):
        for bias in (.2, .3, .45):
            q = IRPolicy(r, bias).law().bit_errors
            low, _ = conditional_point(k, q, gain)
            assert np.all(bound <= low+1e-13)
    previous = np.zeros((2, 4))
    for q in np.linspace(0, 1, 21):
        h, _ = coordinate_components(k, [q]*4)
        assert np.all(h >= previous-1e-13)
        previous = h


def test_conditional_reflection_when_denominators_are_reflected():
    nc, nm = np.array([[200, 210, 190, 180]]), np.array([[90, 80, 110, 120]])
    swap = [2, 3, 0, 1]
    a, b = build_kernel(nc, nm), build_kernel(nc[:, swap], nm[:, swap])
    p = np.array(IRPolicy(.2, .3).law().bit_errors)
    lo, hi = conditional_point(a, p, np.array([False]))
    rlo, rhi = conditional_point(b, p[swap], np.array([False]))
    assert lo == pytest.approx(rlo) and hi == pytest.approx(rhi)


def test_empirical_bernstein_one_sided_and_manual_variance():
    x = np.array([.1, .2, .4, .8])
    delta = .01
    mean = sum(x)/4
    variance = sum((v-mean)**2 for v in x)/3
    rad = sqrt(2*variance*log(2/delta)/4)+7*log(2/delta)/9
    lower = empirical_endpoint(x, delta, 'lower')
    upper = empirical_endpoint(x, delta, 'upper')
    assert lower['sample_variance'] == pytest.approx(variance)
    assert lower['radius'] == pytest.approx(rad)
    assert lower['endpoint'] == max(0., mean-rad)
    assert upper['endpoint'] == min(1., mean+rad)
    complement = empirical_endpoint(1-x, delta, 'upper')
    assert lower['endpoint'] == pytest.approx(1-complement['endpoint'])


@pytest.mark.parametrize('bad', [np.array([[1., 2., 3., 4.]]), np.array([[-1]*4]),
                                 np.array([[True]*4]), np.empty((0, 4), dtype=int),
                                 np.ones((2, 3), dtype=int)])
def test_invalid_denominators(bad):
    with pytest.raises(ValueError):
        build_kernel(bad, np.ones((1, 4), dtype=int))


@pytest.mark.parametrize('q', [[-.1]*4, [1.1]*4, [float('nan')]*4, [.2]*3])
def test_invalid_power_probabilities(q):
    k = build_kernel(np.ones((1, 4), dtype=int), np.ones((1, 4), dtype=int))
    with pytest.raises(ValueError):
        conditional_point(k, q, np.array([False]))


@pytest.mark.parametrize('x,delta,side', [([.2], .01, 'lower'), ([.2, 1.1], .01, 'lower'),
                                       ([.2, .3], 0, 'lower'), ([.2, .3], .01, 'both')])
def test_invalid_mean_bounds(x, delta, side):
    with pytest.raises(ValueError):
        empirical_endpoint(x, delta, side)


def test_invalid_domain_controls():
    a = np.array([[100]*4])
    for kwargs in ({'alpha': 0}, {'q0': float('nan')}, {'truncation': 0}):
        with pytest.raises(ValueError):
            build_kernel(a, a, **kwargs)
    k = build_kernel(a, a)
    for floor, interval in ((.6, (0, .5)), (.01, (.6, .4)), (.01, (-.1, .4))):
        with pytest.raises(ValueError):
            continuous_lower_function(k, floor, interval, np.array([False]))
    with pytest.raises(ValueError):
        conditional_point(k, [.1]*4, [0])


def test_frozen_design_and_envelope(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('power_runner_test', scripts/'run_detector_power.py')
    r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
    p = r.frozen_protocol()
    assert p['mean_functions'] == 2*4*(16+2)
    assert p['direct_cells'] == 2*(4+1)
    assert p['mean_failure_budget']+p['direct_failure_budget'] == .05
    e = r.continuous_envelope([{'endpoint': .4}, {'endpoint': .45}],
                              {'endpoint': .42}, {'endpoint': .48})
    assert e['continuous_minimum_joint95'] == [.4, .48]
    assert not e['balanced_within_five_points']
    with pytest.raises(ArithmeticError):
        r.continuous_envelope([{'endpoint': .9}], {'endpoint': .1}, {'endpoint': .8})
