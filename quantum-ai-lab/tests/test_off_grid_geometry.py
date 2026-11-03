import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import rel_entr

from research.allocation_frontier import BudgetPlan, IRPolicy, broad_pauli_information_bound
from research.calibration_protocol import gate_law, sample_public_batch
from research.exact_comparison import central_exact_pvalues, exact_change_alarm
from research.off_grid_geometry import (
    ERROR_SWAP, reflect_public, public_policy_kl, axis_cells, cell_radius, continuous_envelope,
)


@pytest.mark.parametrize('r,beta', [(.005, .0625), (.04, .4375), (.123, .713), (0., 0.), (1., .5)])
def test_physical_reflection_law(r, beta):
    p, q = IRPolicy(r, beta).law(), IRPolicy(r, 1-beta).law()
    assert np.allclose(q.bit_errors, np.array(p.bit_errors)[[2, 3, 0, 1]], atol=1e-15, rtol=0)
    assert IRPolicy(r, beta).expected_certified_fraction == IRPolicy(r, 1-beta).expected_certified_fraction


def test_public_reflection_involution_and_detector_invariance():
    rng = np.random.default_rng(632)
    c = sample_public_batch(64, 40000, gate_law(), 1., rng)
    m = sample_public_batch(64, 50000, IRPolicy(.07, .137).law(), .35, rng)
    reflected = reflect_public(reflect_public(m))
    for name in ('numerators', 'denominators', 'sifted'):
        assert np.array_equal(getattr(m, name), getattr(reflected, name))
    assert np.array_equal(central_exact_pvalues(reflect_public(c), reflect_public(m)),
                          central_exact_pvalues(c, m)[:, ERROR_SWAP])
    assert np.array_equal(exact_change_alarm(c, m), exact_change_alarm(reflect_public(c), reflect_public(m)))


def pulse_probabilities(policy, eta, fraction):
    # Enumerate the actual one-emission observable alphabet, including random
    # intensity, loss, four sifted strata, mismatch and error-blind disclosure.
    rows = []
    for mu in (.5, .1):
        g = -math.expm1(-mu*eta)
        rows += [.5*(1-g), .5*g*.5]
        for q in policy.law(eta=eta).bit_errors:
            rows += [.5*g/8*(1-fraction), .5*g/8*fraction*(1-q), .5*g/8*fraction*q]
    return np.array(rows)


def test_public_kl_matches_complete_small_alphabet_enumeration():
    a, b = IRPolicy(.053, .173), IRPolicy(.085, .491)
    p, q = pulse_probabilities(a, .15, .4), pulse_probabilities(b, .15, .4)
    assert p.sum() == pytest.approx(1.) and q.sum() == pytest.approx(1.)
    exact = float(rel_entr(p, q).sum())
    assert public_policy_kl(BudgetPlan(3, 2, .4), a, b) == pytest.approx(exact, rel=1e-10)
    assert public_policy_kl(BudgetPlan(1002, 2, .4), a, b) == pytest.approx(1000*exact, rel=1e-10)


@pytest.mark.parametrize('plan,eta', [(BudgetPlan(20, 20, 1.), .15), (BudgetPlan(20, 4, 0.), .15),
                                    (BudgetPlan(20, 4, 1.), 0.)])
def test_zero_observable_exposure(plan, eta):
    assert public_policy_kl(plan, IRPolicy(.01, 0.), IRPolicy(.7, 1.), eta=eta) == 0.
    assert cell_radius(plan, IRPolicy(.1, .5), (.05, .5), (0., 1.), eta=eta)['tv_radius'] == 0.


def test_corner_bound_dominates_interior_and_includes_anchor():
    plan = BudgetPlan(4000000, 1333333, .2500000937499883)
    anchor = IRPolicy(.012, .4375)
    rb, bb = (.011, .013), (.4, .48)
    radius = cell_radius(plan, anchor, rb, bb)
    for r in np.linspace(*rb, 11):
        for beta in np.linspace(*bb, 11):
            assert public_policy_kl(plan, anchor, IRPolicy(r, beta)) <= radius['maximum_corner_kl']+1e-10
    assert 0 <= radius['tv_radius'] <= 1


def test_midpoint_cover_full_domain_no_gap():
    cells = axis_cells([.005, .005625, .00625, .0075, .01], .005, 1.)
    assert cells[0][1][0] == .005 and cells[-1][1][1] == 1.
    for (_, (_, high)), (_, (low, _)) in zip(cells, cells[1:]):
        assert high == low
    for anchor, (low, high) in cells:
        assert low <= anchor <= high


def test_continuous_envelope_truth_vectors_and_vacuous_cells():
    cells = [{'anchor_lower': .4, 'tv_radius': .1}, {'anchor_lower': .7, 'tv_radius': .2}]
    e = continuous_envelope(cells, .6)
    assert e['lower'] == pytest.approx(.3) and e['consistent']
    for theta0 in (.3, .4, .6):
        for theta1 in (.5, .6, .7):
            assert e['lower'] <= min(theta0, theta1)+1e-14
    assert continuous_envelope(cells+[{'anchor_lower': .9, 'tv_radius': 1.}], .6)['lower'] == 0.
    assert not continuous_envelope(cells, .2)['consistent']


def test_balanced_KL_proxy_minimum_and_rate_order_not_power_order():
    plan = BudgetPlan(4000000, 800000, .375)
    for r in (.005, .04, .2, 1.):
        balanced = broad_pauli_information_bound(plan, IRPolicy(r, .5))['joint_kl_nats']
        for beta in np.linspace(0., 1., 17):
            k = broad_pauli_information_bound(plan, IRPolicy(r, beta))['joint_kl_nats']
            assert k >= balanced-1e-9
    ordered = [broad_pauli_information_bound(plan, IRPolicy(r, .5))['joint_kl_nats']
               for r in (.005, .01, .02, .04)]
    assert ordered == sorted(ordered)


@pytest.mark.parametrize('function,args', [
    (reflect_public, (None,)), (axis_cells, ([.2, .1], 0., 1.)),
    (axis_cells, ([True], 0., 1.)), (continuous_envelope, ([], .5)),
    (continuous_envelope, ([{'anchor_lower': .1, 'tv_radius': -1}], .5)),
    (cell_radius, (BudgetPlan(4, 1, .2), IRPolicy(.1, .5), (.2, .3), (0., 1.))),
])
def test_invalid_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_frozen_case_counts_and_confirmation_direction(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('off_grid_runner_test', scripts/'run_off_grid_geometry.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    assert len(runner.frozen_cases('stress')) == 144
    assert len(runner.frozen_cases('confirmation')) == 24
    cells = []
    for pid in runner.PLAN_IDS:
        for case in runner.frozen_cases('confirmation'):
            count = 4096 if case['rate_multiplier'] == 1. and case['beta'] == .5 else 0
            cells.append({'phase': 'confirmation', 'plan_id': pid, **case,
                          'alarms': count, 'trials': 4096, 'simultaneous_ci95': [0., 1.]})
    contrasts = runner.confirmation_contrasts(cells)
    assert len(contrasts) == 16
    assert all(x['difference_first_minus_balanced_floor'] == -1 for x in contrasts)
    assert all(x['bonferroni_lower_power_p'] == 0 for x in contrasts)
    assert all(x['simultaneous_difference_ci95'] == [-1., 1.] for x in contrasts)
