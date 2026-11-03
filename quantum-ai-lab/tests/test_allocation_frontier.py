"""Independent accounting, private-label and joint-divergence checks."""
import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from research.allocation_frontier import (
    BudgetPlan, IRPolicy, broad_pauli_information_bound, constant_sacrifice_plan,
    sample_retained_certified_counts,
)
from research.calibration_protocol import PublicBatch, gate_law, sample_public_batch


@pytest.mark.parametrize('m', [0, 200000, 800000, 1333333, 1600000, 1800000, 2000000])
def test_equal_expected_sacrifice_and_candidates(m):
    plan = constant_sacrifice_plan(4000000, m, 2000000)
    assert plan.calibration_emitted+plan.monitoring_emitted == 4000000
    assert plan.effective_test_emissions == pytest.approx(2000000)
    resource = plan.expected_resources()
    assert resource['total_sacrificed'] == pytest.approx(resource['remaining_candidates'])
    assert resource['calibration_sacrificed']+resource['monitoring_disclosed'] == pytest.approx(resource['total_sacrificed'])


@pytest.mark.parametrize('rate,bias', [(0, 0), (.01, .5), (.08, 0), (.08, 1), (1, .25)])
def test_certified_information_is_basis_balanced_not_basis_bias(rate, bias):
    p = IRPolicy(rate, bias)
    assert p.expected_certified_fraction == pytest.approx(rate/2)
    assert p.eligible(rate/2)
    if rate < 1:
        assert not p.eligible(rate/2+1e-6)
    q = p.law(.03).bit_errors
    # Basis-averaged disturbance is independent of Eve's basis bias.
    assert sum(q)/4 == pytest.approx(.015+.97*rate/4)


def test_retained_only_private_counts_exact_endpoints():
    public = sample_public_batch(200, 50000, gate_law(), .25, np.random.default_rng(3))
    retained = public.sifted-public.denominators[:, :4]
    zero = sample_retained_certified_counts(public, IRPolicy(0, .5), np.random.default_rng(2))
    assert not zero.any()
    for bias in (0, 1):
        known = sample_retained_certified_counts(public, IRPolicy(1, bias), np.random.default_rng(2))
        assert np.array_equal(known[:, :2], retained[:, :2] if bias == 1 else np.zeros_like(retained[:, :2]))
        assert np.array_equal(known[:, 2:], retained[:, 2:] if bias == 0 else np.zeros_like(retained[:, 2:]))
        assert np.all(known <= retained)
    full = sample_public_batch(4, 10000, gate_law(), 1, np.random.default_rng(2))
    assert not sample_retained_certified_counts(full, IRPolicy(1, 1), np.random.default_rng(2)).any()


def test_private_fraction_matches_expectation_without_conditioning():
    policy = IRPolicy(.08, .25)
    public = sample_public_batch(3000, 20000, policy.law(), .25, np.random.default_rng(80))
    known = sample_retained_certified_counts(public, policy, np.random.default_rng(81))
    assert known.sum()/public.remaining_candidates.sum() == pytest.approx(.04, abs=.001)
    assert known.sum(axis=0)/(public.sifted-public.denominators[:, :4]).sum(axis=0) == pytest.approx(
        [.02, .02, .06, .06], abs=.002)
    assert not hasattr(public, 'certified')


@pytest.mark.parametrize('m,f', [(0, .25), (4000000, .25), (1000000, 0), (1000000, 1), (1333333, .25)])
def test_joint_kl_matches_independent_one_pulse_categorical_enumeration(m, f):
    plan, policy = BudgetPlan(4000000, m, f), IRPolicy(.04, .75)
    bound = broad_pauli_information_bound(plan, policy)
    # Each basis/bit class has probability gain/8; undisclosed outcomes carry
    # no error information and have the same mass under both laws.
    gain = (-math.expm1(-.075)-math.expm1(-.015))/2
    q0 = .015
    qa = (.015+.97*.04*(1-.75)/2, .015+.97*.04*.75/2)
    e = (4000000-m)*f
    h = [(m*q0+e*q)/(m+e) for q in qa] if m+e else [q0, q0]
    def categorical(errors, disclosure):
        cells = [1-gain, gain/2, gain*(1-disclosure)/2]
        for error in (errors[0], errors[0], errors[1], errors[1]):
            cells.extend([gain*disclosure*(1-error)/8, gain*disclosure*error/8])
        assert sum(cells) == pytest.approx(1)
        return cells
    def kl(p, q):
        return sum(a*math.log(a/b) for a, b in zip(p, q) if a)
    cal = kl(categorical((q0, q0), 1), categorical(h, 1))
    mon = kl(categorical(qa, f), categorical(h, f))
    independent = m*cal+(4000000-m)*mon
    assert bound['joint_kl_nats'] == pytest.approx(independent, abs=1e-9)
    assert sum(bound['honest_counterpart_pauli_weights'].values()) == pytest.approx(1)


@pytest.mark.parametrize('m,f,eta,noise', [(0, .25, .15, .03), (2000000, 0, .15, .03),
                                        (1000000, .25, 0, .03), (1000000, .25, .15, 1)])
def test_no_public_distinction_endpoints(m, f, eta, noise):
    out = broad_pauli_information_bound(BudgetPlan(4000000, m, f), IRPolicy(.08, .5), noise, eta)
    assert out['joint_kl_nats'] == pytest.approx(0, abs=1e-12)
    assert out['power_ceiling'] == pytest.approx(.05)


def test_optimized_counterpart_minimizes_scalar_kl():
    plan, policy = BudgetPlan(4000000, 800000, .25), IRPolicy(.04, .25)
    bound = broad_pauli_information_bound(plan, policy)
    h = bound['honest_counterpart_errors'][0]
    q0, qa, m, e = .015, policy.law().bit_errors[0], 800000, 800000
    def objective(t):
        return m*(q0*math.log(q0/t)+(1-q0)*math.log((1-q0)/(1-t)))+e*(
            qa*math.log(qa/t)+(1-qa)*math.log((1-qa)/(1-t)))
    assert objective(h) < objective(h-.001)
    assert objective(h) < objective(h+.001)


@pytest.mark.parametrize('bad', [True, -1, 1.1, math.nan, '0.25'])
def test_invalid_policy_and_disclosure_inputs(bad):
    with pytest.raises(ValueError):
        IRPolicy(bad, .5)
    with pytest.raises(ValueError):
        BudgetPlan(100, 50, bad)


@pytest.mark.parametrize('args', [(100, 101, .25), (True, 0, .25), (0, 0, .25), (100, 1.5, .25)])
def test_invalid_budget_inputs(args):
    with pytest.raises(ValueError):
        BudgetPlan(*args)


@pytest.mark.parametrize('args', [(100, 70, 50), (100, 50, 120), (100, -1, 50)])
def test_infeasible_sacrifice_inputs(args):
    with pytest.raises(ValueError):
        constant_sacrifice_plan(*args)


def test_unsupported_information_floor_and_private_inputs():
    with pytest.raises(ValueError):
        IRPolicy(1, .5).eligible(.6)
    with pytest.raises(ValueError):
        sample_retained_certified_counts(gate_law(), IRPolicy(.1, .5), np.random.default_rng(2))


def runner_module():
    path = Path(__file__).resolve().parents[1]/'scripts/run_allocation_frontier.py'
    spec = importlib.util.spec_from_file_location('allocation_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_grid_size_resources_and_floor_eligibility():
    runner = runner_module()
    profiles, grid = runner.plans(), runner.policies()
    assert len(profiles) == 14 and len(grid) == 45
    for profile in profiles:
        plan = profile['plan']
        assert plan.total_emitted == 4000000
        if profile['regime'] == 'fixed_sacrifice':
            assert plan.effective_test_emissions == pytest.approx(2000000)
    for floor in runner.FLOORS:
        ids = runner.candidate_ids(grid, floor)
        assert len(ids) == 15 and all(grid[i].eligible(floor) for i in ids)


def test_predeclared_selection_ties_and_no_evaluation_inputs():
    runner = runner_module()
    profiles, grid = runner.plans(), runner.policies()
    # All search alarms tie. Eve chooses the minimum eligible rate and balanced
    # basis; defender chooses the predeclared one-third tie preference.
    search = [{'plan_id': p['plan_id'], 'policy_id': i, 'alarm_rate': 0}
              for p in profiles for i in range(len(grid))]
    responses, winners = runner.select_responses(search, profiles, grid)
    assert len(responses) == 56 and len(winners) == 8
    for response in responses:
        policy = grid[response['policy_id']]
        assert policy.rate == pytest.approx(2*response['minimum_certified_fraction'])
        assert policy.z_basis_probability == .5
    assert all(profiles[w['plan_id']]['plan'].calibration_emitted == 1333333 for w in winners)
