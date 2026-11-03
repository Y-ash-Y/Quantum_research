"""Independent finite-experiment coverage and real-Aer attack regression tests."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.stats import binom

from adversarial.eavesdropper_game import AttackPolicy, simulate_session
from bb84.honest_nuisance import HonestSettings
from bb84.loss_attacks import simulate_loss_session
from classical_ml.fair_benchmark import BenchmarkConfig
from classical_ml.loss_aware_detectors import LossAwareTests, signal_lower_bound, implied_decoy_probability
from classical_ml.loss_aware_benchmark import run_benchmark, write_results, default_config, attack_batch
from classical_ml.loss_aware_audit import audit_results


@pytest.mark.parametrize('policy', [AttackPolicy(), AttackPolicy(.1, .95, 0),
    AttackPolicy(.8, .5, 0), AttackPolicy(0, .5, .85), AttackPolicy(.2, .9, .3)])
@pytest.mark.parametrize('seed', [12, 23, 34])
def test_nominal_pulse_path_is_exact_regression(policy, seed):
    a = simulate_session(*policy.as_tuple(), rng=np.random.default_rng(seed), return_details=True)
    b = simulate_loss_session(policy, HonestSettings(), np.random.default_rng(seed))
    assert a is not None and b is not None
    assert a.counts == b.counts and a.features == b.features
    assert a.known_bits == b.known_bits and a.sifted_bits == b.sifted_bits


@pytest.mark.parametrize('eta', [.1, .15, .18])
@pytest.mark.parametrize('policy', [AttackPolicy(.1, .95, 0), AttackPolicy(0, .5, .85)])
def test_loss_attack_counts_agree_with_independent_physics(eta, policy):
    b = attack_batch(policy, 600, HonestSettings(eta=eta), np.random.default_rng(97))
    assert b.rejected == 0
    nz, nx, ez, ex, ns, nd, ds, dd = b.counts.sum(axis=0)
    qs = -math.expm1(-.5*eta)
    qforward = qs/(1-math.exp(-.5)*1.5)
    qd = (1-policy.pns_frac)*(-math.expm1(-.1*eta))+policy.pns_frac*qforward*(1-math.exp(-.1)*1.1)
    qz = .03/2 + policy.ir_rate*(1-policy.basis_bias)*.97/2
    qx = .03/2 + policy.ir_rate*policy.basis_bias*.97/2
    for total, observed, p in [(ns, ds, qs), (nd, dd, qd), (nz, ez, qz), (nx, ex, qx)]:
        assert abs(observed-total*p) < 8*math.sqrt(total*p*(1-p))
    A, B = (qs-math.expm1(-.1*eta))/2, (qs+qforward*(1-math.exp(-.1)*1.1))/2
    expected_info = (policy.pns_frac*B+(1-policy.pns_frac)*A*policy.ir_rate/2)/(policy.pns_frac*B+(1-policy.pns_frac)*A)
    total, observed = b.sifted.sum(), b.known.sum()
    assert abs(observed-total*expected_info) < 8*math.sqrt(total*expected_info*(1-expected_info))


def test_attack_path_rejects_undefined_gate_model_and_infeasible_pns():
    with pytest.raises(ValueError, match='ideal local gates'):
        simulate_loss_session(AttackPolicy(), HonestSettings(gate_noise=.01), np.random.default_rng(1))
    with pytest.raises(ValueError):
        simulate_loss_session(AttackPolicy(0, .5, .8), HonestSettings(eta=1), np.random.default_rng(1))
    with pytest.raises(ValueError, match='No usable'):
        attack_batch(AttackPolicy(), 5, HonestSettings(eta=0), np.random.default_rng(2))


@pytest.mark.parametrize('qs', [0, .001, .01, .1, .5, .9, 1])
@pytest.mark.parametrize('ns', [1, 5, 12])
def test_exact_signal_coverage_exhaustively(qs, ns):
    ds = np.arange(ns+1)
    lower = signal_lower_bound(ns, ds, .025)
    probability = sum(binom.pmf(ds, ns, qs)[lower > qs])
    assert probability <= .025+1e-12


@pytest.mark.parametrize('qs', [.001, .01, .1, .5, .9])
def test_decoy_alarm_budget_exhaustive_unfiltered_experiment(qs):
    ns, nd, alpha = 8, 7, .05
    ds, dd = np.meshgrid(np.arange(ns+1), np.arange(nd+1), indexing='ij')
    C = np.zeros((ds.size, 8), dtype=int)
    C[:, 4:6] = [ns, nd]
    C[:, 6], C[:, 7] = ds.ravel(), dd.ravel()
    scores, _ = LossAwareTests(alpha).evaluate(C)
    qd = 1-(1-qs)**.2
    mass = binom.pmf(ds, ns, qs)*binom.pmf(dd, nd, qd)
    alarm = scores['Loss-aware decoy'].reshape(ds.shape) >= -alpha/2
    assert mass[alarm].sum() <= alpha+1e-12


def test_endpoints_have_finite_scores_without_fake_telemetry():
    C = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0, 10, 0]])
    scores, bounds = LossAwareTests().evaluate(C)
    assert all(np.isfinite(v).all() for v in (*scores.values(), *bounds.values()))
    assert all(v[0] == -1 for v in scores.values())
    assert np.array_equal(implied_decoy_probability([0, 1], .2), [0, 1])


@pytest.mark.parametrize('kwargs', [{'alpha': 0}, {'alpha': 1}, {'alpha': True},
    {'intensity_ratio': 0}, {'intensity_ratio': 1}, {'alpha': math.nan}])
def test_invalid_screen_settings(kwargs):
    with pytest.raises(ValueError):
        LossAwareTests(**kwargs)


@pytest.mark.parametrize('C', [[], [[1]*7], [[-1]*8], [[1.5]*8], [[True]*8],
    [[2, 2, 3, 0, 10, 10, 4, 4]], [[2, 2, 0, 0, 1, 1, 2, 2]]])
def test_invalid_screen_counts(C):
    with pytest.raises(ValueError):
        LossAwareTests().evaluate(C)


def test_bound_screens_more_conservatively_than_plugin():
    C = np.array([[20, 20, 0, 0, 3000, 3000, 220, 25]])
    scores, _ = LossAwareTests().evaluate(C)
    assert scores['Loss-aware decoy'][0] < scores['Loss plug-in decoy'][0]
    assert LossAwareTests().cutoffs == {'Loss-aware decoy': -.025, 'Loss-aware joint': -.05/3, 'Loss plug-in decoy': -.05}


@pytest.mark.parametrize('sent,detected,delta', [(4, 5, .05), (4, -1, .05),
    (4.5, 2, .05), (True, 1, .05), (4, 2, 0), (4, 2, 1)])
def test_invalid_bound_helper_inputs(sent, detected, delta):
    with pytest.raises(ValueError):
        signal_lower_bound(sent, detected, delta)


@pytest.mark.parametrize('values,ratio', [([-1], .2), ([math.nan], .2), ([1.1], .2), ([.1], -.2)])
def test_invalid_implied_probability_inputs(values, ratio):
    with pytest.raises(ValueError):
        implied_decoy_probability(values, ratio)


def test_signal_bound_tail_equation_against_independent_finite_sum():
    for n in (1, 5, 12):
        for k in range(1, n+1):
            lower = float(signal_lower_bound(n, k, .025))
            tail = sum(math.comb(n, j)*lower**j*(1-lower)**(n-j) for j in range(k, n+1))
            assert tail == pytest.approx(.025, abs=1e-12)


def test_plugin_has_no_conditional_count_false_alarm_guarantee():
    # A deliberately unbalanced unfiltered emitted-count experiment, not the
    # benchmark's typical random half-signal/half-decoy window. One signal
    # click makes the plug-in implied decoy probability 1 and virtually every
    # honest decoy count reject. Conditional false alarms are about 7%, >5%.
    ns, nd, qs = 1, 100, .07
    qd = 1-(1-qs)**.2
    ds, dd = np.meshgrid(np.arange(ns+1), np.arange(nd+1), indexing='ij')
    C = np.zeros((ds.size, 8), dtype=int)
    C[:, 4:6] = [ns, nd]
    C[:, 6], C[:, 7] = ds.ravel(), dd.ravel()
    scores, _ = LossAwareTests().evaluate(C)
    mass = binom.pmf(ds, ns, qs)*binom.pmf(dd, nd, qd)
    plugin_rate = mass[(scores['Loss plug-in decoy'] >= -.05).reshape(ds.shape)].sum()
    bounded_rate = mass[(scores['Loss-aware decoy'] >= -.025).reshape(ds.shape)].sum()
    assert plugin_rate == pytest.approx(.07, abs=1e-12)
    assert bounded_rate <= .05


@pytest.fixture(scope='module')
def small_run():
    return run_benchmark(BenchmarkConfig(seeds=(1097, 2097), n_train_secure=12, n_train_attack=24,
        n_calibration=30, n_honest_test=20, n_attack_test=10, n_estimators=8))


def test_inventory_and_fixed_conditions(small_run):
    r, e = small_run
    assert default_config().seeds == (1097, 2097, 3097, 4097, 5097)
    assert len(r['conditions']) == 8 and len(r['policies']) == 7 and len(r['summary']) == 15
    assert len(r['runs'][0]['cohorts']) == 44
    assert len(r['runs'][0]['spawn_keys']) == 45
    for run in r['runs']:
        for method in run['methods'].values():
            for key, stat in method['evaluations'].items():
                if 'auc' in stat:
                    assert stat['honest_control'] == key.split('::')[0]+'::honest'
    assert all(b.rejected == 0 for run in e for b in run['cohorts'].values())


def test_independent_audit_and_refused_overwrite(small_run, tmp_path):
    r, e = small_run
    d = tmp_path/'small'
    write_results(deepcopy(r), e, d)
    a = audit_results(d)
    assert a['audit_rows'] == 1152
    assert a['rate_checks'] == 1320
    assert a['independent_screening_score_checks'] == 6480
    assert a['independent_confidence_bound_inversions'] == 2160
    assert a['independent_auc_checks'] == 1050
    with pytest.raises(FileExistsError):
        write_results(deepcopy(r), e, d)


@pytest.mark.parametrize('corruption', ['ratio', 'budget', 'policy', 'summary', 'flags', 'bounds', 'control', 'labels'])
def test_independent_audit_rejects_corruptions(small_run, tmp_path, corruption):
    r, e = deepcopy(small_run)
    key = 'lower transmissivity::PNS anchor'
    if corruption == 'ratio':
        r['screen_budget']['intensity_ratio'] = .3
    elif corruption == 'budget':
        r['screen_budget']['joint'][0] = .02
    elif corruption == 'policy':
        r['policies']['PNS anchor']['pns_frac'] = .2
    elif corruption == 'summary':
        r['summary']['Random forest'][key]['alarm_rate']['mean'] += .1
    elif corruption == 'flags':
        e[0]['flags']['Loss-aware joint'][key][0] = not e[0]['flags']['Loss-aware joint'][key][0]
    elif corruption == 'bounds':
        e[0]['bounds'][key]['signal_lower_decoy'][0] += .01
    elif corruption == 'control':
        r['runs'][0]['methods']['Loss-aware joint']['evaluations'][key]['honest_control'] = 'nominal::honest'
    elif corruption == 'labels':
        r['runs'][0]['cohorts'][key]['label'] = 0
    d = tmp_path/corruption
    write_results(r, e, d)
    with pytest.raises(ValueError):
        audit_results(d)
