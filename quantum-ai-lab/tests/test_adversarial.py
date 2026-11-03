"""Adaptive-search regression tests; Monte Carlo checks use the real Aer LUT.

Tests that replace _sample/simulate_session exercise statistical plumbing only;
they are not substituted for the separate real physics integration checks.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest

from adversarial import eavesdropper_game as game


class FeatureScore:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        score = features[:, 0]
        return np.column_stack([1-score, score])


def detector(threshold=0.5):
    return {'model': FeatureScore(), 'cols': [0], 'thr': threshold}


def batch(score, reps=8, info=0.4):
    return game.SampleBatch(np.tile([score, 0, 0.07, 0.2], (reps, 1)),
                            np.full(reps, info), round(info * 1000 * reps),
                            1000 * reps, reps, 0)


@pytest.mark.parametrize('target', [0, 0.001, 0.05, 0.4, 0.5, 0.8, 0.999, 1])
def test_all_candidates_meet_expected_information_constraint(target):
    policies = game.candidate_policies(target, 20, np.random.default_rng(12))
    assert policies
    assert len(policies) == len(set(policies))
    for policy in policies:
        assert game.expected_information(policy) == pytest.approx(target, abs=1e-12)
        assert all(0 <= p <= 1 for p in policy.as_tuple())
        if target > 0.5:
            assert policy.pns_frac > 0


def test_pns_rate_is_not_sifted_information():
    assert game.expected_information((0, 0.5, 0.4)) == pytest.approx(0.3676661787, abs=1e-10)
    policy = game.policy_for_target(0.4)
    assert policy.pns_frac == pytest.approx(0.4332291765, abs=1e-10)
    assert game.expected_information(policy) == pytest.approx(0.4)
    assert game.pns_forward_prob() == pytest.approx(0.8010343811, abs=1e-10)


@pytest.mark.parametrize('beta', [0, 0.3, 0.5, 0.95, 1])
def test_ir_information_independent_of_basis_bias(beta):
    assert game.expected_information((0.6, beta, 0)) == pytest.approx(0.3)


@pytest.mark.parametrize('bad', [-0.1, 1.1, np.nan, np.inf, True, '0.2'])
def test_invalid_policy_or_target_rejected(bad):
    with pytest.raises(ValueError):
        game.AttackPolicy(ir_rate=bad)
    with pytest.raises(ValueError):
        game.policy_for_target(bad)


def test_ir_cannot_overshoot_target():
    with pytest.raises(ValueError, match='exceeds'):
        game.policy_for_target(0.1, ir_rate=0.3)


def test_zero_target_has_only_null_candidate():
    assert game.candidate_policies(0, 20, np.random.default_rng(0)) == [game.AttackPolicy()]


def test_lut_is_exact_bounded_and_immutable():
    lut = game._lut(0.03)
    assert lut[0] == pytest.approx(0.015)
    assert lut[7] == pytest.approx(0.985)
    assert game._lut.cache_info().maxsize == 64
    with pytest.raises(ValueError):
        lut[0] = 1


@pytest.mark.parametrize('policy, noise, expected_qber, expected_info', [
    ((0, 0.5, 0), 0.03, 0.015, 0),
    ((1, 0.5, 0), 0, 0.25, 0.5),
    ((1, 1, 0), 0, 0.25, 0.5),
    ((0, 0.5, 1), 0.03, 0.015, 1),
    ((0.3, 0.95, 0.25), 0.03, None, None),
])
def test_real_vectorized_physics(policy, noise, expected_qber, expected_info):
    # 1M emitted pulses; about 20K sifted samples, enough for these tolerances.
    out = game.simulate_session(*policy, n_pulses=1_000_000, channel_noise=noise,
                                rng=np.random.default_rng(7), return_details=True)
    assert out is not None
    if expected_info is None:
        expected_info = game.expected_information(policy)
        a, b = game._gains()
        r, _, rho = policy
        normal_weight = (1-rho)*a / ((1-rho)*a + rho*b)
        expected_qber = noise/2 + normal_weight*r*(1-noise)/4
    assert out.features[0] == pytest.approx(expected_qber, abs=0.008)
    assert out.info == pytest.approx(expected_info, abs=0.012)
    assert out.info == out.known_bits / out.sifted_bits
    assert out.features[2] == pytest.approx(-np.expm1(-game.MU_S * game.ETA), abs=0.002)
    a, b = game._gains()
    r, beta, rho = policy
    normal_weight = (1-rho)*a / ((1-rho)*a + rho*b)
    expected_asymmetry = normal_weight*r*abs(2*beta-1)*(1-noise)/2
    assert out.features[1] == pytest.approx(expected_asymmetry, abs=0.016)


def test_simulator_and_training_replay():
    kwargs = dict(n_pulses=6000, return_details=True)
    a = game.simulate_session(0.2, 0.8, 0.1, rng=np.random.default_rng(9), **kwargs)
    b = game.simulate_session(0.2, 0.8, 0.1, rng=np.random.default_rng(9), **kwargs)
    assert a == b
    a = game.build_training_set(8, 30, rng=np.random.default_rng(5), return_details=True)
    b = game.build_training_set(8, 30, rng=np.random.default_rng(5), return_details=True)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.policies, b.policies)
    assert np.bincount(a.labels).tolist() == [8, 30]
    attacks = a.policies[a.labels == 1]
    assert np.all((attacks[:, 0] > 0) | (attacks[:, 2] > 0))
    assert np.any((attacks[:, 0] > 0) & (attacks[:, 2] == 0))
    assert np.any((attacks[:, 0] == 0) & (attacks[:, 2] > 0))
    assert np.any((attacks[:, 0] > 0) & (attacks[:, 2] > 0))


def test_rejections_have_counts_and_empty_shape():
    assert game.simulate_session(n_pulses=1, rng=np.random.default_rng(1)) is None
    result = game._sample((0, 0.5, 0), 5, 1, 0.03, np.random.default_rng(2))
    assert result.features.shape == (0, 4)
    assert result.infos.shape == (0,)
    assert result.attempted == result.rejected == 5
    with pytest.raises(ValueError):
        _ = result.pooled_info


def test_training_retry_budget_is_bounded(monkeypatch):
    calls = []
    def unusable(*args, **kwargs):
        calls.append(1)
        return None
    monkeypatch.setattr(game, 'simulate_session', unusable)
    with pytest.raises(ValueError, match='max_attempts'):
        game.build_training_set(2, 2, max_attempts=5)
    assert len(calls) == 5


def test_training_fills_rejected_rows(monkeypatch):
    calls = []
    def intermittent(*args, **kwargs):
        calls.append(1)
        return None if len(calls) % 2 else ([0.01, 0.02, 0.07, 0.2], 0)
    monkeypatch.setattr(game, 'simulate_session', intermittent)
    rows = game.build_training_set(3, 4, return_details=True)
    assert rows.features.shape == (7, 4)
    assert rows.attempted == 14 and rows.rejected == 7


@pytest.mark.parametrize('s,n', [(0, 10), (10, 10), (5, 10), (50, 100),
                               (4, 4), (9, 9), (21, 21), (23, 23), (0, 4)])
def test_wilson_bounds(s, n):
    low, high = game.wilson_interval(s, n)
    assert 0 <= low <= s/n <= high <= 1
    if s in (0, n):
        assert high - low > 0  # no false certainty from zero/all flagged


@pytest.mark.parametrize('features', [[], np.zeros((2, 3)), [[np.nan]*4], np.zeros((0, 4))])
def test_bad_or_empty_detector_features_rejected(features):
    with pytest.raises(ValueError):
        game.detection_rate(detector(), features)


def test_positive_class_column_is_not_assumed():
    class ReverseScore(FeatureScore):
        classes_ = np.array([1, 0])
        def predict_proba(self, features):
            return super().predict_proba(features)[:, ::-1]
    det = detector()
    det['model'] = ReverseScore()
    assert game.detection_rate(det, [[0.7, 0, 0, 0]]) == 1


def test_all_detected_candidates_still_select_a_policy(monkeypatch):
    monkeypatch.setattr(game, '_sample', lambda *a, **k: batch(0.9))
    result = game.best_response({'test': detector()}, 0.4, n_candidates=3)
    assert result.responses['test'].policy is not None
    assert result.responses['test'].evaluation.rate == 1
    assert len(result.candidates) == 6  # pure PNS, two IR anchors, three mixes


def test_search_and_final_evaluation_are_different_batches(monkeypatch):
    calls, seen_rngs = [], []
    def samples(policy, reps, n, noise, rng):
        seen_rngs.append(rng)
        calls.append(1)
        # One candidate: good on search, poor on independent final evaluation.
        return batch(0.1 if len(calls) == 1 else 0.9, reps)
    monkeypatch.setattr(game, 'candidate_policies', lambda t, n, r: [game.policy_for_target(t)])
    monkeypatch.setattr(game, '_sample', samples)
    result = game.best_response({'test': detector()}, 0.4, reps=8, eval_reps=13)
    response = result.responses['test']
    assert response.selection.rate == 0
    assert response.evaluation.rate == 1
    assert response.evaluation.attempted == 13
    assert seen_rngs[0] is not seen_rngs[1]
    assert len(calls) == 2
    json.dumps(asdict(result))  # result records are exportable without custom encoders


def test_infeasible_candidates_are_not_evaluated(monkeypatch):
    monkeypatch.setattr(game, 'candidate_policies', lambda *a: [game.AttackPolicy()])
    monkeypatch.setattr(game, '_sample', lambda *a: pytest.fail('infeasible policy sampled'))
    with pytest.raises(ValueError, match='No feasible'):
        game.best_response({'test': detector()}, 0.4)


def test_all_rejected_search_is_explicit_failure(monkeypatch):
    empty = game.SampleBatch(np.zeros((0, 4)), np.array([]), 0, 0, 3, 3)
    monkeypatch.setattr(game, '_sample', lambda *a: empty)
    with pytest.raises(ValueError, match='No feasible'):
        game.best_response({'test': detector()}, 0.4)


def test_eval_size_and_detector_order_do_not_change_selection_or_streams(monkeypatch):
    def samples(policy, reps, n, noise, rng):
        scores = rng.random((reps, 4))
        return game.SampleBatch(scores, np.full(reps, 0.4), 400*reps, 1000*reps, reps, 0)
    monkeypatch.setattr(game, '_sample', samples)
    dets = {'a': detector(0.3), 'b': detector(0.7)}
    a = game.best_response(dets, 0.4, reps=8, eval_reps=9,
                          n_candidates=5, rng=np.random.default_rng(22))
    b = game.best_response(dets, 0.4, reps=8, eval_reps=30,
                          n_candidates=5, rng=np.random.default_rng(22))
    c = game.best_response(dict(reversed(list(dets.items()))), 0.4, reps=8,
                          eval_reps=9, n_candidates=5, rng=np.random.default_rng(22))
    assert a.candidates == b.candidates == c.candidates
    assert a.streams == b.streams == c.streams
    for name in dets:
        assert a.responses[name].policy == b.responses[name].policy
        assert a.responses[name].evaluation == c.responses[name].evaluation


def test_frontier_preserves_detector_specific_coordinates(monkeypatch):
    def search(dets, target, **kwargs):
        a = game.BestResponse(game.policy_for_target(target), target,
                             None, game.RateEstimate(1, 10, 10, 0, 0.1, (0.02, 0.4)),
                             target-0.02, 10, 100)
        b = game.BestResponse(game.policy_for_target(target), target,
                             None, game.RateEstimate(2, 10, 10, 0, 0.2, (0.03, 0.5)),
                             target+0.02, 20, 100)
        return game.SearchResult(target, {'a': a, 'b': b}, [], {})
    monkeypatch.setattr(game, 'best_response', search)
    result = game.frontier_curves({'a': detector(), 'b': detector()}, [0.1, 0.3])
    assert result.achieved_infos['a'] == pytest.approx([0.08, 0.28])
    assert result.achieved_infos['b'] == pytest.approx([0.12, 0.32])
    assert result.detection_rates['a'] == [0.1, 0.1]
    assert result.detection_intervals['b'] == [(0.03, 0.5)]*2


def test_calibration_is_not_honest_test(monkeypatch):
    calls = []
    def samples(policy, reps, n, noise, rng):
        calls.append(rng)
        return batch(0.0 if len(calls) == 1 else 1.0, reps)
    monkeypatch.setattr(game, '_sample', samples)
    X = np.array([[0, 0, 0, 0], [1, 1, 1, 1]] * 10)
    dets = game.train_detectors(X, [0, 1]*10, n_estimators=12,
                               n_secure_cal=8, n_secure_test=11)
    assert calls[0] is not calls[1]
    for det in dets.values():
        assert det['calibration'].rate == 0
        assert det['honest_test'].rate == 1
        assert det['honest_test'].accepted == 11
        assert det['nominal_fpr'] == 0.05


def test_real_rf_and_search_integration():
    X, y = game.build_training_set(24, 48, rng=np.random.default_rng(50))
    dets = game.train_detectors(X, y, n_estimators=16,
                               n_secure_cal=20, n_secure_test=25,
                               rng=np.random.default_rng(51))
    result = game.frontier_curves(dets, [0.1, 0.6], reps=6, eval_reps=9,
                                 n_candidates=2, rng=np.random.default_rng(52))
    assert result.target_infos == [0.1, 0.6]
    for name, det in dets.items():
        assert det['honest_test'].attempted == 25
        for search in result.searches:
            response = search.responses[name]
            assert response.expected_info == pytest.approx(search.target_info)
            assert response.evaluation.attempted == 9
            assert response.evaluation.accepted + response.evaluation.rejected == 9
            assert np.isfinite(response.achieved_info)
            assert response.evaluation.interval[0] <= response.evaluation.rate
            json.dumps(asdict(search))


@pytest.mark.parametrize('kwargs', [
    {'n_pulses': 0}, {'n_pulses': 1.2}, {'channel_noise': np.nan},
    {'pns_frac': -0.1}, {'basis_bias': 1.1}, {'rng': 2},
])
def test_simulator_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        game.simulate_session(**kwargs)


def test_empty_frontier_and_bad_search_counts():
    with pytest.raises(ValueError):
        game.frontier_curves({'test': detector()}, [])
    with pytest.raises(ValueError):
        game.best_response({}, 0.2)
    with pytest.raises(ValueError):
        game.best_response({'test': detector()}, 0.2, eval_reps=0)
    with pytest.raises(ValueError):
        game.best_response({'test': detector()}, 0.2, n_candidates=-1)


def test_sampling_information_is_pooled_not_average_of_ratios(monkeypatch):
    sessions = iter([game.Session([0.1, 0, 0.07, 0.2], 10, 20),
                     game.Session([0.1, 0, 0.07, 0.2], 0, 80), None])
    monkeypatch.setattr(game, 'simulate_session', lambda *a, **k: next(sessions))
    result = game._sample(game.AttackPolicy(), 3, 6000, 0.03, np.random.default_rng(8))
    assert result.infos.mean() == 0.25
    assert result.pooled_info == 0.1
    assert result.attempted == 3 and result.rejected == 1


def test_empty_candidate_rejections_are_retained(monkeypatch):
    calls = []
    def samples(*args):
        calls.append(1)
        if len(calls) == 1:
            return game.SampleBatch(np.zeros((0, 4)), np.array([]), 0, 0, 8, 8)
        return batch(0.4)
    monkeypatch.setattr(game, '_sample', samples)
    result = game.best_response({'test': detector()}, 0.4, n_candidates=0)
    assert result.candidates[0].rejected == 8
    assert result.candidates[0].selection_info is None
    assert result.candidates[0].selection_rates == {}
    assert result.responses['test'].policy.ir_rate > 0


def test_pulse_sweep_preserves_legacy_api_and_adds_honest_audit(monkeypatch):
    estimate = game.RateEstimate(1, 8, 8, 0, 0.125, (0.02, 0.47))
    monkeypatch.setattr(game, 'build_training_set', lambda *a, **k: (None, None))
    monkeypatch.setattr(game, 'train_detectors', lambda *a, **k:
                        {'Combined': {**detector(), 'honest_test': estimate}})
    monkeypatch.setattr(game, '_sample', lambda *a, **k: batch(0.8))
    rows = game.stealth_floor_curve([2000], return_details=True)
    assert rows[0].emitted_pulses == 2000
    assert rows[0].honest_test is estimate
    assert rows[0].evaluation.rate == 1
    assert rows[0].expected_info == pytest.approx(0.3676661787, abs=1e-10)
    assert game.stealth_floor_curve([2000]) == [(2000, 0.4, 1)]


@pytest.mark.parametrize('successes,trials,confidence', [
    (0, 0, 0.95), (11, 10, 0.95), (-1, 10, 0.95),
    (1, 10, 0), (1, 10, 1), (1, 10, np.nan),
])
def test_invalid_intervals(successes, trials, confidence):
    with pytest.raises(ValueError):
        game.wilson_interval(successes, trials, confidence)
