"""Calibration, no-leakage and row-level benchmark provenance checks."""
import csv
from dataclasses import replace
import gzip
import hashlib
import json

import numpy as np
import pytest

from classical_ml import fair_benchmark as bench
from classical_ml.benchmark_audit import audit_results


def small_config():
    return bench.BenchmarkConfig(
        seeds=(1, 2), n_train_secure=12, n_train_attack=24, n_calibration=30,
        n_honest_test=20, n_attack_test=10, n_estimators=12)


@pytest.mark.parametrize('m,alpha,k', [(19, 0.05, 19), (1000, 0.05, 951),
                                     (99, 0.1, 90), (10, 0.5, 6)])
def test_conservative_rank_threshold(m, alpha, k):
    threshold = bench.calibrate_threshold(np.arange(m), alpha)
    assert threshold.rank == k
    assert threshold.value == k-1
    assert (m+1-k)/(m+1) <= alpha
    assert not threshold.flags([threshold.value])[0]  # strict >


def test_too_small_calibration_is_not_silently_anti_conservative():
    threshold = bench.calibrate_threshold(np.arange(10), 0.05)
    assert threshold.rank == 11 and threshold.value is None
    assert not threshold.flags([100, 10000]).any()
    json.dumps(bench.asdict(threshold), allow_nan=False)


def test_calibration_ties_are_conservative():
    threshold = bench.calibrate_threshold(np.zeros(1000), 0.05)
    assert not threshold.flags(np.zeros(1000)).any()
    assert threshold.flags([0, 1]).tolist() == [False, True]


@pytest.mark.parametrize('scores,alpha', [([], 0.05), ([np.nan], 0.05),
                                       ([[1]], 0.05), ([1], 0), ([1], 1)])
def test_invalid_calibration(scores, alpha):
    with pytest.raises(ValueError):
        bench.calibrate_threshold(scores, alpha)


def test_tail_scores_match_hand_count_with_ties():
    reference = np.array([[0, 0, 0.1, 0.2], [1, 1, 0.2, 0.3],
                          [1, 2, 0.3, 0.4]])
    X = np.array([[1, 1, 0.2, 0.3], [2, 3, 0.8, 0.1]])
    # First row >= counts: 2,2,3,2 -> corrected tails 3/4,3/4,1,3/4.
    max_score = bench.JointTailScore(reference, 'max')
    sum_score = bench.JointTailScore(reference, 'sum')
    assert max_score.score(X)[0] == pytest.approx(-np.log(0.75))
    assert sum_score.score(X)[0] == pytest.approx(-6*np.log(0.75))
    assert max_score.score(X)[1] == pytest.approx(np.log(4))
    assert sum_score.score(X)[1] == pytest.approx(8*np.log(4))


def test_physical_score_directions():
    X = np.array([[0.01, 0.02, 0.07, 0.2], [0.1, 0.3, 0.07, 0.05]])
    for scorer in (bench.ColumnScore(0), bench.ColumnScore(1), bench.ColumnScore(3, -1)):
        assert scorer.score(X)[1] > scorer.score(X)[0]


def test_scaler_fits_training_rows_only():
    X = np.array([[0, 0, 0, 0], [1, 2, 3, 4]]*10, dtype=float)
    scorers = bench.fit_scorers(X, np.tile([0, 1], 10), 3, 8)
    for name in ('Logistic regression', 'RBF SVM'):
        scaler = scorers[name].model.steps[0][1]
        np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
        before = scaler.mean_.copy()
        scorers[name].score(np.full((4, 4), 1e4))
        np.testing.assert_array_equal(scaler.mean_, before)


@pytest.mark.parametrize('kwargs', [
    {'seeds': (1,)}, {'seeds': (1, 1)}, {'seeds': (True, 2)},
    {'n_pulses': 0}, {'n_attack_test': -1}, {'n_calibration': 2.5},
    {'target_fpr': 0}, {'target_fpr': np.nan}, {'channel_noise': 1.1},
])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        bench.BenchmarkConfig(**kwargs)


def test_real_benchmark_replays_and_records_denominators():
    config = small_config()
    a, ae = bench.run_benchmark(config)
    b, be = bench.run_benchmark(config)
    assert a['runs'] == b['runs']
    assert a['summary'] == b['summary']
    assert a['paired_vs_joint_max'] == b['paired_vs_joint_max']
    for run in a['runs']:
        assert len(set(map(tuple, run['spawn_keys']))) == 9
        for name, method in run['methods'].items():
            assert method['calibrated'] == (name != 'QBER 11% reference')
            assert method['honest_test']['attempted'] == config.n_honest_test
            for family, result in method['attacks'].items():
                assert result['attempted'] == config.n_attack_test
                assert result['accepted']+result['rejected'] == result['attempted']
                assert 0 <= result['auc'] <= 1
                assert result['rate'] == result['flagged']/result['accepted']
        for family, values in run['paired_vs_joint_max'].items():
            for name, paired in values.items():
                assert paired['detection_difference'] == pytest.approx(
                    (paired['ml_only']-paired['stat_only'])/paired['accepted'])
    assert len(ae) == len(be) == 2
    json.dumps(a, allow_nan=False)


def test_honest_test_changes_cannot_tune_training_or_cutoffs(monkeypatch):
    config = small_config()
    before, _ = bench.run_benchmark(config)
    original = bench.sample_cohort
    def poisoned(policy, attempted, cfg, rng):
        cohort = original(policy, attempted, cfg, rng)
        if attempted == cfg.n_honest_test:
            cohort.features[:, 0] = 0.7
        return cohort
    monkeypatch.setattr(bench, 'sample_cohort', poisoned)
    after, _ = bench.run_benchmark(config)
    for a, b in zip(before['runs'], after['runs']):
        assert a['training'] == b['training']
        for name in bench.METHODS:
            assert a['methods'][name]['cutoff'] == b['methods'][name]['cutoff']
            assert a['methods'][name]['calibration'] == b['methods'][name]['calibration']
        assert b['methods']['QBER calibrated']['honest_test']['rate'] == 1


def test_audit_rows_reproduce_counts_and_prevent_overwrite(tmp_path):
    report, evidence = bench.run_benchmark(small_config())
    destination = bench.write_results(report, evidence, tmp_path/'benchmark')
    saved = json.loads((destination/'benchmark.json').read_text())
    assert audit_results(destination)['independent_auc_checks'] == 108
    compressed = destination/'session_audit.csv.gz'
    assert hashlib.sha256(compressed.read_bytes()).hexdigest() == saved['provenance']['audit_sha256']
    with gzip.open(compressed, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    ids = [(r['seed'], r['cohort'], r['session_index']) for r in rows]
    assert len(ids) == len(set(ids))
    for run in saved['runs']:
        for cohort in ('honest_test', *bench.POLICIES):
            cohort_rows = [r for r in rows if int(r['seed']) == run['seed'] and r['cohort'] == cohort]
            for name in bench.METHODS:
                target = (run['methods'][name]['honest_test'] if cohort == 'honest_test'
                          else run['methods'][name]['attacks'][cohort])
                assert len(cohort_rows) == target['accepted']
                assert sum(int(r[f'{name}::flag']) for r in cohort_rows) == target['flagged']
    with pytest.raises(FileExistsError):
        bench.write_results(report, evidence, destination)


def test_auditor_detects_corrupted_summary(tmp_path):
    report, evidence = bench.run_benchmark(small_config())
    report['summary']['QBER calibrated']['honest_fpr']['mean'] += 0.1
    destination = bench.write_results(report, evidence, tmp_path/'corrupt')
    with pytest.raises(ValueError, match='summary'):
        audit_results(destination)


def test_auditor_detects_corrupted_interval(tmp_path):
    report, evidence = bench.run_benchmark(small_config())
    report['runs'][0]['methods']['QBER calibrated']['honest_test']['wilson95'][1] = 0.123
    destination = bench.write_results(report, evidence, tmp_path/'bad-interval')
    with pytest.raises(ValueError, match='Wilson'):
        audit_results(destination)


def test_seed_spread_is_sample_sd_not_confidence_interval():
    stats = bench._spread([0.2, 0.4])
    assert stats['mean'] == pytest.approx(0.3)
    assert stats['seed_sd'] == pytest.approx(np.sqrt(0.02))
    assert stats['seed_min'] == 0.2 and stats['seed_max'] == 0.4
