"""Integer telemetry, exact-null validity, leakage guards and saved evidence."""
import csv
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, fisher_exact

from adversarial.eavesdropper_game import simulate_session
from bb84.telemetry import COUNT_FIELDS, TelemetryCounts
from classical_ml import count_benchmark as bench
from classical_ml.count_audit import audit_results, explicit_binomial_cdf, integer_fisher
from classical_ml.count_detectors import CountTests, METHODS, fisher_pvalue
from classical_ml.fair_benchmark import BenchmarkConfig


def small_config():
    return replace(bench.default_config(), seeds=(1001, 2002), n_train_secure=12,
                   n_train_attack=24, n_calibration=30, n_honest_test=20,
                   n_attack_test=10, n_estimators=12)


@pytest.fixture(scope='module')
def small_run():
    return bench.run_benchmark(small_config())


@pytest.mark.parametrize('kwargs', [
    {'z_errors': 11}, {'signal_detected': 101}, {'decoy_sent': -1},
    {'z_sifted': True}, {'x_errors': 1.2}, {'z_sifted': 25},
])
def test_impossible_counts_rejected(kwargs):
    values = dict(z_sifted=10, x_sifted=10, z_errors=1, x_errors=2,
                  signal_sent=100, decoy_sent=100, signal_detected=10, decoy_detected=10)
    with pytest.raises(ValueError):
        TelemetryCounts(**{**values, **kwargs})


@pytest.mark.parametrize('policy', [bench.AttackPolicy(), bench.AttackPolicy(.1, .95),
                                    bench.AttackPolicy(0, .5, .85)])
def test_counts_are_actual_integer_event_counts(policy):
    out = simulate_session(*policy.as_tuple(), rng=np.random.default_rng(55), return_details=True)
    c = out.counts
    assert c.signal_sent+c.decoy_sent == 6000
    assert c.z_sifted+c.x_sifted == out.sifted_bits
    assert isinstance(c.z_errors, int)
    np.testing.assert_array_equal(c.features(), out.features)
    assert out.known_bits <= out.sifted_bits


def test_empty_basis_counts_are_uninformative_not_a_nan_alarm():
    scores = CountTests(.1).scores(np.zeros((1, 8), dtype=int))
    assert all(v.tolist() == [-1.0] for v in scores.values())
    with pytest.raises(ValueError):
        TelemetryCounts(0, 0, 0, 0, 0, 0, 0, 0).features()


@pytest.mark.parametrize('matrix', [[], np.zeros((3, 7)), np.zeros((2, 8), dtype=float),
                                   np.full((1, 8), np.nan)])
def test_invalid_count_matrix_does_not_get_silently_coerced(matrix):
    with pytest.raises(ValueError):
        CountTests(.1).pvalues(matrix)


def test_exact_pvalues_match_independent_small_table_enumeration():
    for nz, nx in [(0, 5), (4, 5), (20, 23)]:
        for ez in range(nz+1):
            for ex in range(nx+1):
                expected = integer_fisher(nz, nx, ez, ex)
                assert fisher_pvalue(nz, nx, ez, ex) == pytest.approx(expected, abs=1e-12)
    assert fisher_pvalue.cache_info().maxsize == 4096


@pytest.mark.parametrize('p', [0, .01, .1, .5, 1])
def test_explicit_binomial_sum_matches_scipy_including_endpoints(p):
    for n in [0, 1, 20, 3000]:
        for k in sorted({0, min(5, n), n//2, n}):
            assert explicit_binomial_cdf(n, k, p) == pytest.approx(binom.cdf(k, n, p), abs=1e-10)


@pytest.mark.parametrize('q', [0, .01, .03, .2, .5, 1])
def test_fisher_null_is_superuniform_when_common_noise_is_unknown(q):
    nz, nx = 4, 5
    for alpha in (.01, .05, .1, .5):
        false_probability = sum(
            float(binom.pmf(ez, nz, q)*binom.pmf(ex, nx, q))
            for ez in range(nz+1) for ex in range(nx+1)
            if integer_fisher(nz, nx, ez, ex) <= alpha)
        assert false_probability <= alpha+1e-12


def test_binomial_null_is_superuniform_and_joint_rule_has_union_bound():
    # Different sample sizes explicitly change the rejection probabilities.
    for n in (5, 30, 100):
        alpha = .05
        p = .1
        tail = sum(float(binom.pmf(k, n, p)) for k in range(n+1)
                   if explicit_binomial_cdf(n, k, p) <= alpha/2)
        assert tail <= alpha/2+1e-12
    counts = np.array([[50, 50, 0, 15, 1000, 1000, 100, 2],
                       [50, 50, 3, 3, 1000, 1000, 100, 100]])
    values = CountTests(.1).pvalues(counts)
    np.testing.assert_allclose(values[METHODS[2]],
                               np.minimum(1, 2*np.minimum(values[METHODS[0]], values[METHODS[1]])))
    assert (values[METHODS[2]] <= .05).tolist() == [True, False]


def test_no_prior_seeds_and_mimic_equivalence():
    assert not set(bench.default_config().seeds) & {11, 22, 33, 44, 55, 101, 202, 303, 404, 505}
    assert bench.MIMIC_NOISE/2 == pytest.approx(.03/2+.1*(1-.03)/4)
    with pytest.raises(ValueError, match='nominal noise'):
        bench.run_benchmark(replace(small_config(), channel_noise=.02))


def test_real_count_benchmark_replays_and_keeps_analytic_cutoffs(small_run):
    report, _ = small_run
    replay, _ = bench.run_benchmark(small_config())
    assert report['runs'] == replay['runs']
    assert report['summary'] == replay['summary']
    for run in report['runs']:
        assert len(set(map(tuple, run['spawn_keys']))) == 35
        for method in METHODS:
            assert run['methods'][method]['cutoff']['value'] == -.05
            assert run['methods'][method]['cutoff']['operator'] == '>='
            assert not run['methods'][method]['calibrated']
            assert 0 <= run['methods'][method]['mimic_unbiased_ir_auc'] <= 1


def test_test_noise_and_private_knowledge_do_not_enter_count_scorer(monkeypatch, small_run):
    original = bench.sample_cohort
    def poison(policy, attempted, config, rng):
        cohort = original(policy, attempted, config, rng)
        if attempted != config.n_calibration:
            cohort.features[:, 0] = .9
            cohort.known_bits[:] = cohort.sifted_bits
        return cohort
    monkeypatch.setattr(bench, 'sample_cohort', poison)
    changed, _ = bench.run_benchmark(small_config())
    for before, after in zip(small_run[0]['runs'], changed['runs']):
        assert before['training'] == after['training']
        for name in bench.METHODS:
            assert before['methods'][name]['cutoff'] == after['methods'][name]['cutoff']
        for name in METHODS:
            assert before['methods'][name] == after['methods'][name]


def test_historical_saved_session_prefix_is_unchanged():
    # Sampled compatibility check: 50 honest sessions from each prior seed.
    root = Path(__file__).resolve().parents[1]
    with gzip.open(root/'results/benchmarks/robustness-v2/session_audit.csv.gz', 'rt') as stream:
        rows = [r for r in csv.DictReader(stream) if r['cohort'] == 'nominal::honest'
                and int(r['session_index']) < 50]
    cfg = BenchmarkConfig()
    for seed in (101, 202, 303, 404, 505):
        stream = np.random.default_rng(np.random.SeedSequence(seed).spawn(30)[2])
        historical = [r for r in rows if int(r['seed']) == seed]
        current = bench.sample_cohort(bench.AttackPolicy(), 50, cfg, stream)
        np.testing.assert_array_equal(current.features,
            np.array([[float(r[f]) for f in bench.FEATURES] for r in historical]))
        np.testing.assert_array_equal(current.known_bits, [int(r['known_bits']) for r in historical])
        np.testing.assert_array_equal(current.sifted_bits, [int(r['sifted_bits']) for r in historical])


def test_saved_count_rows_and_audit(tmp_path, small_run):
    report, evidence = small_run
    dest = bench.write_results(json.loads(json.dumps(report)), evidence, tmp_path/'counts')
    audit = audit_results(dest)
    assert audit['rate_checks'] == 2*12*34
    assert audit['independent_auc_checks'] == 2*12*29
    assert audit['audit_rows'] == 2*(36+30+4*(20+7*10)+20)
    assert audit['independent_exact_pvalue_checks'] == 2*(30+4*(20+7*10)+20)*3
    with pytest.raises(FileExistsError):
        bench.write_results(report, evidence, dest)


@pytest.mark.parametrize('corruption,message', [
    ('null', 'Physical null'), ('operator', 'operator'), ('summary', 'summary'),
    ('mimic', 'mimic'), ('count_hash', 'Count fingerprint'), ('missing_method', 'inventory'),
])
def test_auditor_rejects_incorrect_count_outputs(tmp_path, small_run, corruption, message):
    report, evidence = small_run
    report = json.loads(json.dumps(report))
    dest = bench.write_results(report, evidence, tmp_path/corruption)
    if corruption == 'null':
        report['count_null']['decoy_click_probability'] = .5
    elif corruption == 'operator':
        report['runs'][0]['methods'][METHODS[0]]['cutoff']['operator'] = '>'
    elif corruption == 'summary':
        report['summary'][METHODS[0]]['nominal::honest']['alarm_rate']['mean'] += .1
    elif corruption == 'mimic':
        report['runs'][0]['methods'][METHODS[0]]['mimic_unbiased_ir_auc'] += .1
    elif corruption == 'count_hash':
        report['runs'][0]['cohorts']['nominal::honest']['counts_sha256'] = '0'*64
    else:
        del report['summary'][METHODS[0]]
    (dest/'counts.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        audit_results(dest)
