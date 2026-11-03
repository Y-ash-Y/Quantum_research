"""Real-Aer and independent-count checks for honest nuisance stress testing."""
from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from adversarial.eavesdropper_game import simulate_session
from bb84.honest_nuisance import HonestSettings, probabilities, sample_honest
from classical_ml.count_detectors import CountTests
from classical_ml.fair_benchmark import BenchmarkConfig
from classical_ml.nuisance_audit import audit_results, physical_reference
from classical_ml.nuisance_benchmark import (
    NULL_PROBABILITY, _score, conditions_for, default_config, run_benchmark, write_results,
)


@pytest.mark.parametrize('noise', [0, .03, 1])
@pytest.mark.parametrize('gate', [0, .02, 1])
def test_aer_gate_counts_match_independent_closed_form(noise, gate):
    settings = HonestSettings(channel_noise=noise, gate_noise=gate)
    actual = probabilities(settings)
    expected = physical_reference(asdict(settings), {'mu_signal': .5, 'mu_decoy': .1})
    for key in actual:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12)


@pytest.mark.parametrize('kwargs', [{'eta': -1}, {'eta': 1.1}, {'gate_noise': math.nan},
                                   {'gate_noise': True}, {'n_pulses': 1.5}, {'n_pulses': 0},
                                   {'channel_noise': -1}])
def test_bad_honest_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        HonestSettings(**kwargs)


def test_sampler_is_reproducible_preserves_global_rng_and_count_features():
    state = np.random.get_state()
    settings = HonestSettings(gate_noise=.02, eta=.10)
    first = sample_honest(300, settings, np.random.default_rng(8))
    second = sample_honest(300, settings, np.random.default_rng(8))
    assert np.array_equal(first.counts, second.counts)
    assert np.array_equal(first.features, second.features)
    now = np.random.get_state()
    assert state[0] == now[0] and np.array_equal(state[1], now[1]) and state[2:] == now[2:]
    nz, nx, ez, ex, ns, nd, ds, dd = first.counts.T
    assert np.all(ns+nd == settings.n_pulses)
    assert np.all(nz+nx <= ds+dd)
    assert np.allclose(first.features[:, 0], (ez+ex)/(nz+nx))
    assert np.allclose(first.features[:, 3], (dd/nd)/(ds/ns))


def test_empty_telemetry_is_not_fabricated_zero():
    with pytest.raises(ValueError, match='No usable'):
        sample_honest(20, HonestSettings(eta=0), np.random.default_rng(9))
    with pytest.raises(ValueError, match='local numpy'):
        sample_honest(20, HonestSettings(), None)
    with pytest.raises(ValueError, match='HonestSettings'):
        probabilities({})


def test_rejected_ids_are_retained_not_silently_resampled():
    batch = sample_honest(1000, HonestSettings(n_pulses=1000), np.random.default_rng(18))
    assert 0 < batch.rejected < batch.attempted
    assert len(batch.features)+batch.rejected == 1000
    assert np.all(np.diff(batch.accepted_attempts) > 0)
    assert batch.accepted_attempts[-1] < 1000
    assert len(batch.accepted_attempts) == len(batch.counts)


def test_aggregate_marginal_means_and_variances_match_closed_form():
    s = HonestSettings(eta=.1, gate_noise=.02)
    b = sample_honest(4000, s, np.random.default_rng(67))
    assert b.rejected == 0
    p = physical_reference(asdict(s), {'mu_signal': .5, 'mu_decoy': .1})
    detection = .5*(p['signal_click']+p['decoy_click'])
    marginal = np.array([detection/4, detection/4, detection*p['q_z']/4,
                         detection*p['q_x']/4, .5, .5, .5*p['signal_click'], .5*p['decoy_click']])
    expected = s.n_pulses*marginal
    variance = s.n_pulses*marginal*(1-marginal)
    assert np.all(abs(b.counts.mean(axis=0)-expected) < 8*np.sqrt(variance/len(b.counts)))
    assert np.allclose(b.counts.var(axis=0, ddof=1), variance, rtol=.15)
    # Mutually exclusive basis counts have the negative multinomial covariance.
    expected_cov = -s.n_pulses*(detection/4)**2
    empirical_cov = np.cov(b.counts[:, :2].T)[0, 1]
    # Covariance is small relative to each marginal variance; sampling error matters.
    assert abs(empirical_cov-expected_cov) < 8*math.sqrt(variance[0]*variance[1]/len(b.counts))


def test_nominal_distribution_agrees_with_existing_pulse_simulator():
    aggregate = sample_honest(500, HonestSettings(), np.random.default_rng(24))
    pulse_rng = np.random.default_rng(25)
    pulse = [simulate_session(rng=pulse_rng, return_details=True) for _ in range(500)]
    assert all(p is not None for p in pulse) and aggregate.rejected == 0
    reference = np.array([p.counts.as_tuple() for p in pulse])
    pooled_variance = (reference.var(axis=0, ddof=1)+aggregate.counts.var(axis=0, ddof=1))/500
    assert np.all(abs(reference.mean(axis=0)-aggregate.counts.mean(axis=0)) < 8*np.sqrt(pooled_variance))
    assert not np.array_equal(reference, aggregate.counts)  # distributional, not RNG identity


def test_protocol_inventory_and_nominal_null_do_not_follow_eta():
    c = default_config()
    assert c.seeds == (1069, 2069, 3069, 4069, 5069)
    conditions = conditions_for(c)
    assert len(conditions) == 7
    assert conditions['lower transmissivity'].eta == .1
    assert conditions['combined shorter window'].n_pulses == 3000
    assert NULL_PROBABILITY == pytest.approx(-math.expm1(-.1*.15))
    with pytest.raises(ValueError, match='fixes nominal'):
        run_benchmark(replace(c, channel_noise=.06))


@pytest.fixture(scope='module')
def small_run():
    config = BenchmarkConfig(seeds=(1069, 2069), n_train_secure=12, n_train_attack=24,
                             n_calibration=30, n_honest_test=20, n_estimators=8)
    return run_benchmark(config)


def test_small_protocol_has_only_honest_evaluation_no_auc(small_run):
    report, evidence = small_run
    for run in report['runs']:
        assert len(run['spawn_keys']) == 9
        assert len({tuple(k) for k in run['spawn_keys']}) == 9
        for method in run['methods'].values():
            assert set(method['evaluations']) == set(report['conditions'])
            assert all('auc' not in stats for stats in method['evaluations'].values())
    for run in evidence:
        for key, batch in run['cohorts'].items():
            assert batch.rejected == 0
            assert batch.attempted == (30 if key == 'calibration' else 20)


def test_exact_scores_do_not_consume_condition_parameters_or_features():
    class LegacyScore:
        def score(self, X):
            return X[:, 0]
    batch = sample_honest(20, HonestSettings(eta=.1), np.random.default_rng(4))
    exact = CountTests(NULL_PROBABILITY)
    before = _score({'legacy': LegacyScore()}, exact, batch)
    batch.features[:] = 100
    after = _score({'legacy': LegacyScore()}, exact, batch)
    for name in exact.scores(batch.counts):
        assert np.array_equal(before[name], after[name])


def test_independent_artifact_audit_and_no_overwrite(small_run, tmp_path):
    report, evidence = small_run
    destination = tmp_path/'small'
    write_results(deepcopy(report), evidence, destination)
    checked = audit_results(destination)
    assert checked['audit_rows'] == 412
    assert checked['rate_checks'] == 192
    assert checked['independent_exact_pvalue_checks'] == 1020
    assert checked['independent_physical_reference_checks'] == 64
    assert checked['current_source_fingerprints'] == 'passed'
    with pytest.raises(FileExistsError):
        write_results(deepcopy(report), evidence, destination)


@pytest.mark.parametrize('corruption', ['null', 'physics', 'summary', 'flags', 'cutoff', 'inventory'])
def test_auditor_rejects_corrupted_evidence(small_run, tmp_path, corruption):
    report, evidence = deepcopy(small_run)
    if corruption == 'null':
        report['trusted_count_null']['eta'] = .1
    elif corruption == 'physics':
        report['runs'][0]['cohorts']['nominal']['physical_reference']['q_x'] += .1
    elif corruption == 'summary':
        report['summary']['Random forest']['nominal']['mean'] += .1
    elif corruption == 'flags':
        evidence[0]['flags']['Random forest']['nominal'][0] = not evidence[0]['flags']['Random forest']['nominal'][0]
    elif corruption == 'cutoff':
        report['runs'][0]['methods']['Count Fisher basis']['cutoff']['value'] = -.01
    destination = tmp_path/corruption
    write_results(report, evidence, destination)
    if corruption == 'inventory':
        import json
        file = destination/'nuisance.json'
        data = json.loads(file.read_text())
        del data['runs'][0]['methods']['Random forest']
        file.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit_results(destination)


@pytest.mark.parametrize('corruption', ['attack-label', 'test-settings', 'fractional-count', 'duplicate'])
def test_auditor_rejects_raw_row_corruption_even_with_updated_file_hash(small_run, tmp_path, corruption):
    import csv
    import gzip
    import hashlib
    import json

    report, evidence = deepcopy(small_run)
    destination = tmp_path/corruption
    write_results(report, evidence, destination)
    path = destination/'session_audit.csv.gz'
    with gzip.open(path, 'rt') as file:
        rows = list(csv.DictReader(file))
    row = next(r for r in rows if r['cohort'] == 'nominal')
    if corruption == 'attack-label':
        row['label'] = '1'
    elif corruption == 'test-settings':
        row['eta'] = '.1'
    elif corruption == 'fractional-count':
        row['z_errors'] = '1.5'
    else:
        rows.append(row.copy())
    with gzip.open(path, 'wt', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report['provenance']['audit_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/'nuisance.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit_results(destination)
