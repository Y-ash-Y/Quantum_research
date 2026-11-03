"""Frozen operating points, matched controls, exact replay and evidence checks."""
from dataclasses import replace
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from classical_ml import robustness_benchmark as robust
from classical_ml.robustness_audit import audit_results


def small_config():
    return replace(robust.default_config(), seeds=(101, 202), n_train_secure=12,
                   n_train_attack=24, n_calibration=30, n_honest_test=20,
                   n_attack_test=10, n_estimators=12)


@pytest.fixture(scope='module')
def small_run():
    return robust.run_robustness(small_config())


def test_fresh_default_seeds_and_strength_support():
    assert not set(robust.default_config().seeds) & {11, 22, 33, 44, 55}
    for name, policy in robust.POLICIES.items():
        strength = policy.ir_rate or policy.pns_frac
        if 'below' in name:
            assert 0 < strength < 0.02
        elif 'above' in name:
            assert strength > 0.60
        else:
            assert 0.02 <= strength < 0.60


def test_condition_configs_do_not_modify_nominal_config():
    config = small_config()
    conditions = robust.conditions_for(config)
    assert conditions['shorter window']['n_pulses'] == config.n_pulses//2
    assert config.n_pulses == 6000 and config.channel_noise == 0.03
    assert conditions['higher noise']['channel_noise'] == 0.06
    with pytest.raises(ValueError, match='shorter window'):
        robust.conditions_for(replace(config, n_pulses=1))


def test_real_aer_run_replays(small_run):
    report, _ = small_run
    again, _ = robust.run_robustness(small_config())
    assert report['runs'] == again['runs']
    assert report['summary'] == again['summary']
    for run in report['runs']:
        assert len(set(map(tuple, run['spawn_keys']))) == 30
        for key, meta in run['cohorts'].items():
            if meta['label']:
                for method in run['methods'].values():
                    stats = method['evaluations'][key]
                    assert stats['honest_control'] == f"{meta['condition']}::honest"
                    assert 0 <= stats['auc'] <= 1


def test_shifted_evaluation_cannot_change_fits_or_cutoffs(monkeypatch, small_run):
    baseline, _ = small_run
    original = robust.sample_cohort
    fits = []
    fit_original = robust.fit_scorers
    def poison(policy, attempted, config, rng):
        cohort = original(policy, attempted, config, rng)
        if attempted != config.n_calibration:
            cohort.features[:, 0] = 0.9
        return cohort
    def tracked_fit(*args):
        fits.append(args[2])
        return fit_original(*args)
    monkeypatch.setattr(robust, 'sample_cohort', poison)
    monkeypatch.setattr(robust, 'fit_scorers', tracked_fit)
    changed, _ = robust.run_robustness(small_config())
    assert fits == [101, 202]  # once per seed, never once per condition
    for before, after in zip(baseline['runs'], changed['runs']):
        assert before['training'] == after['training']
        for name in robust.METHODS:
            assert before['methods'][name]['cutoff'] == after['methods'][name]['cutoff']
            assert before['methods'][name]['calibration'] == after['methods'][name]['calibration']
        assert after['methods']['QBER calibrated']['evaluations']['higher noise::honest']['rate'] == 1


def test_real_usable_honest_and_attack_qber_match_theory():
    # Many pulses test physical expectations, not only file/metric plumbing.
    cfg = replace(small_config(), n_pulses=100_000)
    for noise in (0.01, 0.03, 0.06):
        current = replace(cfg, channel_noise=noise)
        for policy in (robust.AttackPolicy(), robust.AttackPolicy(0.8, 0.95),
                       robust.AttackPolicy(0, 0.5, 0.85)):
            cohort = robust.sample_cohort(policy, 25, current, np.random.default_rng(900))
            errors = np.rint(cohort.features[:, 0]*cohort.sifted_bits).sum()
            qber = errors/cohort.sifted_bits.sum()
            expected = noise/2 + policy.ir_rate*(1-noise)/4
            # PNS and honest have r=0; all policies here are pure-family policies.
            se = np.sqrt(expected*(1-expected)/cohort.sifted_bits.sum())
            assert abs(qber-expected) < 6*se


def test_audit_and_no_overwrite(tmp_path, small_run):
    report, evidence = small_run
    # Avoid mutating the shared fixture when the writer adds provenance.
    report = json.loads(json.dumps(report))
    dest = robust.write_results(report, evidence, tmp_path/'robustness')
    audit = audit_results(dest)
    assert audit['rate_checks'] == 2*9*29
    assert audit['independent_auc_checks'] == 2*9*24
    assert audit['audit_rows'] == 2*(36+30+4*(20+6*10))
    with pytest.raises(FileExistsError):
        robust.write_results(report, evidence, dest)


def test_saved_svm_mechanism_diagnostic_replays(tmp_path, small_run):
    report, evidence = small_run
    report = json.loads(json.dumps(report))
    dest = robust.write_results(report, evidence, tmp_path/'diagnose')
    script = Path(__file__).resolve().parents[1]/'scripts/diagnose_robustness.py'
    diagnose = runpy.run_path(str(script))['diagnose']
    result = diagnose(dest, seed=101)
    assert result['score_replay_max_absolute_error'] <= 1e-10
    assert result['manual_rbf_max_absolute_error'] <= 1e-10
    assert len(result['honest_physics_checks']) == 8
    with pytest.raises(ValueError, match='seed'):
        diagnose(dest, seed=404)


@pytest.mark.parametrize('corruption,message', [
    ('summary', 'summary'), ('control', 'unmatched'),
    ('interval', 'Wilson'), ('policy', 'metadata'), ('fingerprint', 'fingerprint'),
])
def test_auditor_rejects_corrupted_evidence(tmp_path, small_run, corruption, message):
    report, evidence = small_run
    report = json.loads(json.dumps(report))
    dest = robust.write_results(report, evidence, tmp_path/corruption)
    stats = report['runs'][0]['methods']['QBER calibrated']['evaluations']['higher noise::IR anchor']
    if corruption == 'summary':
        report['summary']['QBER calibrated']['higher noise::honest']['alarm_rate']['mean'] += 0.2
    elif corruption == 'control':
        stats['honest_control'] = 'nominal::honest'
    elif corruption == 'interval':
        stats['wilson95'][1] = 0.12345
    elif corruption == 'policy':
        report['runs'][0]['cohorts']['higher noise::IR anchor']['policy']['ir_rate'] = 0.2
    else:
        report['provenance']['source_sha256']['classical_ml/robustness_benchmark.py'] = '0'*64
    (dest/'robustness.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        audit_results(dest)
