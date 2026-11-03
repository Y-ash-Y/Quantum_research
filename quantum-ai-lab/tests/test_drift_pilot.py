"""Frozen pilot invariants, lawful scenarios and complete tiny-run audit."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('drift_pilot_runner_test', scripts/'run_drift_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_counts_budgets_and_expected_resources(runner):
    p = runner.frozen_protocol()
    assert p['cohorts'] == 2*3*4*9 and p['rate_cells'] == 2*p['cohorts']
    assert p['mean_cells'] == 2*3*4 and p['trials'] == 512
    assert p['rate_failure_budget']+p['mean_failure_budget'] == .05
    resources = []
    for _, plan in runner.plans():
        assert plan.effective_test_emissions == pytest.approx(2000000)
        resources.append(plan.expected_resources())
    assert resources[0]['total_sacrificed'] == pytest.approx(resources[1]['total_sacrificed'])
    assert resources[0]['remaining_candidates'] == pytest.approx(resources[1]['remaining_candidates'])


@pytest.mark.parametrize('floor,multiplier', [(f, a) for f in (.005, .01, .02) for a in (0, .5, 1, 1.25)])
def test_scenarios_edges_are_in_envelope_and_attacks_meet_floor(runner, floor, multiplier):
    env, cases = runner.scenarios(floor, multiplier)
    assert len(cases) == 9 and len({c['name'] for c in cases}) == 9
    assert env.z_error == pytest.approx(.97*floor*multiplier/2)
    for c in cases:
        assert c['calibration'].eta == c['monitoring'].eta
        if c['category'] == 'allowed_honest':
            for a, b, d in zip(c['calibration'].bit_errors, c['monitoring'].bit_errors, env.coordinate_allowances):
                assert abs(a-b) <= d
        else:
            assert c['policy']['rate']/2 >= floor


def test_seed_pair_replay_and_independent_stage_streams(runner):
    _, cases = runner.scenarios(.01, .5)
    _, plan = runner.plans()[0]
    first = runner.draw(5, plan, cases[0], (9, 1, 1, 0))
    second = runner.draw(5, plan, cases[0], (9, 1, 1, 0))
    other = runner.draw(5, plan, cases[0], (9, 1, 1, 1))
    assert np.array_equal(first[0].numerators, second[0].numerators)
    assert np.array_equal(first[1].numerators, second[1].numerators)
    assert not np.array_equal(first[0].numerators, other[0].numerators)


def test_tiny_run_full_audit_and_refusal_to_overwrite(runner, tmp_path):
    from audit_drift_pilot import audit
    out = tmp_path/'tiny'
    result = runner.run(out, 3)
    assert result['public_pairs'] == 216*3
    checked = audit(out)
    assert checked['status'] == 'passed'
    assert checked['decision_replays'] == 432*3
    assert checked['independent_scalar_lower_rows'] == 24*3
    fingerprint = (out/'pilot.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(out, 3)
    assert (out/'pilot.json').read_bytes() == fingerprint
    # Bad metadata must fail even without modifying the public archive.
    result['cells'][0]['alarms'] += 1
    (out/'pilot.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Rate count'):
        audit(out)


@pytest.mark.parametrize('trials', [True, 0, -1, 1025, 1.5])
def test_trial_domain_rejected_before_output(runner, trials):
    with pytest.raises(ValueError):
        runner.frozen_protocol(trials)
