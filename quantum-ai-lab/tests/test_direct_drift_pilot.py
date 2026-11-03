"""Frozen Step 12 comparison checks before full evaluation."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('direct_drift_runner_test', scripts/'run_direct_drift_pilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_dimensions_and_budgets(runner):
    p = runner.frozen_protocol()
    assert p['seed'] == 2026100912 and p['seed'] != 2026100911
    assert p['cohorts'] == 216 and p['rate_cells'] == 216*3 and p['effect_cells'] == 216
    assert p['rate_failure_budget']+p['effect_failure_budget'] == .05
    assert p['trials'] == 512


def test_independent_paired_count_bookkeeping(runner):
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, False])
    e = runner.paired_effect(a, b, .01/216)
    assert [e[k] for k in ('both_alarm','wins_direct','losses_direct','neither_alarm')] == [1,2,1,1]
    assert e['net_alarm_rate_difference'] == 1/5
    assert e['ci_joint95'][0] <= .2 <= e['ci_joint95'][1]
    reverse = runner.paired_effect(b, a, .01/216)
    assert reverse['net_alarm_rate_difference'] == -.2
    assert reverse['ci_joint95'] == pytest.approx([-e['ci_joint95'][1],-e['ci_joint95'][0]])


def test_tiny_comparison_complete_audit_and_metadata_failure(runner, tmp_path):
    from audit_direct_drift_pilot import audit
    out = tmp_path/'tiny'
    result = runner.run(out, 3)
    checked = audit(out)
    assert checked['status'] == 'passed' and checked['public_pairs_reconciled'] == 648
    assert checked['decision_replays'] == 1944 and checked['effect_cells'] == 216
    original = (out/'pilot.json').read_bytes()
    with pytest.raises(FileExistsError):
        runner.run(out, 3)
    assert (out/'pilot.json').read_bytes() == original
    result['effects'][0]['wins_direct'] += 1
    (out/'pilot.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Paired effect counts'):
        audit(out)
    result['effects'][0]['wins_direct'] -= 1
    result['source_sha256'].pop('src/research/direct_drift_detector.py')
    (out/'pilot.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='source fingerprints'):
        audit(out)


@pytest.mark.parametrize('trials',[True, 0, -1, 1025, 1.5])
def test_bad_trial_count(runner,trials):
    with pytest.raises(ValueError):
        runner.frozen_protocol(trials)
