import pytest
import json
import numpy as np
import importlib.util
from pathlib import Path

from research.best_response_stability import (
    FLOORS, expanded_grid, eligible_ids, make_shortlist, select_policy,
    grid_envelope, exact_interval,
)


def test_expanded_grid_contains_old_candidates():
    grid = expanded_grid()
    assert len(grid) == 153
    for k in FLOORS:
        ids = eligible_ids(grid, k)
        assert len(ids) == 45 and all(grid[i].rate/2 >= k for i in ids)
        observed = {(grid[i].rate, grid[i].z_basis_probability) for i in ids}
        for m in (1, 1.5, 2):
            for beta in (0, .25, .5, .75, 1):
                assert (round(2*k*m, 12), beta) in observed


@pytest.mark.parametrize('floor', FLOORS)
def test_ties_probes_and_shortlist(floor):
    grid = expanded_grid()
    counts = {i: 0 for i in range(len(grid))}
    ids = eligible_ids(grid, floor)
    old = next(i for i in ids if grid[i].rate == round(2*floor, 12) and grid[i].z_basis_probability == 0)
    result = make_shortlist(counts, counts, grid, floor, old)
    selected = grid[result['winner_pooled']]
    assert selected.rate == round(2*floor, 12) and selected.z_basis_probability == .5
    assert result['independent_searches_agree']
    assert old in result['shortlist'] and len(result['shortlist']) <= 9
    assert all(i in ids for i in result['shortlist'])
    assert {grid[i].z_basis_probability for i in result['shortlist']} >= {0, .5, 1}


def test_independent_winners_and_pooled_selection():
    grid = expanded_grid(); ids = eligible_ids(grid, FLOORS[0])
    a = {i: 100 for i in range(len(grid))}; b = a.copy()
    a[ids[0]], b[ids[1]] = 0, 0
    a[ids[2]], b[ids[2]] = 1, 1
    r = make_shortlist(a, b, grid, FLOORS[0], ids[3])
    assert r['winner_a'] == ids[0] and r['winner_b'] == ids[1]
    assert r['winner_pooled'] == ids[2] and not r['independent_searches_agree']
    assert {ids[0], ids[1], ids[2], ids[3]}.issubset(r['shortlist'])


def test_grid_envelope_and_regret():
    pooled = {0: [.1, .3], 1: [.2, .4], 2: [.15, .5]}
    evaluated = {1: [.22, .28], 2: [.16, .19]}
    r = grid_envelope(pooled, evaluated, [0, 1, 2], 1)
    assert r['grid_minimum_joint95'] == [.1, .19]
    assert r['selected_regret_ceiling'] == pytest.approx(.18)
    assert not r['selected_within_tolerance_certified_on_joint_event']
    # A selected upper bound is not the minimum observed shortlist bound.
    assert grid_envelope(pooled, {1: [.11, .14]}, [0, 1, 2], 1)[
        'selected_within_tolerance_certified_on_joint_event']


def test_inconsistent_bracket_is_not_hidden():
    r = grid_envelope({0: [.7, .9]}, {0: [.1, .2]}, [0], 0)
    assert r['grid_minimum_joint95'] == [.7, .2]
    assert not r['intervals_consistent'] and r['selected_regret_ceiling'] is None
    assert not r['selected_within_tolerance_certified_on_joint_event']


@pytest.mark.parametrize('args', [({}, {}, [], 0), ({0: [0, 1]}, {1: [0, 1]}, [0], 1),
                                   ({0: [-1, 1]}, {0: [0, 1]}, [0], 0)])
def test_invalid_envelope_inputs(args):
    with pytest.raises(ValueError):
        grid_envelope(*args)


@pytest.mark.parametrize('args', [(True, 1, .05), (2, 1, .05), (0, 0, .05), (0, 2, 0)])
def test_invalid_binomial_inputs(args):
    with pytest.raises(ValueError):
        exact_interval(*args)


def test_interval_empty_successes_and_invalid_search():
    assert exact_interval(0, 100, .05)[0] == 0
    assert exact_interval(100, 100, .05)[1] == 1
    with pytest.raises(ValueError):
        select_policy({}, expanded_grid(), [0])
    with pytest.raises(ValueError):
        eligible_ids(expanded_grid(), .123)


def test_numpy_interval_inputs_produce_json_safe_summary():
    r = grid_envelope({0: np.array([.1, .4])}, {0: np.array([.11, .14])}, [0], 0)
    assert type(r['intervals_consistent']) is bool
    assert type(r['selected_within_tolerance_certified_on_joint_event']) is bool
    assert type(r['selected_regret_ceiling']) is float
    assert json.loads(json.dumps(r, allow_nan=False)) == r
    assert all(type(v) is float for v in exact_interval(1, 10, .05))


def test_joint_event_inequalities_by_enumeration():
    pooled = {0: [.1, .4], 1: [.2, .5], 2: [.3, .6]}
    evaluation = {0: [.15, .3], 1: [.25, .35]}
    envelope = grid_envelope(pooled, evaluation, [0, 1, 2], 1)
    # Enumerate truth vectors compatible with both sets of intervals. The
    # lower bound uses all grid policies, not just the selected shortlist.
    for p0 in (.15, .2, .3):
        for p1 in (.25, .3, .35):
            for p2 in (.3, .45, .6):
                truth = min(p0, p1, p2)
                lo, hi = envelope['grid_minimum_joint95']
                assert lo <= truth <= hi
                assert p1-truth <= envelope['selected_regret_ceiling']+1e-14


def test_runner_freezes_before_evaluation_and_counts_shortlists(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('stability_runner_test', scripts/'run_best_response_stability.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    profiles, grid = runner.plans(), expanded_grid()
    search = {phase: [{'plan_id': p['plan_id'], 'policy_id': i, 'alarms': 0}
                       for p in profiles for i in range(len(grid))] for phase in ('search_a', 'search_b')}
    previous = {(p['plan_id'], fid): next(i for i in eligible_ids(grid, k)
                if grid[i].rate == round(2*k, 12) and grid[i].z_basis_probability == .5)
                for p in profiles for fid, k in enumerate(FLOORS)}
    frozen = runner.freeze_shortlists(search, profiles, grid, previous)
    assert len(frozen['responses']) == 56 and len(frozen['defender_plans']) == 8
    assert frozen['evaluation_cells'] == sum(len(r['shortlist']) for r in frozen['responses'])
    assert all(profiles[r['plan_id']]['plan'].calibration_emitted == 1333333 for r in frozen['defender_plans'])
    assert json.loads(json.dumps(frozen)) == frozen
