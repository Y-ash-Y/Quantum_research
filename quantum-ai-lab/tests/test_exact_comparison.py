"""Independent finite-combinatorial checks of the central conditional test."""
from fractions import Fraction
from math import comb
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import fisher_exact

from research.calibration_protocol import PublicBatch, gate_law, sample_public_batch
from research.exact_comparison import central_exact_pvalues, exact_change_alarm, MAX_TWO_SAMPLE_TOTAL


def batch(successes, denominator):
    # One error coordinate; gains account for all matched observations.
    return PublicBatch(denominator, np.array([[successes, 0, 0, 0, denominator, 0]]),
                       np.array([[denominator, 0, 0, 0, denominator, 0]]),
                       np.array([[denominator, 0, 0, 0]]))


def rational_tail(y, nc, nm, total):
    lower, upper = max(0, total-nm), min(nc, total)
    mass = {k: Fraction(comb(nc, k)*comb(nm, total-k), comb(nc+nm, total))
            for k in range(lower, upper+1)}
    left = sum((p for k, p in mass.items() if k <= y), Fraction())
    right = sum((p for k, p in mass.items() if k >= y), Fraction())
    return min(Fraction(1), 2*min(left, right)), mass


@pytest.mark.parametrize('nc', range(1, 8))
def test_exhaustive_small_tables_and_conditional_size(nc):
    for nm in range(1, 8):
        for total in range(nc+nm+1):
            rejected_mass = Fraction()
            for yc in range(max(0, total-nm), min(nc, total)+1):
                expected, mass = rational_tail(yc, nc, nm, total)
                observed = central_exact_pvalues(batch(yc, nc), batch(total-yc, nm))[0, 0]
                assert observed == pytest.approx(float(expected), abs=2e-14)
                if expected <= Fraction(1, 120):
                    rejected_mass += mass[yc]
            assert rejected_mass <= Fraction(1, 120)  # alpha/6 at alpha=.05


@pytest.mark.parametrize('table', [(1, 9, 11, 3), (0, 20, 10, 10), (3, 7, 7, 3), (0, 1, 3, 0)])
def test_one_sided_scipy_agreement_and_symmetries(table):
    a, b, c, d = table
    expected = min(1, 2*min(fisher_exact([[a, b], [c, d]], alternative='less').pvalue,
                          fisher_exact([[a, b], [c, d]], alternative='greater').pvalue))
    p = central_exact_pvalues(batch(a, a+b), batch(c, c+d))[0, 0]
    assert p == pytest.approx(expected)
    assert central_exact_pvalues(batch(c, c+d), batch(a, a+b))[0, 0] == pytest.approx(p)
    assert central_exact_pvalues(batch(b, a+b), batch(d, c+d))[0, 0] == pytest.approx(p)


@pytest.mark.parametrize('yc,nc,ym,nm', [(0, 0, 10, 20), (0, 0, 0, 0), (0, 10, 0, 20), (10, 10, 20, 20)])
def test_uninformative_endpoints(yc, nc, ym, nm):
    c, m = batch(yc, nc), batch(ym, nm)
    assert np.all(central_exact_pvalues(c, m) == 1)
    assert not exact_change_alarm(c, m)[0]


@pytest.mark.parametrize('alpha', [0, 1, -1, float('nan'), 1e-15])
def test_reject_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        exact_change_alarm(batch(1, 20), batch(2, 20), alpha)


def test_public_api_shape_and_domain():
    c = batch(1, 20)
    with pytest.raises(ValueError):
        central_exact_pvalues(gate_law(), c)
    m = sample_public_batch(2, 20, gate_law(), .25, np.random.default_rng(1))
    with pytest.raises(ValueError):
        central_exact_pvalues(c, m)
    with pytest.raises(ValueError):
        central_exact_pvalues(batch(0, MAX_TWO_SAMPLE_TOTAL), batch(0, 1))


def test_includes_observed_atom_and_keeps_inputs_immutable():
    c, m = batch(0, 1), batch(1, 1)
    before = c.numerators.copy()
    p = central_exact_pvalues(c, m)
    assert p[0, 0] == 1  # sf(0), rather than sf(-1), would falsely yield zero.
    assert not p.flags.writeable
    assert np.array_equal(c.numerators, before)


def test_clear_change_and_fresh_stable_control():
    assert exact_change_alarm(batch(0, 100), batch(100, 100))[0]
    rng = np.random.default_rng(2026100504)
    c = sample_public_batch(500, 100000, gate_law(), 1, rng)
    m = sample_public_batch(500, 100000, gate_law(), .25, rng)
    assert exact_change_alarm(c, m).mean() < .09  # broad sanity check, not a proof


def runner_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('exact_runner_test', scripts/'run_exact_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_rule_selection_and_ties(monkeypatch):
    runner = runner_module(monkeypatch)
    profiles, grid = runner.plans(), runner.policies()
    search = [{'plan_id': p['plan_id'], 'policy_id': i,
               'rules': {r: {'alarm_rate': 0.} for r in runner.RULES}}
              for p in profiles for i in range(len(grid))]
    frozen = runner.freeze_choices(search, profiles, grid)
    assert frozen['cp'] == frozen['central_exact']
    for rule in runner.RULES:
        assert len(frozen[rule]['eve_responses']) == 56
        assert len(frozen[rule]['defender_plans']) == 8
        for r in frozen[rule]['eve_responses']:
            assert grid[r['policy_id']].rate == pytest.approx(2*r['minimum_certified_fraction'])
            assert grid[r['policy_id']].z_basis_probability == .5
    # Alter only the new rule's search score; CP's selections must not move.
    before = frozen['cp']
    biased = next(i for i, p in enumerate(grid) if p.rate == .005 and p.z_basis_probability == 0.)
    for c in search:
        c['rules']['central_exact']['alarm_rate'] = 0. if c['policy_id'] == biased else .5
    changed = runner.freeze_choices(search, profiles, grid)
    assert changed['cp'] == before
    assert changed['central_exact']['eve_responses'][0]['policy_id'] == biased


def test_paired_difference_and_multiplicity(monkeypatch):
    runner = runner_module(monkeypatch)
    s = {'cp': np.array([True, False, False, True]),
         'central_exact': np.array([False, True, True, True])}
    paired = runner.paired_summary(s)
    assert (paired['cp_only'], paired['exact_only'], paired['both'], paired['neither']) == (1, 2, 1, 0)
    assert paired['difference_exact_minus_cp'] == .25
    cell = {'trials': 4, 'paired': paired,
            'rules': {'cp': {'alarms': 2}, 'central_exact': {'alarms': 3}}}
    runner.add_evaluation_uncertainty(cell, 112)
    assert paired['mcnemar_bonferroni_p112'] == 1
    lo, hi = paired['difference_ci95_conservative']
    assert lo <= .25 <= hi
    assert paired['difference_simultaneous_ci95_conservative'][0] <= lo
    assert paired['difference_simultaneous_ci95_conservative'][1] >= hi
    same = runner.paired_summary({'cp': s['cp'], 'central_exact': s['cp']})
    assert same['mcnemar_exact_p'] == 1 and same['difference_exact_minus_cp'] == 0
