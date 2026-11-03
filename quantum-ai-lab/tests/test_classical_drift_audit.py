"""Independent finite-table references; grids do not replace uniform proofs."""
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from research.classical_drift_audit import (
    berger_boos_direction, cp_interval, difference_tail, restricted_score_pvalue,
)


def brute(yc, nc, ym, nm, pc, pm):
    total = 0.
    for a in range(nc+1):
        for b in range(nm+1):
            if b*nc-a*nm >= ym*nc-yc*nm:
                total += (math.comb(nc,a)*pc**a*(1-pc)**(nc-a)
                          *math.comb(nm,b)*pm**b*(1-pm)**(nm-b))
    return total


@pytest.mark.parametrize('nc,nm', [(1,3), (4,3), (7,9)])
def test_full_support_tail_independent_enumeration(nc, nm):
    for yc in range(nc+1):
        for ym in range(nm+1):
            for pc, pm in [(0.,.2), (.02,.07), (.25,.3), (.5,.5)]:
                assert difference_tail(yc,nc,ym,nm,pc,pm) == pytest.approx(
                    brute(yc,nc,ym,nm,pc,pm), abs=2e-14)


@pytest.mark.parametrize('y,n', [(0,12),(3,12),(12,12)])
def test_cp_against_separate_scipy_inversion(y,n):
    expected = binomtest(y,n).proportion_ci(.997)
    assert cp_interval(y,n,.003) == pytest.approx((expected.low,expected.high), abs=1e-10)


def test_cell_envelope_by_independent_tables():
    for d in (0.,.04,.3):
        for l,u in ((0.,.1),(.1,.3),(.4,.5)):
            for yc,ym in ((0,4),(1,2),(3,4)):
                upper = difference_tail(yc,4,ym,5,l,min(.5,u+d))
                for pc in np.linspace(l,u,9):
                    assert brute(yc,4,ym,5,pc,min(.5,pc+d)) <= upper+1e-14


@pytest.mark.parametrize('nc,nm,d', [(3,5,0.),(5,7,.1),(12,9,.2)])
def test_actual_small_table_direction_size(nc,nm,d):
    level = .1
    flags = np.array([[berger_boos_direction(a,nc,b,nm,d,level=level).reject
                       for b in range(nm+1)] for a in range(nc+1)])
    for pc in (0.,.01,.1,.3,.5):
        for pm in (0.,.01,.1,.3,.5):
            if pm-pc <= d+1e-15:
                weights = binom.pmf(np.arange(nc+1),nc,pc)[:,None]*binom.pmf(np.arange(nm+1),nm,pm)
                assert weights[flags].sum() <= level+1e-13


def test_cp_coverage_on_finite_grid():
    gamma = .000625
    for n in (3,11,50):
        for p in (0.,.001,.015,.25,.5):
            intervals = [cp_interval(y,n,gamma) for y in range(n+1)]
            uncovered = np.array([not l <= p <= u for l,u in intervals])
            assert binom.pmf(np.arange(n+1),n,p)[uncovered].sum() <= gamma+1e-13


def test_no_search_budget_fails_closed_and_more_budget_cannot_lose_rejections():
    for yc,ym in ((125,265),(130,250),(110,300)):
        a = berger_boos_direction(yc,8000,ym,12000,0.,max_splits=0)
        b = berger_boos_direction(yc,8000,ym,12000,0.,max_splits=64)
        assert not a.reject or b.reject
        if a.status == 'budget-unresolved':
            assert not a.reject


def test_zero_empty_and_screened_cases():
    assert not berger_boos_direction(0,0,0,8,0.).reject
    assert not berger_boos_direction(3,10,2,10,0.).reject
    empty = berger_boos_direction(900,1000,1000,1000,0.)
    assert empty.status == 'empty-confidence-set' and empty.reject


def test_score_zero_margin_matches_pooled_z():
    yc,nc,ym,nm = 15,1000,35,1200
    p = (yc+ym)/(nc+nm)
    from scipy.stats import norm
    expected = norm.sf((ym/nm-yc/nc)/math.sqrt(p*(1-p)*(1/nc+1/nm)))
    assert restricted_score_pvalue(yc,nc,ym,nm,0.) == pytest.approx(expected)


@pytest.mark.parametrize('args', [(True,1,1,1,0.),(2,1,0,1,0.),(0,-1,0,1,0.),
                                 (0,1,0,1,-.1),(0,1,0,1,float('nan'))])
def test_invalid_counts(args):
    with pytest.raises(ValueError):
        berger_boos_direction(*args)
