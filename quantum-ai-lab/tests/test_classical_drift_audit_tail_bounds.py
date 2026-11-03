"""Large-count truncation and near-cutoff checks using reversed tail sums."""
import csv
from functools import lru_cache
import gzip
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from scipy.stats import binom, binomtest, norm

from research.classical_drift_audit import (
    _support, berger_boos_direction, difference_tail, restricted_score_pvalue,
)


def reversed_tail(yc,nc,ym,nm,pc,pm):
    b=np.arange(nm+1,dtype=np.int64)
    max_a=(b*nc-(ym*nc-yc*nm))//nm
    return float(math.fsum(binom.pmf(b,nm,pm)*binom.cdf(max_a,nc,pc)))


@pytest.mark.parametrize('n,lower,upper',[(200,0.,.03),(8000,.01,.02),(1000,.45,.5)])
def test_omission_and_cell_upper_against_full_reversed_sum(n,lower,upper):
    k,omitted=_support(n,lower,upper,1e-12)
    assert len(k)<n+1 and omitted<=1e-12
    yc,ym,nm=int(.015*n),int(.025*int(1.5*n)),int(1.5*n)
    d=.002425
    envelope=difference_tail(yc,n,ym,nm,lower,min(.5,upper+d),support=k)+omitted
    for pc in np.linspace(lower,upper,7):
        mass=binom.cdf(int(k[0])-1,n,pc)+binom.sf(int(k[-1]),n,pc)
        assert mass<=omitted+1e-15
        actual=reversed_tail(yc,n,ym,nm,pc,min(.5,pc+d))
        assert actual<=envelope+2e-13


@lru_cache(None)
def archived_counts(allowance,trial,basis):
    archive=Path(__file__).resolve().parents[1]/'results/benchmarks/direct-drift-pilot-v1/public_counts.csv.gz'
    with gzip.open(archive,'rt',newline='') as stream:
        for row in csv.DictReader(stream):
            if tuple(int(row[k]) for k in ('plan_id','floor_id','allowance_id','cohort_id','trial')) == (10,1,allowance,7,trial):
                result=[]
                for stage in ('cal','mon'):
                    for field in ('numerator','denominator'):
                        result.append(sum(int(row[f'{stage}_{basis}{bit}_error_{field}']) for bit in (0,1)))
                return tuple(result)
    raise AssertionError('Expected saved fixture missing')


@pytest.mark.parametrize('allowance,trial,basis,reject,status',[
    (1,1,'X',True,'bounded-reject'),(0,85,'Z',False,'budget-unresolved')])
def test_near_cutoff_saved_rows_with_full_support_checks(allowance,trial,basis,reject,status):
    yc,nc,ym,nm=archived_counts(allowance,trial,basis)
    d=allowance*.002425
    result=berger_boos_direction(yc,nc,ym,nm,d)
    assert result.reject==reject and result.status==status
    interval=binomtest(yc,nc).proportion_ci(1-(.05/8)/10)
    for pc in np.linspace(interval.low,min(.5,interval.high),17):
        full=reversed_tail(yc,nc,ym,nm,pc,min(.5,pc+d))+(.05/8)/10
        assert full<=result.upper+2e-13
    if reject:
        assert result.upper<=.05/8
    else:
        assert result.lower<.05/8<result.upper


@pytest.mark.parametrize('yc,nc,ym,nm,d',[(0,400,10,500,.002425),
                                        (61,4000,112,6000,.002425),(12,50,21,60,.03)])
def test_score_mle_against_independent_bounded_optimization(yc,nc,ym,nm,d):
    def negative_log_likelihood(p):
        q=p+d
        return -(xlogy(yc,p)+xlogy(nc-yc,1-p)+xlogy(ym,q)+xlogy(nm-ym,1-q))
    fitted=minimize_scalar(negative_log_likelihood,bounds=(0,.5-d),method='bounded',
                           options={'xatol':1e-14})
    p=min((0.,float(fitted.x),.5-d),key=negative_log_likelihood)
    v=p*(1-p)/nc+(p+d)*(1-p-d)/nm
    expected=norm.sf((ym/nm-yc/nc-d)/math.sqrt(v))
    assert restricted_score_pvalue(yc,nc,ym,nm,d)==pytest.approx(expected,abs=2e-8)
