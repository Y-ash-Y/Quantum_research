"""Classical nuisance-confidence-set comparator; no new test theory claimed.

Research audit only. Real-arithmetic validity, not outward-rounded arithmetic.
See docs/classical-audit-protocol.md. Does not change the Step-12 detector.
"""
from dataclasses import dataclass
import heapq
import math
from numbers import Integral

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import beta, binom, norm


@dataclass(frozen=True)
class TailDecision:
    reject: bool
    lower: float
    upper: float
    status: str
    splits: int


def _inputs(yc, nc, ym, nm, drift, level):
    for y, n in ((yc, nc), (ym, nm)):
        if (isinstance(y, bool) or isinstance(n, bool)
                or not isinstance(y, Integral) or not isinstance(n, Integral)
                or not 0 <= y <= n <= 10_000_000):
            raise ValueError('Invalid bounded integer binomial counts')
    if (isinstance(drift, bool) or isinstance(level, bool)
            or not isinstance(drift, (int, float)) or not isinstance(level, (int, float))
            or not math.isfinite(drift) or not 0 <= drift <= .5
            or not math.isfinite(level) or not 1e-10 <= level < 1):
        raise ValueError('Invalid drift or directional level')


def cp_interval(y, n, failure):
    """Two-sided binomial interval before intersecting the known domain."""
    return (0. if y == 0 else float(beta.ppf(failure/2, y, n-y+1)),
            1. if y == n else float(beta.ppf(1-failure/2, y+1, n-y)))


def _support(n, lower, upper, truncation):
    if n <= 100:
        return np.arange(n+1, dtype=np.int64), 0.
    lo = max(0, int(binom.ppf(truncation/4, n, lower)))
    hi = min(n, int(binom.ppf(1-truncation/4, n, upper)))
    omitted = float(binom.cdf(lo-1, n, lower)+binom.sf(hi, n, upper))
    if not math.isfinite(omitted) or omitted > truncation:
        # Conservative recovery; no outcome-dependent removal of rows.
        return np.arange(n+1, dtype=np.int64), 0.
    return np.arange(lo, hi+1, dtype=np.int64), omitted


def difference_tail(yc, nc, ym, nm, pc, pm, *, support=None):
    """Inclusive tail for fixed probabilities; returned sum omits only support.

    Integer products are bounded by input validation in the public caller.
    Supplying a truncated support gives a lower sum; add its omitted mass for
    an upper bound. No normal approximation or variance plug-in enters.
    """
    k = np.arange(nc+1, dtype=np.int64) if support is None else support
    numerator = nm*k + int(ym)*int(nc)-int(yc)*int(nm)
    threshold = (numerator+nc-1)//nc
    terms = binom.pmf(k, nc, pc)*binom.sf(threshold-1, nm, pm)
    value = float(math.fsum(terms))
    if not math.isfinite(value):
        raise ArithmeticError('Nonfinite binomial tail')
    return min(1., max(0., value))


def berger_boos_direction(yc, nc, ym, nm, drift, *, level=.05/8,
                          max_splits=64, gamma_fraction=.1):
    """Reject only after a continuum upper-envelope decision.

    The confidence interval is for calibration p only. All null monitor p
    are covered by monotonicity. Finite search-budget exhaustion is nonreject.
    Bounds describe BB p except for zero-count/screened conservative gates.
    """
    _inputs(yc, nc, ym, nm, drift, level)
    if (isinstance(max_splits, bool) or not isinstance(max_splits, int)
            or not 0 <= max_splits <= 4096
            or not 0 < gamma_fraction < 1):
        raise ValueError('Invalid fixed maximization controls')
    if min(nc, nm) == 0:
        return TailDecision(False, 0., 1., 'zero-denominator', 0)
    if ym/nm-yc/nc <= drift:
        return TailDecision(False, 0., 1., 'screened-nonreject', 0)
    gamma = level*gamma_fraction
    lower, upper = cp_interval(yc, nc, gamma)
    upper = min(.5, upper)
    if lower > upper:
        return TailDecision(True, gamma, gamma, 'empty-confidence-set', 0)
    support, omitted = _support(nc, lower, upper, 1e-12)
    cache = {}

    def tail(pc, pm):
        key = (pc, pm)
        if key not in cache:
            cache[key] = difference_tail(yc, nc, ym, nm, pc, pm, support=support)
        return cache[key]

    best = max(tail(p, min(.5, p+drift)) for p in (lower, (lower+upper)/2, upper))
    if best+gamma > level:
        return TailDecision(False, min(1., best+gamma), 1., 'proved-nonreject', 0)

    def bound(l, u):
        return min(1., tail(l, min(.5, u+drift))+omitted)

    heap = [(-bound(lower, upper), lower, upper)]
    splits = 0
    while True:
        p_upper = min(1., gamma-heap[0][0])
        if p_upper <= level:
            return TailDecision(True, min(1., best+gamma), p_upper, 'bounded-reject', splits)
        if splits >= max_splits:
            return TailDecision(False, min(1., best+gamma), p_upper, 'budget-unresolved', splits)
        _, l, u = heapq.heappop(heap)
        mid = (l+u)/2
        if not l < mid < u:
            return TailDecision(False, min(1., best+gamma), p_upper, 'precision-unresolved', splits)
        best = max(best, tail(mid, min(.5, mid+drift)))
        if best+gamma > level:
            return TailDecision(False, min(1., best+gamma), 1., 'proved-nonreject', splits)
        heapq.heappush(heap, (-bound(l, mid), l, mid))
        heapq.heappush(heap, (-bound(mid, u), mid, u))
        splits += 1


def restricted_score_pvalue(yc, nc, ym, nm, drift):
    """FM-type constrained score with normal tail: APPROXIMATE diagnostic only.

    No finite-sample uniform type-I claim. Uses the known [0,.5] domain.
    Constrained log likelihood on pm=pc+d is concave, so its derivative root
    or an endpoint maximizes it. This is not a Chan exact-score implementation.
    """
    _inputs(yc, nc, ym, nm, drift, .05/8)
    if min(nc, nm) == 0 or ym/nm-yc/nc <= drift:
        return 1.
    right = .5-drift
    if right == 0:
        p = 0.
    elif drift == 0:
        p = min(.5, (yc+ym)/(nc+nm))
    else:
        tiny = min(1e-12, right/4)
        def slope(p):
            q = p+drift
            return yc/p-(nc-yc)/(1-p)+ym/q-(nm-ym)/(1-q)
        if slope(tiny) <= 0:
            p = 0.
        elif slope(right) >= 0:
            p = right
        else:
            p = brentq(slope, tiny, right, xtol=1e-14)
    q = p+drift
    variance = p*(1-p)/nc+q*(1-q)/nm
    excess = ym/nm-yc/nc-drift
    return 0. if variance == 0 else float(norm.sf(excess/math.sqrt(variance)))
