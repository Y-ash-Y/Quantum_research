"""Independent audit numerics: integer Fisher and bounded binomial-tail sums.

Never imported by a production scorer. The omitted tail bound controls series
truncation, not all floating-point rounding. Cross-checks cover large counts.
"""
from functools import lru_cache
import math
from numbers import Integral, Real


def _parameters(n, p, index):
    if (not isinstance(n, Integral) or isinstance(n, bool) or n < 0
            or not isinstance(index, Integral) or isinstance(index, bool)
            or not isinstance(p, Real) or isinstance(p, bool)
            or not math.isfinite(p) or not 0 <= p <= 1):
        raise ValueError('Invalid binomial parameters')


@lru_cache(maxsize=65536, typed=True)
def recurrence_fisher(nz, nx, ez, ex):
    if (any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in (nz, nx, ez, ex))
            or ez > nz or ex > nx):
        raise ValueError('Invalid Fisher table')
    total, errors = nz+nx, ez+ex
    low, high = max(0, errors-nx), min(nz, errors)
    observed = math.comb(nz, ez)*math.comb(nx, ex)
    weight = math.comb(nz, low)*math.comb(nx, errors-low)
    numerator = 0
    for k in range(low, high+1):
        if weight <= observed:
            numerator += weight
        if k < high:
            # Each result is an integer hypergeometric weight; no rounding of
            # the probability ordering or approximate equality/tie handling.
            weight = weight*(nz-k)*(errors-k)//((k+1)*(nx-errors+k+1))
    return numerator/math.comb(total, errors)


def _decreasing_tail(n, k, p, direction):
    """Include k and continue away from the mode until bounded remainder is tiny."""
    term = math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                    +k*math.log(p)+(n-k)*math.log1p(-p))
    terms = [term]
    while (k > 0 if direction < 0 else k < n):
        ratio = k/(n-k+1)*(1-p)/p if direction < 0 else (n-k)/(k+1)*p/(1-p)
        if not 0 <= ratio < 1:
            raise ValueError('Tail must start on decreasing side of the mode')
        following = term*ratio
        # Remaining ratios only decrease, so a geometric series bounds all
        # omitted terms. Absolute truncation error is at most 1e-14.
        if following/(1-ratio) <= 1e-14:
            break
        terms.append(following)
        term, k = following, k+direction
    return min(1.0, math.fsum(terms))


def recurrence_cdf(n, observed, p):
    """P(Binomial(n,p)<=observed), including all probability/count endpoints."""
    _parameters(n, p, observed)
    if observed < 0:
        return 0.0
    if observed >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    if observed < n*p:
        return _decreasing_tail(n, observed, p, -1)
    return max(0.0, 1-_decreasing_tail(n, observed+1, p, 1))


def recurrence_survival(n, minimum, p):
    """P(Binomial(n,p)>=minimum); avoid cancellation in small upper tails."""
    _parameters(n, p, minimum)
    if minimum <= 0:
        return 1.0
    if minimum > n or p == 0:
        return 0.0
    if p == 1:
        return 1.0
    if minimum > n*p:
        return _decreasing_tail(n, minimum, p, 1)
    return max(0.0, 1-_decreasing_tail(n, minimum-1, p, -1))
