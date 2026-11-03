"""Detector-specific conditional integration and continuous IR lower functions.

Power analysis receives simulation probabilities; the deployed alarm remains
count-only and unchanged. Real-arithmetic identities, not certified rounding.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import binom, hypergeom

from research.exact_comparison import MAX_TWO_SAMPLE_TOTAL
from research.identifiability import _unit


def _denominators(values):
    a = np.asarray(values)
    if (a.ndim != 2 or a.shape[1] != 4 or not len(a) or a.dtype.kind not in 'iu'
            or np.any(a < 0) or np.any(a > MAX_TWO_SAMPLE_TOTAL)):
        raise ValueError('Expected nonnegative integer denominator rows of four strata')
    return a.astype(np.int64, copy=True)


def _alpha(value):
    value = _unit(value, 'alpha')
    if not 1e-12 <= value < 1:
        raise ValueError('Unsupported alpha')
    return value


@dataclass(frozen=True)
class ConditionalKernel:
    n_cal: np.ndarray
    n_mon: np.ndarray
    calibration_values: np.ndarray
    weights: np.ndarray
    high_cutoff: np.ndarray
    low_cutoff: np.ndarray
    missing_mass: np.ndarray
    q0: float
    alpha: float

    @property
    def rows(self):
        return len(self.n_cal)//4


def _boundaries(nc, nm, a, tail):
    """Find first high-tail rejection and last low-tail rejection in B.

    For fixed a, Hypergeom(N=m+n,K=a+b,draws=m) is stochastically increasing
    in b: couple success subsets of increasing size. Its CDF at a decreases,
    and its inclusive SF at a increases. Both searches include equality.
    """
    shape = np.broadcast_shapes(nc.shape, a.shape)
    m, n, y = (np.broadcast_to(x, shape) for x in (nc, nm, a))
    active = (m > 0) & (n > 0) & (y <= m)
    high = np.broadcast_to(n+1, shape).copy()
    low = np.full(shape, -1, dtype=np.int64)
    for upper_side in (True, False):
        left, right = np.full(shape, -1, dtype=np.int64), n.copy()+1
        while True:
            todo = active & (right-left > 1)
            if not np.any(todo):
                break
            mid = (left[todo]+right[todo])//2
            total = y[todo]+mid
            if upper_side:
                reject = hypergeom.cdf(y[todo], m[todo]+n[todo], total, m[todo]) <= tail
                right[todo] = np.where(reject, mid, right[todo])
                left[todo] = np.where(reject, left[todo], mid)
            else:
                reject = hypergeom.sf(y[todo]-1, m[todo]+n[todo], total, m[todo]) <= tail
                left[todo] = np.where(reject, mid, left[todo])
                right[todo] = np.where(reject, right[todo], mid)
        if upper_side:
            high[active] = right[active]
        else:
            low[active] = left[active]
    if np.any(low >= high):
        raise ArithmeticError('Rejection tails must be disjoint')
    return high, low


def build_kernel(n_cal, n_mon, q0=.015, alpha=.05, truncation=1e-12):
    nc, nm = _denominators(n_cal), _denominators(n_mon)
    if nc.shape != nm.shape or np.any(nc+nm > MAX_TWO_SAMPLE_TOTAL):
        raise ValueError('Incompatible or unsupported stage denominators')
    q0, alpha = _unit(q0, 'q0'), _alpha(alpha)
    if not np.isfinite(truncation) or not 1e-15 <= truncation < 1e-3:
        raise ValueError('Unsupported calibration truncation')
    nc, nm = nc.ravel(), nm.ravel()
    start = int(np.min(binom.ppf(truncation/2, nc, q0)))
    stop = int(np.max(binom.ppf(1-truncation/2, nc, q0)))
    if len(nc)*(stop-start+1) > 4000000:
        raise ValueError('Conditional support exceeds the bounded research memory domain')
    values = np.arange(start, stop+1, dtype=np.int64)
    weights = binom.pmf(values[None, :], nc[:, None], q0)
    missing = binom.cdf(start-1, nc, q0)+binom.sf(stop, nc, q0)
    if np.any(missing > truncation*1.01+1e-16):
        raise ArithmeticError('Calibration support does not meet truncation budget')
    high, low = _boundaries(nc[:, None], nm[:, None], values[None, :], alpha/12)
    arrays = (nc, nm, values, weights, high, low, missing)
    for a in arrays:
        a.flags.writeable = False
    return ConditionalKernel(*arrays, q0, alpha)


def coordinate_components(kernel, probabilities):
    if not isinstance(kernel, ConditionalKernel):
        raise ValueError('Expected a conditional kernel')
    q = np.asarray(probabilities, dtype=float)
    if q.shape == (4,):
        q = np.tile(q, kernel.rows)
    if q.shape != kernel.n_mon.shape or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError('Expected four monitoring probabilities or one per conditional stratum')
    high = np.sum(kernel.weights*binom.sf(kernel.high_cutoff-1, kernel.n_mon[:, None],
                                       q[:, None]), axis=1)
    low = np.sum(kernel.weights*binom.cdf(kernel.low_cutoff, kernel.n_mon[:, None],
                                       q[:, None]), axis=1)
    if not np.all(np.isfinite(high+low)) or np.any(high+low > 1+1e-11):
        raise ArithmeticError('Invalid conditional power')
    return np.clip(high, 0, 1).reshape(-1, 4), np.clip(low, 0, 1).reshape(-1, 4)


def _gain_mask(gain_alarm, rows):
    a = np.asarray(gain_alarm)
    if a.dtype.kind != 'b' or a.shape != (rows,):
        raise ValueError('Expected a boolean gain-alarm mask')
    return a


def _combine(probabilities, gain):
    return np.where(gain, 1., 1-np.prod(1-probabilities, axis=1))


def conditional_point(kernel, probabilities, gain_alarm):
    """Point power bracket conditional on counts, with omitted calibration mass.

    Independence holds for errors AFTER conditioning on all ancillary counts;
    no independence of the multinomial random denominators is assumed.
    """
    gain = _gain_mask(gain_alarm, kernel.rows)
    high, low = coordinate_components(kernel, probabilities)
    p = np.minimum(1., high+low)
    lower = _combine(p, gain)
    upper = _combine(np.minimum(1., p+kernel.missing_mass.reshape(-1, 4)), gain)
    return lower, upper


def continuous_lower_function(kernel, floor, bias_bounds, gain_alarm, channel_noise=.03):
    """Contained monotone subevent uniformly lower-bounds full r>=2*floor power.

    Only the increased-error tail is retained. No full two-sided power
    monotonicity or exact balanced minimum is asserted.
    """
    floor, noise = _unit(floor, 'floor'), _unit(channel_noise, 'channel_noise')
    if floor > .5 or not math.isclose(kernel.q0, noise/2, rel_tol=0, abs_tol=1e-14):
        raise ValueError('IR floor/noise inconsistent with nominal calibration')
    if len(bias_bounds) != 2:
        raise ValueError('Expected a bias interval')
    blo, bhi = (_unit(x, 'bias') for x in bias_bounds)
    if blo > bhi:
        raise ValueError('Bias bounds are unordered')
    c = (1-noise)/2
    qz, qx = kernel.q0+c*2*floor*(1-bhi), kernel.q0+c*2*floor*blo
    high, _ = coordinate_components(kernel, (qz, qz, qx, qx))
    return _combine(high, _gain_mask(gain_alarm, kernel.rows))


def empirical_endpoint(values, delta, side):
    """One-sided Maurer-Pontil Thm.4; use 1-Z for a lower mean endpoint."""
    x = np.asarray(values, dtype=float)
    delta = _unit(delta, 'delta')
    if (x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x))
            or np.any((x < 0) | (x > 1)) or not 0 < delta < 1 or side not in ('lower', 'upper')):
        raise ValueError('Expected bounded iid values and a supported one-sided bound')
    mean, variance = float(x.mean()), float(x.var(ddof=1))
    log = math.log(2/delta)
    radius = math.sqrt(2*variance*log/len(x))+7*log/(3*(len(x)-1))
    endpoint = max(0., mean-radius) if side == 'lower' else min(1., mean+radius)
    return {'mean': mean, 'sample_variance': variance, 'radius': radius,
            'side': side, 'delta': delta, 'endpoint': endpoint, 'trials': len(x)}
