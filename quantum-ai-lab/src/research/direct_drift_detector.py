"""Direct pooled two-sample bounded-drift test; research-only.

Uses the Step 9 shared-bit iid null. This is a standard-tool construction, not
a new quantum/security theorem. See docs/direct-drift-test.md for the proof,
fixed internal alpha allocation and limits. No private probabilities enter.
"""
import math

import numpy as np
from scipy.stats import beta

from research.calibration_protocol import PublicBatch
from research.drift_detector import _alpha, _envelope

POOLED_COORDINATES = ('Z_error', 'X_error', 'signal_gain', 'decoy_gain')


def pooled_counts(batch):
    """Sum equal-probability bit strata; preserve signal/decoy gain coordinates.

    Validity requires shared bit probabilities, not merely count bookkeeping.
    Pooling is deterministic; it does not filter rows or mutate public batches.
    """
    if not isinstance(batch, PublicBatch):
        raise ValueError('Expected public counts only')
    def pool(values):
        result = np.column_stack((values[:, :2].sum(axis=1),
                                  values[:, 2:4].sum(axis=1), values[:, 4:]))
        result.flags.writeable = False
        return result
    return pool(batch.numerators), pool(batch.denominators)


def _half_upper(y, n, epsilon):
    """One-sided exact upper bound, intersected with the null's [0,.5] domain.

    Called on positive denominators only. All-success counts use .5 after the
    known-domain intersection; no empirical outcome is clipped or relabelled.
    """
    result = np.full(y.shape, .5)
    partial = y < n
    result[partial] = np.minimum(.5, beta.ppf(1-epsilon, y[partial]+1,
                                             n[partial]-y[partial]))
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > .5)):
        raise ArithmeticError('Invalid one-sided variance confidence bound')
    return result


def _error_radius(ybase, nbase, nother, drift, alpha):
    """Positive-denominator Bernstein radius; one of four fixed error directions."""
    epsilon = alpha/16
    upper = _half_upper(ybase, nbase, epsilon)
    other_upper = np.minimum(.5, upper+drift)
    variance = upper*(1-upper)/nbase + other_upper*(1-other_upper)/nother
    b = np.maximum(1/nbase, 1/nother)
    linear = b*math.log(1/epsilon)/3
    return linear + np.sqrt(2*variance*math.log(1/epsilon)+linear*linear)


def direct_drift_directions(calibration, monitoring, envelope, *, alpha=.05):
    """Read-only (rows,4,2) flags: Z,X,signal,decoy; upward then downward.

    Four error directions each spend alpha/16 on a one-sided variance upper
    bound and alpha/16 on Bernstein concentration. Four gain directions each
    spend alpha/8 on a direct Hoeffding tail. These eight internal directions
    jointly control fixed-horizon false alarms at alpha under the approved null.

    Zero denominators disable that coordinate, not the entire transcript.
    CP variance confidence events need not be independent of differences.
    This proof is real-arithmetic, not a floating-point rounding certificate.
    """
    envelope, alpha = _envelope(envelope), _alpha(alpha)
    if (not isinstance(calibration, PublicBatch) or not isinstance(monitoring, PublicBatch)
            or calibration.numerators.shape != monitoring.numerators.shape):
        raise ValueError('Expected paired PublicBatch objects with matching rows')
    yc, nc = pooled_counts(calibration)
    ym, nm = pooled_counts(monitoring)
    flags = np.zeros((len(nc), 4, 2), dtype=bool)
    for j, d in enumerate((envelope.z_error, envelope.x_error, 0., 0.)):
        active = (nc[:, j] > 0) & (nm[:, j] > 0)
        if not np.any(active):
            continue
        a, m = nc[active, j].astype(float), nm[active, j].astype(float)
        difference = ym[active, j]/m - yc[active, j]/a
        if j < 2:
            up = _error_radius(yc[active, j], a, m, d, alpha)
            down = _error_radius(ym[active, j], m, a, d, alpha)
        else:
            up = down = np.sqrt((1/a+1/m)*math.log(8/alpha)/2)
        if not np.all(np.isfinite(up)) or not np.all(np.isfinite(down)):
            raise ArithmeticError('Nonfinite direct-drift threshold')
        flags[active, j, 0] = difference > d+up
        flags[active, j, 1] = -difference > d+down
    flags.flags.writeable = False
    return flags


def direct_drift_alarm(calibration, monitoring, envelope, *, alpha=.05):
    """One public-count-only fixed-horizon rule; no true-noise or policy inputs.

    The OR is over its prebudgeted internal flags, never over separate methods.
    This is not guaranteed to dominate another test or to certify a secret key.
    """
    flags = direct_drift_directions(calibration, monitoring, envelope, alpha=alpha)
    alarms = np.any(flags, axis=(1, 2))
    alarms.flags.writeable = False
    return alarms
