"""Fixed-budget signal-referenced decoy screening, not security certification.

The screening tails are alpha-dependent; they are not globally valid p-values.
No true channel loss/noise, Eve policy or private knowledge is consumed.
"""
import numpy as np
from scipy.stats import beta, binom

from bb84.simulator import _probability
from bb84.telemetry import TelemetryCounts
from classical_ml.count_detectors import fisher_pvalue

METHODS = ('Loss-aware decoy', 'Loss-aware joint', 'Loss plug-in decoy')
BOUND_FIELDS = ('signal_lower_decoy', 'signal_lower_joint')


def signal_lower_bound(sent, detected, delta):
    """One-sided exact frequentist bound, including zero-click endpoints."""
    delta = _probability(delta, 'delta')
    if not 0 < delta < 1:
        raise ValueError('delta must lie strictly between zero and one')
    sent, detected = np.broadcast_arrays(np.asarray(sent), np.asarray(detected))
    if (sent.dtype.kind not in 'iu' or detected.dtype.kind not in 'iu'
            or np.any(sent < 0) or np.any(detected < 0) or np.any(detected > sent)):
        raise ValueError('Expected nonnegative integer clicks not exceeding emitted counts')
    lower = np.zeros(sent.shape, dtype=float)
    active = detected > 0
    lower[active] = beta.ppf(delta, detected[active], sent[active]-detected[active]+1)
    return lower


def implied_decoy_probability(signal_probability, ratio):
    """Eliminate common optical loss using the trusted intensity ratio."""
    ratio = _probability(ratio, 'ratio')
    values = np.asarray(signal_probability, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('Signal probability must lie in [0,1]')
    if ratio == 0:
        return np.zeros(values.shape)
    result = np.ones(values.shape, dtype=float)
    interior = values < 1
    result[interior] = -np.expm1(ratio*np.log1p(-values[interior]))
    return result


class LossAwareTests:
    def __init__(self, alpha=.05, intensity_ratio=.2):
        self.alpha = _probability(alpha, 'alpha')
        self.ratio = _probability(intensity_ratio, 'intensity_ratio')
        if not 0 < self.alpha < 1 or not 0 < self.ratio < 1:
            raise ValueError('alpha and intensity_ratio must lie strictly between zero and one')

    @property
    def cutoffs(self):
        return dict(zip(METHODS, (-self.alpha/2, -self.alpha/3, -self.alpha)))

    def evaluate(self, count_matrix):
        matrix = np.asarray(count_matrix)
        if matrix.ndim != 2 or matrix.shape[1] != 8 or not len(matrix):
            raise ValueError('Expected a nonempty eight-count matrix')
        counts = [TelemetryCounts(*row) for row in matrix]
        sent = np.array([c.signal_sent for c in counts])
        detected = np.array([c.signal_detected for c in counts])
        decoy_sent = np.array([c.decoy_sent for c in counts])
        decoy_detected = np.array([c.decoy_detected for c in counts])
        lower_decoy = signal_lower_bound(sent, detected, self.alpha/2)
        lower_joint = signal_lower_bound(sent, detected, self.alpha/3)
        d = binom.cdf(decoy_detected, decoy_sent, implied_decoy_probability(lower_decoy, self.ratio))
        j = binom.cdf(decoy_detected, decoy_sent, implied_decoy_probability(lower_joint, self.ratio))
        f = np.array([fisher_pvalue(c.z_sifted, c.x_sifted, c.z_errors, c.x_errors) for c in counts])
        point = np.divide(detected, sent, out=np.zeros(len(sent)), where=sent != 0)
        plug_in = binom.cdf(decoy_detected, decoy_sent, implied_decoy_probability(point, self.ratio))
        scores = dict(zip(METHODS, (-d, -np.minimum(j, f), -plug_in)))
        bounds = dict(zip(BOUND_FIELDS, (lower_decoy, lower_joint)))
        if not all(np.isfinite(v).all() for v in (*scores.values(), *bounds.values())):
            raise ValueError('Nonfinite confidence bound or screening score')
        return scores, bounds
