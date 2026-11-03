"""Exact count-aware tests for restricted honest BB84 telemetry assumptions.

Fisher eliminates an unknown *common* error probability across bases. The
one-sided binomial decoy test assumes a known honest decoy click probability.
Neither consumes the true test-time channel noise, an Eve label or knowledge.
"""
from functools import lru_cache

import numpy as np
from scipy.stats import binom, fisher_exact

from bb84.simulator import _probability
from bb84.telemetry import TelemetryCounts

METHODS = ('Count Fisher basis', 'Count binomial decoy', 'Count Bonferroni joint')


@lru_cache(maxsize=4096)
def fisher_pvalue(z_sifted, x_sifted, z_errors, x_errors):
    """Probability-order two-sided Fisher p-value, including discreteness/ties."""
    return float(fisher_exact([[z_errors, z_sifted-z_errors],
                               [x_errors, x_sifted-x_errors]], alternative='two-sided').pvalue)


class CountTests:
    def __init__(self, honest_decoy_click_probability):
        self.decoy_probability = _probability(honest_decoy_click_probability, 'decoy_probability')

    def pvalues(self, count_matrix):
        matrix = np.asarray(count_matrix)
        if matrix.ndim != 2 or matrix.shape[1] != 8 or not len(matrix):
            raise ValueError('Expected a nonempty eight-count matrix')
        # Construct the validated value object before SciPy can silently coerce
        # floats, negative counts or impossible numerator/denominator pairs.
        counts = [TelemetryCounts(*row) for row in matrix]
        basis = np.array([fisher_pvalue(c.z_sifted, c.x_sifted, c.z_errors, c.x_errors)
                          for c in counts])
        decoy = binom.cdf([c.decoy_detected for c in counts],
                          [c.decoy_sent for c in counts], self.decoy_probability)
        return {METHODS[0]: basis, METHODS[1]: decoy,
                METHODS[2]: np.minimum(1.0, 2*np.minimum(basis, decoy))}

    def scores(self, count_matrix):
        # Higher score = stronger evidence. Using -p avoids log(0)/inf and
        # retains the same ranking as a decreasing p-value.
        return {name: -values for name, values in self.pvalues(count_matrix).items()}
