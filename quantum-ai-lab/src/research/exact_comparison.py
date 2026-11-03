"""Research-only fixed-horizon conditional two-sample alarm.

This is the central (doubled-minimum-tail) exact test, NOT SciPy's default
probability-ordered two-sided Fisher test. No nuisance probabilities or private
labels are accepted. Exactness concerns real arithmetic, not certified rounding.
"""
import numpy as np
from scipy.stats import hypergeom

from research.calibration_protocol import PublicBatch
from research.identifiability import _unit

# Bound this research API to an explicit supported numerical domain. Adding
# two stage denominators is done in int64 before comparing with this limit.
MAX_TWO_SAMPLE_TOTAL = 2**31-1


def central_exact_pvalues(calibration, monitoring):
    """Return one two-sided conditional p-value per row and public coordinate.

    Under the stable null, Yc | (Yc+Ym=t,nc,nm) is hypergeometric with population
    nc+nm, t successes and nc draws. p=min(1,2*min(P[Y<=yc],P[Y>=yc])). Empty
    stages or all-success/all-failure totals are uninformative and return one.
    Survival probability uses sf(yc-1), not sf(yc), to include the observed atom.
    """
    if (not isinstance(calibration, PublicBatch) or not isinstance(monitoring, PublicBatch)
            or calibration.numerators.shape != monitoring.numerators.shape):
        raise ValueError('Expected paired PublicBatch objects of the same size')
    nc, nm = calibration.denominators, monitoring.denominators
    yc, ym = calibration.numerators, monitoring.numerators
    population, successes = nc+nm, yc+ym
    if np.any(population > MAX_TWO_SAMPLE_TOTAL):
        raise ValueError('Two-sample denominator exceeds supported numerical domain')
    pvalues = np.ones(yc.shape, dtype=float)
    active = (nc > 0) & (nm > 0) & (successes > 0) & (successes < population)
    lo = hypergeom.cdf(yc[active], population[active], successes[active], nc[active])
    hi = hypergeom.sf(yc[active]-1, population[active], successes[active], nc[active])
    pvalues[active] = np.minimum(1., 2*np.minimum(lo, hi))
    if not np.all(np.isfinite(pvalues)) or np.any(pvalues < 0):
        raise ArithmeticError('Invalid conditional-tail calculation')
    pvalues.flags.writeable = False
    return pvalues


def exact_change_alarm(calibration, monitoring, alpha=.05):
    """Any of six central exact p-values <= alpha/6 gives a change alarm.

    Conditional super-uniformity and a union bound give family-wise false-alarm
    probability <= alpha for a stable iid law. Coordinate independence is not
    needed. No repeated peeking, optional stopping, fitted allocation, honest
    drift attribution or final-key security is covered by this statement.
    """
    alpha = _unit(alpha, 'alpha')
    if not 1e-12 <= alpha < 1:
        raise ValueError('Unsupported alpha')
    return np.any(central_exact_pvalues(calibration, monitoring) <= alpha/6, axis=1)
