"""Research-only public-sample change test; not a QKD security implementation.

Random sample membership is chosen independently of errors after sifting. Only
disclosed errors are generated/exported. The scorer sees counts, never the
simulator's probabilities, undisclosed bits, or Eve's private knowledge.
"""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.stats import beta

from research.identifiability import _unit, honest_bit_errors

COORDINATES = ('Z0_error', 'Z1_error', 'X0_error', 'X1_error',
               'signal_gain', 'decoy_gain')
MAX_COUNT = 2**53-1


def _count(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not minimum <= value <= MAX_COUNT):
        raise ValueError(f'{name} must be an integer in [{minimum}, {MAX_COUNT}]')
    return int(value)


@dataclass(frozen=True)
class ObservationLaw:
    """Simulation-side conditional errors and intensity-independent loss.

    The order is Z0,Z1,X0,X1. Never pass this object to the alarm rule.
    """
    bit_errors: tuple
    eta: float = .15

    def __post_init__(self):
        if len(self.bit_errors) != 4:
            raise ValueError('Expected four bit-conditioned errors')
        object.__setattr__(self, 'bit_errors', tuple(
            _unit(v, 'error_probability') for v in self.bit_errors))
        object.__setattr__(self, 'eta', _unit(self.eta, 'eta'))


def gate_law(channel_noise=.03, gate_noise=0., eta=.15):
    return ObservationLaw(honest_bit_errors(channel_noise, gate_noise), eta)


def pauli_law(weights, eta=.15):
    """Physical I,X,Y,Z channel; matched Z flips under X/Y, X under Z/Y."""
    if len(weights) != 4:
        raise ValueError('Expected I,X,Y,Z weights')
    values = tuple(_unit(v, 'pauli_weight') for v in weights)
    total = math.fsum(values)
    if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Pauli weights must sum to one')
    _, px, py, pz = (v/total for v in values)  # Normalize roundoff only.
    return ObservationLaw((px+py, px+py, pz+py, pz+py), eta)


def _array(values, columns, name):
    raw = np.asarray(values)
    if (raw.ndim != 2 or raw.shape[1] != columns or not len(raw)
            or raw.dtype.kind not in 'iu' or np.any(raw < 0)
            or np.any(raw > MAX_COUNT)):
        raise ValueError(f'Invalid {name} count matrix')
    result = raw.astype(np.int64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PublicBatch:
    """Count-only marginal of an authenticated parameter-estimation transcript.

    numerators: four disclosed-error counts, two detected counts.
    denominators: four disclosed-bit counts, two emitted intensity counts.
    sifted: matched detected bits by preparation bit/basis, not secret bitstrings.
    Unrevealed error counts are deliberately absent. Remaining sifted counts
    are only key *candidates*, not secret-key lengths.
    """
    emitted: int
    numerators: np.ndarray
    denominators: np.ndarray
    sifted: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'emitted', _count(self.emitted, 'emitted'))
        for name, columns in [('numerators', 6), ('denominators', 6), ('sifted', 4)]:
            object.__setattr__(self, name, _array(getattr(self, name), columns, name))
        y, n, s = self.numerators, self.denominators, self.sifted
        if (y.shape != n.shape or len(s) != len(n) or np.any(y > n)
                or np.any(n[:, :4] > s) or np.any(s > self.emitted)
                or np.any(n[:, 4:].sum(axis=1) != self.emitted)
                or np.any(s.sum(axis=1) > y[:, 4:].sum(axis=1))):
            raise ValueError('Inconsistent public count bookkeeping')

    @property
    def disclosed(self):
        return self.denominators[:, :4].sum(axis=1)

    @property
    def remaining_candidates(self):
        return self.sifted.sum(axis=1)-self.disclosed


def sample_public_batch(trials, emitted, law, disclosure_fraction, rng):
    """Exact iid marginal, O(trials) memory/work regardless of emitted pulses.

    Intensity is independent uniform signal(.5)/decoy(.1). Loss is independent
    of basis, preparation bit and error. All rows survive, including zero clicks.
    This NumPy simulation is not a cryptographic random generator.
    """
    trials = _count(trials, 'trials', 1)
    emitted = _count(emitted, 'emitted')
    fraction = _unit(disclosure_fraction, 'disclosure_fraction')
    if not isinstance(law, ObservationLaw) or not isinstance(rng, np.random.Generator):
        raise ValueError('Expected ObservationLaw and a local NumPy Generator')
    decoy = rng.binomial(emitted, .5, trials)
    sent = np.column_stack((emitted-decoy, decoy))
    gains = -np.expm1(-np.array([.5, .1])*law.eta)
    detected = rng.binomial(sent, gains)
    # Each matched (basis, Alice bit) has probability 1/8; mismatch has 1/2.
    classes = rng.multinomial(detected.sum(axis=1), [.125]*4+[.5])
    sifted = classes[:, :4]
    disclosed = rng.binomial(sifted, fraction)
    errors = rng.binomial(disclosed, law.bit_errors)
    return PublicBatch(emitted, np.column_stack((errors, detected)),
                       np.column_stack((disclosed, sent)), sifted)


def simultaneous_intervals(batch, alpha=.05, method='hoeffding'):
    """Twelve calibration/monitor intervals share one family-wise alpha.

    Each stage/coordinate has failure budget delta=alpha/12, split equally
    between its two tails. Both stages use this helper once. Empty denominators
    produce [0,1]. CP is exact in real arithmetic, not a rounding certificate.
    """
    if not isinstance(batch, PublicBatch):
        raise ValueError('Expected public counts only')
    alpha = _unit(alpha, 'alpha')
    if not 1e-12 <= alpha < 1 or method not in ('hoeffding', 'clopper-pearson'):
        raise ValueError('Unsupported alpha or interval method')
    y, n = batch.numerators, batch.denominators
    lower, upper = np.zeros(y.shape), np.ones(y.shape)
    active = n > 0
    if method == 'hoeffding':
        estimate = y[active]/n[active]
        radius = np.sqrt(math.log(24/alpha)/(2*n[active]))
        lower[active] = np.maximum(0, estimate-radius)
        upper[active] = np.minimum(1, estimate+radius)
    else:
        tail = alpha/24
        lo = active & (y > 0)
        hi = active & (y < n)
        lower[lo] = beta.ppf(tail, y[lo], n[lo]-y[lo]+1)
        upper[hi] = beta.ppf(1-tail, y[hi]+1, n[hi]-y[hi])
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ArithmeticError('Nonfinite interval computation')
    return lower, upper


def change_alarm(calibration, monitoring, alpha=.05, method='hoeffding'):
    """Reject a stable honest law when any of six confidence pairs is disjoint.

    Fixed-horizon only: no repeated peeking, post-selection, hidden probability
    inputs, fitting or attack labels. One alpha covers both stages and all six
    coordinates, NOT multiple monitoring windows or separately selected rules.
    """
    if (not isinstance(calibration, PublicBatch) or not isinstance(monitoring, PublicBatch)
            or len(calibration.numerators) != len(monitoring.numerators)):
        raise ValueError('Expected paired public batches of the same size')
    clo, chi = simultaneous_intervals(calibration, alpha, method)
    mlo, mhi = simultaneous_intervals(monitoring, alpha, method)
    return np.any((clo > mhi) | (mlo > chi), axis=1)
