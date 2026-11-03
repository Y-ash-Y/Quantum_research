"""Exact aggregate-count marginal for an independent *honest* BB84 channel.

Not an attack simulator or a pulse log. Local gate noise follows the existing
Aer preparation/measurement circuits; photon loss is marginalized analytically.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from bb84.simulator import _count, _outcome_table, _probability
from bb84.telemetry import TelemetryCounts

MU_SIGNAL, MU_DECOY, DECOY_FRACTION = 0.5, 0.1, 0.5


@dataclass(frozen=True)
class HonestSettings:
    n_pulses: int = 6000
    channel_noise: float = 0.03
    eta: float = 0.15
    gate_noise: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'n_pulses', _count(self.n_pulses, 'n_pulses'))
        for name in ('channel_noise', 'eta', 'gate_noise'):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


@lru_cache(maxsize=64)
def _basis_errors(channel_noise, gate_noise):
    table = _outcome_table(channel_noise, gate_noise=gate_noise)
    return tuple((table[(0, b, b)]+1-table[(1, b, b)])/2 for b in ('Z', 'X'))


def probabilities(settings):
    """Real-Aer bit-averaged basis errors and exact Poisson click probabilities."""
    if not isinstance(settings, HonestSettings):
        raise ValueError('settings must be HonestSettings')
    qz, qx = _basis_errors(settings.channel_noise, settings.gate_noise)
    return {'q_z': qz, 'q_x': qx,
            'signal_click': float(-np.expm1(-MU_SIGNAL*settings.eta)),
            'decoy_click': float(-np.expm1(-MU_DECOY*settings.eta))}


@dataclass
class HonestBatch:
    features: np.ndarray
    counts: np.ndarray
    accepted_attempts: np.ndarray
    attempted: int
    rejected: int


def sample_honest(attempted, settings, rng):
    attempted = _count(attempted, 'attempted')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a local numpy Generator')
    p = probabilities(settings)
    decoy = rng.binomial(settings.n_pulses, DECOY_FRACTION, attempted)
    signal = settings.n_pulses-decoy
    ds = rng.binomial(signal, p['signal_click'])
    dd = rng.binomial(decoy, p['decoy_click'])
    # Actual click counts, then basis matching, then bit-averaged errors.
    bases = rng.multinomial(ds+dd, [.25, .25, .5])
    nz, nx = bases[:, 0], bases[:, 1]
    ez, ex = rng.binomial(nz, p['q_z']), rng.binomial(nx, p['q_x'])
    counts = np.column_stack([nz, nx, ez, ex, signal, decoy, ds, dd])
    usable = ((nz+nx >= 20) & (nz > 0) & (nx > 0)
              & (signal > 0) & (decoy > 0) & (ds > 0))
    ids = np.flatnonzero(usable)
    if not len(ids):
        raise ValueError('No usable honest nuisance sessions')
    counts = counts[usable].astype(int)
    features = np.asarray([TelemetryCounts(*row).features() for row in counts], dtype=float)
    return HonestBatch(features, counts, ids, attempted, attempted-len(ids))
