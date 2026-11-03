"""Fixed-horizon confidence-gap monitoring under bounded honest error drift.

The alarm sees public counts and a predeclared envelope, never true noise or
private Eve labels. Simulation laws and conditional power bounds are separate
analysis helpers. All guarantees concern real arithmetic, not outward-rounded
floating-point security certificates. See docs/bounded-drift-model.md.
"""
from dataclasses import dataclass
import math

import numpy as np

from research.calibration_protocol import (
    ObservationLaw, PublicBatch, _array, simultaneous_intervals,
)
from research.identifiability import _unit


def _half_probability(value, name):
    value = _unit(value, name)
    if value > .5:
        raise ValueError(f'{name} must be in [0, 0.5]')
    return value


def _alpha(value):
    value = _unit(value, 'alpha')
    if not 1e-12 <= value < 1:
        raise ValueError('Unsupported alpha')
    return value


@dataclass(frozen=True)
class DriftEnvelope:
    """Predeclared absolute per-basis error allowances; gains remain stable.

    Values are probability differences, not relative percentages. They must not
    be fitted to monitored errors. The [0, .5] domain covers all differences in
    the approved honest family. It does not model gain/intensity/device drift.
    """
    z_error: float = 0.
    x_error: float = 0.

    def __post_init__(self):
        object.__setattr__(self, 'z_error',
                           _half_probability(self.z_error, 'z_error'))
        object.__setattr__(self, 'x_error',
                           _half_probability(self.x_error, 'x_error'))

    @property
    def coordinate_allowances(self):
        return (self.z_error, self.z_error, self.x_error, self.x_error, 0., 0.)


def _envelope(value):
    if not isinstance(value, DriftEnvelope):
        raise ValueError('Expected a predeclared DriftEnvelope')
    return value


def honest_drift_laws(cal_z, cal_x, mon_z, mon_x, envelope, *, eta=.15):
    """Simulation-only lawful reference/monitor pair, never alarm inputs.

    Basis errors are shared by both preparation bits. Every pair in [0,.5]^2
    admits a physical Pauli representative; this is not a hardware validation.
    Independence, block stationarity and trusted calibration remain assumptions.
    Reject an out-of-envelope proposal rather than clip or label it honest.
    """
    envelope = _envelope(envelope)
    cz, cx, mz, mx = (_half_probability(v, name) for v, name in
                      zip((cal_z, cal_x, mon_z, mon_x),
                          ('cal_z', 'cal_x', 'mon_z', 'mon_x')))
    eta = _unit(eta, 'eta')
    if abs(mz-cz) > envelope.z_error or abs(mx-cx) > envelope.x_error:
        raise ValueError('Monitoring errors exceed the declared honest envelope')
    return (ObservationLaw((cz, cz, cx, cx), eta),
            ObservationLaw((mz, mz, mx, mx), eta))


def drift_change_alarm(calibration, monitoring, envelope, *, alpha=.05,
                       method='hoeffding'):
    """Any confidence-pair gap strictly beyond its allowance gives an alarm.

    The twelve stage/coordinate intervals jointly spend alpha. Under the
    approved iid, error-independent selection law, true per-coordinate changes
    within the envelope imply false-alarm probability <= alpha. CP and Hoeffding
    intervals have this same coverage argument, not equal attained FPR/power.
    Empty stages get [0,1]; gains have zero allowance even if errors allow drift.

    Fixed horizon only: no repeated peeking, fitted envelope, attribution to Eve
    or final-key guarantee. O(rows) work/memory, with six fixed coordinates.
    The returned boolean vector is read-only; inputs are never modified.
    """
    envelope = _envelope(envelope)
    if (not isinstance(calibration, PublicBatch)
            or not isinstance(monitoring, PublicBatch)
            or calibration.numerators.shape != monitoring.numerators.shape):
        raise ValueError('Expected paired PublicBatch objects with matching rows')
    alpha = _alpha(alpha)
    clo, chi = simultaneous_intervals(calibration, alpha, method)
    mlo, mhi = simultaneous_intervals(monitoring, alpha, method)
    allowances = np.asarray(envelope.coordinate_allowances)
    alarms = np.any((mlo-chi > allowances) | (clo-mhi > allowances), axis=1)
    alarms.flags.writeable = False
    return alarms


def hoeffding_family_power_lower(calibration_denominators, monitoring_denominators,
                                 envelope, minimum_fraction, *,
                                 channel_noise=.03, alpha=.05):
    """Analysis-only lower function for continuous fixed-noise iid IR power.

    Inputs are ancillary rows of four error denominators, not mean counts or
    error outcomes. The return is a *conditional* lower function; averaging over
    the ancillary law still needs integration or a sampling confidence budget.
    It bounds the Hoeffding drift alarm only, not CP or the stationary exact test.
    Calibration q0=lambda/2, unchanged gains, r>=2*kappa and beta in [0,1] are
    essential. The attack is not allowed concurrent baseline-noise drift here.
    An uninformative zero does not prove that optimal power is zero.
    """
    envelope = _envelope(envelope)
    floor = _half_probability(minimum_fraction, 'minimum_fraction')
    if floor == 0:
        raise ValueError('minimum_fraction must be positive')
    noise, alpha = _unit(channel_noise, 'channel_noise'), _alpha(alpha)
    nc = _array(calibration_denominators, 4, 'calibration denominators')
    nm = _array(monitoring_denominators, 4, 'monitoring denominators')
    if nc.shape != nm.shape:
        raise ValueError('Expected matching error-denominator rows')
    result = np.zeros(len(nc))
    gamma = math.fsum(((1-noise)*floor, -envelope.z_error, -envelope.x_error))
    active = np.all((nc > 0) & (nm > 0), axis=1)
    if gamma > 0 and np.any(active):
        mc, mm = nc[active].astype(float), nm[active].astype(float)
        scale = math.log(24/alpha)/2
        hmax = np.max(np.sqrt(scale/mc)+np.sqrt(scale/mm), axis=1)
        smax = np.max(1/mc+1/mm, axis=1)
        residual = np.maximum(0., gamma/2-hmax)
        # expm1 avoids losing a small positive bound by subtracting from one.
        result[active] = -np.expm1(-2*residual**2/smax)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ArithmeticError('Invalid conditional power lower function')
    result.flags.writeable = False
    return result
