"""Closed-form feasibility analysis for the explicitly restricted BB84 model.

Nothing here trains, changes, or calibrates a detector. Local gate errors follow
the existing simulator's H/X gate counts. Attacks have ideal local gates, iid
interception and unchanged intensity-independent loss. A matching mean pair is
a matching aggregate-count distribution ONLY with these independence premises.
"""
from dataclasses import dataclass
import math
from numbers import Real


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return value


def honest_bit_errors(channel_noise, gate_noise):
    """Error probabilities (Z0, Z1, X0, X1), NOT Bob=1 probabilities."""
    a = 1-_unit(channel_noise, 'channel_noise')
    t = 1-_unit(gate_noise, 'gate_noise')
    return tuple((1-a*t**k)/2 for k in (0, 1, 2, 3))


def honest_basis_errors(channel_noise, gate_noise):
    z0, z1, x0, x1 = honest_bit_errors(channel_noise, gate_noise)
    return (z0+z1)/2, (x0+x1)/2


def ir_basis_errors(channel_noise, ir_rate, basis_bias):
    noise = _unit(channel_noise, 'channel_noise')
    r, beta = _unit(ir_rate, 'ir_rate'), _unit(basis_bias, 'basis_bias')
    return (noise/2 + (1-noise)*r*(1-beta)/2,
            noise/2 + (1-noise)*r*beta/2)


@dataclass(frozen=True)
class GateMimic:
    channel_noise: float
    gate_noise: float
    nonunique: bool = False


def gate_mimic(q_z, q_x):
    """Invert the honest mean pair; return None outside the physical family.

Use only for theoretical probabilities, not as an estimator on sample rates.
Fully mixed pairs have non-unique inverses; choose lambda=1,g=0 explicitly.
The small tolerance accommodates roundoff at exact physical boundaries only.
"""
    qz, qx = _unit(q_z, 'q_z'), _unit(q_x, 'q_x')
    if qz > .5 or qx > .5:
        return None
    cz, cx = 1-2*qz, 1-2*qx
    if cz == 0:
        return GateMimic(1.0, 0.0, True) if cx == 0 else None
    if cx > cz + 1e-12:
        return None
    t = math.sqrt(min(1.0, cx/cz))
    a = 2*cz/(1+t)
    if a > 1+1e-12:
        return None
    return GateMimic(1-min(a, 1.0), 1-t)


def fixed_z_hidden_limit(channel_noise, gate_cap):
    """Largest IR rate exactly mimicked when honest lambda in [0,1], g<=cap.

Applies ONLY to beta=1, ideal attacked gates, uniform independent Alice bits,
aggregate telemetry and equal eta. This is an overlap boundary, not an achievable
detection rate, a finite-calibration certificate, or a key-security limit.
"""
    noise = _unit(channel_noise, 'channel_noise')
    cap = _unit(gate_cap, 'gate_cap')
    if noise == 1:
        return 1.0  # Every IR rate is fully mixed, even at honest g=0.
    physical_limit = 1-max(0.0, 1-2*noise)**2
    gate_limit = 2*cap-cap*cap
    return min(physical_limit, gate_limit)


def ir_pauli_weights(channel_noise, ir_rate, basis_bias):
    """Equivalent honest Pauli channel for the *complete* IR channel to Bob.

Order: depolarizing lambda after a mixture of identity / Z / X dephasing.
These are probabilities of I,X,Y,Z conjugations, not security/leakage estimates.
"""
    noise = _unit(channel_noise, 'channel_noise')
    r, beta = _unit(ir_rate, 'ir_rate'), _unit(basis_bias, 'basis_bias')
    a = 1-noise
    return {'I': a*(1-r/2)+noise/4,
            'X': a*r*(1-beta)/2+noise/4,
            'Y': noise/4,
            'Z': a*r*beta/2+noise/4}


def bernoulli_kl(p, q):
    """D(Bernoulli(p)||Bernoulli(q)), natural logarithms, in nats."""
    p, q = _unit(p, 'p'), _unit(q, 'q')
    if p == q:
        return 0.0
    if q == 0 or q == 1:
        return math.inf
    result = 0.0
    if p:
        result += p*math.log(p/q)
    if p < 1:
        result += (1-p)*math.log((1-p)/(1-q))
    return max(0.0, result)  # Non-negativity up to floating-point roundoff.


def aggregate_kl_per_pulse(attack_errors, honest_errors, mean_click):
    """Per-emitted-pulse KL of iid aggregate counts with SAME loss in both.

Each basis-matched class occurs with probability mean_click/4. No dark counts,
selection, time correlation, or device-dependent detection probabilities.
Not applicable to gain-changing PNS or filtered conditional distributions.
"""
    gain = _unit(mean_click, 'mean_click')
    if len(attack_errors) != 2 or len(honest_errors) != 2:
        raise ValueError('Expected two basis error probabilities per scenario')
    terms = [bernoulli_kl(p, q) for p, q in zip(attack_errors, honest_errors)]
    return 0.0 if gain == 0 else gain*sum(terms)/4


def power_ceiling(alpha, n_pulses, kl_per_pulse):
    """Pinsker upper bound alpha+sqrt(N*KL/2), capped at 1.

Assumes a valid alpha false-alarm bound at the chosen honest counterpart and
unfiltered independent observations. This ceiling is NOT attainable power.
"""
    alpha = _unit(alpha, 'alpha')
    if isinstance(n_pulses, bool) or not isinstance(n_pulses, int) or n_pulses < 0:
        raise ValueError('n_pulses must be a nonnegative integer')
    if isinstance(kl_per_pulse, bool) or not isinstance(kl_per_pulse, Real):
        raise ValueError('kl_per_pulse must be a nonnegative real number')
    k = float(kl_per_pulse)
    if math.isnan(k) or k < 0:
        raise ValueError('kl_per_pulse must be nonnegative')
    if n_pulses == 0 or k == 0:
        return alpha
    return min(1.0, alpha+math.sqrt(n_pulses*k/2))
