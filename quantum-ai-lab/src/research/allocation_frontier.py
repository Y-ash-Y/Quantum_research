"""Fixed-resource IR experiments and a broad-honest-channel power ceiling.

Research-only. Interception-certified raw knowledge is not total mutual
information, public-transcript leakage, or final-key leakage. Private labels
are never part of PublicBatch or inputs to the alarm rule.
"""
from dataclasses import dataclass
import math

import numpy as np

from research.calibration_protocol import (
    ObservationLaw, PublicBatch, _array, _count,
)
from research.identifiability import (
    _unit, bernoulli_kl, ir_basis_errors, ir_pauli_weights, power_ceiling,
)


@dataclass(frozen=True)
class BudgetPlan:
    total_emitted: int
    calibration_emitted: int
    monitoring_disclosure: float

    def __post_init__(self):
        b = _count(self.total_emitted, 'total_emitted', 1)
        m = _count(self.calibration_emitted, 'calibration_emitted')
        if m > b:
            raise ValueError('Calibration cannot exceed total emissions')
        object.__setattr__(self, 'total_emitted', b)
        object.__setattr__(self, 'calibration_emitted', m)
        object.__setattr__(self, 'monitoring_disclosure',
                           _unit(self.monitoring_disclosure, 'monitoring_disclosure'))

    @property
    def monitoring_emitted(self):
        return self.total_emitted-self.calibration_emitted

    @property
    def effective_test_emissions(self):
        return self.calibration_emitted+self.monitoring_emitted*self.monitoring_disclosure

    def expected_resources(self, eta=.15):
        eta = _unit(eta, 'eta')
        gain = (-math.expm1(-.5*eta)-math.expm1(-.1*eta))/2
        m, n, f = self.calibration_emitted, self.monitoring_emitted, self.monitoring_disclosure
        return {'calibration_sacrificed': m*gain/2,
                'monitoring_disclosed': n*f*gain/2,
                'total_sacrificed': (m+n*f)*gain/2,
                'remaining_candidates': n*(1-f)*gain/2}


def constant_sacrifice_plan(total_emitted, calibration_emitted, test_emissions):
    """Keep M+f*N=C fixed, hence expected sacrifice/candidate counts fixed.

    C is an emitted-pulse exposure equivalent, not an observed bit count. This
    equivalence assumes unchanged, basis-independent gains in both stages.
    """
    b = _count(total_emitted, 'total_emitted', 1)
    m = _count(calibration_emitted, 'calibration_emitted')
    c = _count(test_emissions, 'test_emissions')
    if not m <= c <= b:
        raise ValueError('Require calibration <= test exposure <= total')
    fraction = (c-m)/(b-m) if b != m else 0.
    return BudgetPlan(b, m, fraction)


@dataclass(frozen=True)
class IRPolicy:
    rate: float
    z_basis_probability: float

    def __post_init__(self):
        object.__setattr__(self, 'rate', _unit(self.rate, 'rate'))
        object.__setattr__(self, 'z_basis_probability',
                           _unit(self.z_basis_probability, 'z_basis_probability'))

    @property
    def expected_certified_fraction(self):
        # Uniform Alice basis; loss and public sample membership are independent.
        return self.rate/2

    def eligible(self, minimum_fraction):
        floor = _unit(minimum_fraction, 'minimum_fraction')
        if floor > .5:
            raise ValueError('This IR family cannot certify more than half the raw bits')
        return self.expected_certified_fraction >= floor

    def law(self, channel_noise=.03, eta=.15):
        qz, qx = ir_basis_errors(channel_noise, self.rate, self.z_basis_probability)
        return ObservationLaw((qz, qz, qx, qx), eta)


def sample_retained_certified_counts(public, policy, rng):
    """Private audit labels for unrevealed Alice bits only; scorer never sees them.

    Given retained stratum counts, basis-correct interception probabilities are
    r*beta in Z and r*(1-beta) in X. Labels for publicly tested bits are absent.
    This is the exact iid marginal of the existing basis-match knowledge label.
    """
    if (not isinstance(public, PublicBatch) or not isinstance(policy, IRPolicy)
            or not isinstance(rng, np.random.Generator)):
        raise ValueError('Expected public counts, IRPolicy and private RNG')
    retained = public.sifted-public.denominators[:, :4]
    r, bias = policy.rate, policy.z_basis_probability
    known = rng.binomial(retained, [r*bias]*2+[r*(1-bias)]*2)
    return _array(known, 4, 'private_certified')


def broad_pauli_information_bound(plan, policy, channel_noise=.03, eta=.15, alpha=.05):
    """Construct a stable honest Pauli counterpart and minimize joint KL.

    All publicly sampled errors are included, with random denominators averaged
    exactly. Same gains/selection law in both scenarios. Valid fixed-horizon
    alpha alarms against this honest family obey the reported Pinsker ceiling.
    KL is floating-point diagnostic, not an outward-rounded certificate.
    """
    if not isinstance(plan, BudgetPlan) or not isinstance(policy, IRPolicy):
        raise ValueError('Expected BudgetPlan and IRPolicy')
    noise, eta, alpha = (_unit(channel_noise, 'channel_noise'),
                         _unit(eta, 'eta'), _unit(alpha, 'alpha'))
    m = plan.calibration_emitted
    e = plan.monitoring_emitted*plan.monitoring_disclosure
    total = m+e
    weight = e/total if total else 0.
    counterpart = IRPolicy(weight*policy.rate, policy.z_basis_probability)
    h = ir_basis_errors(noise, counterpart.rate, counterpart.z_basis_probability)
    a = ir_basis_errors(noise, policy.rate, policy.z_basis_probability)
    q0 = noise/2
    gain = (-math.expm1(-.5*eta)-math.expm1(-.1*eta))/2
    # Handle zero resource factors before KL to avoid 0 * infinity at endpoints.
    cal_terms = m*sum(bernoulli_kl(q0, q) for q in h) if m and gain else 0.
    mon_terms = e*sum(bernoulli_kl(p, q) for p, q in zip(a, h)) if e and gain else 0.
    divergence = gain*(cal_terms+mon_terms)/4
    return {'joint_kl_nats': divergence,
            'power_ceiling': power_ceiling(alpha, 1, divergence),
            'honest_counterpart_errors': h,
            'honest_counterpart_pauli_weights': ir_pauli_weights(
                noise, counterpart.rate, counterpart.z_basis_probability),
            'monitoring_test_exposure': e,
            'diagnostic_not_rounding_certificate': True}


def policy_sort_key(policy):
    """Predeclared ties: lower rate, closest to balanced bias, then lower bias."""
    return (policy.rate, abs(policy.z_basis_probability-.5), policy.z_basis_probability)
