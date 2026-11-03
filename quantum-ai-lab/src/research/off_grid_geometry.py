"""Public-law geometry, not proof of detector-optimal Eve or QKD security."""
import math

from research.allocation_frontier import BudgetPlan, IRPolicy
from research.calibration_protocol import PublicBatch
from research.identifiability import _unit, bernoulli_kl

ERROR_SWAP = (2, 3, 0, 1, 4, 5)
SIFT_SWAP = (2, 3, 0, 1)


def reflect_public(batch):
    """Swap Z/X strata only; gains, emission count and row order stay fixed."""
    if not isinstance(batch, PublicBatch):
        raise ValueError('Expected public counts')
    return PublicBatch(batch.emitted, batch.numerators[:, ERROR_SWAP],
                       batch.denominators[:, ERROR_SWAP], batch.sifted[:, SIFT_SWAP])


def public_policy_kl(plan, first, second, channel_noise=.03, eta=.15):
    """KL of paired public laws: calibration cancels; expected test counts suffice.

    Loss, sifting and sample membership have the same distribution. Conditional
    disclosed errors are independent binomials. This is not a private-label KL.
    """
    if not isinstance(plan, BudgetPlan) or not all(isinstance(p, IRPolicy) for p in (first, second)):
        raise ValueError('Expected resource plan and two IR policies')
    noise, eta = _unit(channel_noise, 'channel_noise'), _unit(eta, 'eta')
    exposure = plan.monitoring_emitted*plan.monitoring_disclosure
    gain = (-math.expm1(-.5*eta)-math.expm1(-.1*eta))/2
    if not exposure or not gain:
        return 0.
    p, q = first.law(noise, eta).bit_errors, second.law(noise, eta).bit_errors
    # Each of Z0,Z1,X0,X1 is 1/8 of detected events (half are mismatched).
    return float(exposure*gain/8*math.fsum(bernoulli_kl(a, b) for a, b in zip(p, q)))


def cell_radius(plan, anchor, rate_bounds, bias_bounds, channel_noise=.03, eta=.15):
    """Uniform Pinsker radius from max D(anchor||theta) at rectangle corners.

    KL is separately convex in r and beta, so its maximum is at a corner even
    though the physical probabilities are bilinear in these two parameters.
    Numerical result is NOT outward-rounded or a continuous detector optimum.
    """
    if not isinstance(plan, BudgetPlan) or not isinstance(anchor, IRPolicy):
        raise ValueError('Expected resource plan and IR anchor')
    rlo, rhi = (_unit(x, 'rate_bound') for x in rate_bounds)
    blo, bhi = (_unit(x, 'bias_bound') for x in bias_bounds)
    if rlo > rhi or blo > bhi or not rlo <= anchor.rate <= rhi or not blo <= anchor.z_basis_probability <= bhi:
        raise ValueError('Ordered rectangle must contain its anchor')
    k = max(public_policy_kl(plan, anchor, IRPolicy(r, b), channel_noise, eta)
            for r in (rlo, rhi) for b in (blo, bhi))
    return {'maximum_corner_kl': float(k), 'tv_radius': float(min(1., math.sqrt(k/2)))}


def axis_cells(anchors, lower, upper):
    values = [_unit(v, 'anchor') for v in anchors]
    lower, upper = _unit(lower, 'lower'), _unit(upper, 'upper')
    if not values or values != sorted(set(values)) or not lower <= values[0] <= values[-1] <= upper:
        raise ValueError('Expected increasing anchors inside domain')
    edges = [lower]+[(a+b)/2 for a, b in zip(values, values[1:])]+[upper]
    return [(a, (edges[i], edges[i+1])) for i, a in enumerate(values)]


def continuous_envelope(cells, evaluated_upper):
    """On the anchor coverage event, min_theta power >= min_g(Lg-radius_g)."""
    upper = _unit(evaluated_upper, 'evaluated_upper')
    if not cells:
        raise ValueError('Empty cover')
    lower = min(max(0., _unit(c['anchor_lower'], 'anchor_lower')-
                    _unit(c['tv_radius'], 'tv_radius')) for c in cells)
    return {'lower': float(lower), 'upper': upper, 'consistent': bool(lower <= upper)}
