"""Finite-grid search validation, not a continuous Eve-optimality proof."""
from scipy.stats import binomtest

from research.allocation_frontier import IRPolicy, policy_sort_key

FLOORS = (.0025, .005, .01, .02)
MULTIPLIERS = (1., 1.125, 1.25, 1.5, 2.)
BIASES = tuple(i/8 for i in range(9))


def expanded_grid():
    rates = sorted({round(2*k*m, 12) for k in FLOORS for m in MULTIPLIERS})
    return [IRPolicy(rate, bias) for rate in rates for bias in BIASES]


def eligible_ids(grid, floor):
    if floor not in FLOORS:
        raise ValueError('Unsupported predeclared information floor')
    rates = {round(2*floor*m, 12) for m in MULTIPLIERS}
    return [i for i, p in enumerate(grid) if p.rate in rates and p.eligible(floor)]


def select_policy(counts, grid, ids):
    """All cells have the same trials; minimize alarms then frozen policy ties."""
    if not ids or any(i not in counts for i in ids):
        raise ValueError('Missing eligible search counts')
    return min(ids, key=lambda i: (counts[i], policy_sort_key(grid[i])))


def make_shortlist(counts_a, counts_b, grid, floor, previous_policy):
    """Freeze candidates from search only; old policy and boundary probes retained."""
    ids = eligible_ids(grid, floor)
    pooled = {i: counts_a[i]+counts_b[i] for i in ids}
    ordered_a = sorted(ids, key=lambda i: (counts_a[i], policy_sort_key(grid[i])))
    ordered_b = sorted(ids, key=lambda i: (counts_b[i], policy_sort_key(grid[i])))
    selected = select_policy(pooled, grid, ids)
    probes = [i for i in ids if grid[i].rate == round(2*floor, 12)
              and grid[i].z_basis_probability in (0., .5, 1.)]
    if previous_policy not in ids:
        raise ValueError('Previous frozen policy must belong to the expanded grid')
    shortlist = sorted(set(ordered_a[:2]+ordered_b[:2]+[selected, previous_policy]+probes))
    return {'floor': floor, 'winner_a': ordered_a[0], 'winner_b': ordered_b[0],
            'winner_pooled': selected, 'previous_policy': previous_policy,
            'shortlist': shortlist, 'independent_searches_agree': ordered_a[0] == ordered_b[0]}


def exact_interval(alarms, trials, alpha):
    if (isinstance(alarms, bool) or isinstance(trials, bool)
            or not isinstance(alarms, int) or not isinstance(trials, int)
            or not 0 <= alarms <= trials or trials < 1 or not 0 < alpha < 1):
        raise ValueError('Invalid binomial interval inputs')
    ci = binomtest(alarms, trials).proportion_ci(1-alpha, method='exact')
    return [float(ci.low), float(ci.high)]


def grid_envelope(pooled_intervals, evaluation_intervals, eligible, selected, tolerance=.05):
    """Joint-event bracket on min power over the declared finite grid.

    Every grid policy has a search lower bound. Any evaluated subset policy has
    an upper bound on the grid minimum. Evaluation choices may depend on search
    because interval coverage is simultaneous and fresh evaluation conditional.
    A grid lower bound is NOT a lower bound for the continuous attack family.
    """
    if (not eligible or selected not in evaluation_intervals
            or any(i not in pooled_intervals for i in eligible)
            or not evaluation_intervals or not set(evaluation_intervals).issubset(eligible)
            or not 0 <= tolerance <= 1):
        raise ValueError('Incomplete or inconsistent finite-grid evidence')
    for interval in list(pooled_intervals.values())+list(evaluation_intervals.values()):
        if len(interval) != 2 or not 0 <= interval[0] <= interval[1] <= 1:
            raise ValueError('Invalid probability interval')
    lower = float(min(pooled_intervals[i][0] for i in eligible))
    upper = float(min(ci[1] for ci in evaluation_intervals.values()))
    consistent = bool(lower <= upper)
    regret = float(max(0., evaluation_intervals[selected][1]-lower)) if consistent else None
    return {'grid_minimum_joint95': [lower, upper], 'intervals_consistent': consistent,
            'selected_regret_ceiling': regret, 'tolerance': tolerance,
            'selected_within_tolerance_certified_on_joint_event': consistent and regret <= tolerance}
