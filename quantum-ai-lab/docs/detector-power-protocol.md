# Step 7 frozen protocol: detector-specific continuous lower power bounds

Frozen before new simulations on 2026-10-09. Research only: unchanged central
exact detector, no threshold fitting, production/PDF changes, commits or pushes.
This is a bounded attempt to tighten Step 6's vacuous full-family bounds, not a
promise of balanced optimality or publication novelty.

## Model and question

Keep B=4,000,000, lambda=.03, eta=.15, alpha=.05, stable attack-free calibration,
uniform independent bits/bases/intensities, basis-independent gains/disclosure,
and iid intercept-resend (IR). Use existing equal-expected-sacrifice plans 9
(M=800,000, f=.375) and 10 (M=1,333,333, f=.25000009375). These plans are fixed
in advance, not optimized on this study. Floors kappa=.0025,.005,.01,.02 constrain
r in [2*kappa,1], beta in [0,1]; r/2 is expected certified raw-bit knowledge,
not total information or final-key leakage. No PNS, device drift, correlated
Eve, hardware validation or complete QKD security is covered.

Can a detector-specific continuous lower envelope replace the loose KL cover?
How close can we certify the balanced floor-rate policy to the actual continuous
minimum? The predeclared near-worst tolerance is .05 absolute probability (five
percentage points). Failure to certify does not prove the policy is suboptimal.

## Calculation fixed before results

Condition on public sample denominators and intensity sent/detected counts.
For each error stratum, A~Bin(m,q0), B~Bin(n,q), independently, q0=.015.
The central conditional rule rejects when either hypergeometric inclusive tail
is <=alpha/12. For fixed A=a the increased-monitoring-error rejection event is
B>=u(a); the decreased-error event is B<=l(a). Find both integer boundaries
by monotone binary search, retaining absent-event sentinels n+1 and -1.

Integrate calibration errors and monitoring binomial tails analytically. Keep
calibration support covering at least 1-1e-12 of its mass; explicitly propagate
omitted mass into point upper bounds. Conditional error strata are independent,
but their random denominators are not. Combine conditional coordinate alarm
probabilities by 1-product(1-p_j), then average over ancillary count samples.
Gain alarms are deterministic given the ancillary counts and are included.
Never replace all random denominators by their means.

The increased-error subevent is contained in the unchanged full alarm and is
monotone in each monitoring q, by a common-uniform Bernoulli coupling. For every
r>=2*kappa and beta in [b_lo,b_hi], use qZ>=q0+.485*2*kappa*(1-b_hi) and
qX>=q0+.485*2*kappa*b_lo. Integrating only increased-error tails gives a
pointwise lower function of ancillary counts valid across the whole rate range.
Use 16 equal closed bias cells over [0,.5]; Step 6's analytic reflection law
covers [.5,1]. This does NOT assert monotonicity of full two-sided power.

## Randomness, confidence and direct validation

Seed 2026100907. Per plan, draw 1024 independent ancillary pairs with
SeedSequence([seed,0,plan_id,0]) spawned calibration then monitoring. Reuse them
across all frozen floor/cell calculations: correlation between functions is
allowed; rows within each function must remain iid. Preserve counts, cutoffs,
weights, conditional lower functions and balanced full-alarm upper/lower functions.
Simulation-side probabilities belong only to power analysis, not alarm inputs.

There are 2*4*16=128 continuous-cell lower functions and 2*4*2=16 balanced
point functions: 144 one-sided mean bounds. Give them total failure budget .04,
each delta=.04/144. For bounded observations Z in [0,1], use the one-sided
Maurer-Pontil empirical Bernstein radius

    sqrt(2*s^2*log(2/delta)/T) + 7*log(2/delta)/(3*(T-1)),

with unbiased sample variance s^2. Apply the theorem to 1-Z for lower endpoints
and to Z for upper endpoints; do not present these as per-function two-sided
intervals. The 2009 paper's Theorem 4 controls each side; the union bound controls
the fixed family. Primary reference:
https://www.cs.mcgill.ca/~colt2009/papers/012.pdf . These are standard methods.

For each plan/floor, draw 8192 fresh paired public transcripts at r=2*kappa,
beta=.5 using SeedSequence([seed,1,plan_id,floor_id]) with stage children. Also
draw 8192 stable honest pairs per plan from [seed,2,plan_id,4]. Score only public
counts using the unchanged full detector. All 10 rates receive two-sided exact
binomial intervals with total failure budget .01, each .01/10. Combined with
the analytic-function sampling bounds the declared event has probability at
least .95 in real arithmetic. Detector false-alarm alpha is a separate budget.
No result-dependent enlargement, stopping, policy substitution or plan selection.

For each plan/floor report [min_cell L_cell, U_balanced], a bracket for the
continuous minimum, and U_balanced-min_cell L_cell as an upper bound on balanced
regret. Retain a zero lower endpoint if the calculation is uninformative. Compare
balanced analytic brackets with fresh direct intervals without requiring their
point estimates to match. Report every failed consistency check.

## Validation and numerical limits

Exhaustively enumerate small integer tables using rational hypergeometric masses;
check conditional integration, both boundary inclusions, zero counts, endpoint
probabilities, reflection, truncation, gain alarms and invalid inputs. Check
continuous lower functions against direct conditional full power at interior and
high-rate policies. Audit every saved boundary, mass, confidence endpoint and
envelope; independently sample scalar Fisher boundaries and incomplete-beta power
calculations. Shared SciPy numerics are not independent hardware/physics evidence.

Hash the frozen protocol, sources and archives. Do not overwrite prior outputs.
The distribution identities and coupling bounds are real-arithmetic statements;
floating binomial/hypergeometric computations are not outward-rounded numerical
security certificates. No claim of exact balanced optimality or exhaustive
literature novelty follows even if the five-point regret gate passes.
