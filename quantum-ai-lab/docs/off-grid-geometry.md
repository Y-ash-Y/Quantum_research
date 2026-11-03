# Step 6: what symmetry and off-grid evidence actually establish

Research-only continuation of [Step 5](best-response-stability.md), completed
2026-10-05. [Protocol](off-grid-protocol.md) was frozen before simulations.
[Results](../results/benchmarks/off-grid-geometry-v1/geometry.json) and
[audit](../results/benchmarks/off-grid-geometry-v1/audit.json) are preserved.
The six-coordinate central-exact detector, alpha=.05, budget and physical
assumptions are unchanged. No production/PDF/Git-history updates.

## Outcome in plain language

The model has exact Z/X reflection symmetry. Balanced Eve also minimizes a
specific information-distance proxy and that proxy increases with interception
rate. Neither statement establishes the minimum *actual detector power*.
Fresh nearby off-grid probes found no statistically significant lower-power
counterexample in 16 predeclared contrasts, but do not prove one cannot exist.

We now have a mathematically valid way to extend grid lower bounds to continuous
attack regions. Its current full-family lower endpoint is **zero for both main
allocations at every floor**. That means the certificate is uninformative, not
that actual detection is zero or that a perfectly invisible attack was found.
This exposes the remaining gap between strong finite-grid results and a true
continuous worst-case result. It is useful project validation, not verified
publication novelty.

## Definitions and model

Let B be total emissions, M calibration emissions, N=B-M monitoring emissions,
and f the monitoring disclosure probability. Write e=fN for monitoring test
exposure, measured in emission equivalents, not realized disclosed bits. The
mean click probability is

$$G=\tfrac12[(1-e^{-.5\eta})+(1-e^{-.1\eta})],\qquad \eta=.15.$$

The two intensities .5 and .1 are chosen independently with equal probability.
Detection is independent of Alice's bit/basis and Eve. Uniform independent bases
give a matched (basis, bit) class probability 1/8 conditional on a click: half
the clicks have mismatched bases; the other half split equally among four classes.
Thus expected disclosed counts per error coordinate are MG/8 and eG/8.

Let lambda=.03 be depolarizing channel noise, q0=lambda/2=.015, r the interception
probability, and beta the probability Eve measures Z. Set c=(1-lambda)/2=.485.
The attacked error probabilities are

$$q_Z=q_0+c r(1-\beta),\qquad q_X=q_0+c r\beta.$$

Both Alice-bit strata in a basis share its q. Calibration is attack-free with
all four probabilities q0. The retained raw-knowledge proxy equals r/2 because
Alice's basis is uniform and loss/disclosure do not select Eve's basis matches.
A floor kappa therefore constrains r>=2*kappa. This proxy is not total mutual
information, public-discussion leakage, or final secret-key leakage.

## Proposition 1: exact reflection symmetry

Replacing beta by 1-beta exchanges q_Z and q_X without changing r/2 or gains.
Define T as the permutation (Z0,Z1,X0,X1) -> (X0,X1,Z0,Z1). Apply T to disclosed
numerators, disclosed denominators and sifted counts in both stages; leave signal
and decoy coordinates fixed. Because class selection is uniform and disclosure
is stratum-independent, T maps the entire public law at beta to the law at
1-beta, not merely their mean errors.

The detector forms a central doubled-tail hypergeometric p-value separately for
each coordinate and alarms if any is <=alpha/6. T only permutes those six
p-values, so the alarm event is invariant. If p(r,beta) denotes its population
alarm probability, then

$$p(r,\beta)=p(r,1-\beta).$$

This reduces a valid search to beta in [0,1/2] under the stated model. It does
not force a minimum at 1/2: a symmetric function can have two off-center minima.
Asymmetric gate error, basis-dependent loss/disclosure, unequal basis choices or
coordinate-specific thresholds would invalidate this argument. The symbolic
law identity is a real-arithmetic result. On every fresh sampled transcript,
permuting the saved counts also preserves the numerical p-values exactly.
Reflected copies are **not** extra independent Monte Carlo observations.

## Proposition 2: balanced Eve minimizes broad-honest KL separation

Kullback–Leibler (KL) divergence measures distributional separation, not alarm
probability. For Bernoulli error probabilities p and q, using natural logarithms,

$$d(p\Vert q)=p\log(p/q)+(1-p)\log[(1-p)/(1-q)].$$

Consider a stable honest Pauli counterpart with one error probability h in a
given basis. Its optimal weighted fit to calibration q0 and attacked monitoring
q solves

$$\min_h\{M d(q_0\Vert h)+e d(q\Vert h)\}.$$

Differentiate explicitly: the derivative of d(p||h) is

$$-p/h+(1-p)/(1-h)=(h-p)/[h(1-h)].$$

The total derivative has numerator (M+e)h-Mq0-eq, so for T=M+e>0 the minimizing
probability is h(q)=(Mq0+eq)/T. The second derivative at the zero is positive;
more generally convexity in h gives the minimizer. For the two attacked basis
errors, this pair is physically realizable by the existing Pauli counterpart
with rate (e/T)r and the same beta. Thus it is not an unconstrained, unphysical
mean fit.

Define J(q)=M d(q0||h(q))+e d(q||h(q)). With
F(t)=t log(t)+(1-t) log(1-t), expansion cancels the cross-entropy terms:

$$J(q)=M F(q_0)+e F(q)-T F(h(q)).$$

Since F'(t)=log[t/(1-t)] and h'(q)=e/T,

$$J'(q)=e[\operatorname{logit}(q)-\operatorname{logit}(h)],$$
$$J''(q)=\frac{e}{q(1-q)}-\frac{e^2/T}{h(1-h)}.$$

Here logit(t)=log[t/(1-t)]. Write w=e/T. Direct expansion gives

$$h(1-h)=(1-w)q_0(1-q_0)+w q(1-q)+w(1-w)(q-q_0)^2
\ge w q(1-q).$$

Substitution shows J''>=0: J is convex. Each basis has two bit strata, giving
the complete paired-public KL separation from this stable counterpart as

$$K(r,\beta)=\frac{G}{4}[J(q_Z)+J(q_X)].$$

The average qbar=(q_Z+q_X)/2=q0+cr/2 is independent of beta. Convexity gives
J(q_Z)+J(q_X)>=2J(qbar), with equality at beta=1/2. For M,e>0 and the interior
noise probabilities here, strict convexity gives a unique balanced minimizer
of **K at fixed positive r**. Also h<=q whenever q>=q0; hence J'(q)>=0.
Both q_Z and q_X are nondecreasing with r, so K is nondecreasing with rate for
every fixed beta (strict for r>0 in the nondegenerate setting). Over r>=2*kappa,
this proxy is minimized at (r,beta)=(2*kappa,1/2).

At zero M/e or zero gain, identifiability degenerates and strictness does not
apply. Most importantly, this convex KL proxy is not the exact-test rejection
probability. A Pinsker *upper ceiling* ordered by KL does not order the detector's
actual power. No inference of continuous detector-optimal Eve is made.

## Proposition 3: a uniform continuous extension, with a visible looseness cost

For two policies a and b, gain/selection/sifting distributions coincide and
calibration laws coincide. Conditional on disclosed counts, the four errors are
independent binomials. Their conditional KL is the sum of n_j d(q_aj||q_bj).
Average over their common count law, E[n_j]=eG/8, to obtain the exact public-law
identity

$$D(P_a\Vert P_b)=\frac{eG}{8}\sum_{j=1}^4d(q_{aj}\Vert q_{bj})
=\frac{eG}{4}[d(q_{aZ}\Vert q_{bZ})+d(q_{aX}\Vert q_{bX})].$$

There is no calibration term because its laws are identical. Private knowledge
labels are not included. Zero disclosure or zero gain makes this KL zero. A
one-emission alphabet enumeration unit test checks intensity, loss, mismatch,
each disclosed stratum and its error outcomes, including the factor 1/8.

Total variation distance TV(P,Q)=sup_A |P(A)-Q(A)| bounds the difference in
probability of any fixed alarm event. Standard Pinsker gives TV<=sqrt(D/2) for
KL in nats. These are established information-theory tools, not new results;
see [Polyanskiy–Wu, Sections 7.2 and 7.4](https://people.lids.mit.edu/yp/homepage/data/itbook-2022.pdf).

For a grid anchor g, fix the first KL argument and let theta vary over its
rectangle. For fixed beta the second-argument errors are affine in r; for fixed
r they are affine in beta. Bernoulli KL is convex in that argument, so successive
one-dimensional convexity bounds its maximum by the maximum at the four corners.
Joint convexity in (r,beta) is **not** assumed; the probabilities are bilinear.
Define R_g=min(1,sqrt(max_corner D(P_g||P_theta)/2)). Every policy in that cell has

$$p(\theta)\ge p(g)-R_g\ge L_g-R_g$$

on Step 5's simultaneous anchor coverage event. With cells covering the domain,

$$\min_{\theta\in\mathcal D}p(\theta)
\ge\min_g\max(0,L_g-R_g).$$

The minimum evaluated anchor upper endpoint still upper-bounds the continuous
minimum because those attacks are admissible. This gives a bracket with the
same inherited joint-95% coverage event as Step 5, in ideal arithmetic. It does
not combine that event with new simulation coverage and silently retain 95%:
the inherited event and fresh-cell event each have failure probability .05,
so their conjunction is only bounded by 90% unless a new joint budget is used.

## Continuous-bound results: do not transplant the grid lower endpoint

The cover uses all 14 previous plans and all four floors. The near-floor domain
is [2*kappa,4*kappa] x [0,1]; the full eligible IR domain is [2*kappa,1] x [0,1].
There are 112 covers and 45 cells each, including all midpoint boundaries. The
last rate cell extends to r=1 for the full domain, giving very large radii.

| Floor | Plan 9 near-floor bracket | Plan 10 near-floor bracket | Plan 9 full-family bracket | Plan 10 full-family bracket |
|---|---|---|---|---|
| 0.25% | 0–9.059% | 0–9.172% | 0–9.059% | 0–9.172% |
| 0.5% | 0–19.659% | 0–20.184% | 0–19.659% | 0–20.184% |
| 1% | 0–65.456% | 7.002–62.977% | 0–65.456% | 0–62.977% |
| 2% | 0–99.985% | 0–99.974% | 0–99.985% | 0–99.974% |

These are numerical diagnostic implementations of the inherited-coverage
brackets, not outward-rounded certificates. In total 12/56 near-floor covers
have positive lower endpoints; only 4/56 full-family covers do. Those four all
belong to plan 13, where no monitoring errors are disclosed: their tiny positive
alarm floor concerns the attack-invariant gain-alarm law, not useful error-based
Eve detection. All 112 brackets are consistent.

**Zero lower bound is not zero power.** A cell's conservative radius can erase a
good anchor lower bound. This is particularly stark at high floors: near-unit
grid detection does not rescue a broad rectangle where R_g=1. The calculation
does not exhibit a new stealth attack. It shows why continuous robustness has
not yet been established by this certificate.

## Fresh confirmation results

All rates here use independent 4096-trial cohorts at the minimum allowed rate.
beta=.4375 is off the previous .125-spaced grid; beta=.5 is balanced. Plan 9
uses 20% calibration, plan 10 one third. No winner was selected on these outputs.

| Floor | Plan 9 beta=.4375 / balanced alarms | Plan 10 beta=.4375 / balanced alarms |
|---|---|---|
| 0.25% | 281 / 292 (6.860% / 7.129%) | 279 / 283 (6.812% / 6.909%) |
| 0.5% | 717 / 688 (17.505% / 16.797%) | 679 / 723 (16.577% / 17.651%) |
| 1% | 2538 / 2434 (61.963% / 59.424%) | 2431 / 2377 (59.351% / 58.032%) |
| 2% | 4095 / 4092 (99.976% / 99.902%) | 4084 / 4086 (99.707% / 99.756%) |

The 336 fresh cell rates have simultaneous exact intervals using .05/336 per
cell. Sixteen fixed contrasts compare either the off-grid bias or a 6.25% higher
balanced rate against balanced floor-rate power. One-sided Fisher tests ask
whether the first has lower true power; Bonferroni-adjusted p-values are all 1
(smallest unadjusted p=.103576). **No significant lower-power counterexample**
was found among these comparisons. This is not evidence of exact equality,
equivalence, or population-optimal balanced Eve. The simultaneous contrast
intervals, not overlapping individual intervals, quantify uncertainty.

At the 1% floor, increasing balanced rate from .02 to .02125 increases observed
power by 7.129 and 8.008 percentage points in plans 9 and 10. Their conservative
simultaneous difference intervals are [1.386,12.837] and [2.238,13.739] points,
both positive. This supports a *local* rate ordering at those two tested cases,
not monotonicity over the entire continuous family. At other floors these
contrast intervals include zero. All contrast values are retained in JSON.

Stress minima are descriptive only: one weak-floor cell at plan 10 has an
off-center observed minimum at beta=.1875, and two plan 9 minima use a rate above
the floor. This is expected to be possible with 512 trials across 36 candidates;
it is not a reason to refreeze confirmation or claim a non-monotonic discovery.

## Validation and engineering boundary

Fresh simulations contain **344,064** public pairs: 147,456 stress and 196,608
confirmation, in four bounded compressed blocks. The audit passes all counts,
all 344,064 original decisions, all 344,064 reflected p-value rows, 1008 sampled
independent scalar rule decisions, 336 cell intervals, 16 contrast calculations,
112 continuous covers and **20,160 corner KL calculations** reconstructed using
`rel_entr` rather than the production KL helper. Nine source hashes, one inherited
summary hash and five output hashes reconcile. Shared scorer replay is not an
independent test implementation; sampled scalar and alternate KL APIs still
share SciPy numerical infrastructure. No arbitrary-precision certification.

The full suite passes **667 tests**, including 21 new cases. Source/budget/
detector parameters were frozen before runs. No trial filtering, threshold
tuning, private-label inputs or holdout-based allocation changes were made.
The unchanged Step 5 source/output hashes were checked before consuming its
anchor evidence. PDFs and interview-book claims have not been refreshed.

From the repository root:

```sh
.venv/bin/python quantum-ai-lab/scripts/run_off_grid_geometry.py --output quantum-ai-lab/results/benchmarks/off-grid-geometry-v1
.venv/bin/python quantum-ai-lab/scripts/audit_off_grid_geometry.py quantum-ai-lab/results/benchmarks/off-grid-geometry-v1 --save
.venv/bin/python -m pytest -q
```

Use a new output directory for reruns; archives are never overwritten.

## What to say in an interview—and what not to claim

“I studied a finite-resource intercept-resend detector under a fully stated
aggregate-count model. I proved a basis-reflection reduction and balanced
minimality for a KL proxy, validated off-grid probes on independent cohorts,
and quantified why the finite-grid lower bound does not yet certify continuous
worst-case power.” This is a defensible scoped project contribution.

Do not claim a newly discovered quantum protocol, quantum ML advantage,
continuous minimax detector, composable secure-key yield, or a verified novel
theorem. Pinsker and convexity are standard; novelty of this particular research
question/result combination has not been established by this milestone.

Next proposed milestone, pending confirmation: derive detector-specific
finite-sample power structure (or an explicit counterexample to balanced/rate
ordering) and tighten continuous coverage. Generic KL radii are currently too
loose; merely adding more random policy search is not the missing proof.
