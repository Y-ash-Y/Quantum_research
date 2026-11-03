# Step 10: drift-aware baseline implemented and tested

Date: 2026-10-09. Status: research implementation/validation milestone complete.
The user approved the minimal [Step 9 model](bounded-drift-model.md). No large
power campaign, threshold fitting, new final-report PDF, commit or push occurred.
Publication novelty and finite-budget practical usefulness remain unverified.

## 1. What changed

New module: [drift_detector.py](../src/research/drift_detector.py).
New tests: [test_drift_detector.py](../tests/test_drift_detector.py).
Existing stationary detector, sampler, physics model and saved Step 7 experiment
were not changed. No additional dependency is required.

The implementation is a confidence-gap **baseline**, not an optimal or newly
invented statistical method. It separates a deployable count-only scorer from
helpers that know the simulated physical parameters.

| Entry point | Accepted information | Responsibility |
|---|---|---|
| `DriftEnvelope(z_error, x_error)` | Predeclared absolute error allowances in [0,0.5] | Immutable configuration; four error allowances and two zero gain allowances |
| `drift_change_alarm(calibration, monitoring, envelope, alpha=..., method=...)` | Paired public count batches, envelope, significance level, interval method | Return one read-only alarm per row; no true noise, policy or private knowledge input |
| `honest_drift_laws(cal_z, cal_x, mon_z, mon_x, envelope, eta=...)` | Simulation parameters only | Construct bit-symmetric, stable-gain honest law pairs; reject inadmissible probabilities/changes |
| `hoeffding_family_power_lower(n_cal, n_mon, envelope, minimum_fraction, channel_noise=..., alpha=...)` | Four-stratum ancillary denominator rows and analysis parameters | Conditional lower function for all eligible fixed-noise IR policies; not an alarm input or a measured rate |

The scorer supports Hoeffding and Clopper-Pearson (CP) intervals through the
existing simultaneous-interval helper. Both have the same size-proof framework;
their actual conservatism and power need not be equal. The conditional lower
function implemented here applies **only to the Hoeffding variant**, not CP.

## 2. Decisions and assumptions

For each of six coordinates, compute a calibration interval [Lc,Uc] and a
monitoring interval [Lm,Um]. Twelve intervals together spend one alpha budget;
each interval receives failure probability alpha/12, split between its tails.
Reject if either Lm-Uc>d_j or Lc-Um>d_j. Equality does not reject.
For Z0/Z1 use d_Z; for X0/X1 use d_X; gains use zero allowance.

An empty denominator produces [0,1]. It does not fabricate an observed zero
error rate. Such an error coordinate cannot trigger a gap alarm, but gain
coordinates may still alarm. Rows are never dropped for zero clicks, zero
disclosure or an undefined private-knowledge fraction.

The proof in Step 9 provides a uniform false-alarm ceiling in real arithmetic
over the declared iid, between-block honest-drift family. Code tests check
selected instances of that proof, not every point of the continuum. Random
denominators are preserved; intervals are valid conditional on their appropriate
denominators and the union bound does not require coordinate independence.

The envelope must be declared before monitoring. It is a trust assumption, not
a learned threshold. The family excludes poisoned calibration, arbitrary
within-block drift, nonuniform bit noise, changing gains/intensities, correlated
attacks, repeated peeking and Eve on concurrently drifting baseline noise.
An alarm is not attribution to Eve or composable secret-key certification.

## 3. Validation evidence

Commands run from the repository root using its existing local environment:

```bash
.venv/bin/python -m pytest -q quantum-ai-lab/tests/test_drift_detector.py
.venv/bin/python -m pytest -q
```

- Focused suite: **105 passed** (0.71 seconds in this validation run).
- Full suite: **802 passed** (18.07 seconds in this validation run).
- All **26** recorded Step 7 source/output fingerprints still match.
- Whitespace validation passed. Virtual environments and historical outputs
  were not added to Git or overwritten.

Tests include:

1. Zero-envelope equivalence to the existing confidence-interval rule; input
   immutability, read-only outputs and rejection of probability/private inputs
   by the alarm API.
2. Strict boundary behavior, including one floating-point step below/above
   the computed interval gap; upward/downward symmetry, basis mapping and
   monotonicity as the permitted drift increases.
3. Zero emissions, zero transmission and no disclosed monitoring errors;
   error allowances do not accidentally permit gain changes.
4. Lawful honest pairs, physical Pauli representatives, immutable envelopes,
   and rejection of out-of-envelope pairs rather than silently clipping them.
5. A constructed attack/honest overlap with identical seeded public observations
   and alarm decisions. Equality of the mathematical laws is proved in Step 9;
   shared-seed replay is an implementation check, not empirical discovery.
6. **48** isolated-coordinate two-stage binomial enumerations: both methods,
   stage sizes (0,0), (1,3), (32,45), (80,100), and six lawful probability/drift
   cases, including increases, decreases and endpoints. Combinatorial probability
   masses verify coordinate false alarms <=alpha/6. They do not represent a
   measured six-coordinate deployment FPR. The full family ceiling follows
   from twelve-interval coverage and the union bound, not multiplying rates.
7. Ten strong-attack rate/bias checks comparing the uniform lower function with
   independently enumerated conditional error-union power. Integer-outcome masks
   use the scalar interval formulas, not the scorer. These are numerical unit
   checks, not the project's weak-floor benchmark or a continuous-optimum proof.
8. Manual power-formula reconstruction, basis reflection, overlap/fully mixed
   limits, missing strata, large count inputs, and invalid shapes, booleans,
   negative values, nonfinite parameters or incompatible methods.

The enumeration and CP calculations still use floating arithmetic and, for
some references, SciPy. They are not independently certified rounding or
hardware evidence. The old 697-test count remains historical Step 7 evidence;
802 is the full count after this milestone.

## 4. Power helper: what its numbers mean

Define gamma=(1-lambda)*kappa-d_Z-d_X. For each row of four positive calibration
and monitoring error denominators, compute Hmax as the largest sum of interval
radii and Smax as the largest reciprocal-denominator sum. Put
t=max(0,gamma/2-Hmax). The lower function is 1-exp(-2*t^2/Smax), using `expm1`
to avoid cancellation for small positive values. Return zero if gamma<=0 or
any error stratum lacks a stage denominator.

This is a conditional function of **realized ancillary counts**, independent of
the observed errors or Eve bias. An unconditional lower bound requires averaging
over their joint law. If estimated with Monte Carlo, that average needs its own
confidence/failure budget; an ordinary sample average is not a certificate.
Replacing denominators with their expectations or transferring this bound to
CP is unsupported.

A zero lower function may mean a loose concentration bound, not zero power.
The exact law-overlap result separately bounds *every* valid alarm's worst-case
power by alpha when its conditions hold. Conditional missing-count zeros alone
are not such a whole-distribution impossibility theorem.

## 5. Complexity and numerical guardrails

All four entry points perform constant work per fixed-size coordinate or O(T)
work/memory for T public/ancillary rows; there is no expansion to individual
emitted pulses. CP quantiles have higher numerical cost than elementary
Hoeffding radii. Arrays remain small count-only records, not statevectors or
quadratic training kernels. No GPU, caching, training or parallel search is
needed for these baseline functions.

The common count validator accepts nonnegative integers up to 2^53-1 and rejects
float/bool count arrays. Supported alpha is [1e-12,1). Probabilities and drift
allowances reject booleans, nonfinite values and out-of-domain inputs. Both
batches must have matching row counts. Power-helper inputs have exactly four
columns; detector batches have the existing six-coordinate integrity checks.

Honest-law validation compares the supplied floating-point differences strictly
with the supplied envelope; there is no epsilon enlargement. A decimal intended
to touch an exact boundary may round outside it. Use clearly interior scenarios
or represent/check the intended boundary explicitly in a future protocol;
silently tolerating extra drift would change the null. Neither those checks nor
the floating interval computations constitute an outward-rounded certificate.

## 6. Next milestone, not yet executed

Freeze a bounded drift-aware pilot **before** generating performance data.
Suggested design: nominal calibration q0=0.015, stable eta=0.15, the same
two fixed expected-sacrifice allocations as Step 7, and predeclared drift-envelope
sensitivity values on both sides of the overlap boundary. Keep a separate
independent honest validation set, include honest envelope edges/decreases,
report both methods' attained false alarms and attack power, and retain zeros
and negative results. No performance target is promised.

The original stationary exact test may be an explicitly invalid-under-drift
diagnostic, not a fair drift-valid competitor. Known-noise oracle thresholds
and the paper's threshold classifications also cannot serve as equally valid
resource-matched baselines without aligning their nulls and objectives.
Choose the finite uncertainty budget, scenario grid and stopping rules in that
protocol; do not tune the envelope from its final evaluation rows.

The next decision is approval to freeze and run that small pilot. Tighter CP
integration, nuisance-optimized exact tests, continuous allocation optimization,
concurrent-noise drift, publication priority and independent expert review
remain later work, not claims established by this implementation.
