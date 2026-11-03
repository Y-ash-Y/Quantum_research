# Honest loss/device stress test — predeclared protocol

## Question and fixed design

Do the twelve frozen nominal detectors mistake honest channel loss or local
gate errors for an attack? This milestone tests **honest false alarms only**.
No attack power, AUC, adversarial optimum or repaired-detector claim is inferred.
The protocol is fixed before the full run; outcomes are appended after audit.

- Fresh seeds: **1069, 2069, 3069, 4069, 5069**.
- Per seed, fit the nine legacy scorers on 300 honest and 600 attack sessions
  using the unchanged nominal attack simulator (6000 pulses, lambda=.03, eta=.15,
  ideal local gates). This preserves its training interface and distributions.
- Generate 1000 independent nominal honest calibration sessions using the new
  aggregate-count honest sampler. Calibrate the same eight rank-cutoff methods
  at 5%; retain the 11% QBER reference. Freeze all learned models, reference
  distributions and cutoffs before evaluation.
- The three exact count tests remain unchanged, with analytic alpha=.05 and
  **nominal trusted decoy probability** 1-exp(-.1*.15). They receive measured
  counts, not test-time eta, gate noise, condition labels or private information.
- Each condition gets 1000 attempted honest test sessions per seed. No model or
  cutoff is retuned on this holdout; every method sees the same accepted rows.

| Condition | eta | gate-noise g | channel lambda | emitted pulses |
| --- | --- | --- | --- | --- |
| Nominal | .15 | 0 | .03 | 6000 |
| Lower transmissivity | .10 | 0 | .03 | 6000 |
| Higher transmissivity | .18 | 0 | .03 | 6000 |
| Mild gate errors | .15 | .01 | .03 | 6000 |
| Stronger gate errors | .15 | .02 | .03 | 6000 |
| Combined loss/gates | .10 | .02 | .03 | 6000 |
| Combined shorter window | .10 | .02 | .03 | 3000 |

These are separate stationary cohorts, not trajectories or a physical device
calibration. There will be **44,500 saved rows** if all attempts are usable:
4500 training + 5000 calibration + 35000 evaluation. Record rejected attempts
and accepted attempt IDs; alarms are conditional on usable telemetry. Empty
usable cohorts are errors, not zero-valued measurements. A smaller window may
produce occasional rejections; do not conceal them or claim unconditional bounds.

## Honest count model and independent references

This new sampler models only an **honest** source/channel. It analytically
marginalizes Poisson photon numbers and uniform bit/basis choices; it does not
fabricate counts by multiplying rounded feature ratios. For an independent
phase-randomized Poisson source and intensity-independent photon loss:

`Q_mu=1-exp(-mu*eta)`.

Draw signal/decoy emitted counts from Binomial(N,.5); draw click counts in each
intensity class from independent binomials at Q_signal and Q_decoy. Allocate
the actual total clicks into Z sifted, X sifted and discarded unmatched bases
with a multinomial at probabilities (.25,.25,.5). Draw per-basis errors
conditionally on actual sifted counts at the bit-averaged matched-basis error
probabilities. This is an exact distributional marginal for the retained eight
counts under the stated independent model, not a pulse-record or optical-state
simulator. Its random draw order differs from the old pulse simulator; nominal
equivalence is distributional, not seed-by-seed bitwise identity.

The gate model is the existing Aer depolarizing error after each local X/H gate.
The explicit transmission channel is still applied once between preparation
and measurement. Let a=1-g, with g the per-gate mixing strength. In matched Z,
bit-zero/bit-one paths use 0/1 local gates; in matched X they use 2/3 including
Bob's basis rotation. Depolarization shrinks Bloch-vector length by 1-lambda
and each gate error contributes a. Thus each bit's error probability is
`[1-(1-lambda)*a^k]/2` for gate count k. Averaging uniformly over bits gives:

`q_Z=[2-(1-lambda)*(1+a)]/4`

`q_X=[2-(1-lambda)*(a^2+a^3)]/4`.

When g=0 both are lambda/2. Otherwise basis-dependent honest errors are expected;
Fisher's common-error null is violated even though no Eve exists. The production
sampler gets q_Z/q_X from real Aer density matrices; tests and the artifact
auditor use these independently derived closed forms. Gate parameters are
illustrative, not measured device noise. Source intensities stay .5/.1; no dark
counts, afterpulsing, dead time, basis-dependent efficiency or intensity drift
are modelled. Public parameter-estimation disclosure costs remain unmodelled.

Primary definitions: honest FPR=flagged/accepted; attempted/rejected counts are
retained. Per-seed Wilson 95% intervals are pointwise. Means and sample SD across
seeds are replicate descriptions, not confidence intervals or significance
tests. No model is declared universally superior from seven selected scenarios.
Physical pooled gains and basis QBERs are checked against independent formulas;
zero observed errors stay zero, not missing. The exact-test <=5% null bound is
**not** asserted when its trusted-loss or common-error assumptions are violated.

## Reproduction and scope

From repository root, use a new output directory:

```sh
.venv/bin/python quantum-ai-lab/scripts/run_nuisance_stress.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-nuisance-run
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.nuisance_audit \
  quantum-ai-lab/results/benchmarks/my-fresh-nuisance-run
.venv/bin/python -m pytest -q
```

Existing results and attack-simulator sources are not rewritten. The new artifact
saves source hashes, versions, seed spawn keys, count/feature digests, flags,
scores, theoretical references and independent pooled physical diagnostics.
The independent auditor enumerates integer Fisher weights, explicitly sums
binomial probabilities and recomputes cutoffs/rates/intervals/seed summaries.
No AUC is generated for an honest-only comparison.

The next design, after this evidence, must define trusted source/loss/device
bounds or a separately collected trusted reference. Estimating an honest null
from potentially attacked traffic can absorb the attack signature. Do not add
a correction after seeing these test rows and relabel the evaluation untouched.
Git commits/pushes, final-report PDFs and notebook reruns are outside this milestone.

References: [Aer depolarization definition](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html)
and [NumPy multinomial sampling](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html).

## Completed results (2026-10-03 UTC)

The full run accepted **44,500 sessions**, with no rejected attempts. Of these,
4500 are unchanged nominal training-interface rows, and 40000 calibration/test
rows have actual aggregate draws of all eight integer counts. The artifact was
created at `2026-10-03T18:21:19.745938+00:00`. See
[nuisance.json](../results/benchmarks/nuisance-v2/nuisance.json) and the
[compressed session evidence](../results/benchmarks/nuisance-v2/session_audit.csv.gz).
Raw records include true conditions for audit only; production scorers do not
receive eta, gate_noise or evaluation labels.

Independent audit passes **480 rate checks, 120000 exact p-value checks and
160 physical reference checks**, plus source/raw hashes, counts, ratio
reconstruction, honest-only labels, frozen cutoffs, rank calibration, Wilson
intervals and replicate summaries. Closed-form q_Z/q_X agree with the production
Aer references within 1e-12. The largest absolute pooled physical z-score is
**2.618** over unrejected cohorts, against four expected error/gain probabilities
per cohort. This descriptive agreement is not a multiplicity-adjusted statistical
test or a real-device validation. No cohorts needed exclusion from this diagnostic.

The full suite passes **275 tests**. The added tests exercise independent physical
formulas, sampler moments and covariance, nominal pulse-versus-aggregate moment
checks, local RNG isolation, rejection bookkeeping, no silent overwrites, null
freezing, corrupted count/label/setting rejection and independent artifact replay.
Earlier benchmarks and attack-simulator code are unchanged; their numerical
results are not replaced with this honest-only experiment.

### Honest false alarms: transmission changes

Every rate table uses **percent, mean ± sample SD across five seeds**, not
confidence intervals. The raw artifact retains pointwise Wilson intervals and
all seed rates. There are 1000 accepted sessions per condition per seed.

| Method | Nominal eta=.15 | Lower eta=.10 | Higher eta=.18 |
| --- | --- | --- | --- |
| Calibrated QBER | 4.88 ± 0.61 | 9.42 ± 0.68 | 4.24 ± 0.88 |
| Basis asymmetry | 5.32 ± 0.99 | 10.06 ± 1.37 | 3.22 ± 0.88 |
| Direct decoy ratio | 4.90 ± 0.70 | 9.50 ± 1.56 | 3.32 ± 1.36 |
| Joint max-tail | 5.22 ± 1.03 | 99.82 ± 0.25 | 64.98 ± 6.11 |
| Joint tail-sum | 5.40 ± 1.15 | 68.34 ± 10.89 | 36.36 ± 4.97 |
| Logistic regression | 6.02 ± 0.93 | 51.92 ± 19.45 | 0.38 ± 0.33 |
| Random forest | 4.50 ± 0.59 | 11.72 ± 3.29 | 0.84 ± 0.27 |
| RBF SVM | 5.16 ± 0.79 | 44.42 ± 38.98 | 1.96 ± 3.55 |
| QBER 11% reference | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Count Fisher basis | 0.48 ± 0.24 | 0.12 ± 0.13 | 0.42 ± 0.22 |
| Count binomial decoy | 3.82 ± 0.51 | 75.20 ± 1.31 | 0.14 ± 0.11 |
| Count Bonferroni joint | 2.06 ± 0.42 | 63.84 ± 0.65 | 0.14 ± 0.11 |

The standalone binomial and joint count tests retain eta=.15 in their null,
but honest eta=.10 lowers decoy click probability from .0148880604 to
.0099501663. A lower-tail test for depleted decoy counts then alarms frequently.
This is **null misspecification**, not an arithmetic bug or proof of PNS. Higher
eta increases rather than depletes decoy counts, so the one-sided count test
has very few alarms; it is not a symmetric channel-health monitor.

Gain is included as two-sided deviation in the legacy joint-tail rules. A large
honest loss change therefore correctly looks like a channel change, but calling
every such alarm an attack is incorrect. Joint max-tail's 99.82% honest alarms
at eta=.10 expose that interpretation problem. QBER/asymmetry means stay fixed
without gate errors, but fewer detections widen their sampling distributions.
The ratio baseline is much less affected than the trusted count null, yet its
9.50% observed FPR still exceeds the nominal operating target. No detector has
been selected or tuned using these results.

### Honest false alarms: gates and combined changes

| Method | Gates .01 | Gates .02 | eta=.10, gates .02 | Same, 3000 pulses |
| --- | --- | --- | --- | --- |
| Calibrated QBER | 17.70 ± 0.92 | 35.00 ± 2.08 | 37.10 ± 1.81 | 36.86 ± 2.14 |
| Basis asymmetry | 11.66 ± 1.46 | 22.42 ± 3.36 | 29.04 ± 1.80 | 40.74 ± 1.39 |
| Direct decoy ratio | 5.72 ± 1.08 | 5.30 ± 1.04 | 10.52 ± 1.11 | 18.76 ± 1.38 |
| Joint max-tail | 12.34 ± 2.52 | 26.44 ± 4.02 | 99.86 ± 0.22 | 99.02 ± 0.81 |
| Joint tail-sum | 14.20 ± 1.59 | 27.42 ± 2.81 | 87.94 ± 5.83 | 79.18 ± 1.95 |
| Logistic regression | 13.90 ± 0.83 | 22.40 ± 1.76 | 71.84 ± 12.49 | 65.16 ± 10.95 |
| Random forest | 11.98 ± 2.52 | 22.98 ± 2.37 | 30.24 ± 2.39 | 39.82 ± 3.21 |
| RBF SVM | 11.32 ± 1.70 | 18.54 ± 3.49 | 50.04 ± 41.12 | 51.50 ± 38.55 |
| QBER 11% reference | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.80 ± 0.37 |
| Count Fisher basis | 1.24 ± 0.24 | 3.30 ± 0.76 | 1.34 ± 0.38 | 0.30 ± 0.12 |
| Count binomial decoy | 4.72 ± 0.79 | 4.28 ± 0.72 | 75.12 ± 0.91 | 46.56 ± 1.27 |
| Count Bonferroni joint | 2.70 ± 0.33 | 3.30 ± 0.56 | 63.48 ± 0.51 | 33.80 ± 0.96 |

At gate mixing g=.01, honest expected basis QBERs are
q_Z=.017425 and q_X=.0270282425; at g=.02 they are
q_Z=.019850 and q_X=.0388639400. These are consequences of different local gate
counts, not Eve. Unlike the version-1 transmission-noise placement bug, the new
device errors are intentional, explicitly separate, independently checked and
reported as benign device effects.

Fisher/joint count alarms remain below 5% on average for both gate-only cases.
That empirical fact must be retained: this run does **not** establish a large
gate-only FPR failure for those methods. Nevertheless q_Z != q_X means their
common-error null is false; its exact <=5% guarantee cannot be applied. Sparse
errors and conservative discrete cutoffs limit power against modest asymmetry.
Higher device error, more pulses or different basis effects could produce a
different rate. No such untested extension is claimed here. Combined-loss count
alarms are largely explained by decoy-null misspecification; the shorter window
has less power to reject that null, not necessarily a better detector.

### Defensible conclusion and next design

**Share with caveats:** the specified simulation and calculations are audited,
but the present nominal monitoring design is not suitable for arbitrary loss or
device drift. Channel-health alarms and attack-specific evidence must not be
conflated. The run adds a reproducible failure-mode study, not an algorithmic
discovery, cryptographic theorem or improved attack detector.

Next, specify a joint signal/decoy count test under a shared unknown honest loss
parameter, with honest-error assumptions handled separately. A signal gain can
constrain honest decoy gain through Q_d=1-(1-Q_s)^(mu_d/mu_s), but uncertainty in
Q_s must be propagated, not plugged in as if exact. Such a model can still
absorb attacks whose counts fit its honest family; it is not a security
certificate. Compare new designs against existing methods on **fresh** honest
and attack cohorts, reporting both FPR and power, with no test-driven cutoff
changes. A realistic disclosed parameter-estimation sample remains separate work.
