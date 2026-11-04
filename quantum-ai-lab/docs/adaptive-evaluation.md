# Adaptive-evaluation milestone: repaired API, benchmark pending

This milestone repairs the restricted adaptive eavesdropper experiment. It does
not establish novel physics, final secret-key security, a globally optimal Eve,
or improved ML performance. The version-2 physics specification is in
[physics-model.md](physics-model.md). Final PDFs, LaTeX reports and old saved
figures have **not** been regenerated.

Verification: **161 tests passed** (91 physics and 70 adaptive regressions) with
the pinned Python 3.12/macOS arm64 environment. Notebook schema/Python syntax and
dependency consistency also passed. Full historical notebook execution remains
pending; the integration test is not a publishable performance benchmark.

## 1. Defects repaired

| Previous problem | Current behaviour |
| --- | --- |
| Independent zeroing of IR/PNS created 25% null policies among positive labels | Explicit IR, PNS or mixed families, with strictly positive active strengths |
| PNS pulse-mixture fraction was treated as Eve's sifted information fraction | Gain-weighted information equation solved for each candidate |
| Candidates were not checked against the target | Check long-run expected information within 1e-10 before sampling |
| Best candidate's selection rate was also reported as final performance | Freeze policy using selection data, then evaluate on independent sessions |
| All candidates at detection rate 1 left a null selected policy | The first usable feasible candidate is selected, even at rate 1 |
| False alarms assumed equal to nominal calibration target | Separate honest-test rate and 95% Wilson interval recorded |
| Short sessions were silently dropped, sometimes yielding empty-array/NaN behaviour | Explicit rejection counts, bounded training retries, empty-sample errors |
| Different detectors' achieved information was averaged into one x-axis | Detector-specific observed pooled information coordinates |
| Missing basis data was assigned zero error rate | Reject undefined telemetry instead of fabricating measurements |
| PNS feasibility formula was duplicated | Use the shared validated physics implementation |

Positive means a nonzero *attempted attack policy*. In a finite session it does
not guarantee that Eve intercepted a detected photon. That distinction is not a
label bug; stochastic outcomes must not be used to redefine the training label.

## 2. Information definition and derivation

Eve's information proxy is the fraction of **Alice's raw bits known among
signal and decoy sifted observations**. Decoys are included to retain the existing
experiment's definition. A deployed protocol normally removes decoy observations
from key generation. Therefore this is deliberately **not** a signal-key leakage
fraction, mutual information, Holevo bound, or post-privacy-amplification secret
key leakage. The channel may flip Bob's bit even when Eve knows Alice's.

Let d be the decoy probability (0.5), mu_s/mu_d the signal/decoy intensities
(0.5/0.1), and eta the per-photon transmission/detection probability (0.15).
The honest gains are Q_s = 1-exp(-mu_s eta) and Q_d = 1-exp(-mu_d eta).
Let q be the validated PNS multiphoton forwarding probability and
M_d = 1-exp(-mu_d)(1+mu_d) the decoy multiphoton probability.

Define the intensity-averaged detection probabilities:

$$A=(1-d)Q_s+dQ_d,\qquad B=(1-d)Q_s+dqM_d.$$

A applies to ordinary/IR pulses; B applies to PNS pulses. PNS is assigned with
probability rho. Conditional on not being PNS, IR is assigned with probability r.
Eve chooses basis Z with probability beta; Alice and Bob each choose bases
uniformly and independently. Eve's basis matches Alice with probability
beta/2+(1-beta)/2=1/2, so IR knowledge probability is r/2, independent of beta.

Sifting also occurs with probability 1/2. Expected known sifted observations per
emitted pulse are

$$N=\tfrac12[\rho B+(1-\rho)Ar/2].$$

Expected total sifted observations per pulse are

$$D=\tfrac12[\rho B+(1-\rho)A].$$

The long-run fraction is the ratio I=N/D, **not** the finite-sample expectation
of a ratio conditional on session acceptance:

$$I=\frac{\rho B+(1-\rho)Ar/2}{\rho B+(1-\rho)A}.$$

Set h=r/2. Multiplying by the denominator and collecting terms gives

$$I\rho B+I(1-\rho)A=\rho B+(1-\rho)Ah,$$
$$A(I-h)(1-\rho)=\rho B(1-I),$$
$$A(I-h)=\rho[B(1-I)+A(I-h)],$$
$$\boxed{\rho=\frac{A(I-h)}{B(1-I)+A(I-h)}}.$$

The solver requires h <= I. At I=1 it chooses pure PNS. At I=0 it chooses the
honest baseline. Pure IR can only reach I<=0.5; it is **excluded**, not clipped,
above 0.5. For I in (0,1), mixed candidates sample r within [0,min(1,2I)]
and beta within [0.5,1] and solve rho. This is a restricted policy family, not
all physically allowed attacks; bases below beta=0.5 are not searched.

For these fixed source parameters:

| Policy / target | Analytical long-run value |
| --- | --- |
| Pure PNS pulse fraction rho=0.4 | I=0.3676661787 |
| Desired I=0.4, pure PNS | rho=0.4332291765 |
| Pure IR r=0.6, any beta | I=0.3 |

These are analytical model checks, not new detector benchmark results.

The evaluation reports **pooled observed information**
sum(known bits)/sum(sifted bits). Averaging per-session information ratios would
weight small and large sessions equally and answer a different question.
Observed information can differ from the requested long-run target due to finite
sampling and acceptance conditioning. We do not discard evaluation sessions
because their realized information is inconvenient. If a future question requires
a high-probability finite-session leakage constraint, that needs a different
constraint, estimator and validation protocol.

## 3. Evaluation flow and random streams

1. Generate labelled training sessions. A nonzero family is assigned before
   simulation. Collect the exact requested number of usable rows with bounded
   retries; record attempts/rejections.
2. Fit each fixed-feature random forest on training rows only.
3. Calibrate each threshold on independent honest calibration sessions.
4. Audit frozen thresholds on separate fresh honest-test sessions.
5. Generate expected-information-feasible attack candidates.
6. Select each detector's best *found* policy using search sessions only.
7. Freeze those policies and test them on new sessions. **Never minimize the
   final evaluation rates or use them to expand/select the policy search.**

NumPy SeedSequence children separate calibration and honest testing. Another
three-child hierarchy separates candidate generation, selection sessions and
final evaluation. Final children are indexed by candidate, so detector iteration
order does not change a selected policy's evaluation samples. Detectors that
select the same candidate share its final batch. Root entropy and spawn keys are
stored in the returned records. Full reproducibility also requires the source
revision, parameters and pinned environment. Independent seeds are not a cure
for test-set reuse across repeated human tuning.

See [NumPy's parallel random-stream documentation](https://numpy.org/doc/stable/reference/random/parallel.html).
Models use a fixed forest seed 0 and 300 trees by default; integration tests use
smaller forests for speed. A fixed seed is not evidence of low training variance.

## 4. Rates, confidence intervals and usable sessions

The feature vector is total QBER, absolute Z/X QBER difference, signal gain and
decoy/signal gain ratio. A usable session must contain at least 20 sifted
observations, both sifted bases, both emitted intensities, and nonzero detected
signal gain. Otherwise the needed features are undefined/too sparse.

All returned alarm rates have denominator **accepted sessions**, not attempted
sessions. attempted = accepted + rejected. Rejection can depend on attack,
channel conditions and pulse count. Rates must be described as conditional on
usable telemetry; silence/rejection itself is not treated as a security alarm.
A deployment would need an explicit fail-closed/fail-open policy. The code must
not hide this choice by silently counting rejected sessions as detected or safe.

Training retries are globally bounded (default ten times requested rows).
Calibration, honest-test, search and final evaluation use fixed attempt budgets
and retain their acceptance counts. Empty usable batches raise errors, not NaN
or a misleading zero rate.

Scores are forest positive-class probabilities. A threshold is the calibration
(1-fpr) quantile and flags use **strictly greater than**. Forest score ties and
finite calibration size mean the realized operating FPR need not equal fpr.
The stored nominal_fpr is a target, calibration is an in-sample calibration
diagnostic, and honest_test is the independent audit. No threshold is adjusted
using that audit.

For k flagged usable sessions out of n, p=k/n. Let z be the standard-normal
97.5th percentile for a two-sided 95% interval, approximately 1.96. Inverting
the binomial score inequality gives the Wilson interval with

$$c=\frac{p+z^2/(2n)}{1+z^2/n},\qquad
 w=\frac{z\sqrt{p(1-p)/n+z^2/(4n^2)}}{1+z^2/n},$$

and endpoints max(0,c-w), min(1,c+w). Zero observed alarms are not proof that
the true alarm probability is zero. These are per-point conditional intervals
for independent sessions and frozen models/policies. They do not include model
training variation, adaptive reuse of final test sets, simultaneous coverage
over a frontier, or uncertainty that the restricted search missed a better Eve.
No confidence interval for secret-key leakage is claimed.

## 5. API migration and notebook provenance

`best_response(...)` now returns SearchResult rather than tuples. Its responses
contain the selected policy, expected information, selection rate estimate,
fresh evaluation estimate, observed pooled information and raw counts.
`frontier_curves(...)` now returns FrontierResult with:

- target_infos and full searches;
- detection_rates[name] from fresh evaluation;
- achieved_infos[name] for that detector's selected policy;
- detection_intervals[name] for plotting.

The legacy `stealth_floor_curve` name remains for compatibility. It returns the
old tuples by default; return_details=True adds actual honest-test rates and
attack intervals/counts. Its x-axis is **emitted pulse count**, not sifted
length or final secret-key length. Retraining at every size confounds sample
size with model estimation and is not a finite-key security proof.

Notebook cells 0 and 24–29 were archived, with original outputs and source, in
`notebooks/archive/adaptive-v1-cells.json`. Changed code outputs/execution stamps
were cleared. The migrated plotting code uses per-detector coordinates and
Wilson alarm-rate bars, removes “invisibility floor” claims, audits FPRs, and
uses new filenames rather than overwriting historical figures.

Notebook schema and Python syntax are validated. Real-Aer Monte Carlo checks
and a small actual-forest/search integration run are tested. **The full notebook
has not been executed in this milestone**: historical earlier chapters also
need benchmark-method repairs; the minimal environment does not contain the
full plotting/Torch/Jupyter stack. To execute later, install the full notebook
dependencies, use a clean kernel, and rerun after the benchmark design is fixed.
An exploratory rerun alone does not turn historical claims into valid claims.

## 6. Verification and next milestone

From repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests exercise the information constraint, pure/mixed policy bounds and
endpoints, real vectorized physics, deterministic replay, non-null positive
labels, bounded retries, accepted/rejected bookkeeping, calibration/test
separation, final evaluation stream separation, all-detected candidates,
infeasible/empty searches, detector-order invariance and plotting coordinates.
Mock sampling tests cover plumbing; separate tests use real Aer probabilities
and analytical physical references. The environment is pinned for Python 3.12
on macOS arm64, not a claim of identical floating-point output across platforms.

The subsequent [fixed-policy benchmark](fair-benchmark.md) now provides fresh
five-seed physical/statistical-versus-ML results with a common calibration target,
actual FPR audits and row-level evidence. It is separate from an adaptive search
rerun and does not establish exact equality of operating FPRs.

The subsequent [robustness benchmark](robustness-benchmark.md) adds fresh-seed
noise/window shifts and attack strengths outside training, without retuning.
It uses fixed policies, not a new adaptive search. Remaining benchmark work
includes repeated training/search seeds; expanded search budgets; realistic
loss/device changes; nuisance-aware monitoring; and honest-vs-attack temporal
evaluation. Predefine
what wins, then keep a final test cohort untouched. No new headline recall,
novelty claim or project rating is justified by these repairs alone.
