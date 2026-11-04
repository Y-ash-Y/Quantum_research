# Fixed-policy fair benchmark: protocol v1

This protocol is fixed before the first benchmark run. It is an internal
predeclared design, not an externally registered study. It evaluates version-2
simulated BB84 telemetry, not hardware or secret-key security.

## Question and scope

Do classical supervised ML models improve on simple statistically calibrated
telemetry rules under the restricted IR/PNS model, when all calibrated methods
use the same nominal honest false-alarm target?

The answer must report **actual honest-test FPRs** alongside detection rates.
Common calibration targets do not make realized operating FPRs exactly equal.
No ML superiority, universal robustness or novelty is presumed. The earlier
0.8% versus 87.4% comparison used different rules and is not this experiment.

## Fixed setup

- Replicate seeds: 11, 22, 33, 44, 55. A replicate regenerates training,
  calibration and test sessions and refits all models.
- 6,000 emitted pulses/session; transmission depolarization lambda=0.03.
  Ideal gates, source intensities 0.5/0.1, decoy probability 0.5 and eta=0.15.
- Per seed: 300 honest + 600 nonzero attempted-attack training rows;
  1,000 attempted honest calibration sessions; 1,000 attempted fresh honest-test
  sessions; 500 attempted test sessions for each of six fixed attack policies.
- Generator acceptance rules are unchanged from the adaptive milestone.
  All rates are conditional on usable sessions; counts/rejections are retained.
- Training contains explicit IR, PNS and mixed families with active strengths
  in [0.02,0.6). It is not exclusively matched to the six test policies.
- No hyperparameter search or tuning on the final tests.

| Fixed test cohort | IR r | Eve Z-basis probability beta | PNS rho |
| --- | --- | --- | --- |
| IR weak | 0.1 | 0.95 | 0 |
| IR medium | 0.3 | 0.95 | 0 |
| PNS weak | 0 | 0.5 | solve for expected I=0.05 |
| PNS medium | 0 | 0.5 | solve for expected I=0.2 |
| Mixed weak | 0.05 | 0.9 | solve for expected I=0.05 |
| Mixed medium | 0.2 | 0.9 | solve for expected I=0.2 |

I is raw Alice-bit knowledge among **signal and decoy sifted observations**,
not final secret-key leakage. See [the derivation](adaptive-evaluation.md).

## Compared methods and design choices

All methods receive the same four features: QBER, absolute basis-QBER difference,
signal gain and decoy/signal gain ratio.

1. Calibrated QBER: high total QBER score.
2. Basis asymmetry: high absolute Z/X difference score.
3. Decoy ratio: low ratio score.
4. Joint max-tail: largest marginal honest-reference anomaly score.
5. Joint tail-sum: Fisher-style sum of marginal log-tail scores.
6. Logistic regression: standardization fitted only on training, C=1.
7. Random forest: 300 trees; seed varies with the replicate.
8. RBF SVM: training-only standardization, C=1, gamma=scale; decision scores.
9. **Uncalibrated reference only:** QBER>0.11. This is not a competing
   5%-target detector and not a universal security/intrusion criterion.

Statistical joint rules learn only from the honest training subset; supervised
models receive all labelled training rows. This is intentional: it tests whether
attack labels/nonlinear learning add value beyond a normal-channel reference.
Both use calibration on the same independent honest cohort.

For joint rules, anomaly directions are predeclared by physics: high QBER,
high asymmetry, two-sided signal-gain median deviation, and low decoy ratio.
For an oriented feature a and m honest reference values, define

$$p_j(a)=\frac{1+\#\{a^{ref}_{ij}\ge a_j\}}{m+1}.$$

This keeps ties and never yields zero. The max-tail score is
max_j[-log(p_j)]; tail-sum is -2 sum_j log(p_j).
The sum is **not** compared with a chi-square distribution: the features are
correlated and empirical reference p-values are discrete. Instead the combined
score is independently calibrated using its observed honest distribution.
This avoids an unjustified independent-feature null assumption.

These are accessible, strong first baselines, not optimal likelihood-ratio
tests or full decoy-state parameter estimation. Raw photon-count likelihood
baselines and nuisance channel/device shifts remain separate extensions.

## Common conservative calibration

For m usable calibration scores sorted increasingly, alpha=0.05, take
k=ceil((m+1)(1-alpha)) and threshold=score_(k), using one-based rank.
An alarm requires score **strictly greater** than the threshold.
If k>m, there are insufficient rows for a nontrivial conservative threshold;
the method records a no-alarm cutoff rather than silently miscalibrating.

Conditional on an already-fitted scoring function and exchangeable honest
sessions, the new score's rank among m+1 continuous scores is uniform.
The chance of exceeding the kth calibration score is
(m+1-k)/(m+1) <= alpha; ties with strict > are conservative.
This is a **marginal** statement over calibration and fresh honest sampling,
not a guarantee that every realized frozen model/threshold has FPR<=alpha.
It also does not guarantee a sequential/session-family false-alarm bound.
Channel shifts or nonexchangeable observations invalidate that simple argument.

At m=1000, k=951 and the marginal bound is 50/1001.
The implemented rational rank arithmetic prevents floating-point off-by-one
errors at exact integer ranks.

This follows the split-rank calibration principle described in
[Angelopoulos and Bates (2021)](https://arxiv.org/abs/2107.07511).
The SVM uses decision scores without optional internal probability fitting;
only score ordering is required for this calibration.
[scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

## Data separation, metrics and interpretation

NumPy SeedSequence creates separate children for training, calibration, honest
testing and each attack cohort. Every method sees the exact same accepted rows
within a cohort. Scaling/reference distributions use training data only.
Thresholds use honest calibration only. Honest-test and attack-test outputs
cannot refit models, choose thresholds, or choose policies.

Per seed/method/cohort, record flagged/accepted/attempted/rejected counts,
conditional alarm rate and 95% Wilson interval. AUC compares each fixed attack
against that seed's common honest-test scores. It is threshold-independent;
high AUC is not evidence of a particular operational recall/FPR.

Across seeds report arithmetic mean, **sample standard deviation**, minimum and
maximum. SD describes replicate variability; it is not a confidence interval.
Per-seed Wilson intervals do not include retraining variation. Reusing one
honest-test cohort for several attack AUCs makes those estimates correlated.

The predeclared paired comparator is Joint max-tail (not whichever statistical
method wins on the test). Paired ML-only/statistical-only detections on the
same attack rows and detection/FPR differences are retained. No significance
claim or multiple-comparison-adjusted superiority claim is made from five seeds.
Higher detection with materially higher FPR is not a fair advantage.

## Reproduction and evidence

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python quantum-ai-lab/scripts/run_fair_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-run
```

The runner refuses an existing output directory. Choose a new path for a
rerun. It stores benchmark.json (configuration, full per-seed metrics, source
hashes, dataset digests and environment versions) and session_audit.csv.gz
(training features/policies plus calibration/test features, per-method scores
and flags, and per-session raw knowledge/sifting counts).
Training raw knowledge counts and model scores are deliberately blank, not zero.
The audit key is seed + cohort + session_index; test/calibration indices refer to
original attempts, while training indices identify accepted training rows.

Source hashes matter because the working tree includes uncommitted repairs.
UTC capture time is provenance, not evidence of telemetry freshness: all rows
are synthetic. Physical constants, attack assumptions and model parameters are
inspectable in source. A version string alone is not a source revision.

Before publishing claims, independently reconcile flags/rates, rank cutoffs,
pooled information, AUCs and summaries from the audit rows. Failures or
surprising numbers must be investigated before changing the protocol; an
altered protocol needs a new run and cannot reuse these tests for tuning.

## Remaining limits

This is a fixed-policy, stationary-channel benchmark. It does not evaluate
adaptive best responses, out-of-family attacks, dark counts, device drift,
hardware calibration, temporal honest-vs-attack detection or quantum kernels.
The legacy full notebook/PDF/LaTeX reports remain historical and are not
overwritten by this runner. Fresh numerical results belong to this protocol only.

## First recorded run — 3 October 2026

The frozen five-seed protocol completed: **29,500 simulated sessions**, including
4,500 training rows, 5,000 calibration sessions, 5,000 honest-test sessions and
15,000 fixed-attack test sessions. No session was rejected in this run.
Saved evidence:

- [Full configuration, per-seed metrics and source/environment provenance](../results/benchmarks/fixed-policy-v2/benchmark.json)
- [Compressed session-level feature/score/flag audit](../results/benchmarks/fixed-policy-v2/session_audit.csv.gz)

All table values below are **percentages, mean ± sample SD across five seeds**.
The six attack columns are detection rates, not FPRs. The 11% row is the
uncalibrated reference and must not be treated as a common-target competitor.

| Method | Honest FPR | IR weak | IR medium | PNS weak | PNS medium | Mixed weak | Mixed medium |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QBER calibrated | 5.5 ± 0.8 | 59.8 ± 2.6 | 99.2 ± 0.5 | 6.0 ± 1.7 | 5.4 ± 1.2 | 29.4 ± 2.7 | 88.2 ± 1.1 |
| Basis asymmetry | 4.9 ± 1.0 | 49.7 ± 3.6 | 96.6 ± 0.5 | 4.4 ± 0.8 | 5.2 ± 1.1 | 20.0 ± 2.4 | 73.5 ± 1.8 |
| Decoy ratio | 5.6 ± 0.5 | 5.0 ± 1.1 | 4.7 ± 1.3 | 10.5 ± 1.2 | 29.0 ± 3.7 | 6.8 ± 1.1 | 13.6 ± 2.2 |
| Joint max-tail | 5.0 ± 1.0 | 49.7 ± 1.8 | 99.0 ± 0.3 | 7.2 ± 1.1 | 14.7 ± 2.0 | 18.8 ± 0.5 | 82.6 ± 1.4 |
| Joint tail-sum | 5.5 ± 0.6 | 51.1 ± 3.9 | 98.5 ± 0.2 | 7.0 ± 1.0 | 12.3 ± 1.5 | 21.3 ± 2.2 | 84.4 ± 0.8 |
| Logistic regression | 5.8 ± 1.0 | 37.8 ± 2.3 | 93.9 ± 1.7 | 9.1 ± 1.4 | 22.5 ± 3.1 | 20.2 ± 2.0 | 80.9 ± 2.4 |
| Random forest | 5.7 ± 0.7 | 42.2 ± 3.7 | 98.1 ± 0.3 | 9.4 ± 2.1 | 25.1 ± 3.2 | 17.6 ± 2.8 | 80.1 ± 2.8 |
| RBF SVM | 5.3 ± 1.0 | 32.8 ± 4.2 | 92.4 ± 6.4 | 10.3 ± 2.8 | 28.9 ± 3.4 | 15.5 ± 2.5 | 74.9 ± 5.5 |
| QBER 11% reference | 0.0 ± 0.0 | 0.1 ± 0.1 | 17.9 ± 0.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.2 ± 0.7 |

### What these results establish

- A low alarm rate under QBER>0.11 does not mean QBER contains no useful
  information. For weak IR, calibrated QBER detects 59.8% versus 42.2% for RF,
  at mean measured honest FPRs of 5.5% and 5.7%. Their score AUCs are 0.891
  and 0.800. This is a descriptive counterexample to a blanket ML advantage,
  not an exact-equal-FPR superiority test.
- Direct decoy-ratio monitoring detects 29.0% of medium PNS versus 25.1% for
  RF and 28.9% for SVM. Mean FPRs are 5.6%, 5.7% and 5.3%, respectively.
  The simple physical signature does not require ML under these assumptions.
- Against the **predeclared joint max-tail comparator**, RF detects more medium
  PNS (25.1% versus 14.7%), but fewer weak IR (42.2% versus 49.7%).
  RF has mean FPR 5.7% versus 5.0% for joint max-tail, so the PNS detection
  difference alone is not evidence of a strictly matched-operating-point gain.
- Weak PNS remains difficult for every method: mean detection ranges from
  approximately 4% to 11% among the calibrated methods. That is a result of
  this stationary simplified model and specified sample budget, not a
  fundamental security limit or universal stealth theorem.
- No tested learned model beats the appropriate direct physical statistic
  across all fixed families. Selecting QBER for known IR and decoy ratio for
  known PNS after seeing the test is an **oracle comparison**, not a deployable
  detector that knows the attack family. Both scalar rules and fixed joint rules
  are reported explicitly so this distinction remains visible.

The study answers a narrower question than the old headline table: a calibrated,
physically motivated baseline can be strong, and model choice/training
distribution affects the trade-off. These six fixed policies and five seeds
do not establish general robustness, research novelty or hardware performance.
The test cohorts are now consumed for this analysis. Any future tuning needs
new held-out seeds, not repeated evaluation against these same rows.

### Independent validation

The arithmetic auditor independently reconciled all 29,500 unique session keys,
360 per-seed/cohort/method rate records, 270 AUCs using direct pairwise
Mann–Whitney enumeration rather than sklearn, conservative calibration cutoffs,
raw-bit information totals, Wilson intervals, paired discordant detections,
seed summaries and the compressed-file SHA-256 digest.
This confirms serialization/arithmetic consistency, not the validity of all
physical assumptions. Separate physics regressions cover the specified channel.

The final combined regression suite passed **190 tests**, without skipped tests
or warnings. Dependency consistency and patch whitespace checks also passed.
The new benchmark runner was executed; the old full notebook was not rerun.

The analysis-validation workflow led to explicit realized-FPR qualifications,
paired comparisons and independent AUC/count checks. The next benchmark should
study honest channel shifts and attack strengths outside training, with fixed
choices and fresh test cohorts, before any stronger portfolio or research claim.
