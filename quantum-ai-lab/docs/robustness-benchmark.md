# Frozen-detector robustness benchmark — protocol

This protocol is specified before the full benchmark is run. It is a simulation
stress test, not a security proof or a claim of novel physics.

## Question and fixed design

Do nominally trained/calibrated detectors preserve low honest false-alarm rates
and discriminate attacks when the telemetry distribution changes?

- Five fresh seeds: **101, 202, 303, 404, 505**, separate from the earlier
  fixed-policy benchmark. Each seed independently refits the nine unchanged
  scorers from that benchmark on 300 honest and 600 attack training sessions.
- Training and 1,000 honest calibration sessions use 6,000 emitted pulses and
  depolarizing strength lambda=0.03. Eight methods use the conservative 5% rank
  cutoff; the 11% QBER rule remains an uncalibrated historical reference.
- After calibration, **no model, scaler, reference distribution, hyperparameter,
  policy or cutoff is changed** using evaluation data.
- Four evaluation conditions: nominal (lambda=0.03, 6,000 pulses), lower noise
  (0.01, 6,000), higher noise (0.06, 6,000), shorter window (0.03, 3,000).
  These are independent stationary cohorts, not a time-dependent drift process.
  Transmissivity, source intensities and decoy fraction stay fixed. Loss drift,
  gate/readout noise and hardware data are outside this milestone.
- Per condition: 1,000 attempted honest sessions and 500 attempted sessions per
  each of six fixed attack policies. All methods see identical rows.
- Intercept-resend policies: in-support anchor r=0.10, below-support r=0.01,
  above-support r=0.80; all beta=0.95, rho=0.
- PNS policies: in-support anchor chosen for nominal expected raw knowledge 0.20,
  below-support rho=0.01, above-support rho=0.85; all r=0, beta=0.5.
  Training active strengths are drawn from [0.02,0.60), so the latter strengths
  are outside training support. Attack *families* are not new/unseen.

There are 89,500 accepted rows if no attempted telemetry is rejected. Rejection
counts, attempted-session IDs, sifted/known-bit counts, scores and flags are saved.
Rates are conditional on usable sessions; an empty usable cohort is an explicit
error, not a fabricated zero. No detector or condition is selected as a winner
after seeing the results.

## Metrics and interpretation

Honest FPR = honest alarms / accepted honest sessions. Attack detection = attack
alarms / accepted attacked sessions. Each run includes a Wilson 95% interval.
Across-seed mean and sample SD describe replicate variation; SD is not a
confidence interval. Observations across conditions are not paired sessions.

Every attack AUC uses that condition's independent **matched honest test** rows,
not the nominal honest test rows. AUC measures ranking, not the safety of the
frozen operating threshold. A high detection rate accompanied by a high honest
FPR is not useful attack-specific detection.

The calibration guarantee relies on exchangeability with future honest data;
these noise/window shifts deliberately violate that assumption. Exceeding 5%
under shift does not refute the rank calculation or justify retuning on test data.
No multiplicity-adjusted hypothesis tests or significance/superiority claims are
planned. Individual intervals are pointwise descriptive intervals.

For honest depolarization, expected sifted QBER is lambda/2: 1.5%, 0.5%, 3.0%,
and 1.5% for the four conditions. For intercept-resend fraction r it is
lambda/2 + r(1-lambda)/4. A basis-unbiased IR policy can therefore imitate an
increase in honest aggregate noise. Per-basis asymmetry and decoy telemetry can
help in restricted models, but telemetry alone is not a cryptographic certificate.
Shorter windows increase estimator variance without changing those expected
means. Expected raw knowledge and pooled observed knowledge are also saved;
they are not final-key leakage or mutual information.

## Reproduction and audit

From the repository root:

```sh
.venv/bin/python quantum-ai-lab/scripts/run_robustness_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/robustness-v2
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.robustness_audit \
  quantum-ai-lab/results/benchmarks/robustness-v2 --historical-sources
.venv/bin/python -m pytest -q
```

An existing output directory is refused before computation. Source fingerprints,
package versions, stream spawn keys and the compressed CSV digest accompany JSON
results. The auditor independently recomputes cutoffs, flags, counts, Wilson
intervals, condition-matched AUCs and seed summaries from saved rows. A matching
audit is arithmetic evidence, not independent validation of real-device physics.

## Completed results (2026-10-03 UTC)

The predeclared full run accepted **89,500 rows**, with no training or evaluation
rejections. The arithmetic auditor checks **1,305 rate records and 1,080 AUCs**,
plus rank cutoffs, row/source digests, counts, pooled information and summaries.
The combined regression suite passes **202 tests** with real Aer physics.

### Honest false alarms

All values below are mean ± sample SD across the five seeds, in percent. Honest
false alarms are NOT Eve detections. No significance claim follows from the SD.

| Method | Nominal | Lower noise | Higher noise | Shorter window |
| --- | --- | --- | --- | --- |
| Calibrated QBER | 5.2 ± 0.4 | 0.1 ± 0.1 | 36.5 ± 1.8 | 10.6 ± 1.3 |
| Basis asymmetry | 4.8 ± 1.2 | 0.5 ± 0.3 | 14.4 ± 1.4 | 13.9 ± 1.2 |
| Direct decoy ratio | 5.1 ± 1.5 | 5.1 ± 0.7 | 5.0 ± 1.5 | 12.3 ± 1.8 |
| Joint max-tail | 4.6 ± 0.4 | 2.3 ± 0.5 | 21.9 ± 3.3 | 21.7 ± 1.6 |
| Joint tail-sum | 4.8 ± 0.9 | 0.9 ± 0.5 | 21.6 ± 1.0 | 20.3 ± 2.9 |
| Logistic regression | 5.3 ± 0.5 | 0.8 ± 0.2 | 23.5 ± 1.5 | 13.0 ± 1.8 |
| Random forest | 4.6 ± 0.7 | 2.5 ± 0.7 | 20.1 ± 3.4 | 12.0 ± 1.5 |
| RBF SVM | 5.1 ± 1.3 | 2.3 ± 0.6 | 16.8 ± 2.6 | 14.1 ± 2.7 |
| QBER 11% reference | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

The decoy-ratio score is comparatively insensitive to isotropic depolarization
here because photon detection is independent of qubit errors in this model.
That is a modelling assumption, not evidence of robustness to real hardware
noise. Halving the window still more than doubles its observed false-alarm rate.
The historical 11% reference has no honest alarms here, but misses weak attacks;
zero false alarms alone is not a useful detector.

### Weak and strong strengths outside training

Nominal-condition attack detection percentages (mean ± seed SD):

| Method | IR r=0.01 | IR r=0.80 | PNS rho=0.01 | PNS rho=0.85 |
| --- | --- | --- | --- | --- |
| Calibrated QBER | 7.8 ± 0.4 | 100.0 ± 0.0 | 5.9 ± 0.6 | 6.1 ± 0.6 |
| Basis asymmetry | 6.9 ± 0.9 | 100.0 ± 0.0 | 5.2 ± 0.9 | 6.9 ± 2.1 |
| Direct decoy ratio | 5.0 ± 1.6 | 4.4 ± 0.8 | 5.2 ± 1.7 | 100.0 ± 0.0 |
| Joint max-tail | 5.6 ± 1.2 | 100.0 ± 0.0 | 4.4 ± 0.8 | 99.5 ± 0.6 |
| Joint tail-sum | 6.6 ± 1.4 | 100.0 ± 0.0 | 5.0 ± 1.4 | 60.5 ± 6.7 |
| Logistic regression | 7.3 ± 0.4 | 100.0 ± 0.0 | 6.3 ± 0.8 | 98.8 ± 0.6 |
| Random forest | 5.9 ± 1.1 | 99.9 ± 0.3 | 4.9 ± 0.5 | 92.0 ± 3.8 |
| RBF SVM | 6.4 ± 1.7 | 80.6 ± 41.6 | 5.3 ± 1.8 | 97.8 ± 4.0 |
| QBER 11% reference | 0.0 ± 0.0 | 99.9 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |

All nine mean AUCs on these very weak attacks are between **0.500 and 0.555**
(rounded). Random forest means are 0.524 (weak IR) and 0.509 (weak PNS).
This documents poor discrimination at this pulse budget; it does not prove
information-theoretic indistinguishability or exclude a better detector.
The strength parameter is outside the training support, but finite-session
feature distributions can still overlap training features.

PNS rho is not a sifted-information percentage: for rho=0.01 and 0.85, expected
raw knowledge fractions are 0.008733 and 0.831714. Pooling actual known/sifted
bits across all nominal test seeds gives 0.008805 and 0.831819 respectively.
The corresponding IR r=0.01 and 0.80 expectations are 0.005 and 0.40; pooled
observations are 0.004937 and 0.400019. These are raw Alice-bit knowledge proxies,
not compromised final secret-key fractions.

### Higher detection can conceal a worse operating point

For the same in-support PNS anchor (nominal expected raw knowledge 0.20):

| Detector / condition | Honest FPR (%) | Detection (%) | Matched AUC |
| --- | --- | --- | --- |
| RF / nominal | 4.6 ± 0.7 | 21.3 ± 1.6 | 0.727 ± 0.011 |
| RF / higher noise | 20.1 ± 3.4 | 37.2 ± 1.9 | 0.654 ± 0.019 |
| RF / shorter window | 12.0 ± 1.5 | 26.8 ± 3.7 | 0.644 ± 0.010 |
| Joint max / nominal | 4.6 ± 0.4 | 12.8 ± 3.7 | 0.680 ± 0.007 |
| Joint max / higher noise | 21.9 ± 3.3 | 29.0 ± 3.3 | 0.581 ± 0.015 |
| Joint max / shorter window | 21.7 ± 1.6 | 33.1 ± 4.4 | 0.607 ± 0.016 |
| Decoy / nominal | 5.1 ± 1.5 | 30.2 ± 3.0 | 0.791 ± 0.021 |
| Decoy / shorter window | 12.3 ± 1.8 | 34.6 ± 3.3 | 0.711 ± 0.003 |

Detection rates can rise while honest FPR rises and AUC falls. In particular,
QBER alone has higher-noise PNS-anchor detection 36.6% versus honest FPR 36.5%,
with matched AUC 0.501: these alarms do not meaningfully distinguish PNS from
the same noisy honest channel. An unmatched nominal control would give a
misleading impression of attack discrimination.

### Audited SVM failure (post-hoc diagnostic)

Nominal strong-IR detection for SVM across seeds 101, 202, 303, 404, 505 is
**6.2%, 100%, 100%, 96.6%, 100%**. The mean 80.6 ± 41.6% must not hide seed 101.
In that seed, 31/500 sessions exceed cutoff **1.0096203340**, whereas their mean
score is **0.9788370835**. AUC is **0.955454**; it summarizes ranking over all
possible thresholds, not the calibrated operating point.

`scripts/diagnose_robustness.py` refits solely on that seed's saved training rows:
the saved test scores replay exactly (maximum absolute difference zero).
The explicit binary RBF decision calculation agrees within 3.56e-15:

`f(x) = sum_i a_i exp(-gamma ||z(x)-s_i||^2) + b`

Here z(x) is training-fitted standardized telemetry, s_i are support vectors in
that standardized space, a_i are learned signed coefficients, gamma controls
kernel locality, and b is the intercept. This is the standard
[scikit-learn RBF SVM formulation](https://scikit-learn.org/stable/modules/svm).
The score is not constrained to grow monotonically with physical attack strength.
Far from support, the kernel terms can decay and the score can approach b.
In this replay b=0.9561822830, below the cutoff; 44.6% of tested scores are within
0.01 of b. This supports a bounded/local-score explanation, not an assertion
that every test point has negligible kernel similarity.

The diagnostic was selected after observing the failure. It is not another
untouched benchmark or a universal SVM impossibility result. No fitted score or
threshold has been changed to improve these reported test results. Reproduce:

```sh
.venv/bin/python quantum-ai-lab/scripts/diagnose_robustness.py \
  quantum-ai-lab/results/benchmarks/robustness-v2 --seed 101 \
  --output /tmp/bb84-svm-diagnostics-new.json --historical-sources
```

The same diagnostic reconstructs pooled error counts for all 20 honest test
cohorts and compares QBER with lambda/2. Maximum absolute binomial z-score is
1.59. This supports agreement for these idealized cohorts; it does not validate
the full physical attack space, or hardware and finite-key security.

## What this milestone adds, and the next design

The defensible contribution is a reproducible, negative-result robustness study
in one tested codebase: transparent controls, fresh replicates, row-level evidence
and a concrete operating-point failure. None of those establish literature
novelty or a 9/10 project rating by themselves. It is not ready for real QKD
intrusion deployment; **share with these simulation and robustness caveats**.

The next milestone should specify a new design before running a new holdout:

1. Preserve actual error/detection counts and denominator sizes, rather than
   relying only on four estimated ratios. Model window-dependent uncertainty.
2. Separate channel-health alarms from attack-specific evidence. For biased IR,
   investigate count-based per-basis homogeneity tests that treat unknown common
   honest error rate as a nuisance; isotropic/unbiased IR remains confounded with
   unknown isotropic noise without additional assumptions or trusted controls.
3. Calibrate count-aware combined tests and ML on independently generated honest
   references covering predeclared conditions. Compare measured FPR and power,
   not only AUC. Do not infer a trusted noise estimate from potentially attacked
   traffic and then subtract away Eve's effect.
4. Validate any physics-informed monotone guard for strong attacks on a NEW
   cohort. Do not add a guard after seeing this holdout and relabel it untouched.
5. Keep adaptive-search, device/loss drift, temporal bursts and real telemetry as
   separately specified follow-ups; weak-signal improvement is not guaranteed.

The existing fixed-policy artifacts are unchanged. Notebook plots and reports
remain historical; this milestone does not regenerate them, commit or push.

## Subsequent count-aware milestone: historical-source audit

The count-aware implementation adds optional count telemetry to the same
vectorized simulation, without consuming new random numbers. Its source hashes
therefore differ from this run's pre-instrumentation source snapshot. Do not
replace the recorded hashes or call this artifact a new count-aware run.
A regression replays 250 saved historical sessions and compares their features
and known/sifted counts exactly; this is a sampled compatibility check.

For this unchanged historical artifact, explicitly skip current-source matching
while still auditing the saved file digest, rows, cutoffs, metrics and summaries:

```sh
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.robustness_audit \
  quantum-ai-lab/results/benchmarks/robustness-v2 --historical-sources
```

The auditor reports current-source fingerprints as **not checked**, not passed.
The same `--historical-sources` flag is required to replay the historical SVM
diagnostic with the newer instrumentation. New count-aware artifacts are audited
against current sources normally. See [count-aware-benchmark.md](count-aware-benchmark.md).
