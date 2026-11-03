# Count-aware detectors — predeclared protocol

This milestone asks whether exact measurement counts can control honest alarms
under the noise/window shifts that broke nominally calibrated ratio detectors.
It does not presume improved power or novel physics.

## Design fixed before the full run

Fresh seeds: **1001, 2002, 3003, 4004, 5005**. Each seed independently trains the
same nine legacy scorers on 300 honest and 600 attack sessions and calibrates on
1,000 nominal honest sessions (lambda=0.03, 6,000 pulses). Nothing is fitted or
retuned on evaluation cohorts. Source, transmissivity and decoy fraction remain
mu_s=0.5, mu_d=0.1, eta=0.15 and 0.5.

Three new count tests are specified, with analytic cutoffs at alpha=0.05:

1. **Fisher basis test:** compare [Z errors, Z correct] with [X errors, X correct]
   using the probability-order two-sided exact p-value. Unknown common honest
   bit-error probability is eliminated by conditioning on the total errors.
2. **Binomial decoy test:** test a lower-than-expected decoy click count given
   actual decoy pulses emitted. The trusted null is q_d=1-exp(-mu_d eta).
   This assumes known source/loss settings; it is not robust to unknown loss.
3. **Bonferroni joint:** adjusted p=min(1, 2 min(p_basis,p_decoy)); alarm if
   adjusted p <=0.05. This combines the two tests without assuming independence.

These are established statistical methods, not new algorithms. They use more
explicit denominators and a trusted physical null than the old four-ratio ML
input, so this is a system-design comparison, not a pure algorithm-only ablation.
No test receives true test-time channel_noise, Eve's policy label or known bits.
Scores are -p (higher is more suspicious); AUCs include ties. Exact tests may be
conservative and achieve less than the target 5% rate. Observed FPRs are reported,
not forced equal. No mid-p or randomized rejection is used to inflate power.

Evaluation has the same four conditions as the earlier robustness run: nominal,
lower-noise lambda=0.01, higher-noise lambda=0.06 (all 6,000 emitted pulses), and
a 3,000-pulse window at lambda=0.03. Each has 1,000 honest attempts and 500 attempts
per policy: IR r=0.10/0.01/0.80 with beta=0.95; the nominal 0.20-information PNS
anchor, PNS rho=0.01/0.85; and **unbiased IR r=0.10, beta=0.5**.

One additional 1,000-session honest control per seed uses lambda=0.0785 and
6,000 pulses. Its expected bit-error probabilities match unbiased IR r=0.10 at
lambda=0.03 exactly. This separately labelled noise-mimic control illustrates
an identifiability limitation; it does not replace condition-matched controls
for the primary AUCs. No attack cohorts are evaluated under the mimic condition.
There will be **104,500 rows** if all sessions are usable. Every evaluation and
calibration row saves eight exact counts; training rows preserve the unchanged
four-feature training interface and do not invent unrecorded counts.

## What is and is not a success

Primary guardrail: observed honest FPR stays compatible with the analytic <=5%
bound across the predeclared conditions, under these model assumptions. Each
seed/condition gets a pointwise Wilson interval; means and sample SD describe
replicate variation, not a confidence interval or multiplicity-adjusted test.
Power is reported beside FPR for every policy, including weak and unbiased IR.
Any loss of power versus ML is retained. No method is selected as a winner after
evaluation. These are independent stationary cohorts, not temporal drift.

Rates are conditional on usable telemetry. Rejected sessions and attempted IDs
are retained. The usual unconditional exact-test bounds must not be relabelled
as a theorem about arbitrary rejection, correlated pulses or real QKD devices.
No rejected sessions would simplify this run's empirical comparison, not prove
that rejections are universally impossible.

## Derivation and assumptions

Write n_Z,n_X for actual sifted counts, e_Z,e_X for their errors, N=n_Z+n_X and
E=e_Z+e_X. Under an independent common error probability q, the probability of
e_Z=k and e_X=E-k is C(n_Z,k) C(n_X,E-k) q^E (1-q)^(N-E). Summing over feasible
k gives C(N,E) q^E (1-q)^(N-E), by the binomial coefficient convolution.
Dividing eliminates q:

`P(e_Z=k | E,n_Z,n_X) = C(n_Z,k) C(n_X,E-k) / C(N,E)`.

Feasible k runs from max(0,E-n_X) to min(n_Z,E). The two-sided p-value sums
probabilities of tables no more probable than the observed table. A larger common
noise level changes E but not this conditional null. Basis-dependent honest
device errors violate the null and may produce alarms even without Eve.

For decoys, given emitted count m, D~Binomial(m,q_d), so
`p_decoy = sum_(k=0)^D C(m,k) q_d^k (1-q_d)^(m-k)`.
Here m is emitted decoy pulses, not detected or sifted bits. Changing m changes
the null uncertainty automatically. Both exact p-values are super-uniform under
their respective null assumptions. By the union bound,
`P(p_basis<=alpha/2 OR p_decoy<=alpha/2) <= alpha/2+alpha/2=alpha`.
No independent-feature assumption is needed. See the primary
[SciPy Fisher documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
and [binomial documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

With unbiased IR, q_Z=q_X=lambda/2+r(1-lambda)/4. Set
lambda_eff=lambda+r(1-lambda)/2; then lambda_eff/2 equals that same q_Z=q_X.
IR does not change click probabilities in this model. Thus all retained counts
have the same distribution as an honest isotropic channel at lambda_eff, under
the independent-pulse assumptions. No detector using only this telemetry can
universally separate those two hypotheses without extra trusted assumptions.
For lambda=0.03,r=0.10, lambda_eff=0.03+0.10*0.97/2=0.0785.
The empirical mimic AUC is a check of this model limitation, not a new theorem.

## Compatibility, audit and reproduction

Count instrumentation consumes no new random numbers and does not alter the
physical model. Existing Session(features,known,sifted) construction remains valid
with an optional counts field. A compatibility replay compares old saved rows
against fresh rows. Historical artifacts are not overwritten: their recorded
source fingerprints belong to the pre-instrumentation source snapshot.

From repository root, a NEW output directory is mandatory:

```sh
.venv/bin/python quantum-ai-lab/scripts/run_count_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-count-run
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.count_audit \
  quantum-ai-lab/results/benchmarks/my-fresh-count-run
.venv/bin/python -m pytest -q
```

The independent auditor uses integer combinatorial table enumeration for Fisher
and explicit binomial sums, not the production SciPy p-value functions. It also
reconciles legacy rank cutoffs, flags, Wilson intervals, matched AUCs, mimic AUCs,
count-derived features, source hashes and seed summaries. Outcomes are appended
only after the full run is checked. Notebook/report regeneration and git push
are outside this milestone.

## Completed results (2026-10-03 UTC)

The predeclared full run accepted **104,500 sessions**, with zero rejected
attempts. Of these, 4,500 are training rows and 100,000 calibration/evaluation
rows retain all eight actual integer counts. No count is fabricated from a
rounded ratio. Results were created at `2026-10-03T17:54:54.477588+00:00`.
The current source/package fingerprints and compressed raw file digest are
recorded in [counts.json](../results/benchmarks/count-aware-v2/counts.json).
The [session audit](../results/benchmarks/count-aware-v2/session_audit.csv.gz)
records rows, scores, flags and cohort metadata.

The independent full audit passes **2,040 rate checks, 1,740 AUC checks and
300,000 exact p-value checks**. It also passes frozen cutoff checks,
count-derived feature reconstruction, condition-matched controls, seed summaries,
Wilson intervals, row provenance and current-source fingerprints. Production
p-values use SciPy; auditing Fisher values uses integer table weights and
auditing binomial values uses explicit log-space sums. Agreement is arithmetic
evidence, not a hardware experiment or cryptographic security proof.

The full suite passes **240 tests** with real Qiskit Aer, including small-null
enumeration, invalid-count rejection, exact-test independence from Eve's private
knowledge and unchanged replay of 250 historical sessions. The historical-source
robustness audit also passes its 89,500 saved rows with current-source matching
explicitly marked **not checked**; that artifact has not been rewritten.

### Honest false-alarm rates

All table entries below are percent, **mean ± sample SD across five seeds**.
These are measured rates, not theoretical values or confidence intervals. Each
primary honest cohort has 1,000 independent accepted sessions per seed. The
noise-mimic column is an extra honest control, not a fifth primary attack test.

| Method | Nominal | Lower noise | Higher noise | Shorter window | Noise mimic |
| --- | --- | --- | --- | --- | --- |
| Calibrated QBER | 5.46 ± 0.98 | 0.16 ± 0.09 | 35.64 ± 4.33 | 9.90 ± 1.78 | 58.34 ± 3.83 |
| Basis asymmetry | 5.06 ± 0.59 | 0.36 ± 0.18 | 14.28 ± 0.71 | 12.60 ± 1.45 | 19.62 ± 0.32 |
| Direct decoy ratio | 5.58 ± 1.15 | 5.32 ± 0.84 | 5.66 ± 0.50 | 12.44 ± 1.43 | 5.10 ± 0.48 |
| Joint max-tail | 4.04 ± 0.85 | 2.12 ± 1.27 | 22.14 ± 2.89 | 18.72 ± 2.80 | 41.64 ± 3.07 |
| Joint tail-sum | 5.30 ± 1.16 | 0.60 ± 0.26 | 22.28 ± 2.38 | 17.88 ± 2.15 | 36.42 ± 3.61 |
| Logistic regression | 5.10 ± 0.51 | 0.60 ± 0.35 | 21.52 ± 3.78 | 11.96 ± 1.13 | 35.24 ± 4.02 |
| Random forest | 5.02 ± 0.79 | 2.60 ± 0.62 | 21.38 ± 2.41 | 12.14 ± 2.25 | 39.36 ± 3.08 |
| RBF SVM | 4.76 ± 0.79 | 2.64 ± 0.68 | 16.16 ± 4.48 | 13.16 ± 1.12 | 28.96 ± 5.35 |
| QBER 11% reference | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.05 |
| Count Fisher basis | 0.36 ± 0.11 | 0.00 ± 0.00 | 1.36 ± 0.27 | 0.06 ± 0.05 | 1.92 ± 0.59 |
| Count binomial decoy | 4.38 ± 0.48 | 4.22 ± 0.53 | 4.62 ± 0.47 | 3.76 ± 0.42 | 4.06 ± 0.70 |
| Count Bonferroni joint | 2.06 ± 0.50 | 2.00 ± 0.37 | 2.96 ± 0.47 | 2.02 ± 0.34 | 2.70 ± 0.78 |

The joint test is conservative: its observed nominal rate is about 2%, although
its analytic alarm budget is 5%. All 25 joint-test seed/condition Wilson 95%
upper endpoints are below 5% (maximum 4.944%); those are **pointwise**, not a
simultaneous confidence statement. The standalone binomial test has one
higher-noise seed with 52/1,000 honest alarms. A finite sample can exceed a null
rate bound, and that seed's interval includes 5%; do not claim every empirical
binomial replicate is below 5%.

This supports improved false-alarm robustness to these isotropic noise/window
changes under fixed, trusted source/loss. It does not cover loss drift,
basis-dependent detector errors, correlations or changing source intensities.
The 11% reference's very low FPR alone is not evidence of a useful detector.

### Detection power at the fixed operating points

Nominal-condition attack detection percent, mean ± seed SD. There are 500
accepted sessions per policy per seed. `IR .10/.01/.80` use beta=0.95; unbiased
IR uses r=.10,beta=.5. The PNS anchor has nominal expected raw knowledge .20,
not rho=.20. Its exact policy is recorded in JSON.

| Method | IR .10 | IR .01 | IR .80 | PNS anchor | PNS .01 | PNS .85 | Unbiased IR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Count Fisher basis | 13.64 ± 1.26 | 0.36 ± 0.17 | 100.00 ± 0.00 | 0.24 ± 0.33 | 0.20 ± 0.24 | 0.24 ± 0.43 | 1.72 ± 0.33 |
| Count binomial decoy | 4.12 ± 1.01 | 5.20 ± 0.24 | 4.28 ± 0.91 | 28.48 ± 1.85 | 4.04 ± 0.86 | 99.92 ± 0.11 | 3.76 ± 0.71 |
| Count Bonferroni joint | 8.76 ± 1.14 | 2.72 ± 0.33 | 99.92 ± 0.18 | 18.60 ± 2.25 | 1.76 ± 0.62 | 99.88 ± 0.18 | 2.24 ± 0.71 |
| Random forest | 43.12 ± 2.18 | 5.84 ± 1.11 | 99.96 ± 0.09 | 21.40 ± 3.96 | 3.84 ± 1.07 | 86.08 ± 13.28 | 40.56 ± 4.58 |
| Joint max-tail | 51.08 ± 3.23 | 5.60 ± 1.17 | 100.00 ± 0.00 | 12.88 ± 3.87 | 4.08 ± 1.95 | 99.48 ± 0.73 | 43.56 ± 4.39 |

The count joint test loses substantial weak-IR power. Nominal RF uses a 5.02%
observed FPR; the count joint test uses 2.06%, and the latter intentionally
does not treat an aggregate noise increase as basis-asymmetric evidence.
Conservative discrete p-values, sparse errors and the split alarm budget further
reduce its power. The comparison does not isolate any one cause. Strong IR/PNS
remain easy for the joint test at this nominal pulse budget; weak attacks are
not solved. No threshold has been adjusted after seeing these rows.

Nominal IR-anchor AUC is 0.692 ± 0.013 for the joint count test, versus
0.807 ± 0.016 for RF. Nominal PNS-anchor AUC is 0.772 ± 0.014 versus
0.711 ± 0.011 for RF. These are different attack-specific trade-offs, not a
blanket ML or count-test win. All twelve methods' detection rates and matched
AUCs for every primary condition/policy are retained in JSON.

Fisher's p-value is conditional on the total errors, so its discrete distribution
can change with that total even when both bases share the same error probability.
Super-uniform p-values control rejection probability; they do **not** guarantee
identical score distributions across different common noise levels. Therefore
the standalone Fisher test's nominal unbiased-IR AUC of 0.582 is not a violation
of its null calculation, and is not evidence that it can universally identify
unbiased Eve. The matched-noise comparison below checks the relevant limit.

### Honest-noise mimic versus unbiased Eve

Compare nominal unbiased IR with the separately generated honest
lambda_eff=0.0785 control, at the same 6,000-pulse budget. This is not an AUC
against nominal honest traffic. Mean ± sample SD of these AUCs:

| Method | Mimic-control AUC |
| --- | --- |
| Calibrated QBER | 0.512 ± 0.018 |
| Basis asymmetry | 0.505 ± 0.011 |
| Direct decoy ratio | 0.489 ± 0.023 |
| Joint max-tail | 0.512 ± 0.014 |
| Joint tail-sum | 0.509 ± 0.011 |
| Logistic regression | 0.504 ± 0.015 |
| Random forest | 0.505 ± 0.017 |
| RBF SVM | 0.507 ± 0.015 |
| QBER 11% reference | 0.512 ± 0.018 |
| Count Fisher basis | 0.505 ± 0.013 |
| Count binomial decoy | 0.494 ± 0.021 |
| Count Bonferroni joint | 0.500 ± 0.021 |

The observed near-chance ranking agrees with the theoretical equality of retained
telemetry distributions within this independent isotropic model. AUC sampling
variation is expected; this is not a claim that every realized AUC equals .5.
RF's 40.56% nominal unbiased-IR detection has to be read alongside its 39.36%
honest noise-mimic alarm rate. A high alarm rate can identify channel changes
without establishing that Eve caused them.

### Engineering scope and next steps

The defensible addition is count-preserving monitoring with explicit null
assumptions, a predeclared fresh-cohort comparison, independent numerical audit,
and an honest power/robustness trade-off. Fisher, binomial tails, Bonferroni and
noise confounding are established ideas, not claimed discoveries.

All sifted error counts are observable to the simulator. In a real protocol,
Alice and Bob disclose a parameter-estimation sample; those bits and the
finite-key estimation/security costs must be accounted for before claiming a
deployable detector. Unknown loss cannot be estimated from potentially attacked
traffic and silently treated as trusted: that could absorb the PNS signature.

Before the next holdout, specify trusted loss/source bounds and basis-dependent
honest-device models. Stress-test those assumptions, then investigate stronger
count-aware tests or learned scorers with separate selection/calibration data.
Do not tune on this completed cohort or relabel it untouched. Correlated pulses,
temporal detection and a constrained adaptive Eve require separate protocols.

The subsequent [honest loss/device stress test](nuisance-stress.md) is now complete.
It uses a separate honest-only sampler and fresh seeds, leaving these artifacts
and recorded source fingerprints unchanged. It demonstrates that the nominal
trusted-loss decoy null produces high honest alarms when eta decreases; these
earlier isotropic-noise results must not be promoted to loss robustness.
