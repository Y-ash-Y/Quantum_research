# Loss-aware signal/decoy monitoring — predeclared protocol

## Question and design fixed before the full run

Can measured signal counts remove the nominal-loss false-alarm failure without
discarding meaningful PNS detection? These are simulation monitoring tests, not
new quantum physics, a full decoy-state security estimator or a security proof.

Fresh seeds: **1097, 2097, 3097, 4097, 5097**. Each independently fits the same
nine legacy scorers on 300 honest + 600 attack nominal sessions, then calibrates
on 1000 separate nominal honest sessions. Training is the unchanged nominal
pulse simulator; honest calibration is the tested honest aggregate-count sampler.
All fitted models, references and rank cutoffs are frozen before evaluation.
The previous three nominal count tests also remain unchanged.

Three added score constructions, alpha=.05 and trusted intensity ratio c=.1/.5=.2:

1. **Loss-aware decoy:** allocate alpha/2 to a one-sided signal-gain lower
   confidence bound and alpha/2 to a lower-tail decoy test at the implied lower
   honest decoy gain. No nominal eta is supplied.
2. **Loss-aware joint:** allocate alpha/3 each to signal-bound noncoverage,
   the implied decoy test and the existing Fisher basis-homogeneity test.
   Combine the two rejection events by OR. Common honest basis-error probability
   remains an assumption; this is not an arbitrary-device robust joint rule.
3. **Loss plug-in decoy (no bound):** insert Ds/Ns into the honest gain relation
   and alarm at decoy tail <=alpha. This comparator omits estimation uncertainty;
   it is not assigned the confidence-bound test's guarantee.

Scores are negative screening tails (joint: negative minimum of its two tails).
Signal confidence levels depend on the fixed design alpha, so the resulting
scores are **not ordinary globally valid p-values**. AUC describes the ranking
of this fixed construction; sweeping its numerical cutoff does not preserve
the advertised alarm-budget proof. Do not retune alpha on evaluation rows.

| Evaluation condition | eta | channel lambda | gate mixing | pulses | attacks evaluated? |
| --- | --- | --- | --- | --- | --- |
| Nominal | .15 | .03 | 0 | 6000 | yes |
| Lower transmission | .10 | .03 | 0 | 6000 | yes |
| Higher transmission | .18 | .03 | 0 | 6000 | yes |
| Shorter window | .15 | .03 | 0 | 3000 | yes |
| Higher isotropic noise | .15 | .06 | 0 | 6000 | yes |
| Mild gate errors | .15 | .03 | .01 | 6000 | no |
| Stronger gate errors | .15 | .03 | .02 | 6000 | no |
| Combined loss/gates | .10 | .03 | .02 | 6000 | no |

Each condition has 1000 honest attempts per seed. Each of the five ideal-gate
conditions additionally has 500 attempts per seven fixed policies: biased IR
r=.01/.10/.80,beta=.95; unbiased IR r=.10,beta=.5; PNS rho=.01/.85 and the earlier
nominal .20-expected-information anchor (its nominal rho is frozen across eta).
These families are seen in training; weak/strong strengths are outside its active
support. Each attack's AUC uses independent honest controls from the **same**
condition. Gate-only/combined-gate tests establish honest FPR, not attack power.
No new model is selected after observing results. Expected maximum saved rows:
**137000** = 4500 training + 5000 calibration + 40000 honest + 87500 attack.

Honest counts use the existing aggregate marginal. A new parameterized pulse
attack path varies eta while preserving the old default path and random draw
order at eta=.15. It uses the same exact Aer probabilities and feasible PNS
gain-matching rule. Gate noise with attacks is explicitly unsupported, rather
than silently treating noisy Alice/Eve/Bob gates as the same device. Every row
records count denominators, true settings/policy for audit, known/sifted raw bits,
scores, flags and two signal lower bounds. Detectors receive counts/features,
not true eta, channel noise, Eve policy or known bits.

## Derivation, error budgets and assumptions

Given emitted counts Ns,Nd, honest clicks are independent binomials:
Ds~Binomial(Ns,qs), Dd~Binomial(Nd,qd). A shared intensity-independent honest
loss eta with a Poisson source gives qs=1-exp(-mu_s eta), qd=1-exp(-mu_d eta).
Eliminate eta: 1-qs=exp(-mu_s eta); raise to c=mu_d/mu_s to obtain
exp(-mu_d eta)=(1-qs)^c, hence qd=f(qs)=1-(1-qs)^c.
Since f'(p)=c(1-p)^(c-1)>0 for 0<p<1, f is increasing.

Let delta be the signal noncoverage budget. For Ds>0, the one-sided exact
Clopper–Pearson lower bound L solves
`sum_(k=Ds)^Ns C(Ns,k) L^k (1-L)^(Ns-k) = delta`.
Equivalently L=BetaQuantile(delta; Ds,Ns-Ds+1); for Ds=0 or Ns=0 set L=0.
This is frequentist binomial inversion, not an assumed Bayesian prior.
Its coverage property gives P(L>qs)<=delta. On the event L<=qs,
f(L)<=qd. The decoy tail p_d=P(Binomial(Nd,f(L))<=Dd) is conservative for an
honest qd>=f(L). Independence given Ns,Nd permits conditioning on the signal
count; the decoy rejection probability on this event is at most epsilon.
Splitting by whether coverage fails and applying the union bound gives
P(decoy alarm)<=delta+epsilon. Choosing delta=epsilon=alpha/2 yields <=alpha.
For the joint construction, add Fisher's common-basis-error rejection budget
gamma=alpha/3 and choose delta=epsilon=alpha/3; total <=alpha. No independence
between Fisher and the click tests is needed for this union bound.

These bounds concern the **unselected independent count experiment** under the
specified honest family. Legacy usability filtering depends on counts; measured
benchmark rates are conditional on usable sessions, with attempted/rejected IDs
reported. Do not relabel the unconditional proof as a bound after arbitrary
filtering. No observed rejections would simplify this comparison, not prove they
are universally impossible. Empty usable cohorts are errors, not fabricated zeros.

Source intensity ratio, common optical loss, independent pulses and negligible
dark counts are assumptions. Local gate errors in the honest model affect qubit
bits but not click rates; the standalone decoy bound still uses that click model.
The joint Fisher assumption fails for gate-dependent honest basis errors.
An attack compatible with the honest gain family can evade these observations;
using traffic to constrain a null does not certify that traffic as secure.
Raw sifted errors/known bits are not final-key leakage, and real public
parameter-estimation sampling/disclosure costs remain unmodelled.

## Reproduction, audit and success criteria

Success must report both honest FPR and attack power, with any loss of weak-PNS
power retained. Rates and pointwise Wilson intervals are saved; across-seed mean
and sample SD are not confidence intervals. This is not equal-realized-FPR
superiority or a novelty claim. Loss-aware power may be conservative at 6000 pulses.

```sh
.venv/bin/python quantum-ai-lab/scripts/run_loss_aware_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-loss-aware-run
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.loss_aware_audit \
  quantum-ai-lab/results/benchmarks/my-fresh-loss-aware-run
.venv/bin/python -m pytest -q
```

Existing output directories are refused. No previous source/artifact is rewritten.
The auditor checks confidence bounds using explicit binomial-sum residuals, not
production Beta quantiles, and reconstructs all count-test scores, flags, rank
cutoffs, intervals, matched AUCs, raw/source hashes and seed summaries. Independent
physics tests cover gain matching, IR errors and nominal-path equivalence.
Outcomes are appended only after the full run and audit. No git push, notebook
rerun or final-report PDF generation is included in this milestone.

Primary implementation references: [SciPy exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and [Beta quantile documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html).

## Completed five-seed results — 4 October 2026

The frozen run saved **137,000 rows** with **zero rejected attempts**. Results
below belong to seeds 1097/2097/3097/4097/5097, not earlier experiments.
Every rate is percent, mean ± sample SD across independently generated seeds;
these SDs are not confidence intervals. Session-level Wilson intervals remain
in the JSON. All fifteen methods and all 43 evaluation cohorts are recorded.

### Honest alarms: the nominal-loss failure is removed in this tested family

| Honest condition | Old nominal joint | Loss-aware joint | Loss-aware decoy | Plug-in decoy | Random forest |
| --- | --- | --- | --- | --- | --- |
| nominal | 1.86 ± 0.25 | 0.18 ± 0.18 | 0.32 ± 0.25 | 6.00 ± 0.47 | 4.60 ± 1.46 |
| lower transmissivity | 64.26 ± 0.97 | 0.26 ± 0.05 | 0.40 ± 0.07 | 5.32 ± 1.09 | 13.60 ± 4.14 |
| higher transmissivity | 0.14 ± 0.17 | 0.12 ± 0.11 | 0.34 ± 0.17 | 5.76 ± 0.40 | 0.80 ± 0.25 |
| shorter window | 1.76 ± 0.55 | 0.22 ± 0.19 | 0.24 ± 0.18 | 5.22 ± 0.74 | 13.34 ± 1.62 |
| higher isotropic noise | 2.34 ± 0.34 | 0.34 ± 0.05 | 0.24 ± 0.18 | 4.82 ± 0.37 | 21.52 ± 2.72 |
| mild gate errors | 2.58 ± 0.52 | 0.56 ± 0.17 | 0.48 ± 0.16 | 5.90 ± 0.49 | 11.08 ± 1.34 |
| stronger gate errors | 3.54 ± 0.80 | 1.06 ± 0.46 | 0.36 ± 0.15 | 5.34 ± 0.79 | 21.56 ± 2.32 |
| combined loss/gates | 64.50 ± 1.64 | 0.40 ± 0.20 | 0.44 ± 0.21 | 5.42 ± 0.81 | 28.68 ± 6.51 |

Under lower transmission, the old joint test's **64.26%** alarms occur even
without Eve. The loss-aware joint's **0.26%** is not obtained by supplying true
eta: its signal-count confidence bound permits the shared honest loss to vary.
The new rules are much more conservative than a nominal 5% budget. The plug-in
comparator has nominal honest alarms **6.00 ± 0.47%**, consistent with the concern
that treating estimated signal gain as exact can exceed the intended alarm budget.
This benchmark alone is not a proof of that comparator's behavior for every null.
A separate exhaustive unfiltered count example shows why it lacks the same
conditional guarantee: with Ns=1, Nd=100, qs=.07 and qd=1-(1-.07)^.2, Ds=1
occurs with probability .07. The plug-in then implies decoy probability 1,
rejecting whenever Dd<100, for an honest conditional alarm probability
`.07*(1-qd^100)` (approximately 7%, above 5%). The confidence-bound screen
respects its budget in that experiment. This deliberately unbalanced emitted-count
example is not the benchmark's typical half-signal/half-decoy window and does
not establish its unconditional alarm probability under random source allocation.

Low joint alarms under the three nonzero-gate conditions are empirical results
only: different honest basis-error probabilities invalidate its Fisher assumption.
The standalone decoy screen still concerns the stated gate-independent click
model, not arbitrary devices, dark counts or correlated/intensity-dependent loss.

### Attack power: preserve the negative result

Nominal, 6000-pulse detection percentages, with the same honest controls:

| Fixed policy | Old nominal joint | Loss-aware joint | Loss-aware decoy | Plug-in decoy | Random forest |
| --- | --- | --- | --- | --- | --- |
| IR anchor | 9.04 ± 2.30 | 5.64 ± 1.31 | 0.52 ± 0.23 | 5.96 ± 0.79 | 42.28 ± 2.23 |
| IR below support | 2.36 ± 0.43 | 0.32 ± 0.18 | 0.40 ± 0.20 | 6.28 ± 0.78 | 6.68 ± 1.46 |
| IR above support | 99.84 ± 0.17 | 99.80 ± 0.24 | 0.40 ± 0.24 | 5.44 ± 1.21 | 100.00 ± 0.00 |
| PNS anchor | 18.92 ± 2.22 | 3.20 ± 1.09 | 5.60 ± 0.95 | 29.08 ± 3.13 | 21.04 ± 5.26 |
| PNS below support | 2.20 ± 0.73 | 0.28 ± 0.11 | 0.24 ± 0.09 | 6.32 ± 1.65 | 5.48 ± 2.56 |
| PNS above support | 100.00 ± 0.00 | 96.80 ± 0.65 | 98.28 ± 0.79 | 100.00 ± 0.00 | 91.00 ± 4.56 |
| IR unbiased | 3.12 ± 0.50 | 0.80 ± 0.20 | 0.48 ± 0.23 | 5.92 ± 0.84 | 40.12 ± 3.36 |

The loss-aware joint retains **99.80%** detection of strong biased IR and
**96.80%** detection of strong PNS, but detects only **3.20%** of the moderate PNS
anchor (old joint **18.92%**, RF **21.04%**). Their nominal honest alarms differ
(0.18%, 1.86%, 4.60% respectively), so this is **not equal-realized-FPR superiority**.
The standalone decoy rule is not an IR detector: it has no basis-error evidence.
Unbiased IR is also compatible with an appropriate unknown isotropic noise level;
these monitoring counts cannot establish its presence without additional assumptions.
The PNS anchor's nominal 20%-expected-raw-knowledge policy is frozen across eta;
do not call its observed information exactly 20% in every condition or session.

### Window/loss/noise trade-off for the loss-aware joint

| Condition | Honest FPR | Moderate PNS detection | Strong PNS detection | Strong biased IR detection |
| --- | --- | --- | --- | --- |
| nominal | 0.18 ± 0.18 | 3.20 ± 1.09 | 96.80 ± 0.65 | 99.80 ± 0.24 |
| lower transmissivity | 0.26 ± 0.05 | 1.80 ± 0.84 | 83.04 ± 2.13 | 97.24 ± 0.46 |
| higher transmissivity | 0.12 ± 0.11 | 4.08 ± 0.70 | 99.24 ± 0.17 | 99.96 ± 0.09 |
| shorter window | 0.22 ± 0.19 | 1.40 ± 0.68 | 65.08 ± 1.91 | 87.96 ± 0.89 |
| higher isotropic noise | 0.34 ± 0.05 | 3.16 ± 0.55 | 97.44 ± 1.09 | 99.48 ± 0.18 |

The shorter 3000-pulse window reduces strong-PNS detection to **65.08%** despite
honest FPR **0.22%**; lower transmission reduces it to **83.04%**. A low alarm
rate cannot be interpreted as adequate sensitivity. This run demonstrates a
controlled robustness/power trade-off, not completion of all research goals.

### Independent audit and verification

The auditor passed **3,300 rate/Wilson checks**, **2,625 condition-matched AUC
checks**, **795,000 independently reconstructed screening-score checks** and
**265,000 signal-bound inversions using explicit binomial sums**. It also checks
880 independent physical marginal references; the maximum absolute descriptive
pooled residual is 3.373 standard deviations across these many comparisons. This
is not a multiple-testing discovery criterion. All current source hashes, raw
hashes, count-derived features, cohort/label/policy inventories and frozen cutoffs
match. The full real-Aer regression suite passes **360 tests**.

Audit scope: all six count-based score columns are independently recomputed for
every nontraining row; legacy learned score columns are saved predictions whose
finite values, frozen flags, calibration ranks, rates and AUCs are audited.
The auditor does not independently refit/reimplement every learned estimator.
Physics agreement is separately tested against closed forms and the original
nominal pulse path. A single observed run with no rejections does not upgrade
the unselected-count theorem into a universal post-selection guarantee.

Artifacts: [results JSON](../results/benchmarks/loss-aware-v2/loss_aware.json),
[session audit](../results/benchmarks/loss-aware-v2/session_audit.csv.gz),
[independent audit record](../results/benchmarks/loss-aware-v2/independent_audit.json).
Earlier sources and reviewed benchmark artifacts remain unchanged.
Current count-aware, nuisance and loss-aware source fingerprints still match
their manifests, and all five reviewed raw files retain their recorded checksums.
The earlier fixed-policy/robustness manifests retain historical source hashes
that predate the previous count-instrumentation/auditor changes; they were not
rewritten to pretend those historical sources match the present checkout.

### What is now defensible, and what comes next

Defensible portfolio claim: implemented a count-based screen with an explicit
honest-model false-alarm budget, caught a trusted-loss misspecification, and
measured its sensitivity cost using reproducible, independently audited controls.
This is established statistical methodology applied to a restricted BB84 model;
**not a novel-discovery, deployable-QKD-security or quantum-advantage claim**.

Recommended next milestone (not executed here): predeclare a more efficient
joint signal/decoy test or pulse-window power study, controlling unknown loss
without treating its estimate as exact. Compare at documented operating budgets,
include nuisance misspecification and weak attacks, and account for public
parameter-estimation disclosure before making practical-security claims.
No new poster, historical PDF/report rebuild, commit or push was performed.
