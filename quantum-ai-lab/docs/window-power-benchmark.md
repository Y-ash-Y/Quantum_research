# Fixed-window power study — predeclared design

## Question fixed before the full run

Can more observations recover moderate-attack sensitivity without retuning the
loss-aware screens? This is a resource/power study, **not a new detector**, an
ML comparison, a real-hardware experiment or a novel quantum-security claim.

Fresh seeds: **1117, 2117, 3117, 4117, 5117**. Independent windows contain
**6000, 20000, 60000 and 120000 emitted pulses**. For every window and seed:

- Nominal: eta=.15, transmission depolarization lambda=.03, ideal local gates.
- Lower transmission: eta=.10, lambda=.03, ideal local gates.
- Honest device stress: eta=.15, lambda=.03, per-X/H gate mixing g=.02.

Each condition has 500 attempted honest sessions. Each ideal-gate condition also
has 250 attempts for each of four frozen policies: biased IR r=.10,beta=.95;
unbiased IR r=.10,beta=.5; the previous nominal 20%-expected-raw-knowledge PNS
anchor (rho fixed across eta/windows); weak PNS rho=.01. No attack is generated
under gate mixing, because that local-device attack model remains unspecified.
Expected maximum: **70000 saved rows**, no training or calibration rows.

Six unchanged count-based comparators share every cohort: Fisher basis, nominal
binomial decoy, nominal Bonferroni joint, loss-aware decoy, loss-aware joint and
the unbounded plug-in decoy. Alpha=.05 throughout; new designs use the previously
fixed alpha/2 and alpha/3 budgets. Trusted intensities mu_s=.5, mu_d=.1 and
decoy fraction .5 remain fixed. No detector receives true eta, Eve policy, raw
knowledge, gate settings or an attack label. There is no learned estimator and
no test-time calibration, fitted intensity or selection of a winning window.

Fresh cohorts at different windows are independent, not a repeatedly inspected
growing trajectory. Report attempted/rejected/accepted IDs; do not silently
resample usability failures. AUC uses honest controls under the same window
and optical condition. Empty cohorts fail. Existing artifacts are never overwritten.

Success question: does moderate PNS/biased IR mean detection reach 80% at 120000
pulses at both ideal-gate transmission settings? This fixed reporting target is
not a rule for selecting a model/window or omitting failures. All four windows,
weak attacks, honest device stress and six methods must remain in the results.
Verify observed ideal-gate honest rates alongside power; a theoretical <=5%
budget is not an expectation of exactly 5% observed alarms or a guarantee for
arbitrary device settings. Investigate any consequential discrepancy.

## Why larger windows can help, and what they cost

For independent clicks D~Binomial(N,q), E[D]=Nq and Var[D]=Nq(1-q).
Dividing D by N gives E[D/N]=q and Var[D/N]=q(1-q)/N, so its standard
deviation is sqrt(q(1-q)/N). A fixed honest/attack mean separation becomes
easier to resolve as N increases. Exact confidence bounds and integer rejection
regions still determine actual power; this square-root relation is intuition,
not a derivation of the experiment's recall.

The exact CP signal-bound and decoy-tail budgets/derivation are unchanged; see
[the loss-aware protocol](loss-aware-benchmark.md). Their honest theorem is for
unselected independent counts with shared intensity-independent loss, trusted
intensities and negligible dark counts. Joint Fisher additionally requires a
common honest basis-error probability. Telemetry usability is count-dependent,
so measured rates remain conditional on usable sessions. No rejections in one
run would not establish an arbitrary post-selection theorem.

Twenty times more emitted pulses (120000 versus 6000) is not an algorithm-only
win. At fixed pulse emission rate R pulses/second, collecting a window costs
N/R seconds; no hardware rate is assumed here. Larger windows can delay alarms
and mix nonstationary behavior. They are not inherently better for burst attacks.
Checking many windows or stopping when an alarm first appears requires separate
sequential/multiple-testing control: the single-window alpha budget is not a
bound on ever raising a false alarm over an unlimited stream.

The honest g=.02 device condition intentionally violates the joint common-basis
assumption. A larger window may reveal device asymmetry and **increase false
alarms**, even as attack power improves. This is a stress limitation, not a
failure of a theorem outside its assumptions. Standalone click screening does
not depend on this particular gate-bit model, but it is not a complete IR detector.
Unbiased IR remains confounded with an appropriate unknown isotropic noise level.
Raw Eve knowledge among sifted observations is not final-key leakage. Public
parameter-estimation disclosure, dark counts, correlated loss, hardware drift,
general/adaptive Eve and final-key security remain unmodelled.

## Implementation and independent numerical audit

Use the existing validated honest aggregate sampler and loss-parameterized
pulse attack path unchanged. Preserve all eight integer counts, four legacy
features, raw known/sifted bits, bounds, six scores and flags. Seed child streams
are disjoint across conditions/windows/policies. The new runner refuses existing
output directories.

The independent auditor does not call the production Beta quantile, binomial
CDF or Fisher scorer. A decreasing-binomial-mass recurrence sums tails with a
geometric omitted-remainder bound <=1e-14; it checks signal CP inversion residuals
and every click score. Integer combinatorial recurrence checks probability-order
Fisher exactly before conversion to float. Endpoint, exhaustive small-count and
large-count tests cross-check both numerical paths. Rates, Wilson intervals,
matched AUCs, feature reconstruction, frozen budgets, row inventories and
raw/source fingerprints are independently reconciled.

Reproduction commands and outcomes are added after implementation and the run.
No commit/push, poster or historical report/PDF rebuild is part of this milestone.

## Reproduction

```sh
.venv/bin/python quantum-ai-lab/scripts/run_window_power_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-window-power-run
PYTHONPATH=quantum-ai-lab/src .venv/bin/python -m classical_ml.window_power_audit \
  quantum-ai-lab/results/benchmarks/my-fresh-window-power-run
.venv/bin/python -m pytest -q
```

## Completed results — 4 October 2026

**70000 rows**, zero rejected attempts, all four windows and six methods retained.
These values belong to the five new seeds in this protocol; do not substitute
the previous loss-aware run's numbers. Rates below are percent, mean ± sample
SD across seeds, not confidence intervals. Per-cohort Wilson intervals remain
in the JSON. The same frozen count rules are applied at every window.

### Moderate PNS and biased IR recover with more observations

Loss-aware **joint** rule:

| Emitted pulses | Nominal PNS moderate | Lower-transmission PNS moderate | Nominal IR biased | Lower-transmission IR biased |
| --- | --- | --- | --- | --- |
| 6000 | 3.20 ± 1.17 | 1.76 ± 0.46 | 6.40 ± 0.94 | 2.40 ± 0.89 |
| 20000 | 17.52 ± 3.64 | 9.04 ± 2.51 | 45.44 ± 5.48 | 26.72 ± 1.78 |
| 60000 | 71.04 ± 2.13 | 46.96 ± 2.60 | 96.88 ± 1.73 | 82.72 ± 3.41 |
| 120000 | 97.76 ± 0.73 | 87.12 ± 0.52 | 100.00 ± 0.00 | 99.52 ± 0.33 |

The predeclared 80%-mean-detection question is answered **yes at 120000 pulses**
for both moderate PNS and biased IR at both transmission settings. This is a
resource result: 120000 pulses cost **20 times** the emitted observations of a
6000-pulse window, not a more sample-efficient algorithm. Do not label the
observed crossing an exact minimum required window; only four independent
budgets were tested. Do not infer a latency without a specified emission rate.

### Honest false alarms: model assumptions still determine validity

| Pulses | Joint nominal | Joint lower transmission | Joint honest gate stress | Standalone decoy honest gate stress |
| --- | --- | --- | --- | --- |
| 6000 | 0.28 ± 0.23 | 0.20 ± 0.14 | 0.96 ± 0.48 | 0.32 ± 0.23 |
| 20000 | 0.92 ± 0.23 | 0.56 ± 0.33 | 8.24 ± 1.11 | 0.40 ± 0.24 |
| 60000 | 1.12 ± 0.41 | 1.56 ± 0.48 | 34.48 ± 2.64 | 0.56 ± 0.33 |
| 120000 | 1.44 ± 0.38 | 1.60 ± 0.73 | 67.84 ± 0.84 | 0.64 ± 0.46 |

Under the specified ideal-gate honest family, mean joint false alarms remain
below 5% at every tested window/transmission setting (maximum **1.60%**). This
empirical observation is not a proof after arbitrary usability filtering and
not a simultaneous/sequential-alarm guarantee. The g=.02 honest gate condition
**violates** the joint common-error assumption: its honest FPR grows from
**0.96%** to **67.84%**. The standalone click screen stays at **0.64%** at 120000
pulses in this gate-independent click model, but does not solve IR detection.

This finding must accompany the improved attack recall. Larger windows give
more statistical evidence about *any* systematic basis asymmetry, including
honest device effects. It would be wrong to advertise the 120000-pulse joint
rule as a deployable device-robust detector or call all its alarms eavesdropping.

### Weak/indistinguishable attacks remain limitations

| Pulses | Joint nominal weak PNS | Joint nominal unbiased IR | Unbiased IR mean AUC |
| --- | --- | --- | --- |
| 6000 | 0.16 ± 0.22 | 0.80 ± 0.57 | 0.5578 ± 0.0175 |
| 20000 | 1.28 ± 1.11 | 1.28 ± 0.18 | 0.5360 ± 0.0213 |
| 60000 | 1.68 ± 1.31 | 1.92 ± 0.33 | 0.5087 ± 0.0178 |
| 120000 | 1.92 ± 0.33 | 1.76 ± 0.67 | 0.5056 ± 0.0134 |

At 120000 pulses, weak-PNS detection is **1.92%**, versus **1.44%** nominal honest
joint alarms; meaningful protection against rho=.01 is not established. Unbiased
IR AUC is about **0.506** in that window. The monitor's basis-homogeneity premise
permits arbitrary common error rates; an appropriate unknown isotropic honest
noise level can reproduce unbiased IR telemetry in the restricted model. More
counts do not resolve that observational ambiguity. This experiment is not a
new noise-mimic control rerun or a claim about all real QKD devices.

### All six comparator alarm rates and AUCs

The complete [results JSON](../results/benchmarks/window-power-v2/window_power.json)
retains every cohort's rates/intervals/AUCs and all seed summaries. Selected honest
rates below prevent a single-method interpretation:

| Pulses / honest condition | Fisher basis | Nominal binomial decoy | Nominal Bonferroni joint | Loss-aware decoy | Loss-aware joint | Plug-in decoy |
| --- | --- | --- | --- | --- | --- | --- |
| 6000 / nominal | 0.48 ± 0.46 | 4.68 ± 0.73 | 2.48 ± 0.59 | 0.48 ± 0.18 | 0.28 ± 0.23 | 5.60 ± 0.97 |
| 6000 / lower transmission | 0.24 ± 0.17 | 75.44 ± 1.56 | 63.64 ± 2.04 | 0.32 ± 0.18 | 0.20 ± 0.14 | 4.84 ± 0.95 |
| 6000 / honest gate stress | 2.84 ± 0.95 | 3.92 ± 0.82 | 3.20 ± 0.71 | 0.32 ± 0.23 | 0.96 ± 0.48 | 5.04 ± 1.23 |
| 20000 / nominal | 2.08 ± 0.64 | 4.68 ± 0.63 | 3.20 ± 1.10 | 0.48 ± 0.11 | 0.92 ± 0.23 | 5.84 ± 1.16 |
| 20000 / lower transmission | 1.64 ± 0.52 | 99.80 ± 0.14 | 99.52 ± 0.30 | 0.44 ± 0.17 | 0.56 ± 0.33 | 7.12 ± 1.92 |
| 20000 / honest gate stress | 17.84 ± 1.87 | 4.92 ± 0.81 | 12.92 ± 1.79 | 0.40 ± 0.24 | 8.24 ± 1.11 | 6.88 ± 1.22 |
| 60000 / nominal | 4.08 ± 0.48 | 4.04 ± 0.70 | 3.56 ± 0.54 | 0.28 ± 0.30 | 1.12 ± 0.41 | 6.60 ± 0.51 |
| 60000 / lower transmission | 3.48 ± 0.41 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.36 ± 0.17 | 1.56 ± 0.48 | 7.60 ± 1.59 |
| 60000 / honest gate stress | 51.04 ± 2.54 | 6.12 ± 1.36 | 42.12 ± 3.06 | 0.56 ± 0.33 | 34.48 ± 2.64 | 8.08 ± 1.45 |
| 120000 / nominal | 3.72 ± 0.87 | 4.32 ± 0.78 | 3.92 ± 0.41 | 0.44 ± 0.17 | 1.44 ± 0.38 | 6.08 ± 0.74 |
| 120000 / lower transmission | 4.32 ± 0.90 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.40 ± 0.37 | 1.60 ± 0.73 | 7.04 ± 1.73 |
| 120000 / honest gate stress | 81.20 ± 0.86 | 4.72 ± 1.55 | 73.32 ± 1.31 | 0.64 ± 0.46 | 67.84 ± 0.84 | 6.08 ± 1.54 |

Do not claim equal-realized-FPR superiority from recall alone. The plug-in
comparator has no inherited signal-bound guarantee. The nominal known-loss
count null remains deliberately frozen under lower transmission, so its large
alarm rates represent misspecification, not a valid detection advantage.

### Verification and scope

Independent auditing passed **1320 rate/Wilson checks**, **960 matched AUC checks**,
**420000 independent score reconstructions**, **140000 CP-bound inversions** and
**880 physical marginal references**. Maximum absolute descriptive pooled physics
residual is **3.533** standard deviations over these many comparisons, not a
multiple-testing discovery criterion. Current/raw fingerprints, integer counts,
features, private audit-only knowledge, source inventory, fresh child streams,
matched windows/controls and frozen cutoffs all reconcile.

The full real-Aer regression suite passes **434 tests**. Numerical tests compare
all small Fisher tables to exact integer enumeration, large tables to SciPy,
and binomial tails at large counts to independent library calculations. Series
truncation has a geometric remainder bound <=1e-14; floating-point rounding is
separately tested/tolerated and is not implied to share that exact bound. A new
independent-auditor cache validation edge case was fixed: typed keys prevent
`True == 1` from bypassing integer-only input validation on a cache hit.
Production detectors and previous reviewed sources/results remain unchanged.

[Raw sessions](../results/benchmarks/window-power-v2/session_audit.csv.gz) and
[independent audit record](../results/benchmarks/window-power-v2/independent_audit.json)
are saved. This verifies the simulation comparison, not hardware, privacy
amplification, public parameter-estimation leakage, adaptive Eve or final-key
security. Fixed-window bounds cannot be reused for unlimited sequential checks.

### Next milestone, not executed here

Replace the unrealistic common-basis honest null with a **predeclared
physics-aware device/noise family**, then test honest gate-error control and
attack power on fresh data. A valid design must distinguish nuisance uncertainty
from an attack and retain irreducible observational ambiguities. Choose device
parameter bounds before holdout and keep parameter-estimation disclosure costs
explicit. The current study supplies a reproducible window/power requirement
and a device-model failure case, not a novel-physics claim or a free 20x gain.
No commit/push, poster, notebook rerun or historical report/PDF rebuild was done.
