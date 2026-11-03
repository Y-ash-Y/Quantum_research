# Step 2: finite calibration and public-sample protocol

Protocol frozen before the pilot run: 2026-10-05. This is a research-only
fixed-horizon prototype, not a change to the existing production detectors.
The user confirmed attack-free, stable calibration and broad Pauli controls.
Novelty is **not established**. This step operationalizes the restricted
[identifiability model](../src/research/identifiability.py); standard confidence intervals
and a change-detection rule are not claimed as discoveries.

## 1. Question and trust boundary

Can a count-only alarm detect a change from independently obtained calibration
without oracle noise parameters, while controlling false alarms for stable
honest device/channel conditions?

The null is stability, not a label meaning "Eve absent under arbitrary drift."
Conditional error probabilities for Z0,Z1,X0,X1 and signal/decoy gains must
remain unchanged from calibration to monitoring. The null includes the narrow
H/X gate model, arbitrary physical Pauli channels, and other fixed conditional
Bernoulli probabilities under the sampling premises below. It is deliberately
broader than a gate-only model. A change alarm does not identify its cause.

Simulation assumptions:

- Calibration is attack-free, authenticated, independent of the monitoring
  sample, and taken under the same stable honest conditions. Authentication,
  attack-free access and stability are assumptions, not features implemented here.
- Pulses and Alice/Bob choices are iid within each stage. A uniformly random
  preparation bit and two independently uniform bases give four matched
  bit/basis strata, each with probability 1/8, and mismatches with probability 1/2.
- Signal intensity 0.5 and decoy intensity 0.1 are independently chosen with
  probability 1/2. Click probabilities are 1-exp(-mu*eta). Loss is independent
  of preparation bit, basis, error and sampling membership.
- Error-test membership is drawn independently of errors, **after** sifting.
  The membership randomness is unavailable to a future Eve before interaction.
  This prototype uses NumPy offline randomness, not a cryptographic generator.
- The two stages use the same six observation definitions. No accepted-session
  filter, threshold tuning, optional stopping, or outcome-based sampling is used.

Source flaws, basis-dependent losses, coherent/memory attacks and hardware
drift require additional analysis. An attacker present during calibration could
become part of the baseline and evade this test; the trust boundary is essential.

## 2. Public sampling and sacrificed-bit accounting

All matched calibration bits are test-only and sacrificed. Monitoring uses a
Bernoulli public test sample: each matched detected bit is included independently
with predeclared probability f=0.25. Revealing a test bit means revealing the
positions/basis/preparation value needed to compare Alice/Bob outcomes through
an authenticated channel. Both parties discard those positions from candidate
key material. This is a transcript **count marginal**, not a complete network
exchange or a byte-cost accounting implementation.

For stratum j, let S_j be matched detected bits, T_j publicly tested bits,
and E_j errors among those tested bits. Then

$$T_j\mid S_j\sim\mathrm{Binomial}(S_j,f),\qquad
E_j\mid T_j\sim\mathrm{Binomial}(T_j,q_j).$$

Here q_j is the stable conditional wrong-bit probability; it is known to the
simulator but **not passed to the scorer**. For each stage the public scorer
receives (E_j,T_j) for j=Z0,Z1,X0,X1, and (K_i,N_i) for signal/decoy, where N_i
is emitted pulses at intensity i and K_i is detected pulses at that intensity.
S_j is retained only for budget reconciliation. Undisclosed error counts, raw
bitstrings and Eve's known bits are not generated/exported in this module.

Let barQ=(Q_signal+Q_decoy)/2 be mean click probability and N the monitoring
emissions. Each stratum has expected S_j=N*barQ/8. Therefore

$$\mathbb{E}[\text{monitor disclosed}]=fN\bar Q/2,$$
$$\mathbb{E}[\text{remaining candidates}]=(1-f)N\bar Q/2,$$
$$\mathbb{E}[\text{calibration sacrificed}]=M\bar Q/2.$$

Every row obeys tested<=sifted, errors<=tested, detections<=sent,
signal_sent+decoy_sent=emitted, and disclosed+remaining=sifted. Remaining
candidates are **not** a secret-key length: error correction, leakage,
privacy amplification and a composable security proof remain absent.

## 3. Finite-sample honest region

For one paired experiment, the total false-alarm budget is alpha=0.05. There
are d=6 coordinates and two stages. Allocate delta=alpha/(2*d)=alpha/12 to
each interval. Its two tails receive delta/2=alpha/24. Budgeting two stages
is necessary: treating a noisy calibration estimate as known would invalidate
the guarantee. Each rule below is evaluated separately; their alarms are not ORed.

For observed successes k among n public trials, n=0 gives [0,1] and cannot
create a separation. Define qhat=k/n when n>0.

### Conservative Hoeffding baseline

For independent Bernoulli variables, Hoeffding's standard inequality is

$$\Pr(|\hat q-q|\ge\epsilon)\le2\exp(-2n\epsilon^2).$$

Set the right side equal to delta. Dividing by two, taking natural logarithms,
then multiplying by -1 yields 2*n*epsilon^2=log(2/delta). Hence

$$\epsilon=\sqrt{\frac{\log(24/\alpha)}{2n}},\qquad
C=[\max(0,\hat q-\epsilon),\min(1,\hat q+\epsilon)].$$

This bound is uniform in q but does not adapt to rare errors and can be loose.
It is included as a transparent baseline, not a competitive novelty claim.

### Clopper–Pearson baseline

An interval can instead invert exact binomial tail probabilities. For 0<k<=n,
the lower endpoint L solves

$$\sum_{j=k}^{n}\binom nj L^j(1-L)^{n-j}=\delta/2.$$

For 0<=k<n, the upper endpoint U solves

$$\sum_{j=0}^{k}\binom nj U^j(1-U)^{n-j}=\delta/2.$$

The standard binomial/incomplete-beta identity gives
L=BetaInverse(delta/2;k,n-k+1) and
U=BetaInverse(1-delta/2;k+1,n-k). BetaInverse is the quantile of the beta
probability distribution, a continuous distribution on [0,1]. Endpoint
conventions are L=0 at k=0 and U=1 at k=n. The real-arithmetic coverage is
at least 1-delta; discrete binomial tails generally make it conservative.

Implementation uses SciPy beta quantiles and is checked against a separate
binomial confidence-interval interface plus independent finite binomial
enumeration. These checks are not an outward-rounded numerical certificate.
The analytical guarantee is in real arithmetic; extreme machine-rounding
behavior is not certified. Supported alpha is restricted to [1e-12,1).
[SciPy's binomial-test interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
documents the exact confidence-interval option used as a cross-check.

### Alarm and the uniform stability guarantee

Let C_cal,j and C_mon,j be the two intervals for coordinate j. Alarm if **any**
pair is strictly disjoint. Equality of endpoints is not an alarm. Under a
stable honest law both intervals concern the same q_j. If all twelve intervals
contain their true probabilities, every corresponding pair overlaps. Thus

$$\{\text{alarm}\}\subseteq
\bigcup_{s\in\{cal,mon\},j=1}^{6}\{q_j\notin C_{s,j}\},$$
$$\Pr_H(\text{alarm})\le\sum_{s,j}\Pr_H(q_j\notin C_{s,j})
\le12\delta=\alpha.$$

The union bound needs no independence across coordinates. Random denominators
are allowed: conditional on a denominator n, its errors are binomial with the
same q; averaging the conditional coverage over n preserves the bound.
Independent error-blind sampling and the specified detection model justify
that conditional distribution. For each allowed fixed honest law the same
bound holds, so its supremum over this continuous stable family is <=alpha.
This does not mean the finite control grid proves the theorem.

The guarantee is **unconditional over paired experiments**. It is not a
per-realized-calibration conditional guarantee, a sequential/repeated-window
bound, a final-key security statement, or a rule allowing ordinary drift without
false alarms. Running both rules and choosing the more favorable one post hoc
would require a new multiple-testing budget.

## 4. Broad-noise controls and attribution

An honest I,X,Y,Z Pauli mixture with weights summing to one has conditional
errors q_Z=p_X+p_Y and q_X=p_Z+p_Y, the same for preparation bit 0 and 1.
All such stable mixtures are in the null, including high-error mixtures outside
the narrow depolarizing H/X gate family.

The IR-equivalent Pauli channel from Step 1 is included as a stable honest
control at both stages. It must not alarm merely because it resembles an attack.
Separately, attack-free nominal calibration followed by an IR interaction changes
the observation law and can alarm. An honest change from nominal to that same
Pauli channel would produce the same data. Trusted stability is what makes that
change anomalous, **not** a way of attributing its cause from Bob's data alone.
An explicit honest-noise-drift case is labeled outside the null to show this limit.

## 5. Frozen small validation study

No fitting or hyperparameter search. Seed 20261005; 400 independent repetitions
per resource/scenario cell; both rules use the same paired transcripts. The
three (calibration M, monitoring N) emission budgets are:

- (2000, 2000000): scarce calibration despite abundant monitoring.
- (200000, 200000): smaller paired resource budget.
- (2000000, 2000000): larger paired resource budget.

Honest controls, all stable: noiseless, nominal depolarizing, H/X gate error,
IR-equivalent Pauli, high-error Pauli, fully mixed, and zero transmission.
Diagnostic attacks: fixed-Z IR at r=0.0396 and r=1, with lambda=0.03 and eta=0.15.
The former is the aggregate-overlap boundary from Step 1 with ideal attacked
gates; the latter is only a strong positive control. Neither is an optimized Eve.
An honest lambda change from 0.03 to 0.3 is an outside-stability stress case,
not a false-alarm control. No PNS benchmark or adaptive search is performed here.

All cells include zero-denominator rows. Alarm-rate uncertainty uses marginal
95% exact binomial intervals over repeated paired experiments. They are not
simultaneous across cells and are not calibration-parameter confidence regions.
Resource panels do not hold total emissions or disclosed bits equal: they
validate mechanisms, not an optimal budget allocation.

Reproduce from repository root (new output directory required):

```sh
.venv/bin/python -m pytest -q quantum-ai-lab/tests/test_calibration_protocol.py
.venv/bin/python quantum-ai-lab/scripts/run_calibration_pilot.py \
  --output quantum-ai-lab/results/benchmarks/calibration-protocol-v1
.venv/bin/python quantum-ai-lab/scripts/audit_calibration_pilot.py \
  quantum-ai-lab/results/benchmarks/calibration-protocol-v1
.venv/bin/python -m pytest -q
```

The sampler draws exact count marginals, not millions of simulated pulse logs.
Its work/memory scale with repeated experiments and six coordinates, not N.
Results retain every public count row, scenario definitions, library versions,
source hashes and an output hash. No raw secret key is produced.

## 6. Pilot results and what they establish

Saved [results](../results/benchmarks/calibration-protocol-v1/pilot.json)
and [audit](../results/benchmarks/calibration-protocol-v1/audit.json).
The runner regenerates the complete public-count CSV locally; that generated
archive is not bundled with this release.
There are 12000 paired transcripts and 60 rule/resource/scenario result cells.
Each cell uses 400 independent paired experiments; method outputs share those
experiments and must not be treated as independent replications.

For weak fixed-Z IR (r=0.0396), observed detection is:

| Calibration M | Monitoring N | Hoeffding alarms | CP alarms | CP rate, marginal 95% exact interval |
| ---: | ---: | ---: | ---: | --- |
| 2000 | 2000000 | 0/400 | 1/400 | 0.25%, [0.0063%, 1.3850%] |
| 200000 | 200000 | 0/400 | 15/400 | 3.75%, [2.1138%, 6.1097%] |
| 2000000 | 2000000 | 0/400 | 400/400 | 100%, [99.0820%, 100%] |

The first and last rows hold monitoring emissions and disclosure fraction fixed;
calibration increases by a factor of 1000. Their expected monitoring disclosure
budget is the same, about 10893 test bits. In the observed weak-attack cells,
mean calibration sacrifice increases from 43.875 to 43599.4075 bits, and mean
monitor disclosure changes only by sampling fluctuation (10892.2425 versus
10896.255). This is evidence for a calibration-resource bottleneck **in this
frozen model/rule**, not a minimal required budget or universal discovery.
The middle row changes both budgets, so it cannot isolate calibration's effect.

The conservative Hoeffding rule fails to detect this weak shift even at the
largest tested budget. CP substantially improves that setting without a new
physics claim. This is an established statistical improvement, not a novel
algorithm. A 400/400 result is a Monte Carlo estimate, not guaranteed power.

Across the 42 stable-honest cells (7 controls x 3 resources x 2 rules), observed
false-alarm rates range from 0 to 1%. The highest is 4/400, with marginal exact
95% interval [0.2731%, 2.5405%]. These finite controls do not prove uniform
false-alarm control; the declared iid sampling assumptions and interval/union
argument supply that real-arithmetic guarantee. Rates from different cells are
not pooled into one population, and no cell was discarded.

Full fixed-Z IR is a positive control: CP alarms on 338/400 with scarce
calibration, then 400/400 in both other resource panels. Honest noise drift
alarms on 400/400 for CP in the two larger-calibration panels. This is **not**
an in-null false-alarm failure: that stress case violates stability. It shows
why a successful alarm cannot prove that Eve, rather than honest drift, caused
the change. No claims of attack classification or final-key security follow.

Verification: 64 new focused tests pass, and the full suite passes 563 tests.
The audit reconciles all 12000 public pairs, all 24000 rule decisions and all
60 summaries/resource means. Three predeclared rows per cell are independently
checked with scalar Hoeffding/binomial-CI calculations (180 sampled decisions).
Full replay shares the scorer; the independent calculation is sampled, not a
claim of independent full coverage. Source/output hashes agree.

## 7. Completion and next decision

Step 2 is complete within this scope. Production detectors, existing PDFs,
the final report and Git remote were not changed. Next would be a
budget-controlled detectability study: stronger exact
two-sample tests or likelihood methods, disclosure fractions, calibrated attack
information constraints and matched baselines. That is a separate milestone,
not an automatic next action. Repeated-window monitoring, calibration poisoning,
drift-tolerant nulls, real hardware and publication priority remain open.
