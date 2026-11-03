# Step 5: independent Eve searches and a finite-grid uncertainty bracket

Protocol frozen on 2026-10-05 before new simulations. Research-only continuation
of [Step 4](exact-detector-comparison.md). The central exact detector, alpha=.05,
14 resource plans, B=4000000, stable attack-free calibration and the raw-knowledge
proxy remain unchanged. No production, historical PDF, commit or push updates.

## Motivation and predeclared search

The earlier weak-floor searches sometimes selected a biased policy even though
a separately evaluated balanced policy had lower observed detection. That is
finite Monte Carlo selection noise, not proof of the selected policy's true
optimality. Increase the per-floor grid from 15 to 45 policies: rate multipliers
{1,1.125,1.25,1.5,2} times 2*kappa and Eve Z-basis probability {0,1/8,...,1}.
The union across four floors has 153 unique policies. Eligibility is confined
to each floor's declared five rates; no broader union-rate substitution.

Two independently seeded searches A/B each use 512 pairs per plan/policy,
versus Step 4's 192. The primary score is central exact only; do not tune its
threshold or combine it with CP. Select A and B winners separately and a pooled
winner from 1024 trials, with frozen ties: lower rate, closest beta to .5,
then lower beta. Pooled counts also select a defender allocation from search
only using the earlier allocation tie rule. Search independence is checked by
exact winner agreement, not presented as a probability of being correct.

Before any evaluation, freeze a shortlist per plan/floor containing the top two
from A, top two from B, pooled winner, previous Step 4 central-exact response,
and rate-floor probes beta=0,.5,1. Deduplicate within a plan/floor only; this is
at most nine policies. Keep policies repeated across floors as separate cohorts.
All evaluations use 2048 fresh pairs and both frozen detectors score the same
public observations. Private retained-knowledge labels use a separate stream;
they are absent from public scorer inputs. Four stable controls use 1024 pairs
per plan, with no filtering or threshold tuning.

Master seed 2026100505; phase codes A=0, B=1, evaluation=2, control=3, with
plan and case identifiers in local seed sequences. Save all public counts and
decisions in compressed numeric blocks per phase/plan. Archive axes are
(case,trial,coordinate), with metadata mapping cases to policies. Search blocks
are bounded to about 10 MB raw count arrays per plan; streaming compressed
blocks avoids retaining the full experiment in memory. Private knowledge has
separate evaluation archives, not hidden error counts or bitstrings.

## A valid finite-grid bracket, not merely the lowest observed rate

Write p_g for the central exact alarm probability under a grid policy g.
Allocate confidence failure probability .025 to simultaneous exact binomial
intervals on the 2142 pooled search cells (14 plans times 153 policies). Allocate
another .025 to fresh evaluation, divided across twice the frozen evaluation
cell count because both detector rates are reported. Conditional on the frozen
shortlist, the evaluation intervals simultaneously cover all its rates. The
search and evaluation failure events together have probability at most .05.
No coordinate-level detector alpha is reused as this Monte Carlo confidence
budget: false alarms and confidence coverage are different probability claims.

For one plan/floor, let G be all its 45 eligible policies and S the frozen
shortlist. On the joint coverage event:

$$L=\min_{g\in G}L_g\le \min_{g\in G}p_g
\le \min_{s\in S}U_s=U.$$

The left inequality holds because every L_g<=p_g. The right holds because S is
a subset of G and every p_s<=U_s. Thus [L,U] brackets true minimum power over
the **declared finite grid**, even when the observed lowest policy is selected.
Search need not identify the true minimum to make this inequality valid.

For the pooled search-selected policy g*,

$$p_{g*}-\min_{g\in G}p_g\le U_{g*}-L.$$

Report this regret ceiling and whether it is at most a predeclared five percentage
points. Failure to certify that tolerance means insufficient evidence, not a
proved suboptimal policy. Keep inconsistent numerical brackets visible rather
than clipping away a possible coverage failure or implementation problem.

For the continuous IR family, min_continuous p <= min_grid p. Consequently the
grid lower bound does **not** certify continuous worst-case robustness. The
evaluated upper bound remains a constructive continuous upper bound because the
evaluated attacks are admissible; no unsearched attack is assumed easier.
Exact binomial bounds remain floating numerical calculations, not rounding
certificates or hardware/quantum-security guarantees.

## Completion boundary and planned checks

Verify expanded-grid containment of previous policies, eligibility, separate RNG
phases, source/output hashes, all saved counts and decisions, independent sampled
one-sided Fisher checks, private/public separation, frozen top-two selection,
all pooled intervals and shortlist intervals, and the bracket/regret arithmetic.
Exhaustive small synthetic selection fixtures check tie and envelope logic.
The two detector guarantees remain separate; no alarm OR or honest drift null.

## Frozen-run results

The completed evidence is in
[`best-response-stability-v2/stability.json`](../results/benchmarks/best-response-stability-v2/stability.json).
It contains 2,877,440 simulated calibration/monitoring pairs: 2,193,408 search
pairs, 626,688 fresh attack-evaluation pairs, and 57,344 stable-control pairs.
There are 306 frozen evaluation cells and 56 honest-control cells. Each pair is
an independent aggregate simulation of its declared emission budget; these are
not 2.88 million individual photons or experimental hardware observations.

Search A/B winner agreement is **26/56 plan/floor combinations**, broken down
as 2/14, 4/14, 10/14 and 10/14 for increasing knowledge floors. Agreement on an
exact policy is not agreement on its true power, and disagreement can reflect
near-ties as well as Monte Carlo noise. A single pooled selected response uses a
rate above its floor (plan 13, 1% floor); that plan discloses no monitoring errors,
so this is not evidence that stronger interception is generally stealthier.
All 56 finite-grid brackets are numerically consistent. The five-percentage-point
regret tolerance is certified on the joint coverage event for **17/56** responses
(5, 2, 1 and 9 for increasing floors). The other 39 responses are not certified;
this does not prove they are more than five points from optimal.

### Equal expected sacrifice, search-frozen defender plans

Every plan in this regime has the same expected 43,572.29 sacrificed sifted bits
and 43,572.29 retained raw candidates. These are expectations, not an exact
realized count match or final secret-key yield. Every rate below has denominator
2048 fresh pairs. `beta` denotes Eve's probability of choosing the Z basis.

| Expected certified raw-knowledge floor | Calibration M/B | Selected r; beta | CP alarms | Central exact alarms | Joint-95% bracket on finite-grid minimum power | Selected-policy regret ceiling |
|---|---:|---|---:|---:|---|---:|
| 0.25% | 33.333325% | .005; .75 | 1 (0.049%) | 168 (8.203%) | 3.446–9.172% | 7.508 percentage points |
| 0.5% | 20% | .01; .5 | 9 (0.439%) | 330 (16.113%) | 11.313–19.659% | 8.345 percentage points |
| 1% | 33.333325% | .02; .5 | 91 (4.443%) | 1199 (58.545%) | 54.097–62.977% | 8.880 percentage points |
| 2% | 20% | .04; .5 | 1529 (74.658%) | 2043 (99.756%) | 98.130–99.985% | 1.854 percentage points |

The brackets concern the **true minimum over 45 declared attacks**, not a
confidence interval for the selected policy's alarm rate. The selected rate can
exceed the bracket's upper endpoint without contradiction. For example, at the
weakest floor the selected beta=.75 policy has marginal exact 95% interval
7.051–9.477%, while the balanced beta=.5 probe gives **136/2048 = 6.641%**
(marginal interval 5.601–7.807%). Its simultaneous evaluation upper bound,
9.172%, provides the grid-minimum upper endpoint. These are different estimands.
The lower observed balanced rate does not, by itself or by overlapping intervals,
prove a population ordering or make it the continuous best response. We do not
replace the search-frozen selection using this holdout.

For the fixed-disclosure-fraction regime, the four frozen defender plans allocate
20%, one third, one third and one third of emissions to calibration. Their
selected central-exact rates are 143/2048 (6.982%), 334/2048 (16.309%),
1238/2048 (60.449%) and 2044/2048 (99.805%). Different allocations and attack
policies across runs prevent treating Step 4 versus Step 5 as a paired detector
improvement. The 1% equal-sacrifice allocation changed from 20% in Step 4 to one
third here; neither search establishes a unique population-optimal allocation.

### Stable controls and interpretation

Across 56 predeclared control cells, the largest central-exact alarm rate is
**54/1024 = 5.273%**, with marginal exact 95% interval 3.986–6.825%. The largest
CP rate is 5/1024 = 0.488%. These observed maxima do not prove either false-alarm
guarantee, and the controls are not equal-attained-FPR calibration. All
zero-transmission controls produce zero alarms. With zero monitoring disclosure,
gain coordinates can still alarm; nominal plan 13 has 12/1024 exact alarms,
which cannot diagnose Eve's error disturbance.

The result is consistent with the earlier restricted-model theory: weak IR
disturbances can remain hard to separate from honest aggregate variation, while
stronger attacks are readily detected. The exact test's observed gain over CP
does not establish quantum advantage, a new detector, attack attribution, or
composable security. This study improves the credibility of the benchmark and
makes finite-search uncertainty explicit; **publication novelty remains
unverified**. No all-attacks robustness claim follows from the high 2% grid lower
bound because a lower bound over a grid need not hold between its points.

## Reproduction and audit boundary

Run from the repository root in the project's existing environment:

```sh
.venv/bin/python quantum-ai-lab/scripts/run_best_response_stability.py --output quantum-ai-lab/results/benchmarks/best-response-stability-v2
.venv/bin/python quantum-ai-lab/scripts/audit_best_response_stability.py quantum-ai-lab/results/benchmarks/best-response-stability-v2 --save
.venv/bin/python -m pytest -q
```

Do not overwrite an archived evidence directory when rerunning; use a new output
name. The first v1 execution completed its simulations but failed while
serializing a NumPy boolean, leaving a zero-byte summary. Its archives are
preserved as an **incomplete run**, not evidence for the numerical claims above.
The v2 rerun uses the identical seed/protocol after converting summary values to
native JSON types; the summary is now serialized before its file is opened.
A regression test guards this repair. No result-driven search tuning occurred.

The current full suite passes **646 tests**, including 19 new Step 5 tests.
The saved [audit](../results/benchmarks/best-response-stability-v2/audit.json)
passes: 2,877,440 public pairs and 626,688 private rows reconcile; **3,561,472
rule decisions** replay; 4,646 summary cells and 2,142 pooled search intervals
reconcile; all 56 shortlists, 8 defender selections and 56 envelopes are
reconstructed. The audit checks 8 source hashes, 2 previous-evidence hashes and
the complete 71-file output-hash manifest, including the frozen shortlist.
It independently checks **1,836 sampled rule decisions** through scalar APIs.
Full decision replay uses the frozen scorers; sampled scalar
Fisher/Clopper–Pearson checks use different APIs but share SciPy's distribution
machinery. Neither is a wholly independent physics simulator or a certified
arbitrary-precision computation.

## Follow-up status

The proposed next milestone was an off-grid/analytic attack-characterization study:
investigate rate monotonicity, basis symmetry and the error-power geometry before
choosing a justified continuous search. [Step 6](off-grid-geometry.md) now implements
that study with a separately frozen confirmation set and the same detector and
resource constraints. It establishes reflection and KL-proxy properties but
does not establish balanced detector optimality or a useful full-family lower
bound at the main allocations. A literature comparison and stronger
detector-specific argument remain required before proposing a novelty claim.
