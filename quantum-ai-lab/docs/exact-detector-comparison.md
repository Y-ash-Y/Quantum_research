# Step 4: exact conditional detector versus interval separation

Protocol frozen on 2026-10-05 before running the new search and evaluation.
Research-only continuation of [Step 3](allocation-frontier.md); no production
detector, historical PDF, commit or push changes are included.

## Question and unchanged resources

Does direct conditional two-sample testing recover detection power lost by the
conservative CP interval-separation rule, at the same stable-null false-alarm
budget? Keep Step 3's 14 plans, B=4000000, two resource regimes, four expected
interception-certified raw-knowledge floors, 45-policy union grid and each
floor's 15-policy candidate set. Loss, gates, sampling and the broad honest
Pauli family remain unchanged. Raw-knowledge proxies are not final-key leakage.
Stable attack-free calibration remains a trust assumption, not a learned fact.

## Intuition and exact conditional derivation

For a public coordinate, calibration provides yc successes in nc observations
and monitoring ym in nm. A success means a disclosed error for four coordinates,
or detection for two intensity-gain coordinates. Under the stable null both
stages have the same unknown success probability p, conditional on their public
denominators. Let t=yc+ym and L=nc+nm. Independence of the two binomial counts
gives

$$P(Y_c=k,Y_m=t-k)=\binom{n_c}{k}\binom{n_m}{t-k}
p^t(1-p)^{L-t}.$$

Summing over k yields Binomial(L,p)'s probability of t, by Vandermonde's
identity. Dividing joint by that sum cancels the unknown p:

$$P(Y_c=k\mid t,n_c,n_m)=
\frac{\binom{n_c}{k}\binom{n_m}{t-k}}{\binom{L}{t}},$$

for max(0,t-nm)<=k<=min(nc,t). This is a hypergeometric distribution: allocate
the t observed successes among L trials, drawing nc trials for calibration.
Error counts are generated only for publicly disclosed bits; hidden error
counts and Eve labels never enter this conditioning or the scorer.

Define the **central exact** two-sided p-value

$$p_{central}=\min\{1,2\min[P(Y_c\le y_c),P(Y_c\ge y_c)]\}.$$

The upper tail is `sf(yc-1)`, including the observed atom. Using `sf(yc)` is
an off-by-one defect, explicitly tested. Each tail is a super-uniform discrete
tail probability under its conditional null. Splitting a coordinate's rejection
budget delta equally between tails gives P(pcentral<=delta)<=delta. Set
delta=alpha/6, with alpha=.05, and alarm if any of six coordinates rejects.
A union bound gives family-wise false-alarm probability <=alpha without
coordinate independence. Empty stages and degenerate pooled successes return
p=1; there is no fabricated evidence from an absent error sample.

This is a standard central doubled-tail conditional test, **not** the default
probability-ordered two-sided Fisher p-value, and not a new detector invention.
The conditioning law and one-sided conventions are documented in
[SciPy's Fisher documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
The distinction between two-sided conventions is documented in
[R's Fisher documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html).
Small-table rational combinatorial tests independently check the formula and
conditional size; SciPy one-sided Fisher calls cross-check the two tails.

The comparator CP rule is unchanged: six coordinates, two stage intervals,
one overall alpha=.05. Both methods are individually valid fixed-horizon rules;
their OR is not another authorized 5% rule. Numerical tail calls are floating
point, not outward-rounded certificates. This research API refuses total
coordinate denominators above 2^31-1, covering the present four-million budget
without claiming unrestricted large-count support.

## Frozen search and paired comparison

Seed 2026100504; 192 paired search trials per plan/policy. Each public transcript
is scored by **both** rules, while private knowledge uses a separate RNG stream.
Each rule independently selects its Eve response and defender allocation from
search only, with Step 3's predeclared ties. Persist both selections before any
evaluation. Previous held-out values are neither reused to tune the rule nor
pooled with the new experiment.

Fresh evaluation uses 600 paired trials per plan/floor/selection-rule, under
that selection-rule's frozen Eve policy. Both rules score each identical public
transcript. This yields 112 evaluation cells. If the two rules select the same
policy for a plan/floor, their two evaluation cohorts remain independently seeded;
they are reported separately, not silently pooled. Four stable honest controls
use 400 fresh pairs per plan, scored by both rules; none tunes a threshold.

For every held-out cell report both alarm rates, exact marginal 95% binomial
intervals, and simultaneous 95% intervals adjusted across 224 rule/cell rates.
Report exact-only and CP-only alarms from the same rows. Their paired difference
is (exact_only-CP_only)/trials. A conservative marginal 95% difference interval
subtracts each rule's 97.5% exact bounds; simultaneous difference bounds subtract
the corresponding 224-rate simultaneous bounds. These intervals are conservative
and not paired-score optimal, but they have stated coverage without assuming
independent detector outcomes. Do not decide significance by interval overlap.

The paired equal-power test conditions on the total discordant pairs and uses
an exact Binomial(discordant,1/2) test (McNemar's exact conditional test). Zero
discordances give p=1. Adjust paired p-values across the 112 evaluation cells
with Bonferroni when discussing cellwise differences. A different selected Eve
policy can change a method's own-response rate, so do not subtract rates from
different attacks and call that a matched detector improvement.

## Verification and completion boundary

Verify combinatorial conditional size, tail inclusivity, symmetries, zero
denominators, immutable public inputs, count-domain limits, source/output hashes,
all saved decisions and denominators, every selected response, paired differences
and uncertainty, and sampled independent one-sided Fisher replays. Private labels
must remain absent from public exports. All rows survive; no usability filter.

This comparison does not remove the model's indistinguishable honest-channel
cases, strengthen final-key security, cover drift or stopping rules, establish
global best responses, or prove publication priority. Results are recorded below.
A poor result is retained, not followed by
unregistered tuning on the same evaluation set.

## Results: matched improvement, not a new quantum discovery

Saved [summary](../results/benchmarks/exact-comparison-v1/comparison.json),
[frozen choices](../results/benchmarks/exact-comparison-v1/frozen_selections.json),
and [audit](../results/benchmarks/exact-comparison-v1/audit.json) cover 630 search,
112 fresh evaluation and 56 control cells: 210560 public pairs and 188160 private
rows. Both rules score every public pair.
The runner regenerates the public-count and private-proxy CSV archives locally;
these bulk archives are not bundled with this release.

The table uses the **central-exact rule's own search-selected defender allocation
and Eve response** in the fixed-sacrifice regime. Both scores in each row come
from the same 600 fresh paired trials, not separate attacks. The first weak-floor
policy uses beta=0; the others use beta=0.5. All rates are r=2*kappa.

| Expected certified raw fraction (%) | Calibration share (%) | Eve beta | CP alarms / 600 | Central exact alarms / 600 | CP marginal 95% interval (%) | Exact marginal 95% interval (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | approximately 33.33 | 0 | 3 | 66 | 0.103–1.454 | 8.610–13.781 |
| 0.50 | 20 | 0.5 | 2 | 94 | 0.040–1.199 | 12.850–18.826 |
| 1.00 | 20 | 0.5 | 27 | 360 | 2.986–6.480 | 55.955–63.946 |
| 2.00 | approximately 33.33 | 0.5 | 437 | 600 | 69.084–76.356 | 99.387–100.000 |

At the 1% floor the paired gain is 55.5 percentage points; its conservative
simultaneous 95% interval is 43.96–65.26 points. At the 2% floor it is 27.17
points with interval 19.27–34.28 points. Bonferroni-adjusted exact paired-test
p-values across all 112 evaluation cells are approximately 1.28e-98 and
1.92e-47, respectively. These are matched tests, not interval-overlap arguments.
The 600/600 result is an empirical point estimate, not guaranteed 100% power;
its simultaneous lower bound is approximately 98.495%.

The CP-selected 1% cohort uses approximately 33.33% calibration and balanced
Eve; it gives 33/600 CP versus 352/600 exact alarms. CP-selected 2% uses 20%
calibration and gives 452/600 versus 598/600. These independently seeded cohorts
corroborate the paired improvement without pooling trials across plans.

At fixed 2% knowledge all exact-selected policies in the equal-sacrifice plans
are balanced, allowing the following same-attack resource comparison:

| Calibration share (%) | CP alarms / 600 | Central exact alarms / 600 | Paired gain (percentage points) |
| --- | --- | --- | --- |
| 5 | 105 | 403 | 49.67 |
| 10 | 295 | 586 | 48.50 |
| 20 | 451 | 598 | 24.50 |
| approximately 33.33 | 437 | 600 | 27.17 |
| 40 | 317 | 582 | 44.17 |
| 45 | 144 | 480 | 56.00 |
| 50 | 0 | 9 | 1.50 |

The 50% endpoint reveals no monitoring errors. Its nine exact alarms arise from
unchanged gain-coordinate fluctuations, not identifiable attack evidence. Step
3's complete-law overlap still limits valid power to alpha=5% at this plan.
The new rule does not circumvent that bound. The resource trade-off persists,
though its observed shape and rates depend on the detector.

### False alarms and honest interpretation

Both rules target the same **nominal 5% ceiling**, not the same attained
false-alarm rate. CP is substantially more conservative. The direct rule uses
the allowed error budget less conservatively; improved power alone does not
prove superiority at equal measured false-alarm rates.

| Stable control | CP observed range across plans (%) | Central exact observed range across plans (%) |
| --- | --- | --- |
| Nominal | 0–1.00 | 2.00–5.50 |
| Gate error | 0–0.25 | 1.50–5.00 |
| High-error Pauli | 0–0.25 | 1.00–6.00 |
| Zero transmission | 0 | 0 |

Each control cell has 400 pairs. The largest exact empirical rate is 24/400=6%
under high-error Pauli noise at 40% calibration in the fixed-sacrifice regime.
Its marginal exact 95% interval is 3.882–8.796%. A one-sided exact binomial test
against 5% gives p=0.2073 (before correcting for inspecting 56 cells). This is
not statistically significant evidence of inflation, but neither the test nor
finite controls establish the theoretical guarantee. Do not round down the
observation, claim every empirical rate is below 5%, pool unrelated controls,
or tune alpha after seeing this result. The stable-null guarantee comes from
the conditional-test derivation in real arithmetic, with numerical limitations
stated above.

Across 112 held-out cells there are 15017 exact-only decisions and no CP-only
decisions. These heterogeneous totals are a replay observation, not a pooled
power estimate or proof of uniform dominance. In 105 cells the positive paired
difference survives Bonferroni correction; all remaining cells are retained.

### Limits, verification and next decision

Weak-floor search remains noisy. The CP-selected balanced 0.25% cohort at
approximately one-third calibration gives 1/600 CP versus 41/600 exact alarms
(6.83% exact), below the exact-selected biased cohort's 11% exact point estimate.
Neither frozen choice is retrospectively changed. This demonstrates why
selected-policy power is **not verified worst-case power**, even within a small
search grid. No continuous global optimum is established. Step 3's information
ceilings remain necessary, not achievable power targets.

The new module adds **25 tests**; the full suite passes **627 tests**. Small
conditional tables are exhaustively enumerated for stage sizes 1–7 using exact
rational combinatorial probabilities. The audit reconciles 210560 public pairs,
188160 private rows and 798 summary cells; replays 421120 rule decisions; checks
128 selections and six source hashes; and independently replays 672 rule
decisions on 336 predetermined held-out rows. Paired equal-power p-values are
recomputed with exact integer combinatorial sums. Full replay shares the scorers;
sampled one-sided Fisher checks share SciPy's distribution machinery, whereas
the small-table rational tests do not. These scopes are not fully independent
validation of every numerical result.

Reproduce from the repository root with NumPy, SciPy and pytest in the existing
environment. Use a new output path; the runner refuses to overwrite evidence:

```sh
python quantum-ai-lab/scripts/run_exact_comparison.py --output quantum-ai-lab/results/benchmarks/exact-comparison-rerun
python quantum-ai-lab/scripts/audit_exact_comparison.py quantum-ai-lab/results/benchmarks/exact-comparison-rerun
python -m pytest quantum-ai-lab/tests/test_exact_comparison.py -q
python -m pytest -q
```

**Claim boundary:** a reproducible resource-matched comparison shows that a
standard conditional test recovers substantial power lost by the project's
conservative interval baseline, using public counts rather than oracle
probabilities or private labels. This strengthens the testbed but does not
establish a novel quantum principle, composable security or publication priority.
The disclosure-budget frontier remains a candidate research framing.

**Proposed next milestone, requiring confirmation:** validate best-response
stability with a higher-resolution, larger independent Eve search, particularly
at weak knowledge floors, keeping the new standard detector as a baseline.
Resolve selected-policy uncertainty before advancing a worst-case frontier claim.
Literature priority and drift-aware attribution remain separate unresolved work.
