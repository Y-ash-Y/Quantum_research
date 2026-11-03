# Classical-baseline audit: the direct detector is not the contribution to pursue

Date: 2026-11-03. **Share with caveats.** The bounded existing-data comparison
is complete. The broader optimality, practical-utility and publication-priority
questions remain open. New audit-only files were added; original detector code,
Step-12 evidence, historical reports and Git history were not changed.

## Decision in ordinary language

A standard classical nuisance-confidence-set construction recovers much of
the sensitivity lost by our conservative CP/Bernstein detector. Consequently,
continuing to tune that detector or searching more Eve policies against it is
not a justified novelty strategy at this point.

This does not invalidate the old experiment. Its saved counts and decisions
replay correctly. It changes their interpretation: poor observed sensitivity
outside the overlap boundary was partly a property of that conservative rule,
not evidence that every valid detector has the same poor sensitivity.

The population law-overlap obstruction is unchanged. This audit establishes
neither an optimal classical detector nor a new quantum/statistical method.
The implementation is an application of established methods, not a new discovery.

## What was compared

The [protocol](classical-audit-protocol.md) was written before the new replay.
This is nevertheless a **retrospective** audit: the underlying Step-12 data
and earlier method results had already been seen. It is not a fresh holdout.

We selected all 512 saved pairs in each of 24 cohorts: plans 9/10, the 1%
expected exactly-known retained raw-bit floor, zero/half-overlap allowances,
all five honest controls and the balanced floor-rate attack. That is 12,288
existing calibration/monitoring pairs. No simulator or new measurement campaign
was run. Each pair retains the previous four-million-emission resource plan;
those emissions are not new work performed in this audit.

Three methods use identical pooled public counts and unchanged gain flags:

1. The original direct CP/Bernstein rule.
2. A restricted-score normal-tail diagnostic, of Farrington–Manning type.
   It is approximate; its nominal threshold is **not** treated as a uniform
   finite-sample false-alarm guarantee.
3. A Berger–Boos confidence-set construction using binomial tails and a
   conservative continuous-nuisance upper envelope. Its size argument is
   finite-sample in real arithmetic under the same approved honest model.

For the two finite-sample rules, each of four error directions receives
alpha/8, with the remaining alpha/2 retained for the four original gain
directions. Total per-rule alpha is .05. This is equal **upper-alpha** control,
not an equal-attained-FPR comparison. The methods are not OR-ed together.

## Focal fixed-probe results

The attack has r=.02, beta=.5 and therefore the 1% raw-bit knowledge proxy.
That is not a claim of 1% final-key leakage. Half the overlap boundary is
0.2425 QBER percentage points of allowed change per basis.

| Plan | Allowance | Original direct | Classical BB | Approximate score |
|---|---|---:|---:|---:|
| 9 | Zero | 168/512 (32.81%) | 378/512 (73.83%) | 415/512 (81.05%) |
| 9 | Half boundary | 12/512 (2.34%) | 79/512 (15.43%) | 107/512 (20.90%) |
| 10 | Zero | 151/512 (29.49%) | 353/512 (68.95%) | 394/512 (76.95%) |
| 10 | Half boundary | 18/512 (3.52%) | 76/512 (14.84%) | 120/512 (23.44%) |

Do not substitute the approximate score's attractive numbers for the
finite-sample comparator's guarantee. Nor is 15% detection useful robust
protection merely because it is better than 2%.

Paired BB-minus-direct effects, in percentage points:

| Plan / allowance | Gain | Nominal simultaneous paired interval |
|---|---:|---:|
| 9 / zero | +41.02 | [31.97, 48.76] |
| 9 / half | +13.09 | [6.85, 18.95] |
| 10 / zero | +39.45 | [30.48, 47.17] |
| 10 / half | +11.33 | [5.43, 16.92] |

All four have zero pairs alarming only under direct: BB adds 210, 67, 202
and 58 alarms respectively. This is a pointwise observation, not a theorem
of uniform dominance over transcripts, honest parameters or attack policies.

All 72 rates have nominal fixed-rule binomial intervals, with .025/72 allocated
per rate for their joint family. The 24 effects use intervals for the mutually
exclusive BB-only/direct-only counts, .025/(24*2) per count interval. Their
endpoint differences give conservative paired intervals without assuming the
counts independent. The .05 total failure accounting applies to a fixed-analysis
interpretation. Because comparator design and scope are retrospective, these
are **not independent confirmatory intervals covering an unrestricted adaptive
research process**. Marginal intervals and all controls remain in the JSON.

## Honest controls and computational limits

Across the 20 selected honest cohorts, the largest counts are 2/512 for direct,
11/512 for BB and 16/512 for approximate score. The methods do not attain the
same honest alarm rate. More alarms on honest rows are false alarms, not gains.
These controls alone do not prove uniform validity or validate approximate score.

All 2,048 zero-transmission rows stay in the replay. There are 49,152 error
direction evaluations: 14,239 proved nonreject, 25,506 conservatively screened
nonreject, 1,212 bounded rejects, 8,192 zero-denominator directions and 3
budget-unresolved directions. Those three return nonreject. The search was not
extended or tuned after seeing that outcome. The aggregate method may still
alarm if another prebudgeted direction rejects. There were no observed gain
alarms in this selected replay; that branch is covered by the unchanged rule
and earlier tests rather than empirically exercised here.

## Why the classical comparator controls false alarms

Conditional on positive error-independent denominators, let

    T = Ym/nm - Yc/nc,
    Yc ~ Bin(nc,theta), Ym ~ Bin(nm,mu),
    0 <= theta,mu <= .5, mu-theta <= d.

For fixed p, the upper allowed monitoring probability is q(p)=min(.5,p+d).
The event T>=t is increasing in monitoring successes and decreasing in
calibration successes. Its boundary-tail p-value G_p(t) therefore upper-bounds
the honest tail for true theta=p and any allowed mu.

Set a=alpha/8, gamma=a/10. The calibration count gives a two-sided CP set C
with coverage >=1-gamma, intersected with [0,.5]. Define

    p_BB = min(1, gamma + sup_{p in C} G_p(T_observed)).

On the confidence-coverage event, the supremum includes the fixed true-parameter
tail p-value. Thus

    P(reject) <= P(theta not in C) + P(G_theta(T) <= a-gamma)
              <= gamma + (a-gamma) = a.

No independence of C and T is assumed. Both random error counts are integrated
in G; the reference distribution is not conditioned on observed Yc. If C is
empty, assigning p=gamma spends the same confidence-failure event, not another
budget. Swapping stages provides the downward direction. Integrating conditional
bounds over the random public denominators preserves validity.

### A continuum bound rather than an uncertified parameter grid

For a cell [l,u] covering possible calibration p, couple binomials monotonically:
calibration at l has no more successes than calibration at p, and monitoring
at min(.5,u+d) has no fewer successes than monitoring at q(p). Therefore that
corner's tail upper-bounds every parameter in the cell. The corner need not
itself satisfy the honest drift bound; it is a dominating distribution.

We subdivide the largest upper-envelope cell up to 64 times and reject only
when the maximum cell bound plus gamma is below the directional level. A mere
maximum over evaluated boundary points would not justify this guarantee.

Tail inclusion uses integer arithmetic:

    monitoring threshold(k) = ceil((nm*k + ym_observed*nc - yc_observed*nm)/nc).

This avoids losing equality mass through floating `ceil` rounding. Binomial
survival probabilities are evaluated at threshold-1. Truncated calibration
support retains a non-renormalized sum and adds a conservative omitted-mass
bound using confidence-interval endpoint distributions.

This is an established confidence-set p-value principle with an explicit
conservative implementation. It is not a new inequality, an exact solution
of the global nuisance maximum, or a replica of Chan's score ordering.

## Verification and source integrity

- All original 11 Step-12 source fingerprints and the public-count archive
  fingerprint match before and after replay.
- Every selected original direct decision matches the source archive.
- A separate checker reconciles 12,288 selected keys, 72 unique rate cells and
  24 unique paired effects, and independently inverts binomial intervals with
  SciPy's binomial-test interface instead of the runner's beta-quantile helper.
- New tests use independent finite-table sums for inclusive tails, actual
  null-size grids, CP coverage grids, empty/zero-count cases and truncation
  envelopes. Large-count references reverse the integration: monitoring PMFs
  times calibration CDFs, with full support. Near-cutoff saved rows check a
  rejection just below .00625 and a max-split unresolved interval straddling it.
- A separate read-only mathematical/implementation review found no flaw in
  the scoped real-arithmetic argument. This is another computational review,
  not human peer review or external-laboratory replication.

The final complete suite passed **896 tests in 64.33 seconds**, including
27 new tests across the two classical-audit test modules. Tests check selected finite examples; they do
not replace the uniform proof. Reference calculations share SciPy numerics.
Neither old nor new rule has an outward-rounded floating-point certificate.

Saved evidence:

- `results/benchmarks/classical-drift-audit-v1/audit.json`
- `results/benchmarks/classical-drift-audit-v1/decisions.csv.gz`
- `results/benchmarks/classical-drift-audit-v1/verification.json`
- `src/research/classical_drift_audit.py`
- `scripts/run_classical_drift_audit.py`
- `scripts/verify_classical_drift_audit.py`

## What this decision gate settles — and what it does not

**Settled enough to stop:** the bespoke direct detector should not be the
center of a methodological novelty claim. An established classical construction
provides much greater fixed-probe sensitivity with the same stated model-wide
upper false-alarm budget. Further Eve-search tuning of the old rule does not
address this fundamental comparator gap.

**Not settled:** publication priority of a genuinely QKD-specific resource or
identifiability result; exact minimax power under drift; optimal allocation;
the best classical test; real-device utility; or composable final-key security.
No Chan score-ordered global exact-test reproduction was performed. A stronger
exact comparator might improve further. The approximate score is a diagnostic
only. These omissions do not rescue a claim of detector invention.

Even the elementary overlap theorem needs practical motivation and a priority
check before being a paper contribution. A standard method beating our rule
does not logically prove every possible quantum-monitoring direction non-novel.
Conversely, these improved numbers are not our new discovery.

**Recommended stopping point:** do not start the old proposed continuous-bias
campaign, more ML models or sequential extensions. Either obtain one precise
QKD-specific theorem/operational question that is not a direct classical
corollary, or freeze the project as a reproducible application/limitations study.
No next experiment is automatically authorized by this audit.

## Review scope and coverage

Totals below are scoped comparison/claim units, not exhaustive verification
percentages. Zero observed defects does not mean an entire field was checked.
The underlying coverage record is retained with the thread's working evidence.

### Audit usefulness and presentation

| Category | Observed defects | Assessment |
|---|---|---|
| Usefulness/completeness | 0 / 1 | The frozen 24-cohort comparison is complete; global optimality and publication review are explicitly outside it. |
| Analytical clarity | 0 / 1 | Exact-in-real-arithmetic, approximate and retrospective evidence are distinguished. |
| Visual/interaction consistency | 0 / 1 | Numeric chart bindings are checked; browser rendering remains unverified because the local headless browser executable is absent. |

### Analytical correctness and robustness

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority/confidence | 0 / 3 | Original rule, classical construction and approximate score are scoped separately; no full Chan replication is claimed. |
| Value accuracy | 0 / 5 | Four focal paired comparisons plus the honest-control summary reconcile with saved decisions. |
| Within-chart agreement | 0 / 1 | Four Plan-9 points and nominal marginal interval bindings match the saved summary. |
| Complete source details | 0 / 1 | Source archive, selection, proof scope and retrospective caveats are retained. |
| Cross-artifact consistency | 0 / 1 | Original fingerprints and all selected original decisions remain unchanged. |
| Data-quality controls | 0 / 1 | Selected keys, trial IDs, zero denominators and unresolved outcomes are retained/reconciled. |
| Conclusion support | 0 / 2 | The old detector is not recommended as the contribution; no new novelty or optimality claim is asserted. |

## Literature and limits of the comparison

- [Farrington & Manning (1990)](https://pubmed.ncbi.nlm.nih.gov/2281232/),
  nonzero risk-difference nulls and restricted estimation.
- [Chan (1998)](https://pubmed.ncbi.nlm.nih.gov/9682328/), exact unconditional
  nonzero-margin two-binomial comparisons. Used as a priority/method lead,
  not presented as code reproduced in this audit.
- [Berger & Boos (1994)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476836),
  confidence-set nuisance maximization. Direct publisher access was blocked
  during this run; the construction and scoped proof are stated above rather
  than claiming a new full-text replication.

No full systematic literature refresh or hardware validation was performed.
The newly identified QKD drift preprint and older missing full texts remain
relevant to a later narrowly specified research claim, not prerequisites to
recognizing the classical-comparator weakness established here.

## Reproduction

From the project root and project environment:

```sh
python -m pytest -q
python quantum-ai-lab/scripts/run_classical_drift_audit.py --output quantum-ai-lab/results/benchmarks/classical-drift-audit-v2
python quantum-ai-lab/scripts/verify_classical_drift_audit.py quantum-ai-lab/results/benchmarks/classical-drift-audit-v1
```

The runner refuses overwrite and replays the same fixed scope. A new output
directory does not authorize a new simulator campaign or protocol tuning.
