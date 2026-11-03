# Bounded classical-baseline audit — frozen before replay

Date: 2026-11-03. This is a retrospective audit of existing Step-12 observations,
not a new independent holdout campaign, a detector invention, or a priority claim.
Original detectors, protocols and archives must remain unchanged.

## Question and selection

Does an established finite-sample nuisance-confidence-set approach materially
change the interpretation of the direct CP/Bernstein rule? Replay all 512 rows
of plans 9 and 10, floor_id=1 (1% raw-knowledge floor), allowance_id=0,1 (zero
and half-overlap allowance), cohort_id=0,1,2,3,4,7 (all five declared honest
controls and the balanced floor-rate probe). This is 24 cohorts / 12,288 pairs.
Selection follows the previous discussion, not new comparator results. It does
not characterize continuous worst-case power, best allocation or other attacks.

## Comparators and fairness

1. Original Step-12 direct rule, recomputed and checked against saved decisions.
2. A restricted-score (Farrington–Manning-type) normal-tail diagnostic. Constrain
   the fitted boundary probabilities to the declared [0,.5] error domain.
   This is **approximate**, not a uniform finite-sample-size comparator.
3. A Berger–Boos construction using exact binomial tail probabilities with
   conservatively bounded nuisance maximization. This is not a reproduction of
   Chan's score-ordered test and is not claimed optimal or the strongest method.

All methods use the same pooled public counts. Retain original gain flags and
allocate alpha/8 to each of four error directions, alpha/2 total, with the
unchanged gain directions using the other alpha/2. No alpha tuning. Original
unpooled CP rates can be retained as historical context, not the sole baseline.

## Finite-sample construction

For an upward error direction, let T=Ym/nm-Yc/nc, null pm-pc<=d, probabilities
in [0,.5]. Set a=alpha/8 and gamma=a/10. A two-sided CP interval [L,U] for pc
has coverage at least 1-gamma; intersect with [0,.5]. For fixed pc, monotonicity
in pm bounds the null tail by pm=min(.5,pc+d). Define

    p_BB = min(1, gamma + sup_{pc in [L,U]} P(T>=T_observed)).

The tail includes equality and is computed by summing the calibration binomial
PMF times monitoring survival probabilities with integer-exact thresholds.
The observed confidence set need not be independent of T: on coverage, the
fixed true-parameter tail p-value is at most the supremum; the complement costs
gamma. An empty domain intersection can alarm within that coverage failure.

For a nuisance cell [l,u], its entire boundary tail is bounded above by the
tail at calibration probability l and monitoring probability min(.5,u+d).
This follows by coupling the monotone event (increasing in Ym, decreasing in Yc).
Subdivide the largest upper cell at most 64 times. Reject only when all upper
bounds plus gamma are <=a. An unresolved maximization does not reject. A
midpoint/end-point lower bound above a merely stops futile refinement.
T<=d is conservatively screened to nonrejection. Zero denominator disables
that direction. Reflect stages for the downward direction.

Calibration support can be truncated using quantiles at the interval endpoints;
the full omitted mass is bounded by the left endpoint's lower-tail probability
plus the right endpoint's upper-tail probability and is **added** to upper
tails, never renormalized away. Target tail truncation is 1e-12. These validity
arguments are in real arithmetic; SciPy evaluations are not outward-rounded
certificates. Finite grid checks are not substitutes for the uniform argument.

## Validation and output

- Verify original source/archive hashes before reading and recheck after replay.
- Reconcile all selected keys and all original per-row direct decisions.
- Independently enumerate small binomial tables, check CP coverage and test
  null-size grids including parameter-domain boundaries and empty/zero counts.
- Verify cell upper envelopes against independent full-support tail sums.
- Record unresolved decisions and approximation limits, not just alarm rates.
- Save every selected row decision, summary rates and paired effects. Use
  joint exact-binomial intervals: .025 across 72 method rates, .025 across
  24 BB-minus-direct effects (two binomial endpoints each). Score intervals
  describe simulation alarm frequency, not a model-wide size guarantee.

No publication conclusion follows from a fixed power-gain threshold. If a
classical construction makes the bespoke rule unnecessary, stop treating the
rule as a candidate methodological contribution. Any remaining QKD-specific
theorem or operational benefit needs a separate explicit argument.

## Primary sources and implementation scope

- Farrington & Manning (1990), DOI 10.1002/sim.4780091208:
  https://pubmed.ncbi.nlm.nih.gov/2281232/
- Chan (1998), exact nonzero-margin binomial tests:
  https://pubmed.ncbi.nlm.nih.gov/9682328/
- Berger & Boos (1994), DOI 10.1080/01621459.1994.10476836:
  https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476836

The audit's raw-difference ordering and calibration-only confidence set are
explicit choices within Berger–Boos, not claims to replicate every choice in
the cited papers. Chan's full score-ordered global maximization remains a
possible stronger comparator; failure to beat the present comparator would not
establish optimality or novelty.
