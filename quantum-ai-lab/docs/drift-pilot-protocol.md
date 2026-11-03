# Step 11 frozen protocol: bounded-drift baseline pilot

Frozen on 2026-10-09 before generating any pilot results. This is a small fixed
simulation study of the approved Step 9 model and Step 10 baselines. No fitting,
allocation selection, enlarged search after results, hardware claim, final-key
claim, commit or push. Historical and stationary outputs must remain unchanged.

## Question, estimands and resources

How much alarm power do the two confidence-gap baselines retain when required
to allow bounded honest between-block drift? Alpha=.05 per rule per fixed-horizon
pair, not .05 for the OR of the two rules. Both rules score identical public
observations. Their attained FPRs need not be equal.

Total B=4,000,000 emissions, lambda=.03, nominal q0=.015, eta=.15. Calibration
is independent and attack-free. Bits/bases/intensities/disclosure/loss obey the
existing iid public-count sampler. Calibrations disclose all their sifted bits.
Keep C=M+f*(B-M)=2,000,000 fixed for equal expected sacrifice and candidate counts:

- Plan 9: M=800,000, monitoring N=3,200,000, f=.375.
- Plan 10: M=1,333,333, N=2,666,667, f=666667/2666667.

These plans are fixed, not optimized here. Realized counts are random and remain
in every record. Expected cost equivalence is not equal hardware cost or secure
key yield. Floors kappa=.005,.01,.02 constrain the expected exactly-known retained
raw-bit proxy r/2; it is not mutual information or final-key leakage.

For each floor define d_star=(1-lambda)*kappa/2. Set d_Z=d_X=a*d_star for
a in {0,.5,1,1.25}. These are declared sensitivity scenarios, not measured device
realism. The continuous family admits an exactly overlapping attack at a>=1,
with positive error-observation resources in both stages. A nonzero lower bound
on detection over the whole family cannot exceed alpha there.

## Cohorts, randomness and scoring

Per plan/floor/allowance, evaluate nine named cohorts, 512 independent paired
transcripts each. There are 24 settings, 216 cohorts and 110,592 public pairs.

Five honest cohorts, all gains stable within a pair:

1. Stable nominal q0 in both stages.
2. Both monitoring basis errors at the upper envelope edge.
3. Z at the upper and X at the lower edge.
4. Both at the lower edge.
5. Zero transmission, with otherwise nominal error probabilities.

For finite floats, positive/decreased edge probabilities move one representable
step toward q0 if needed to ensure their supplied absolute difference <=d.
Record the actual probabilities; do not silently accept a larger envelope.
At a=0 the first four laws coincide but remain independently seeded cohorts;
do not average them into a single FPR or pretend they are distinct physical laws.

Four fixed-noise attacks, all with nominal trusted calibration and unchanged
gains: floor rate r=2*kappa at beta=0,.25,.5; and higher rate r=4*kappa at
beta=.5. Eve's basis is Z with probability beta. These are fixed probes, not
continuous best responses, temporal/adaptive strategies or a searched optimum.
The attacked baseline does not drift concurrently with Eve.

Use seed 2026100911. Each cohort uses
SeedSequence([seed,plan_id,floor_id,allowance_id,cohort_id]), with separate
spawned children for calibration and monitoring. Methods share the pair;
distinct cohorts do not. Never filter a row or regenerate a favorable seed.
No private knowledge labels are generated or passed to the scorers.

Scoring: existing drift_change_alarm with either Hoeffding or Clopper-Pearson
intervals. Save both binary decisions and all six public numerator/denominator
coordinates plus four sifted counts at each stage. Save the schema, actual laws,
plan/envelope metadata, environment versions, frozen source hashes and archive
hash. Existing output directories are refused, not overwritten.

## Uncertainty and bound calculation

There are 432 alarm rates (216 cohorts times two methods). Save marginal 95%
exact binomial intervals and simultaneous intervals with per-rate failure
delta=.04/432, computed before any rate-based selection. Correlation between
the two method decisions does not invalidate this union bound.

For each of 24 settings, use the stable nominal cohort's actual four error
denominators in hoeffding_family_power_lower. This conditional lower function
is policy-independent in the approved family. Compute its mean and a one-sided
Hoeffding mean lower endpoint, max(0,mean-sqrt(log(24/.01)/(2*512))). Each mean
receives failure delta=.01/24. Combined rate/mean coverage is at least 95% in
real arithmetic. The alpha=.05 detector budget is separate from this uncertainty.
Do not call a sample mean a certificate, or transfer the function to CP.

The nominal cohort's denominator law is the same as that under every attacked
policy because loss/disclosure do not depend on errors. Row functions remain
iid; sharing observations with rate estimates is allowed by the union bound.
Save every function value for replay. Retain zero endpoints and uninformative
bounds. A minimum among four empirical probe rates is not a continuous minimum.

## Validation and conclusions

Before the full run: unit-test scenario counts, lawful edge construction,
resource invariants, refusal to overwrite, interval/failure budgets, seed replay
and audit reconstruction on a tiny temporary run. After the run: replay every
saved decision and conditional lower value, reconcile counts/resources/intervals,
independently check scalar decisions on fixed first/middle/last rows per rate,
and check source/output fingerprints. Duplicate/missing IDs must fail the audit.

Create a new executed companion notebook, not edits to the legacy main notebook.
Include paired-method power plots, honest-drift FPR plots with the 5% target,
bound limitations, expected/realized resource checks and source links. Validate
its outputs before writing the result summary; do not retune to rescue weak power.

Report observed rates with their denominators and uncertainty. Uniform false
alarm validity comes from the model/proof, not low observed counts. An empirical
honest rate above alpha is not by itself proof of invalidity; use its uncertainty.
No comparison to the 2026 paper's averaged threshold-classification accuracy is
valid here. Review readiness is limited to the declared simulation scope.

Next decision depends on the retained results: a tighter drift-valid method or
power integration may be needed. Do not promise that this pilot establishes
novelty, continuous optimality, useful weak-attack sensitivity or publication.
