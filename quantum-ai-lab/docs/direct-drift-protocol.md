# Step 12 frozen comparison protocol

Frozen 2026-10-09 after derivation and unit tests, before generating comparative
results. The [new test/proof](direct-drift-test.md) is fixed; there is no tuning
on this evaluation. No commit, push, paper submission or novelty claim is implied.

## Design and scope

Use the same Step 11 model, laws, two resource allocations, floors (.005,.01,.02),
allowance multipliers (0,.5,1,1.25), five honest controls and four fixed attacks.
The scalar drift allowance is .97*kappa*multiplier/2 for each basis. All parameters
are sensitivity assumptions, not measured device drift. Blockwise iid, trusted
attack-free calibration, shared bit errors and stable gains remain essential.
The attacked baseline does not drift concurrently; no global Eve optimum is claimed.

Each pair uses B=4,000,000 emissions, C=2,000,000 effective test emissions,
lambda=.03 and eta=.15 (or eta=0 for the zero-transmission honest control).
Plan 9 has M=800,000 and f=.375; plan 10 has M=1,333,333 and f=666667/2666667.
Expected sacrifice/candidate counts match; realized counts are random and not
final-key yield. Each of 216 cohorts has 512 independent pairs: 110,592 pairs.

New seed **2026100912**, distinct from Step 11. Each cohort uses
SeedSequence([seed,plan_id,floor_id,allowance_id,cohort_id]).spawn(2) for independent
calibration and monitoring streams. Preserve exact public-count sampling and
all rows. Numerical honest edges use the already frozen Step 11 construction.
Use the existing Step 11 scenario builder without changing its source/output.

## Fixed methods and estimands

Score every pair with alpha=.05 separately for:

1. Original Hoeffding confidence-gap rule.
2. Original Clopper–Pearson confidence-gap rule.
3. Direct pooled Bernstein/CP-variance rule defined in the new proof.

There are **648** alarm-rate cells. No OR across the three rules is evaluated.
The proof's eight internal directions belong only to the direct rule. Compare
upper-alpha-valid methods, not equal attained FPR or a secret-key security claim.

For each cohort save the 2×2 paired direct-versus-CP decision counts: both alarm,
only direct (wins), only CP (losses), neither. The net alarm-rate difference is
(wins-losses)/512. A positive number is higher pointwise attack detection for
attack probes, but **higher false alarms**, not a benefit, for honest controls.
Never pool honest/attack rows into accuracy or classify allowed weak attacks as
honest. Keep every comparison, including losses, zero counts and negative effects.

## Uncertainty fixed before results

Save marginal 95% exact-binomial intervals and joint rate intervals with failure
0.04/648 per rate. For each of 216 paired effects, form separate exact-binomial
two-sided intervals for P(only direct) and P(only CP), each with failure
0.01/(216*2). Subtract their endpoints as [L_win-U_loss, U_win-L_loss]. The
effect interval's failure is at most 0.01/216 despite dependence between the
two counts. The union of all rate/effect intervals covers jointly with at least
95% probability in real arithmetic. Do not infer a paired difference from two
unpaired rate-interval endpoints or their apparent overlap.

A minimum over four empirical attack probes is not a continuous-family lower
bound. No power certificate is computed for the new rule, and the old Hoeffding
conditional lower function is not transferred. No new best-response search,
allocation choice, stopping/tuning rule or enlarged campaign is allowed.

## Evidence and acceptance checks

Save every stage's six numerator/denominator counts, four sifted counts, three
decisions and row IDs. Save cohort/law/envelope/resource metadata, exact budgets,
environment versions, source hashes and raw-archive hash. Refuse existing output
directories. Before the full run, reconstruct a tiny temporary run in tests,
check paired-count/effect bookkeeping and detect altered metadata.

After the run, regenerate every public pair from saved seeds; replay all 331,776
method decisions; check every rate/effect interval and realized resource mean;
independently check first/middle/last scalar decisions for each method/cohort.
Scalar CP reference calculations share SciPy and do not constitute a separate
library or outward-rounded floating-point certificate. Recheck Step 7 and Step
11 fingerprints. Create a separate executed notebook for interpretation.

No predetermined positive performance result is required. Report whether weak
attack sensitivity remains inadequate and distinguish population overlap from
finite-sample conservatism. Novelty remains a separate unresolved literature question.
