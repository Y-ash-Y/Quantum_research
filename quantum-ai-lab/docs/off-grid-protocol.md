# Step 6 frozen protocol: symmetry, off-grid confirmation and continuity

Frozen before simulations on 2026-10-05. No detector tuning, attack attribution,
production changes, PDFs, commits or pushes. The Step 5 summary and its hashes
control inherited grid evidence; fresh observations are never used to replace
its frozen allocations or responses.

## Questions and boundaries

1. Prove Z/X reflection invariance under uniform basis/bit choices, depolarizing
   nominal calibration, identical loss and disclosure across strata, and the
   unchanged six-coordinate central-exact rule. Check the implementation by
   permuting each fresh transcript, not by treating reflected copies as new
   independent trials. Symmetry does not prove a minimum at beta=1/2.
2. Derive balanced-policy minimality of the *broad-honest KL separation*, not of
   the actual alarm probability. Prove this KL proxy increases with rate when
   calibration and monitoring exposure are positive. Do not import that ordering
   into a discrete conditional detector without proof.
3. Extend Step 5's simultaneous anchor lower bounds to continuously covered
   rectangles using public-law KL and Pinsker. Report both the near-floor band
   r in [2*kappa,4*kappa] and the full eligible IR domain r in [2*kappa,1],
   beta in [0,1]. A vacuous zero lower bound is a result, not a license to retain
   the finite-grid lower bound. Real-arithmetic proof is distinct from floating
   diagnostic endpoints.
4. Check rate/bias behavior using fresh, predeclared off-grid policies; neither
   significance failure nor an observed minimum establishes continuous optimality.

## Frozen simulation design

B=4,000,000, nominal channel noise .03, eta .15, alpha=.05, fixed horizon.
Use equal-expected-sacrifice plans 9 (20% calibration, f=.375) and 10
(M=1,333,333; f=.25000009375). Both preserve exposure C=B/2, not exact realized
bit sacrifice. Four expected certified raw-knowledge floors .0025,.005,.01,.02;
the proxy equals r/2 and is not total/final-key information.

Stress phase: per floor, rate multipliers {1,1.0625,1.1875,1.375,1.75,2} times
2*kappa and beta in {0,.0625,.1875,.3125,.4375,.5}. These span the half-domain
only because the reflection theorem supplies the other half. There are 144 cells
per plan, 288 total, with 512 trials each. All cells are fixed, not selected
from Step 5 holdout minima. This is a geometry probe, not a full-family search.

Confirmation phase: per floor, multipliers {1,1.0625}, beta in {.4375,.5,.5625},
4096 trials per cell, 24 cells per plan, 48 total. The candidates are frozen now,
not chosen from stress outputs. Repeat coordinates across phases are independent
cohorts. Each cell has independent calibration/monitoring streams generated from
SeedSequence([2026100506, phase_id, plan_id, case_id]), spawned in stage order.
No hidden probability, private bit label or Eve metadata reaches the detector.
No private knowledge draws are needed: the expected r/2 eligibility constraint
and retained-only label law are unchanged and already tested in prior milestones.

Every public count and alarm is saved in four compressed blocks (phase,plan).
Reflected p-values must exactly equal the original p-values with their four
error columns permuted; reflected alarms must exactly equal original alarms.
Do not count copies as extra evidence or fit a threshold using them.

All 336 fresh cell rates receive exact binomial intervals with total Monte Carlo
failure budget .05 (per cell .05/336). Detector alpha is a separate statement.
Sixteen predeclared confirmation contrasts use independent binomial cohorts:
for each plan/floor, (i) beta=.4375 versus beta=.5 at floor rate; (ii) balanced
rate multiplier 1.0625 versus balanced floor rate. Report differences and
conservative simultaneous intervals formed from the cell intervals. One-sided
Fisher tests ask whether the first policy has *lower* true power; adjust all 16
with Bonferroni. No significance failure is a proof of monotonicity or equality.

## Continuous cover and confidence accounting

The 45 Step 5 anchors per plan/floor define rectangular cells with midpoint
edges in r and beta. Domain endpoints are r_min=2*kappa and either 4*kappa
(near floor) or 1 (full eligible); bias endpoints are 0,1. The last rate cell
extends to 1 for the full-domain cover. Every region is covered, including shared
boundaries; no unknown region is assigned zero observed alarms.

For anchor g, set radius_g=min(1,sqrt(max_corner KL(P_g||P_theta)/2)). The
continuous minimum lower endpoint is min_g max(0,L_g-radius_g). Its upper
endpoint is the existing Step 5 shortlist minimum upper endpoint. The confidence
event is inherited from Step 5's .025 search plus .025 holdout allocation;
there is no new simulation confidence cost for a deterministic mathematical
extension. Fresh-cell simultaneous coverage is a *separate* 95% statement.
Their conjunction is at least 90%, not automatically 95%; no joint claim combining
the inherited and new events will be labeled 95%.

Derivations use standard joint convexity and Pinsker, verified in Polyanskiy and
Wu, *Information Theory*, author manuscript (2022), Sections 7.2 and 7.4,
https://people.lids.mit.edu/yp/homepage/data/itbook-2022.pdf . Applying these tools
to this model is not itself a verified literature novelty claim.

## Validation gate

Tests: physical reflection law; public bookkeeping/permutation and involution;
exact alarm invariance on synthetic samples; small binomial-law KL enumeration;
zero exposure/loss; corner dominance on interior probes; rectangular cover;
envelope algebra; proxy Jensen/rate order on representative values; frozen case
coverage, confirmation contrasts and invalid inputs. Archive audit replays every
decision, sampled independent scalar Fisher decisions, all counts/intervals,
all contrasts, full cover arithmetic, frozen choices and source/input/output
hashes. Shared scorer replay and sampled scalar APIs are not an independent
physics simulator. Results and limitations go in a separate results note.
