# Step 3: fixed resources and interception-certified raw knowledge

Protocol frozen before simulations: 2026-10-05. Research-only continuation of
[Step 2](calibration-protocol.md). Stable, attack-free calibration remains an
explicit simulation assumption. No final-key security or publication priority
is claimed. Existing production detectors and historical PDFs remain unchanged.

## 1. Primary question and fair resource accounting

At total emitted-pulse budget B=4000000, how does splitting emissions between
calibration M and monitoring N=B-M affect detection of an IR policy required
to certify a minimum expected fraction kappa of retained raw Alice bits?

Two comparisons are predeclared:

1. **Fixed emissions and disclosure fraction:** f=0.25; M/B is
   0.05,0.1,0.2,1/3,0.5,0.7,0.9. This holds emissions equal, but not sacrifice
   or candidate yield. More calibration consumes additional test resources.
2. **Fixed emissions and expected sacrifice:** fix C=M+f*N=B/2. Use M/B
   0.05,0.1,0.2,1/3,0.4,0.45,0.5 and solve f=(C-M)/(B-M).
   The last plan has f=0 and deliberately no disclosed monitoring errors.
   Integer M uses floor(B*fraction); N is the exact remainder. Expected
   sacrifice and candidate counts are equal, not their realized random values.

With mean gain barQ and stable intensity-independent loss:

$$E[\mathrm{sacrifice}]=\bar Q(M+fN)/2=\bar Q C/2,$$
$$E[\mathrm{candidates}]=\bar Q N(1-f)/2=\bar Q(B-C)/2.$$

This is not an equal latency, energy, financial cost, or secret-key yield
comparison. Gains, intensities, authentication and hardware rate are held fixed
or assumed; practical costs and final-key security are separate questions.

## 2. A precise attack-information proxy

IR rate r is the probability of intercepting a pulse. Eve chooses Z with
probability beta, independent of Alice's uniform basis. In this project's
restricted attack family, the private "known" label means Eve measured in
Alice's preparation basis and therefore learned that Alice bit exactly.
This matches `known = ir & (eve_basis == alice_basis)` in the existing pulse
simulator; it is not a general information-theoretic leakage quantity.

For retained Z bits that label has probability r*beta; for retained X bits
it has probability r*(1-beta). Loss and sample membership are independent,
and Alice's basis is uniform, giving

$$I_{\mathrm{cert}}=\tfrac12 r\beta+\tfrac12 r(1-\beta)=r/2.$$

Thus an expected floor I_cert>=kappa requires r>=2*kappa. The study uses
kappa=0.0025,0.005,0.01,0.02 (0.25%,0.5%,1%,2%). Eligibility is based on this
expectation **before simulation**. Trials are never dropped because the realized
knowledge fraction is lower or higher. Retained-only knowledge labels are
generated in a separate private stream and file, never passed to the scorer.
Publicly tested bits are sacrificed and excluded from this metric.

The proxy omits partial information, inference from public metadata, error
correction leakage, quantum memory, photon-number splitting and privacy
amplification. It is NOT "Eve's total information" or final-key compromise.
Zero candidate counts give undefined fraction, not an invented zero-security
claim. All such rows remain in the experiment and have zero known counts.

Attacked local gates remain ideal, with nominal channel depolarization
lambda=0.03 and eta=0.15. Matched errors are

$$q_Z=\lambda/2+(1-\lambda)r(1-\beta)/2,$$
$$q_X=\lambda/2+(1-\lambda)r\beta/2.$$

Preparation-bit errors within each basis are equal. The scorer sees only the
same six public coordinate counts as Step 2; no policy or knowledge inputs.

## 3. An all-valid-alarms upper bound, not a predicted detection rate

The stable honest family includes arbitrary Pauli mixtures, not just gate
depolarization. Let q0=lambda/2, e=f*N, and t=M+e. For each basis b, choose

$$h_b=(M q_0+e q_b)/t\quad(t>0).$$

This is physical: take the convex mixture of the nominal depolarizing channel
and the IR-equivalent Pauli channel from Step 1 with weight e/t. It is an
honest-noise counterpart for this hypothesis comparison, not an Eve model.

The public sampling law does not depend on errors. Conditional on denominators,
errors are binomial and their sufficient counts preserve the likelihood ratio.
Each bit/basis stratum has expected disclosed denominator M*barQ/8 in
calibration and e*barQ/8 in monitoring. The two preparation bits share each
basis error probability. Independence of stages gives

$$D=\bar Q/4\sum_{b\in\{Z,X\}}
\{M d(q_0\Vert h_b)+e d(q_b\Vert h_b)\},$$

where d(p||h)=p*log(p/h)+(1-p)*log((1-p)/(1-h)) uses natural logarithms.
Gain and sampling-count laws agree, so contribute zero divergence.

For fixed p, derivative with respect to h is

$$\partial_h d(p\Vert h)=-p/h+(1-p)/(1-h)
=(h-p)/[h(1-h)].$$

Consequently the derivative of M*d(q0||h)+e*d(qb||h) is
[(M+e)h-M*q0-e*qb]/[h(1-h)], whose zero is the weighted mean above.
The negative-log terms are convex, so this minimizes the divergence among
these stable basis-error counterparts. Endpoints use continuity.

For any alarm with false-alarm probability <=alpha at that honest counterpart,
total variation and the standard Pinsker inequality imply

$$\mathrm{power}\le\min\{1,\alpha+\sqrt{D/2}\}.$$

This upper bound concerns any valid **fixed-horizon** alarm under the stated
stable iid family. It is not achievable power, a numerical rounding certificate,
a lower bound, a final-key result or a guarantee after usability filtering.
At M=0 or f=0, D=0: errors appear in only one stage and a stable honest
counterpart matches the complete public law. No valid attribution/change test
can have power above alpha for that overlap. More private known bits do not
make them available to the public scorer.

At a given kappa the balanced r=2*kappa,beta=0.5 policy supplies a constructive
ceiling on the minimax problem, whether or not the tested CP rule's grid-selected
response is balanced. We will report the bound separately from empirical power.
It may be loose or equal to one. Known statistical inequalities and channel
equivalence are not advertised as original discoveries.

## 4. Search and untouched evaluation

Primary detector is the existing CP interval-change rule, alpha=0.05 per paired
experiment. This choice was made from Step 2, not from the new held-out data.
No new classifier or detector threshold is fitted.

For each kappa the Eve search grid uses rates 2*kappa times {1,1.5,2} and
beta in {0,0.25,0.5,0.75,1}. Duplicates across floors are evaluated once per
plan. This is 45 unique policies and 14 plans, not a continuous or global
best-response proof. Grid points omitted by this design may be harder.

Search: 192 independent paired trials per plan/policy. For each floor/plan,
select the eligible searched policy with the lowest observed search alarm rate.
Eligibility is confined to that floor's three declared rates, not every rate
in the union grid. Ties prefer lower rate, bias closest to 0.5, then lower bias.
Within each regime/floor the defender selects the plan with the highest
grid-minimum search rate. Plan ties prefer closest to one-third calibration,
then fewer calibration emissions. Freeze those choices before evaluation.

Evaluation: 600 fresh paired trials for each plan/floor under its frozen Eve
policy. Report all plans, not only the selected winner. Evaluation is not used
to choose an allocation or rerun Eve's search. These are held-out selected-policy
rates, NOT verified worst-case power across all attacks.

Controls: 400 fresh paired trials per plan/control for stable nominal noise,
gate noise g=0.02, high-error Pauli noise (I,X,Y,Z)=(0,0.2,0.7,0.1), and zero
transmission. No honest control is used to tune or screen a plan.

Seed 2026100503; separate phase/plan/policy seeds, separate calibration,
monitoring and private-label streams. All public count rows are retained.
Separate private rows record known retained bits. Source/output hashes and
reconciliation checks accompany results. Streaming compressed CSVs avoid keeping
every trial as a large in-memory object.

Monte Carlo rates have marginal 95% exact binomial intervals. The 56 held-out
plan/floor cells additionally have Bonferroni simultaneous 95% intervals;
conditional on the independent search these cover all frozen responses by a
union bound. They do not certify that an unsearched attack is easier, and do
not turn a data-selected maximum into a global optimum.

## 5. Verification and completion boundary

Check emission and expected-sacrifice invariants; retained-only known labels;
private/public separation; eligibility and selection provenance; every saved
alarm, summary denominator and hash; finite sampling/zero-resource endpoints;
and the KL construction using independent categorical and scalar calculations.

This milestone ends with a bounded resource/attack study and an inspectable
conditional power ceiling. Temporal Eve, general noise with memory, drift-aware
nulls, a composable key proof, hardware validation and publication priority are
not silently added. The frozen run and audit below complete this milestone.

## 6. Results of the frozen run

Outputs: [summary](../results/benchmarks/allocation-frontier-v1/frontier.json),
[frozen choices](../results/benchmarks/allocation-frontier-v1/frozen_selections.json),
and [audit](../results/benchmarks/allocation-frontier-v1/audit.json).
The public-count and private-proxy CSV archives are regenerated locally by the
runner rather than bundled with this release.

The run contains 630 search cells, 56 held-out evaluation cells and 56 honest
control cells: 176960 paired public transcripts and 154560 private-count rows.
Each evaluation cell uses 600 fresh trials. The following table reports the
fixed-sacrifice regime at kappa=2%. All seven frozen Eve responses have r=0.04
and beta=0.5. They certify 2% of remaining raw bits in expectation, not a 2%
secret-key leak.

| Calibration emissions (% of B) | Monitoring disclosure f | Alarms / 600 | Detection (%) | Marginal 95% interval (%) | Simultaneous 95% interval (%) |
| --- | --- | --- | --- | --- | --- |
| 200000 (5%) | 0.473684 | 99 | 16.50 | 13.62–19.72 | 11.84–22.04 |
| 400000 (10%) | 0.444444 | 276 | 46.00 | 41.96–50.08 | 39.24–52.86 |
| 800000 (20%) | 0.375000 | 433 | 72.17 | 68.39–75.72 | 65.74–78.01 |
| 1333333 (approximately 33.33%) | 0.250000094 | 439 | 73.17 | 69.43–76.67 | 66.80–78.93 |
| 1600000 (40%) | 0.166667 | 293 | 48.83 | 44.76–52.91 | 42.02–55.68 |
| 1800000 (45%) | 0.090909 | 169 | 28.17 | 24.60–31.95 | 22.29–34.61 |
| 2000000 (50%) | 0 | 0 | 0.00 | 0.00–0.61 | 0.00–1.28 |

The simultaneous intervals account for all 56 frozen evaluation rates, not only
these seven. At every plan in this regime, expected sacrifice and expected
remaining candidates are approximately 43572.29 each. Realized means fluctuate;
these plans do not have identical realized sample counts or certified key yield.

Intermediate allocations perform better descriptively than the extremes in this
tested grid. Spending more on calibration eventually reduces monitoring evidence.
At 50%, no monitoring errors are disclosed; the complete public attack law has
a stable honest counterpart, giving the exact theoretical power ceiling alpha.
The empirical zero is not a claim that all detectors must always return zero.

The defender's **search-selected** allocation at this floor was 20%, with search
grid-minimum alarm rate 0.671875 and held-out rate 0.721667. The 33.33% plan's
slightly higher held-out point estimate does not authorize replacing that frozen
choice. No formal pairwise difference test was performed; overlapping intervals
do not prove equality, and neither allocation is a proven unique optimum.

For comparison, fixed-fraction f=0.25 produces held-out rates 13.00%, 29.67%,
59.33%, 71.33%, 70.83%, 56.67% and 16.50% at calibration shares 5%, 10%, 20%,
33.33%, 50%, 70% and 90%, respectively, at the same 2% floor. Its search-selected
allocation was 50%, not the maximum held-out point. Unlike the table above,
this regime changes expected sacrifice and candidate yield with allocation.

### Weak attacks and the distinction between detector and information limits

Search-selected plans in the fixed-sacrifice regime give the following fresh
results. All listed selected policies have beta=0.5 and r=2*kappa.

| Expected certified raw fraction (%) | Selected calibration share (%) | Alarms / 600 | Detection (%) | Marginal 95% interval (%) |
| --- | --- | --- | --- | --- |
| 0.25 | approximately 33.33 | 1 | 0.17 | 0.004–0.925 |
| 0.50 | 45 | 2 | 0.33 | 0.040–1.199 |
| 1.00 | 20 | 32 | 5.33 | 3.676–7.446 |
| 2.00 | 20 | 433 | 72.17 | 68.394–75.720 |

Very weak constrained attacks remain difficult for this conservative CP rule.
They also differ from Step 2's fixed-Z demonstration: Eve can distribute errors
between bases, and the new comparison fixes total rather than monitoring pulses.
The earlier success therefore cannot be promoted to general robustness.

At kappa=0.25%, the balanced attack gives all-valid-alarm ceilings between
0.2766 and 0.5324 in the fixed-fraction plans and between 0.05 and 0.5496 in
the fixed-sacrifice plans. Under the declared broad stable honest family, no
valid fixed-horizon alarm can reach 80% power against that admissible attack at
any of these **tested plans**. This is a necessary ceiling, not achievable power
or a conclusion over every continuous allocation. At kappa=2%, most Pinsker
ceilings equal one and are uninformative; the no-monitoring endpoint remains
0.05. The low CP rates and the fundamental ceiling are distinct statements.

Across the 56 stable honest-control cells, the largest observed false-alarm rate
is 0.5%, below the declared 5% target. Finite control samples do not prove the
mathematical guarantee, and unrelated control cells are not pooled. Honest drift
is outside this study's stable null; as established in Step 2, a change alarm
cannot uniquely attribute the cause to Eve.

## 7. Checks, reproduction and next decision

The new module adds 39 tests; the full suite passes **602 tests**. The saved audit
reconciles all 176960 public pairs, 154560 private rows and 742 summary cells;
replays all 176960 decisions; checks 64 frozen selections and four source hashes;
and independently checks 112 categorical KL/bound constructions. Independent
scalar decision checks cover 168 predetermined held-out rows. Full replay shares
the original scorer, so it is not a fully independent implementation of every
decision. Private labels are absent from the public export and scorer inputs.

From the repository root, with NumPy, SciPy and pytest available in the existing
environment:

```sh
python quantum-ai-lab/scripts/run_allocation_frontier.py --output quantum-ai-lab/results/benchmarks/allocation-frontier-rerun
python quantum-ai-lab/scripts/audit_allocation_frontier.py quantum-ai-lab/results/benchmarks/allocation-frontier-rerun
python -m pytest quantum-ai-lab/tests/test_allocation_frontier.py -q
python -m pytest -q
```

The runner refuses an existing output directory. Choose a new path for each
run; the auditor requires the saved source versions to match. Counts are exact
public marginals of the declared iid model, not pulse traces, hardware runs or
general attacks. The seeded draw sequence also depends on recorded NumPy/SciPy
versions; recorded counts support decision replay without resimulating them.

**Contribution boundary:** this milestone is a reproducible budget-constrained
simulation study plus a scoped stable-null information limit. It does not
establish a new quantum principle, a new optimal detector, publication priority
or a composable key guarantee. Literature priority remains unverified.

**Proposed next milestone, requiring confirmation:** compare a stronger exact
two-sample test with the CP interval rule at the same false-alarm budget, fixed
resource plans and expected raw-knowledge constraint. Freeze selection separately
and evaluate on fresh trials, so detector conservatism is not mistaken for an
unavoidable information limit. Temporal attacks and drift-aware attribution are
separate subsequent choices. No production detector, historical PDF, Git commit
or push was changed as part of Step 3.
