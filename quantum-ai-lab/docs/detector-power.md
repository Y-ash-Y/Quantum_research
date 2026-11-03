# Step 7: detector-specific power bounds over a continuous IR family

Completed 2026-10-09. [Frozen protocol](detector-power-protocol.md),
[saved results](../results/benchmarks/detector-power-v1/power.json), and
[audit](../results/benchmarks/detector-power-v1/audit.json). Research-only;
production alarms, historical PDFs, notebook performance claims and Git history
are unchanged. The central-exact rule is not a new statistical invention.

## Outcome in plain language

We no longer need Step 6's loose generic information-distance radius to extend
this study beyond the attack grid. A contained increased-error rejection event
is monotone, even though the complete two-sided detector has not been proved
monotone. It supplies a lower function valid for every eligible interception
rate in a bias interval. Conditional binomial integration and fresh ancillary
count sampling turn these functions into simultaneous continuous lower bounds.

All eight full-family brackets are nonvacuous. Weak-floor lower endpoints remain
below the observed honest false-alarm rates: their positivity alone is NOT
operationally useful weak-Eve sensitivity. Strong-floor detection is bounded
away from zero across the whole restricted family. Balanced Eve passes the
predeclared five-percentage-point near-worst gate only at the 2% floor; saturation
makes that gate easier and it does not identify a unique optimum.

No exact balanced optimality, best allocation, arbitrary-attack robustness,
final-key security, hardware realism or publication novelty is established.

## 1. Model, conditioning and rejection boundaries

Keep B=4,000,000 emitted pulses, q0=lambda/2=.015, lambda=.03, eta=.15, alpha=.05.
Calibration is attack-free and stable. Plans 9/10 respectively allocate 20%
and approximately one third to calibration, with equal expected sacrifice
exposure C=B/2. Actual disclosed counts are random, not replaced by their means.
The eligible iid intercept-resend family is r in [2*kappa,1], beta in [0,1].
Eve's expected certified knowledge of retained raw Alice bits is r/2, not
mutual information, public-discussion leakage or final secret-key leakage.

Condition on the ancillary counts D: emitted intensity counts, detected counts,
matched strata and disclosed denominators in both stages. These have the same
law for every IR policy here because gain and disclosure are independent of Eve.
For error stratum j, let m_j and n_j be calibration and monitoring denominators.
Let A_j be calibration errors and B_j monitoring errors. Conditional on D,

    A_j ~ Bin(m_j,q0),   B_j ~ Bin(n_j,q_j),

with independent error draws across stages and strata. Random denominators
themselves are NOT independent. The two gain-coordinate p-values are already
determined by D; write g(D)=1 when either triggers the unchanged alarm.

For fixed A=a, the conditional null law for A given total t=a+b is
Hypergeom(population=m+n, successes=t, draws=m). The central doubled-tail test
rejects at alpha/6 iff either inclusive hypergeometric tail is <=alpha/12.

Couple uniformly chosen success subsets of successive sizes t by adding one
success. A fixed m-position sample then has a nondecreasing success count.
Consequently F(a;t)=Pr(A<=a|t) decreases as t increases, and
S(a;t)=Pr(A>=a|t) increases. Holding a fixed and increasing b increases t.
The rejection sets in b therefore have integer boundaries:

    increased monitoring errors: b >= u(a),
    decreased monitoring errors: b <= l(a).

Find u as the first b with F<=alpha/12, l as the last b with S<=alpha/12.
Absent events use u=n+1 or l=-1. They are disjoint since F+S>=1 while
2*(alpha/12)<1. Equality is rejection. Inclusive survival uses sf(a-1), not
sf(a). Empty stages return no rejection, consistent with the original scorer.

## 2. Integrating errors instead of simulating every error draw

Let w_m(a)=BinomialPMF(a;m,q0). Conditional upper- and lower-tail powers are

    H_j(q)=sum_a w_m(a) Pr[Bin(n,q)>=u(a)],
    L_j(q)=sum_a w_m(a) Pr[Bin(n,q)<=l(a)].

Their sum p_j(q) is the full coordinate rejection probability. Conditional
independence, not independence of raw denominators, gives

    R(D,q)=g(D)+(1-g(D)) [1-product_j(1-p_j(q_j))].

Unconditional full power is E_D R(D,q). This is a conditional-expectation
variance reduction calculation (often called Rao-Blackwellization), not an
additional detector, trainable model, quantum advantage or new data source.

The implementation retains a calibration support S with omitted mass
epsilon_j=Pr(A_j not in S)<=1e-12. Truncated H+L is a lower bound on p_j; adding
epsilon_j and clipping at one gives an upper bound. Propagate those bounds
through the product before averaging. Weights are not renormalized, which
would incorrectly remove the omission uncertainty. Full integration is exact
in real arithmetic; truncation is explicit, and floating tails are not
outward-rounded numerical certificates.

## 3. Continuous lower bound without assuming full-power monotonicity

Represent a monitoring Bernoulli error as 1{U<=q} using common uniform U. Raising
q cannot decrease the count B. Hence Pr(B>=u) is nondecreasing in q for each a,
and H_j(q) is nondecreasing after averaging positive calibration weights.
The full H+L need not inherit this order because the decreased-error event
shrinks. We only use the increased-error subevent, contained in the full alarm.

Write c=(1-lambda)/2=.485. For r>=2*kappa and beta in [b_lo,b_hi],

    qZ=q0+c*r*(1-beta) >= q0+c*2*kappa*(1-b_hi)=qZ_min,
    qX=q0+c*r*beta     >= q0+c*2*kappa*b_lo=qX_min.

The corresponding four-stratum high-tail union, including g(D), gives a
deterministic conditional lower function h_cell(D)<=R(D,r,beta) everywhere
in that rectangle, with the rate axis extending all the way to r=1. Using
truncated calibration weights further lowers the function and is safe.

Use 16 closed bias cells on [0,.5]. The analytic Z/X reflection law from Step 6
maps [.5,1] to this half-domain in population power. No claim of samplewise
reflection of an unpermuted denominator row is made. These cells cover the
entire bias domain; no interpolation is silently assumed between anchor rates.

If L_cell bounds E h_cell from below simultaneously, then

    min_{r>=2*kappa,beta in [0,1]} power(r,beta) >= min_cell L_cell.

The balanced floor-rate policy (r=2*kappa,beta=.5) is admissible. An upper
bound U_balanced on its full power therefore upper-bounds the continuous
minimum. The difference U_balanced-min_cell L_cell bounds balanced regret:
how much MORE detectable balanced Eve could be than the true hardest Eve.
This direction matters: we are minimizing Eve's detectability, not maximizing it.

## 4. One combined confidence event, not several relabeled events

Per plan, 1024 iid ancillary pairs are reused across all predeclared functions.
For any fixed bounded function Z=h(D), let mean be its sample average and s^2
the unbiased variance with denominator T-1. The one-sided empirical Bernstein
radius is

    sqrt(2*s^2*log(2/delta)/T) + 7*log(2/delta)/(3*(T-1)).

Apply Maurer-Pontil (2009), Theorem 4, to Z for an upper endpoint and to 1-Z
for a lower endpoint. This is an established concentration inequality, not
our invention: https://www.cs.mcgill.ca/~colt2009/papers/012.pdf . Rows must
be iid; functions may be correlated because a union bound does not require
independence between them.

There are 128 cell lower functions and 16 balanced lower/upper functions.
Each side receives delta=.04/144, so all mean endpoints hold jointly with
probability at least .96. Ten fresh direct rates each receive exact two-sided
binomial intervals with failure .01/10; their joint coverage is at least .99.
By the union bound the combined event is at least .95. The detector's alpha=.05
false-alarm statement is separate from this Monte Carlo uncertainty budget.
No peeking, stopping, threshold fitting or result-based cell refinement occurred.

## 5. Full continuous-family results

Percentages below bracket the TRUE minimum power in the declared continuous
family, on the combined confidence event, subject to numerical qualifications.
They are not selected-policy intervals or observed rates. The upper endpoint
comes from a balanced candidate, not a found adversarial optimizer.

| Raw-knowledge floor | Plan 9 continuous minimum | Plan 10 continuous minimum | Balanced regret upper, plan 9 / 10 |
|---|---|---|---|
| 0.25% | 2.790–10.276% | 2.747–10.157% | 7.486 / 7.410 percentage points |
| 0.5% | 12.202–19.837% | 11.975–19.516% | 7.635 / 7.541 points |
| 1% | 54.376–62.809% | 52.774–61.139% | 8.433 / 8.366 points |
| 2% | 97.721–100% | 97.565–100% | 2.279 / 2.435 points |

All eight lower endpoints are positive, unlike Step 6's zero full-family
endpoints at these plans. This improves the BOUND, not the unchanged detector.
The five-point near-worst gate passes in 2/8 plan/floor cases, only at the
strongest floor. All other cases remain unresolved, not proven suboptimal.
This is not a uniform result for all possible knowledge floors or allocations.
Strong-floor saturation makes a small regret bound less diagnostic of attack
geometry. No exact minimum at balanced Eve is established.

## 6. Fresh direct validation

Each entry uses 8192 independently drawn calibration/monitoring pairs.
Intervals below are simultaneous over all ten direct cells with joint coverage
at least 99%, included in the combined 95% event above.

| Case | Plan 9 observed alarms (interval) | Plan 10 observed alarms (interval) |
|---|---|---|
| Balanced 0.25% floor | 556/8192 = 6.787% (5.907–7.748%) | 570/8192 = 6.958% (6.067–7.929%) |
| Balanced 0.5% floor | 1365/8192 = 16.663% (15.333–18.054%) | 1364/8192 = 16.650% (15.321–18.041%) |
| Balanced 1% floor | 4984/8192 = 60.840% (59.050–62.610%) | 4802/8192 = 58.618% (56.814–60.406%) |
| Balanced 2% floor | 8178/8192 = 99.829% (99.621–99.941%) | 8174/8192 = 99.780% (99.553–99.912%) |
| Stable honest | 326/8192 = 3.979% (3.305–4.740%) | 348/8192 = 4.248% (3.551–5.030%) |

All eight direct intervals overlap their balanced analytic brackets. Overlap
is a consistency check, not an equality proof. The higher strong-floor lower
power bound is a model-scoped guarantee; observed near-unit rates are not
guaranteed perfect detection. Weak-floor bounds do not establish useful excess
power over the honest false-alarm baseline.

## 7. Verification and interpretation

The full suite passed 697 tests, including 30 new power tests. They enumerate
small integer tables via rational combinatorial tails independently of SciPy's
hypergeometric scorer, cover integration/endpoints/reflection/truncation and
check the contained lower function at interior and high-rate policies.

The archive audit passes 2,277,376 saved boundary-entry checks, 256 sampled
scalar Fisher boundary checks, 147,456 conditional-function row reconstructions
through incomplete-beta/log-gamma identities, all 144 mean endpoints and eight
envelopes. It replays all 81,920 direct decisions and samples 120 independent
scalar six-coordinate decisions. Ancillary pairs (2048) are NOT extra direct
alarm trials. Evidence fingerprints bind the frozen protocol, sources and
archives. These paths share SciPy numerical infrastructure; no independent
hardware simulator or physical measurements were used.

Reproduce in a fresh output directory:

```bash
.venv/bin/python quantum-ai-lab/scripts/run_detector_power.py \
  --output quantum-ai-lab/results/benchmarks/my-detector-power
.venv/bin/python quantum-ai-lab/scripts/audit_detector_power.py \
  quantum-ai-lab/results/benchmarks/my-detector-power
.venv/bin/python -m pytest -q
```

This is meaningful progress from finite-grid diagnostics to continuous,
detector-specific bounds within the restricted model. The coupling/integration/
concentration tools are standard. Publication novelty requires a focused
closest-work comparison; no claim follows merely because these calculations
were not previously in this repository.

Next decision, requiring user confirmation: either tighten the remaining weak/
medium-floor regret gap using a separately frozen precision study, or first
seek independent expert/literature review of the present contribution. Do not
silently retune this completed study or call the exact optimum proved.
