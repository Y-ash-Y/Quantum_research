# Step 9: bounded honest drift - assumptions and proof note

Date: 2026-10-09. Status: analytical model/proof draft completed; **not implemented
or experimentally evaluated**, and novelty is not established. This analytical
note precedes the subsequent implementation and experimental evaluation.
It does not change the frozen stationary Step 7 model, results or detector.

Follow-up: the user approved this model and [Step 10](drift-implementation.md)
now implements/tests the confidence-gap baseline and conditional Hoeffding
lower function. This note records the preceding analytical milestone; its
"not implemented" statements describe that original handoff, not current status.

## 1. The question in ordinary language

Calibration estimates normal channel behavior from a finite trusted block.
Later, a small legitimate change may look like eavesdropping. We want an alarm
that rarely fires anywhere within a stated honest-change allowance, and then
want to know how much attack detection remains for a fixed measurement budget.

Three quantities must stay separate:

- **Drift allowance:** an assumption about how much honest error probability
  may change; it is not something learned freely from attacked data.
- **False-alarm level alpha:** an upper limit on alarm probability for every
  channel admitted as honest by that assumption.
- **Power:** alarm probability at a particular attacked channel, or its minimum
  over the declared attack family.

The first goal is to locate observational overlap. If honest behavior is allowed
to exactly reproduce an attack's public observations, more samples cannot solve
that attribution problem. Beyond overlap, more samples can reduce uncertainty.

## 2. Minimal proposed model

There are two independent blocks, calibration c and monitoring m. Parameters
are constant within each block; only the between-block change is allowed here.
Pulses are independent and identically distributed (iid) within a block. This
is **not** arbitrary time-dependent drift within a block.

Keep the existing public-count protocol and six coordinates:
Z0 error, Z1 error, X0 error, X1 error, signal gain and decoy gain. The scorer
sees error/detected counts and their actual public denominators, not private
bits, simulator probabilities, true Eve information or attack labels.
Bits/bases/intensities are independent uniform choices; intensities remain
0.5/0.1. Loss and random disclosure are independent of errors, basis and bit.
Do not reject or filter zero-click/zero-disclosure rows.

### Honest family H_d

In calibration, the error probabilities are theta_Z for both Z preparation
bits and theta_X for both X preparation bits. In monitoring they are mu_Z and
mu_X. For b in {Z,X},

$$0\le\theta_b,\mu_b\le1/2,\qquad |\mu_b-\theta_b|\le d_b.$$

The nonnegative constants d_Z,d_X are agreed before collecting monitoring
results. They are **absolute probability differences**, not relative percent
changes. Errors and gains are sampled using the same public marginal law as
`sample_public_batch`; all four bit/basis strata remain explicit.

This rectangle has physical representatives: for any q_Z,q_X in [0,1/2], the
Pauli mixture with weights

$$w_I=1-q_Z-q_X,\quad w_X=q_Z,\quad w_Y=0,\quad w_Z=q_X$$

is nonnegative and sums to one. X flips matched Z measurements; Z flips matched
X measurements. It therefore realizes the stated pair for both bit values.
These are possible honest channels, not a claim that real devices sweep the
whole rectangle or that a fitted Pauli explanation rules out Eve.

Gains are stable across the two blocks with one common unknown eta, independent
of basis/bit/errors, and intensity gains 1-exp(-intensity*eta). Initially set
gain drift allowances to zero. Intensity drift, dark counts, detection bias,
nonuniform bit noise, gate-count asymmetry and calibration poisoning are outside
this model. The zero-transmission case is honest but offers no error power.

Calibration is assumed genuinely attack-free; authenticating messages alone
does not establish this. A drift envelope inferred from a finite device study
would require a separate validation/failure budget. Until such evidence exists,
chosen envelopes are sensitivity assumptions, not measured realism.

### Restricted attack family A_kappa

For the first comparison, keep the attack branch from Step 7: trusted calibration
with theta_Z=theta_X=q0=lambda/2, ideal local attack gates and one downstream
depolarizing channel with the **same fixed lambda**, iid IR rate r and Eve Z-basis
probability beta. Gains/disclosure are unchanged. Define c=(1-lambda)/2. Then

$$q_Z^A=q_0+cr(1-\beta),\qquad q_X^A=q_0+cr\beta.$$

The private proxy remains I_cert=r/2, so r in [2*kappa,1], beta in [0,1] and
0<kappa<=1/2. It is expected exactly-known raw Alice bits, not mutual information,
chance-correct guesses, min-entropy or final-key leakage. It does not enter the
alarm. The detector, resource plan and drift bounds are known to Eve.

**Important asymmetry:** the honest null permits drift, but this initial attack
branch has fixed baseline noise. It asks whether that attack can be mistaken
for legitimate drift. It does not cover Eve acting on a concurrently drifting
channel. That larger alternative must be derived separately; the boundary below
must not be presented as a guarantee against it.

## 3. Exact observational-overlap boundary

The necessary-and-sufficient statement below assumes both stages have positive
emissions, eta>0 and positive monitoring disclosure, so each basis error
probability is identifiable from its population public-count law. For the
retained-bit interpretation also keep monitoring disclosure below one. A finite
row may still have zero counts; those rows are not discarded. If calibration
is absent, monitoring errors are never disclosed, or transmission is zero,
additional observational overlaps occur regardless of the displayed drift sum.

Let Delta_Z=cr(1-beta) and Delta_X=cr*beta be the attack's excess errors relative
to its calibration. Both are nonnegative and

$$\Delta_Z+\Delta_X=cr.$$

The attack law belongs to H_d exactly when

$$cr(1-\beta)\le d_Z,\qquad cr\beta\le d_X.$$

For cr>0 these inequalities give the bias interval

$$\max(0,1-d_Z/(cr))\le\beta\le\min(1,d_X/(cr)).$$

An interval exists if and only if cr<=d_Z+d_X. Necessity follows by adding
the two inequalities. For sufficiency, split the nonnegative total cr between
two nonnegative parts each no greater than its allowance; the displayed
interval gives the corresponding beta. All resulting attack probabilities
remain at most 1/2 because q0+c*r<=q0+c=1/2.

The smallest allowed rate is 2*kappa. Therefore there is an exactly overlapping
attack in the family if and only if

$$\boxed{(1-\lambda)\kappa\le d_Z+d_X.}$$

This is an exact **population law-overlap criterion for the stated count model**,
not an optimality theorem for a particular detector or an operational key limit.
For lambda=1 every attack is fully mixed and overlap always holds; the bias
interval division is not used at cr=0.

For equal allowances d_Z=d_X=d, balanced floor-rate Eve is a witness whenever

$$d\ge(1-\lambda)\kappa/2.$$

At lambda=0.03, the illustrative thresholds are:

| Expected certified raw-bit floor kappa | Minimum rate r=2*kappa | Equal-basis overlap allowance d | Absolute QBER change, percentage points |
|---|---:|---:|---:|
| 0.25% | 0.005 | 0.0012125 | 0.12125 |
| 0.5% | 0.01 | 0.002425 | 0.2425 |
| 1% | 0.02 | 0.00485 | 0.485 |
| 2% | 0.04 | 0.0097 | 0.97 |

These are formula evaluations, not new experimental observations. Unequal
allowances need not have a balanced overlap witness. Once the sum condition
holds, Eve can allocate its disturbance to fit the larger allowed direction.
That does **not** prove this bias minimizes finite-sample power outside overlap.

### Why equality of probabilities is enough here

The calibration law matches exactly. Click/intensity counts, sifted classes
and disclosure denominators have the same joint law and do not depend on error
probabilities. Conditional on these ancillary counts, each disclosed-error
count is binomial with the same probability in the matched honest and attack
cases. Both preparation bits in each basis agree as well. Conditional joint
laws and their mixing law therefore agree, so the **complete public-count
distribution** is equal, not just its expected QBER.

This conclusion relies on iid observations and the stated telemetry. Unmodeled
timestamps, correlations, side-channel observables or independently trusted
device measurements could change distinguishability; they cannot be silently
added to the current claim.

### Consequence for every valid alarm

Let phi(T) in [0,1] be the probability of alarming at public transcript T;
this includes deterministic rules and randomized alarms. Require

$$\sup_{H\in\mathcal H_d}E_H[\phi(T)]\le\alpha.$$

For a matched attack A and honest law H*, equality of transcript laws gives

$$E_A[\phi(T)]=E_{H^*}[\phi(T)]\le\alpha.$$

Hence if the overlap condition holds, the worst-case attack power of **any**
uniformly valid alarm is at most alpha. This remains true for any emission,
calibration or disclosure budget with the same observation protocol. It is an
application of an established identifiability principle, not a claimed new
quantum-security theorem. A failure to alarm does not certify a secret key.

## 4. A valid conservative baseline, defined but not implemented

The stationary central conditional test is not automatically valid when
calibration and monitoring probabilities differ. Enlarging its threshold by
intuition, subtracting an estimated drift, or reusing old power curves has no
uniform false-alarm justification.

A transparent baseline is to compare simultaneous confidence intervals with
the permitted gap. Index coordinates j=1,...,6 and stages s in {c,m}. Write
Y_sj for successes/errors and n_sj for their public denominators. When n_sj>0,
set phat_sj=Y_sj/n_sj and

$$h_{sj}=\sqrt{\log(24/\alpha)/(2n_{sj})},$$
$$L_{sj}=\max(0,\widehat p_{sj}-h_{sj}),\qquad
U_{sj}=\min(1,\widehat p_{sj}+h_{sj}).$$

For n_sj=0 use [L_sj,U_sj]=[0,1], not an imputed observed rate. Use allowances
(d_Z,d_Z,d_X,d_X,0,0). Alarm when at least one coordinate satisfies

$$L_{mj}-U_{cj}>d_j\quad\hbox{or}\quad L_{cj}-U_{mj}>d_j.$$

The inequality is strict. Merely touching the allowed boundary is not evidence
against the null. Each interval uses failure probability alpha/12; there are
twelve stage-coordinate intervals, not twelve additional detectors.

### Uniform false-alarm proof

For a binomial proportion with positive denominator, the two-sided Hoeffding
bound is 2*exp(-2*n*h^2). Substituting the chosen h gives alpha/12. Error
coverage holds conditional on its random disclosed denominator; independence
of sample membership from error is essential. Gain coverage holds conditional
on its sent-intensity denominator. Integrating preserves these marginal bounds.
The twelve intervals jointly cover their true probabilities with probability
at least 1-alpha by the union bound. Coordinate independence is unnecessary.

On this coverage event, for an allowed honest law,

$$L_{mj}-U_{cj}\le\mu_j-\theta_j\le d_j,$$
$$L_{cj}-U_{mj}\le\theta_j-\mu_j\le d_j.$$

No alarm can occur. Thus the rule has false-alarm probability at most alpha
for **every** member of H_d, including zero denominators, in real arithmetic.
It is fixed-horizon only: repeated windows, optional stopping or fitting d to
the monitored data would need a different guarantee.

Replacing each interval with a Clopper-Pearson interval of the same failure
budget preserves this coverage proof. The finite-power formula below is
derived specifically for the Hoeffding baseline, not automatically for CP or
a nuisance-optimized exact test. Confidence-box compatibility ignores some
shared-bit/physical constraints, making the rule conservative rather than invalid.
Unioning it with the old detector at full alpha would invalidate the combined
alpha ceiling; it would need a new proof or split error budget.

## 5. Finite-sample power bound beyond overlap

First condition on the full ancillary counts (clicks, classes, disclosure),
which fix error denominators m_j=n_cj and n_j=n_mj. Error draws in the two
blocks are independent conditional on these counts. Only the four error
coordinates are needed for the bound; additional gain alarms cannot decrease
power. Treat a zero error denominator as contributing lower bound zero.

For a chosen coordinate with positive denominators, let its actual attack
increment be Delta_j and H_j=h_cj+h_mj. Define

$$t_j=\Delta_j-d_j-H_j,\qquad S_j=1/m_j+1/n_j.$$

Clipping implies L_mj-U_cj>=phat_mj-phat_cj-H_j. Therefore nonrejection
in the increased-error direction implies

$$\widehat p_{mj}-\widehat p_{cj}\le d_j+H_j,$$
$$[(\widehat p_{mj}-p_{mj})-(\widehat p_{cj}-p_{cj})]\le-t_j.$$

For t_j>0, a weighted Hoeffding inequality gives

$$P(\text{no full alarm}\mid\text{ancillary})
\le\exp(-2t_j^2/S_j).$$

To see the scaling, the centered difference is a sum of n_j independent
monitor Bernoulli terms with weight 1/n_j and m_j independent calibration
terms with weight -1/m_j. The sum of their squared ranges is
n_j/n_j^2+m_j/m_j^2=S_j. Hoeffding's moment bound is exp(s^2*S_j/8).
Applying Markov's inequality to the negative exponential gives
exp(-s*t_j+s^2*S_j/8). Minimize over s>=0: the derivative is
-t_j+s*S_j/4, giving s=4*t_j/S_j and exponent -2*t_j^2/S_j.

Thus each positive t_j supplies a conditional lower power bound
1-exp(-2*t_j^2/S_j). Use zero if t_j<=0 or a denominator is zero. The maximum
of the four such bounds is also valid, since they bound the same full alarm
and the maximizing coordinate depends on parameters/ancillary counts, not
observed error outcomes. Average over the actual ancillary law to obtain an
unconditional bound. Do not replace random denominators with their expectations.

### A uniform but potentially loose whole-family lower bound

In the separated regime define

$$\gamma=(1-\lambda)\kappa-d_Z-d_X>0.$$

For every eligible attack, the sum of the basis residuals is
cr-d_Z-d_X>=gamma. At least one basis has Delta_b-d_b>=gamma/2.
Its two preparation-bit coordinates share that increment. If all four error
denominators are positive, put

$$H_{\max}=\max_{j\le4}H_j,\qquad
S_{\max}=\max_{j\le4}S_j,\qquad t_*=\gamma/2-H_{\max}.$$

Set ell=0 if a denominator is zero or t_*<=0. Otherwise set

$$\ell=1-\exp(-2t_*^2/S_{\max}).$$

For the basis with sufficient residual, t_j>=t_* and S_j<=S_max, so the
conditional power is at least ell, uniformly for r>=2*kappa and every beta.
The ancillary law is independent of policy. Consequently

$$\inf_{r\ge2\kappa,\,\beta\in[0,1]}P_A(\text{alarm})
\ge E_{\rm ancillary}[\ell].$$

This is a conservative analytical bound, **not a computed project result**.
It may be zero at every sampled resource setting. That would mean this bound
is uninformative there, not prove that the optimal detector has no power.
It does show a route to uniform consistency for fixed lambda<1, fixed
positive gamma, eta>0 and allocations with positive calibration and monitoring
disclosure fractions as their budgets grow: all four denominators grow,
H_max and S_max tend to zero, and ell tends to one. Allocations that starve
either block, or zero transmission, do not satisfy this argument.

Tighter finite-budget results would require exact/CP rejection boundaries and
continuous policy integration with their own validity checks. We have not
proved that balanced Eve or any allocation is finite-budget optimal.

## 6. Resources and implementation gate

Retain the Step 3 accounting: total emissions B=M+N; disclose all calibration
sifted bits and a random fraction f of monitoring sifted bits. With unchanged
mean gain Qbar,

$$E[\text{sacrifice}]=\bar Q(M+fN)/2,$$
$$E[\text{remaining candidates}]=\bar Q N(1-f)/2.$$

If C=M+fN is fixed, these become Qbar*C/2 and Qbar*(B-C)/2. They count
expected raw resources, not realized sample equality or final secret-key yield.
The drift allowance is a separate trust resource; increasing it cannot be
portrayed as merely collecting more data. Below overlap, splitting resources
controls two-stage uncertainty. At overlap, no split removes indistinguishability.

No empirical protocol is frozen here. Before the next implementation milestone:

1. Confirm this between-block, per-basis drift model and the fixed-noise attack
   alternative, or request the harder jointly drifting alternative instead.
2. Choose envelope scenarios explicitly as sensitivity values unless supported
   by independent measurements; retain cases inside and outside overlap.
3. Implement the confidence-gap baseline separately from the stationary APIs.
   Validate boundary inequalities, zero counts, iid conditional laws, alpha
   accounting and independence of disclosure before any performance campaign.
4. Only then freeze resources, candidate comparisons, uncertainty budgets and
   continuous-power calculation. Do not reuse the Step 7 proof unchanged.
5. Compare against drift-aware rules at the same null, observations and resource
   budget. A known-noise oracle or original threshold without drift validity is
   a diagnostic, not an equally valid deployment competitor.

Decision at the end of this analytical milestone: approve this minimal model
for implementation (subsequently approved and completed in Step 10). The
paper-specific access gap is resolved, but the missing 2022 source, a targeted
priority review for this exact frontier and independent expert review remain.

## 7. Checks and claim limits

The four overlap numbers were checked against `ir_basis_errors`. Another 216
deterministic rate/allowance cases checked the bias-interval feasibility condition
and constructed matching witnesses. The general bias inequalities,
confidence-gap size proof and weighted concentration bound were derived above.
A deterministic enumeration checked the concentration
tail in 192 small binomial settings: m,n=1..4, probability pairs (0,0.2),
(0.015,0.025), (0.1,0.35), (0.4,0.48), and deviations 0.01,0.05,0.1.
This is not a substitute for a proof,
a regression suite for an unimplemented detector, or certified floating rounding.
All 26 recorded Step 7 source/output fingerprints remained unchanged.

Do not claim new quantum physics, composable security, general adaptive attack
coverage, optimality, hardware robustness or publication priority. The potential
contribution is a useful **quantitative frontier/design result** built on these
known statistical principles, if tighter evaluated bounds and literature checks
eventually establish its significance. This note alone does not establish it.
