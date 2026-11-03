# Step 12: direct pooled bounded-drift test

Design and proof frozen on 2026-10-09 **before any new comparative evaluation**.
This is a research-only application of standard concentration and exact-binomial
tools to the approved [honest drift model](bounded-drift-model.md). It is not a
claim of new statistical inequalities, quantum physics, publication priority,
hardware validation or final-key security. Step 11 motivates the design;
Step 11 is not reused as its final evaluation data.

## 1. Why another test?

The confidence-gap baseline estimates four preparation-bit error probabilities
separately, although the model assigns the same probability to both preparation
bits in each basis. It also compares two separate stage intervals. This can
waste statistical sensitivity. We instead pool the two **public** error strata
within a basis and concentrate their calibration-to-monitoring difference.

Pooling is valid because the approved model has a shared per-basis probability
and error-independent sampling. It would not be justified under arbitrary
bit-dependent noise, changed detection efficiencies or pooled unequal probabilities.
No private bits, true error rates, Eve policies or information labels enter the rule.
Both upward and downward differences are tested; no direction is selected after
seeing the outcomes. The original confidence-gap baselines stay unchanged.

## 2. Public counts and the null

For basis b in {Z,X}, let Y_cb,Y_mb be the sums of the two disclosed-error
numerators, and n_cb,n_mb the sums of the corresponding public denominators.
Subscripts c and m mean calibration and monitoring. Conditional on these
denominators, independence and the shared-bit assumption give

$$Y_c\sim\operatorname{Binomial}(n_c,\theta),\qquad
Y_m\sim\operatorname{Binomial}(n_m,\mu),$$

with independent stages, 0<=theta,mu<=1/2 and |mu-theta|<=d_b. The counts retain
all rows, including zero denominators. For an active coordinate define

$$D=Y_m/n_m-Y_c/n_c.$$

Signal/decoy gain coordinates stay separate with their emitted-intensity
denominators, unknown probabilities in [0,1] and **zero** allowed change. Their
coverage arguments condition on sent-intensity denominators, not click counts.
One must not condition a gain test on its own observed successes.

## 3. A safe variance envelope (not a plug-in variance)

Let alpha be the per-rule fixed-horizon false-alarm budget. Set
epsilon=alpha/16. For Y~Binomial(n,p), n>0, use a one-sided Clopper–Pearson
upper endpoint U_raw with undercoverage probability at most epsilon:

$$U_{\rm raw}(Y,n)=
\begin{cases}
1,&Y=n,\\
F_{\mathrm{Beta}(Y+1,n-Y)}^{-1}(1-\epsilon),&Y<n.
\end{cases}$$

The binomial-CDF inversion defining this endpoint gives
P_p(U_raw<p)<=epsilon. Under the known physical domain p<=1/2,
U=min(1/2,U_raw) retains this coverage: on U_raw>=p both operands in the
minimum are >=p. This is not truncating an observation or widening the null.

For the **upward** test use calibration U_c. On its coverage event,
theta<=U_c and mu<=theta+d<=U_c+d. Because v(p)=p(1-p) has derivative
1-2p>=0 for p in [0,1/2], define

$$u_m=\min(1/2,U_c+d),\qquad
\widehat V_+=\frac{U_c(1-U_c)}{n_c}
+\frac{u_m(1-u_m)}{n_m}.$$

The actual variance of D is

$$V=\frac{\theta(1-\theta)}{n_c}
+\frac{\mu(1-\mu)}{n_m},$$

so V_hat_plus>=V on that coverage event. For the **downward** test exchange
the stages: take monitoring upper endpoint U_m, bound theta by
min(1/2,U_m+d), and form V_hat_minus. The stage exchange is fixed in advance.
The confidence event and D may be dependent; the proof below does not assume
that the estimated variance is independent of the test statistic.

## 4. Direct Bernstein concentration: derivation

Write S=D-(mu-theta) as the sum of n_m centered monitoring Bernoulli terms
divided by n_m and n_c negative centered calibration terms divided by n_c.
These terms are independent conditional on the denominators, have mean zero,
sum of variances V and absolute values bounded by

$$b=\max(1/n_c,1/n_m).$$

For any centered summand Z with |Z|<=b, and t>=0 with tb<3, expand its
moment-generating function. Since E[Z]=0 and
|E[Z^k]|<=b^(k-2)E[Z^2] for k>=2, and k!>=2*3^(k-2),

$$E[e^{tZ}]=1+\sum_{k=2}^{\infty}\frac{t^kE[Z^k]}{k!}
\le1+\frac{t^2E[Z^2]}2\sum_{k=2}^{\infty}(tb/3)^{k-2}
=1+\frac{t^2E[Z^2]}{2(1-tb/3)}.$$

The geometric sum converges because tb/3<1. Apply log(1+x)<=x and
independence to obtain

$$\log E[e^{tS}]\le\frac{t^2V}{2(1-tb/3)}.$$

Markov's inequality gives P(S>=h)<=exp(-th+t^2V/[2(1-tb/3)]).
For V>0 choose t=h/(V+bh/3). Substitution gives

$$P(S\ge h)\le\exp\left[-\frac{h^2}{2(V+bh/3)}\right].$$

If V=0 every centered summand is zero, so the positive-tail probability is
zero directly; no division by V is required. This is a standard Bernstein bound.
For L=log(1/epsilon), solve h^2=2L(V+bh/3):

$$h^2-(2bL/3)h-2VL=0,$$
$$h(V,b,L)=bL/3+\sqrt{2VL+(bL/3)^2}.$$

This positive root is increasing in V. Define the upward alarm by
D>d+h(V_hat_plus,b,L), and the downward alarm by
-D>d+h(V_hat_minus,b,L), using **strict** inequalities.

## 5. Why data-dependent thresholds remain valid

Fix an allowed honest law and positive denominators. Let E be the appropriate
variance coverage event. P(E^c)<=epsilon. On E,
h(V_hat,b,L)>=h(V,b,L). Under the upward null mu-theta<=d, an alarm implies

$$S=D-(\mu-\theta)\ge D-d>h(\widehat V,b,L)\ge h(V,b,L).$$

Therefore, without requiring E and S to be independent,

$$P(\mathrm{upward\ alarm})\le P(E^c)+P(S>h(V,b,L))
\le2\epsilon=\alpha/8.$$

For the downward null use the exchanged-stage argument. Integrating over the
ancillary error denominators preserves the bound. If either denominator is
zero the coordinate never alarms; there is no imputed observed proportion.

## 6. Gains and the total budget

For either stable gain coordinate the centered difference has summand range
lengths 1/n_c and 1/n_m. The sum of squared lengths is 1/n_c+1/n_m.
Hoeffding gives

$$P(D\ge h_g)\le\exp[-2h_g^2/(1/n_c+1/n_m)].$$

Choose h_g=sqrt((1/n_c+1/n_m)*log(8/alpha)/2), and test both D>h_g and
-D>h_g. Each gain direction spends alpha/8 without a variance confidence event.

| Components | Directions | Failure per direction | Total |
|---|---:|---:|---:|
| Z/X pooled errors | 4 | alpha/16 for variance coverage + alpha/16 for tail | alpha/2 |
| Signal/decoy gains | 4 | alpha/8 for direct Hoeffding tail | alpha/2 |

The OR of these eight **internal** directions has false-alarm probability at
most alpha for every law in the approved honest family. Independence across
coordinates is not needed for this union bound. OR-ing this detector with a
different externally selected detector would need a new joint budget/proof.

## 7. What is and is not improved

The test pools valid shared-bit observations, concentrates a difference directly,
and uses low-noise variance information conservatively. It may improve the
pilot's sensitivity but **does not uniformly dominate** the CP baseline; this
must be measured on fresh observations. It has no nuisance optimizer, training
set, adjustable significance allocation, selected direction or noise oracle.
It is O(number of transcript rows) work/memory with a fixed number of coordinates,
apart from special-function evaluation. No per-emitted-pulse simulation is needed.

It does not change the population overlap boundary. Nor is there a new uniform
continuous-family power certificate for this rule. The earlier Hoeffding lower
function belongs to the earlier confidence-gap rule and must not be transferred.
The proof is in real arithmetic; ordinary floating operations and beta quantiles
are **not** an outward-rounded numerical security certificate.

Same scope limits remain: trusted attack-free calibration, blockwise iid and
error-independent selection, shared preparation-bit error probabilities, fixed
envelopes and stable gains. Concurrent attacked-channel drift, adaptive/temporal
Eve, general quantum attacks and final-key security are outside the claim.

## References and priority

The concentration inequality and exact binomial intervals are established tools,
not discoveries of this project. See the Bernstein theorem and its historical
references in [Jothimurugesan et al., 2018, appendix Theorem 2](https://www.ece.iastate.edu/snt/files/2018/10/strsaga-nips-2018.pdf),
and [SciPy's exact-binomial interval documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
(citing Clopper & Pearson, 1934). The derivation above is self-contained and
uses no Gaussian approximation. Neither the derivation nor lack of a located
exact-match paper establishes publication novelty. Existing close-paper and
prior-art limitations still apply.
