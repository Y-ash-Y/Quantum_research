# Physics milestone: reference model version 2

The simulator correctness milestone is implemented. **91 tests passed** with real
Qiskit-Aer and independent analytical/density-matrix references on Python 3.12.14,
Qiskit 2.5.2, Aer 0.17.2 and NumPy 2.5.3 (macOS arm64). These tests validate the
specified simplified model. They do not establish novel research, hardware accuracy,
ML performance, or security of a generated secret key.

The old notebook outputs, figures, final PDF and LaTeX reports have not been
regenerated. They are version-1 historical artifacts and include known numerical
and interpretation errors. They must not be presented as version-2 measurements.

## 1. States and channel placement

Alice prepares bit 0/1 in basis Z/X:

| Bit | Basis | State | Preparation |
| --- | --- | --- | --- |
| 0 | Z | zero computational state | no gates |
| 1 | Z | one computational state | X |
| 0 | X | plus state | H |
| 1 | X | minus state | X then H |

Bob applies H when measuring in X, then measures in Z. Measurement in the same
basis is deterministic without noise; the conjugate basis gives probability 1/2
for either bit. The X-then-H order for the minus state is tested explicitly.

Each circuit places exactly one explicit depolarizing QuantumError instruction
between preparation and measurement:

$$\mathcal{E}_{\lambda}(\rho)=(1-\lambda)\rho+\lambda I/2.$$

Here rho is the single-qubit density matrix, I is the two-dimensional identity,
and lambda=`channel_noise` is a mixing parameter in [0,1]. It is **not QBER**.
For ideal gates and matched bases, the intended outcome receives probability
`(1-lambda)+lambda/2=1-lambda/2`; the wrong outcome therefore receives `lambda/2`.
At lambda=0.03, both honest basis error rates are **0.015**.

The previous implementation attached noise after H/X gates. That made local gate
counts determine noise exposure and gave different honest Z/X rates. Version 2
applies transmission noise even when preparing Z-basis bit zero, with no local
preparation gates. An explicit non-unitary channel avoids relying on an identity
marker that an optimizer could remove. Tests compare independent 2x2 matrix
calculations with Aer, including after transpilation at optimization levels 0/3.

`gate_noise`, default zero, is separate depolarizing noise after local H/X gates.
Nonzero device noise can create genuine *modelled device asymmetry*. It must not
be described as transmission noise or evidence of Eve. The tests cover both zero
and nonzero gate noise. These gate errors are an illustrative device model, not
a calibrated description of a physical source or detector.

Honest theoretical Z/X rates are equal with ideal gates. Individual finite samples
can differ, and their **absolute** observed difference has positive sampling bias;
we do not assert that every sampled session has zero asymmetry.

[Qiskit depolarizing-channel definition](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.depolarizing_error.html).

## 2. Intercept–resend reference equations

Eve is idealized: she intercepts before the modelled downstream channel, measures
in Z with probability beta (otherwise X), and resends the measured state. A
wrong-basis interception yields a random resend bit. Let r be the interception
probability and lambda the transmission parameter, with ideal local gates.

For Alice Z, Eve chooses the wrong basis with probability `1-beta`. Without such
an interception, error probability is `lambda/2`; wrong-basis interception gives
error probability 1/2. Thus:

$$Q_Z=[1-r(1-\beta)]\lambda/2+r(1-\beta)/2
      =\lambda/2+r(1-\beta)(1-\lambda)/2.$$

For Alice X, the wrong-basis probability is beta:

$$Q_X=(1-r\beta)\lambda/2+r\beta/2
      =\lambda/2+r\beta(1-\lambda)/2.$$

With uniform Alice bases:

$$Q=(Q_Z+Q_X)/2=\lambda/2+r(1-\lambda)/4,$$
$$|Q_X-Q_Z|=r|2\beta-1|(1-\lambda)/2.$$

Full noiseless symmetric interception gives Q=0.25. Full noiseless fixed-Z
interception gives Q_Z=0, Q_X=0.5 and Q=0.25. The session simulator's `biased_ir`
uses beta=1 for eve_basis=Z or beta=0 for eve_basis=X. The temporal simulator's
continuous attack uses beta=1/2. Tests enumerate all Alice and Eve basis/bit
outcomes and check these equations, then exercise sampled session/sequence paths.

## 3. Photon number, losses and PNS

The photon abstraction assumes phase-randomized weak coherent pulses, ideal
intensity-independent detector response, no dark counts, and a Poisson mixture:

$$P(k|\mu)=e^{-\mu}\mu^k/k!,$$

where mu is mean photons per pulse and k is photon count. A normal pulse is
detected with probability `1-(1-eta)^k`, where eta in [0,1] is the effective
per-photon transmission/detection probability. Averaging:

$$Q_{\mu}=1-\sum_{k=0}^{\infty}e^{-\mu}\frac{[\mu(1-\eta)]^k}{k!}
          =1-e^{-\mu}e^{\mu(1-\eta)}=1-e^{-\mu\eta}.$$

The idealized PNS policy blocks k<=1. It forwards k>=2 with intensity-independent
probability q over a lossless line, retaining a photon for Eve. The simulator
models only Bob's resulting gain and faithful qubit forwarding, not a full
multi-photon optical state or Eve's imperfect storage/measurement hardware.

$$P_{\ge2}(\mu)=1-e^{-\mu}(1+\mu),$$
$$q=\frac{1-e^{-\mu_s\eta}}{P_{\ge2}(\mu_s)},\qquad
Q_s^{PNS}=qP_{\ge2}(\mu_s)=Q_s.$$

The same q applies to decoy pulses, so `Q_d^PNS=q*P_ge2(mu_d)`. This is an
assumption of the restricted attack: Eve does not choose q based on intensity.
Decoy-state estimators in a full protocol are more general than this gain-ratio
illustration.

For mu_s=0.5, mu_d=0.1 and eta=0.15:

| Quantity | Closed-form expectation |
| --- | --- |
| q | 0.8010343811 |
| Honest signal gain | 0.0722565137 |
| Gain-matched PNS signal gain | 0.0722565137 |
| Honest decoy/signal gain ratio | 0.2060445438 |
| PNS decoy/signal gain ratio | 0.0518695359 |

Values q>1 are physically infeasible here and now raise ValueError. A minuscule
1e-12 floating-point tolerance is permitted at q=1; this is not a policy for
masking infeasible gains. Zero multiphoton probability also raises ValueError.
The small-intensity Poisson tail uses its series to avoid cancellation. Tests
cover a weak-intensity case, the feasible boundary, and independent Monte-Carlo
detection rates for signal/decoy and honest/PNS policies.

PNS adds no bit error in this ideal forwarding model. With channel noise present,
its matched-basis error probability remains lambda/2. This does not imply that
the resulting raw key is secure or that real PNS hardware would behave identically.

## 4. Exact probabilities and reproducibility

The eight bit/preparation-basis/measurement-basis probabilities are computed from
real Aer density matrices using save_probabilities **before measurement**. The
table is no longer estimated from 40,000 unseeded shots. Density-matrix evaluation
is exact for this small circuit up to numerical precision; later sampled bit
outcomes still have ordinary Monte-Carlo uncertainty.

The immutable lookup has a bounded cache. Clearing it or starting a fresh process
does not introduce a random table estimate. The legacy shots argument is accepted
and validated but has no effect on the probabilities.

run_bb84 uses a local NumPy generator for photons, detection and bit/basis draws;
a derived seed controls Aer measurement sampling. simulate_key_sequence uses a
local Python generator. Neither entry point changes global NumPy/Python random
state. Tests check repeated calls, cache clearing, and separate fresh processes
for both sessions and sequences. Seed repeatability is established in the tested
runtime; it is not a promise of bitwise identity across arbitrary library versions,
machines or backend algorithms.

## 5. Bursty interception domain

For 0<s<1 and mean ON burst length B emitted pulses, use transition probabilities
`b=1/B` from ON to OFF and `a=s/[(1-s)B]` from OFF to ON. Their stationary ON
fraction is:

$$\pi_{ON}=a/(a+b)=s.$$

Require B>=1 and a<=1. An invalid transition is rejected rather than treated as a
probability. At s=0/1, absorbing OFF/ON states are used; the mean finite run-length
interpretation does not apply at these boundaries. Initialization is stationary.
Transitions occur in emitted-pulse time **before sifting**, so B is not a count of
consecutive sifted observations.

Continuous and bursty attacks share the expected mean at fixed s, but correlations
alter session-mean variance. Tests verify the shared mean and nonidentical sample
variance for a fixed diagnostic configuration. This is not a theorem that an
aggregate classifier can never discriminate them.

## 6. API, counts and limitations

- n_bits remains the legacy argument name for **emitted pulses**.
- key_length counts basis-matched detections from **signal and decoy pulses**. It
  is not a signal-only distilled secret-key length.
- model_version=2 and raw signal/decoy sent/detected counts plus per-basis
  sifted/error counts accompany every successful session.
- No detection/matched basis gives None. Missing individual denominators retain
  legacy zero-valued statistics for compatibility; inspect counts before treating
  zero as a measured rate. Correct missing-data policy remains benchmark work.
- build_qubit_circuit now accepts keyword-only channel_noise; an external caller
  must pass it to include transmission noise. build_noise_model now handles only
  keyword-only gate_noise. Applying a gate-noise model is not a substitute for
  including the explicit channel.
- Public entry points reject invalid probabilities, intensities, counts, attack
  names, bases, seeds and Markov transitions. The private pulse inner loop assumes
  validated inputs.
- No dark counts, afterpulsing, dead time, intensity drift, detector mismatch,
  error correction, authentication, privacy amplification or composable finite-key
  security estimator is implemented by this milestone.
- Raw error sequences are simulator-observable. Operational monitoring must define
  telemetry available without revealing retained key bits, e.g. disclosed and
  sacrificed parameter-estimation samples with their sampling masks.

## 7. Running and scope of verification

From repository root, with a Python 3.12 environment activated:

```bash
python -m pip install -r quantum-ai-lab/requirements-physics.txt
python -m pytest -q quantum-ai-lab/tests/test_physics.py
```

The pinned file includes the tested minimal physics/test dependency set. Full
training dependencies remain separate. Fresh-process tests require normal native
OpenMP shared-memory access; the desktop sandbox initially prevented this, so the
physics suite was verified outside that restriction. No physics test was skipped
or mocked. The subsequent adaptive-evaluation suite includes labelled plumbing
mocks as well as separate real-Aer physics and actual-forest integration checks.
Stochastic-rate tests use fixed seeds and six-standard-deviation envelopes plus
one-count resolution, not exact equality to theory. Correlated bursty sequences
are checked with session-level uncertainty, not an independent-bit assumption.

**Subsequent adaptive milestone:** null positive labels are repaired, long-run
expected-information targets are enforced, attack selection and final evaluation
are separated, and honest-test false alarms/intervals are recorded. See
[adaptive-evaluation.md](adaptive-evaluation.md) for the precise information
definition and API migration.

**Subsequent benchmarks:** comparable physical/statistical baselines and
repeated-seed fixed-policy results are now in [fair-benchmark.md](fair-benchmark.md).
Frozen-detector noise/window and attack-strength shifts are now in
[robustness-benchmark.md](robustness-benchmark.md).

**Still pending:** honest-vs-attack temporal data; realistic telemetry;
loss/device drift; nuisance-aware monitoring and adaptive-search benchmark reruns.
No headline training rerun, new AUC/novelty claim, report regeneration, commit or
push was part of the physics milestone.
