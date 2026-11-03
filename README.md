# Quantum AI Lab

## Finite-Sample Monitoring of BB84 Quantum Key Distribution

**Event: Research Scholar Day 2026 — March 2026**

**Author: Yash Yadav · Indian Institute of Technology Kanpur**

This project investigates how reliably public measurement statistics can detect
restricted eavesdropping in simulated BB84 quantum key distribution (QKD).
It combines a physically explicit noise model, classical machine-learning
baselines, finite-sample statistical tests, and calibration/disclosure resource
accounting.

[Read the final report](output/pdf/quantum_ai_lab_final_report.pdf)

The central question is not simply whether a classifier recognizes synthetic
attack labels. It is whether an alarm can distinguish an attack from allowed
honest channel variation while controlling false alarms and using finite
reference data.

## What the project implements

- BB84 state preparation and measurement with one explicit transmission
  depolarizing channel, separate from optional local device noise.
- Basis-biased intercept–resend (IR) and a restricted gain-matched
  photon-number-splitting (PNS) abstraction.
- Classical classification and calibrated physical/statistical baselines.
- Count-aware and loss-aware monitoring with honest-noise/device negative controls.
- Independent calibration blocks, randomly disclosed monitoring bits, and
  comparisons at matched expected sacrifice exposure.
- Restricted attack selection, stationary-model power analysis, bounded-drift
  identifiability, and a classical nuisance-parameter comparison.
- Tests, reproducible runners, saved synthetic replay evidence, and independent
  bookkeeping/interval checks.

The repository also contains exploratory quantum-kernel, temporal-CNN, and
adaptive-search components. They are not evidence of quantum advantage or
validated superiority over the corrected baselines.

## Physical model and observations

Alice prepares the four BB84 states; Bob independently selects the Z or X
measurement basis. With ideal local gates, the transmission channel is

$$\mathcal{E}_{\lambda}(\rho)=(1-\lambda)\rho+\lambda I/2,$$

so the honest matched-basis quantum bit error rate (QBER) is
$q_0=\lambda/2$. For example, `channel_noise=0.03` means theoretical
honest QBER `0.015`, not `0.03`.

For interception probability $r$ and Eve's Z-basis selection probability $\beta$,

$$q_Z=q_0+(1-\lambda)r(1-\beta)/2,\qquad
q_X=q_0+(1-\lambda)r\beta/2.$$

Uniform basis selection makes the mean QBER independent of Eve's basis bias,
although the per-basis errors change. Full noiseless interception gives mean
QBER 25%.

The weak-coherent-pulse abstraction uses Poisson photon counts and idealized
independent transmission. Its restricted PNS policy can preserve signal gain
and QBER while changing the decoy/signal gain ratio. It is not a complete
optical-device attack or a decoy-state security proof.

The public-count rules use disclosed error counts and their observed
denominators, together with signal/decoy detection counts. Hidden attack labels,
true nuisance parameters, and retained private bits are not alarm inputs.

[Physical assumptions and checks](quantum-ai-lab/docs/physics-model.md)

## Main results

### Classical ML does not dominate every attack-specific baseline

The corrected fixed-policy campaign contains 29,500 synthetic sessions, with
6,000 emissions per session. Entries below are percentages, reported as the
five-seed mean ± sample standard deviation; they are not confidence intervals.

| Method | Honest false alarms | Weak IR detection | Medium PNS detection |
|---|---:|---:|---:|
| Calibrated QBER | 5.5 ± 0.8 | 59.8 ± 2.6 | 5.4 ± 1.2 |
| Decoy gain ratio | 5.6 ± 0.5 | 5.0 ± 1.1 | 29.0 ± 3.7 |
| Joint max-tail | 5.0 ± 1.0 | 49.7 ± 1.8 | 14.7 ± 2.0 |
| Random forest | 5.7 ± 0.7 | 42.2 ± 3.7 | 25.1 ± 3.2 |

The useful statistic depends on the attack. Actual false-alarm rates differ,
so these numbers do not establish superiority at exactly equal attained size.

[Benchmark protocol](quantum-ai-lab/docs/fair-benchmark.md) ·
[Saved summary](quantum-ai-lab/results/benchmarks/fixed-policy-v2/benchmark.json)

### Honest variation can look like an attack

In a separate robustness campaign, frozen random-forest honest alarms rise
from 4.6% to 20.1% under higher honest noise. A device-noise negative control
drives a misspecified joint rule to 67.84% honest alarms at its largest tested
window. Collecting more observations cannot repair a wrong null model.

[Robustness](quantum-ai-lab/docs/robustness-benchmark.md) ·
[Loss-aware screening](quantum-ai-lab/docs/loss-aware-benchmark.md) ·
[Window and device limits](quantum-ai-lab/docs/window-power-benchmark.md)

### Bounded honest drift creates a model-specific observation limit

Let $d_Z,d_X$ bound absolute per-basis changes in honest QBER between the trusted
reference block and monitoring. The restricted raw-bit knowledge constraint
is $r/2\ge\kappa$; this proxy is not final-key leakage.

Under the declared model, an eligible IR attack can share the complete
public-count law of an allowed honest channel when

$$ (1-\lambda)\kappa\le d_Z+d_X. $$

For equal allowances, $\lambda=0.03$ and $\kappa=0.01$, the boundary is
**0.485 QBER percentage points per basis**. A uniformly level-$\alpha$ alarm
cannot have power above $\alpha$ on a matching attack. This is an application
of statistical identifiability to the restricted model, not evidence that
BB84 security is broken.

[Model, assumptions, and derivation](quantum-ai-lab/docs/bounded-drift-model.md)

### A stronger established classical comparator recovers sensitivity

A retrospective comparison replays 12,288 existing calibration/monitoring
pairs in 24 selected cohorts. Each focal attack cell below contains 512 paired
trials with the same observations for both detectors.

| Allocation | Allowed drift per basis | Direct CP/Bernstein | Classical Berger–Boos |
|---|---|---:|---:|
| Plan 9: 20% calibration | Zero | 168/512 (32.81%) | 378/512 (73.83%) |
| Plan 9: 20% calibration | 0.2425 percentage points | 12/512 (2.34%) | 79/512 (15.43%) |
| Plan 10: approximately one-third calibration | Zero | 151/512 (29.49%) | 353/512 (68.95%) |
| Plan 10: approximately one-third calibration | 0.2425 percentage points | 18/512 (3.52%) | 76/512 (14.84%) |

Both finite-sample constructions have a model-conditional 5% upper false-alarm
ceiling in real arithmetic, but different attained honest alarm rates.
These are fixed-probe results, not a proof of uniform dominance or continuous
worst-case power. The comparator uses established statistical machinery; its
improvement is not presented as a new statistical inequality. The comparison
is retrospective, not independent confirmatory evaluation.

[Detailed interpretation](quantum-ai-lab/docs/classical-baseline-audit.md) ·
[Replay summary](quantum-ai-lab/results/benchmarks/classical-drift-audit-v1/audit.json)

## Scope and limitations

- All datasets are synthetic; there is no empirical hardware validation.
- Calibration is assumed attack-free. The current attacked branch has a fixed
  background channel rather than concurrent adversarial and honest drift.
- Restricted IR/PNS abstractions do not cover arbitrary coherent attacks,
  calibration poisoning, or all source/detector side channels.
- The raw-bit knowledge proxy is not mutual information, min-entropy, or a
  composable final-key leakage measure.
- Ordinary floating-point calculations are not outward-rounded numerical
  certificates. Mathematical validity remains conditional on the declared model.
- This monitoring framework does not replace authentication, reconciliation,
  parameter estimation, privacy amplification, or a complete QKD security proof.
- The project does not establish quantum computational advantage, an optimal
  detector, or a new quantum effect.

## Repository layout

```text
quantum-ai-lab/
  src/
    bb84/          physical model, datasets, public telemetry, loss abstractions
    classical_ml/  classifiers, statistical rules, benchmarks, auditors
    research/      calibration, attack geometry, power, bounded-drift tests
    adversarial/   restricted policies, session simulation, search components
    quantum_ml/    exploratory statevector fidelity-kernel implementation
    sequence_ml/   exploratory temporal datasets and CNN
  tests/           physics, inference, sampling, and regression checks
  scripts/         experiment runners, notebook builders, replay verification
  docs/            technical assumptions, protocols, and interpretation
  notebooks/       companion drift-study notebooks
  results/
    benchmarks/    selected synthetic summaries and compressed replay evidence
  requirements-physics.txt
  requirements-adaptive.txt
  requirements-notebooks.txt
output/pdf/
  quantum_ai_lab_final_report.pdf
pytest.ini
```

## Setup and verification

From the repository root, use Python 3.12 and create a local environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r quantum-ai-lab/requirements-adaptive.txt
python -m pytest -q
```

On Windows, activate with `.venv\Scripts\activate` instead. Environments are
local-only and are not distributed with the repository.

Check the saved classical comparison without generating new observations:

```bash
python quantum-ai-lab/scripts/verify_classical_drift_audit.py \
  quantum-ai-lab/results/benchmarks/classical-drift-audit-v1
```

Experiment runners require a new output directory and preserve existing
evidence. For example:

```bash
python quantum-ai-lab/scripts/run_fair_benchmark.py \
  --output quantum-ai-lab/results/benchmarks/my-fresh-run
```

To regenerate the direct drift comparison and then the classical comparison
from the implementation, use separate new directories:

```bash
python quantum-ai-lab/scripts/run_direct_drift_pilot.py \
  --trials 512 --output quantum-ai-lab/results/benchmarks/reproduced-direct
python quantum-ai-lab/scripts/run_classical_drift_audit.py \
  --source quantum-ai-lab/results/benchmarks/reproduced-direct \
  --output quantum-ai-lab/results/benchmarks/reproduced-classical
python quantum-ai-lab/scripts/verify_classical_drift_audit.py \
  quantum-ai-lab/results/benchmarks/reproduced-classical \
  --source quantum-ai-lab/results/benchmarks/reproduced-direct
```

These are full experiments, not quick installation checks. Fixed settings and
seeds support reproduction with the pinned environment. Runtime fields and
compressed-file metadata need not be byte-identical between executions;
cross-platform numerical differences should be checked rather than assumed absent.
Precomputed checkpoints are not required. Large generated arrays remain local
and can be regenerated by their experiment runners.

Optional notebook/plotting dependencies are listed in
`quantum-ai-lab/requirements-notebooks.txt`; the broader exploratory ML
environment is separate in `quantum-ai-lab/requirements.txt`.

The published scope is code, technical documentation, the final PDF, and
selected synthetic evidence. Personal working files, report sources,
environments, and disposable build artifacts are excluded.

## License

See [LICENSE](LICENSE).
