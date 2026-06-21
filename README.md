# Quantum AI Lab

**Detecting sub-threshold eavesdropping in BB84 QKD with machine learning.**

## Overview
The textbook BB84 security check raises an alarm only when the quantum bit error
rate (QBER) exceeds ~11%. A careful eavesdropper can stay *below* that threshold
and remain invisible to the standard test. This project shows that machine
learning on richer channel statistics catches such **sub-threshold** attacks that
the QBER test misses.

Two attacks are modelled, both kept below the 11% threshold:
- **Basis-biased intercept–resend** — produces a per-basis QBER *asymmetry*
  (`QBER_X ≫ QBER_Z`) while the average stays low.
- **Gain-matched photon-number-splitting (PNS)** — Eve masks the channel loss so
  *both* QBER and signal gain look normal; only the *decoy/signal gain ratio*
  collapses (the regime that motivated decoy-state QKD).

## Key results
| Detector | Detection rate @ 11% | ROC AUC |
|---|---|---|
| QBER threshold test | **0.8%** | 0.726 |
| ML (rich features) | **87.4%** | 0.963 |

- **Gain-matched PNS is invisible to QBER** (AUC 0.51 ≈ random) and to signal gain,
  yet detected **perfectly** (AUC 1.00) via the decoy ratio alone.
- The decisive features are **QBER asymmetry** and **decoy/signal ratio**, not QBER.
- A naive **quantum-kernel SVM fails** (AUC 0.58) due to exponential kernel
  concentration; once the encoding is tuned it becomes **competitive** (AUC 0.95)
  but never beats the classical baselines — no quantum advantage here.
- The simulator is validated against closed-form QKD theory, and runs are
  reproducible via a seed.

### Detector coverage under an adaptive eavesdropper
Treating detection as a game where Eve adaptively picks her attack *channel*:
- **Detector robustness = feature coverage.** A QBER-only or asymmetry-aware
  detector is evaded at **every** information level (Eve switches to PNS); only a
  **decoy-aware/combined** detector is robust.
- The residual low-information gap is a **finite-key stealth floor**: at a fixed
  attack, combined-detector detection rises **0.14 → 0.64 → 0.96** as the key grows
  (2k → 6k → 20k pulses).

## Status & relation to prior work
This is a **completed, self-contained reproduction / learning project**, not a claim
of novel results. Its findings are consistent with — and were verified against — the
recent literature:
- The adaptive-eavesdropper detection game with decoy-aware/temporal detectors and a
  finite-key stealth floor is developed more rigorously (entropy-accumulation bounds,
  more attack families) in [arXiv:2603.03502](https://arxiv.org/abs/2603.03502).
- Multi-attack DV-QKD detection with CNN/LSTM/quantum-LSTM appears in
  [arXiv:2509.14282](https://arxiv.org/abs/2509.14282).
- Decoy-state detection of PNS is standard (Lo–Ma–Chen, 2005); game-theoretic
  detection-probability framings predate this work.

The value here is a **theory-validated, transparent, single-codebase testbed** that
independently reproduces these effects (sub-threshold detection gap, quantum-kernel
concentration pitfall, coverage/stealth-floor behaviour) with honestly reported
negative results. "Adversarial" means an *adaptive eavesdropper choosing a physical
attack strategy* — distinct from input-perturbation adversarial *examples* on
quantum classifiers.

## Project layout
```
src/
  bb84/          photon/decoy-state BB84 simulator + sub-threshold dataset
  classical_ml/  classical models and evaluation utilities
  quantum_ml/    quantum-kernel SVM (exact statevector ZZFeatureMap kernel)
  sequence_ml/   1D-CNN on raw sifted-error sequences (temporal attacks)
  adversarial/   eavesdropper-vs-detector game (channel palette, best-response)
notebooks/
  main.ipynb     end-to-end experiment driver (generates all figures, ~30 s)
results/figures/ generated plots
paper/report.tex LaTeX report
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running
```bash
jupyter notebook notebooks/main.ipynb   # run all cells; regenerates results/figures/
```

## Method notes
- The simulator (`src/bb84/simulator.py`) runs real single-qubit Qiskit-Aer
  circuits through a depolarizing channel, with a Monte-Carlo photon-number/decoy
  layer for loss, PNS, and gain statistics. Default channel is high-loss
  (`eta=0.15`) — the regime where PNS is a genuine threat.
- The quantum kernel is computed exactly from statevectors
  (`K(xi,xj)=|⟨φ(xi)|φ(xj)⟩|²`), which is exact and fast.
