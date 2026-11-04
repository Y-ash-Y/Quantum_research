# BB84 simulation and observations

| File | Role |
|---|---|
| [simulator.py](simulator.py) | BB84 states, measurement, channel/local noise and restricted attack simulation |
| [dataset.py](dataset.py) | Reproducible synthetic session datasets |
| [telemetry.py](telemetry.py) | Disclosed counts and observation bookkeeping |
| [loss_attacks.py](loss_attacks.py) | Restricted loss/gain attack abstractions |
| [honest_nuisance.py](honest_nuisance.py) | Allowed honest channel/device variation |

Transmission depolarization and local gate noise are distinct parameters. Under
ideal local gates, honest QBER is half the transmission depolarizing parameter.
The IR/PNS policies are declared abstractions, not a complete hardware attack.

Read the [physical model](../../docs/physics-model.md) before changing simulator
settings. The [physics tests](../../tests/test_physics.py) compare real Aer
probabilities with independent density-matrix calculations and analytical limits.
There is no hardware-validation claim.
