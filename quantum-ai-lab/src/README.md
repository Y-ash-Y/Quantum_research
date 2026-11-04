# Source packages

| Package | Responsibility |
|---|---|
| [bb84](bb84/README.md) | Physical simulation, datasets and public telemetry |
| [classical_ml](classical_ml/README.md) | Classifiers, count rules, nuisance controls and benchmark audits |
| [research](research/README.md) | Calibration allocation, restricted policy geometry, power and drift tests |
| [adversarial](adversarial/README.md) | Restricted Eve policies and adaptive-search bookkeeping |
| [quantum_ml](quantum_ml/README.md) | Exploratory simulated fidelity-kernel SVM |
| [sequence_ml](sequence_ml/README.md) | Exploratory temporal error-sequence CNN |

The root `pytest.ini` adds this directory to the test import path. The runnable
entry points are in [scripts/](../scripts/README.md); no package move is needed
to use the repository.

Keep simulation truth separate from public alarm inputs. Hidden attack labels,
the true channel parameter and retained private bits may support evaluation,
but are not legitimate detector inputs in the maintained monitoring studies.
