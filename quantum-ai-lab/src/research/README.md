# Finite-sample research methods

These modules study explicitly restricted statistical models. Their guarantees
depend on the declared observations, calibration assumptions and nuisance bounds.

| Modules | Subject |
|---|---|
| [calibration_protocol.py](calibration_protocol.py), [allocation_frontier.py](allocation_frontier.py) | Independent reference blocks and resource allocation |
| [identifiability.py](identifiability.py), [off_grid_geometry.py](off_grid_geometry.py) | Observation overlap and restricted policy geometry |
| [exact_comparison.py](exact_comparison.py), [best_response_stability.py](best_response_stability.py), [detector_power.py](detector_power.py) | Detector comparisons, search stability and stationary power |
| [drift_detector.py](drift_detector.py), [direct_drift_detector.py](direct_drift_detector.py) | Bounded-drift alarm constructions |
| [classical_drift_audit.py](classical_drift_audit.py) | Established classical nuisance-parameter comparator |

Read [bounded-drift assumptions](../../docs/bounded-drift-model.md) and the
[classical audit](../../docs/classical-baseline-audit.md) before interpreting the
saved comparison. Standard confidence bounds and nuisance-parameter methods are
not presented as newly discovered inequalities.

Overlap limits are model-specific. Floating-point calculations are not
outward-rounded numerical certificates, and a raw-bit knowledge proxy is not
composable final-key leakage. Use [scripts/](../../scripts/README.md) for the
declared experiment configurations and audits.
