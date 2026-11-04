# Runners and audits

Run commands from the repository root after installing `requirements.txt`.
Use `--help` to inspect a runner before starting a full campaign.

| Group | Entry points |
|---|---|
| Classical campaigns | `run_fair_benchmark.py`, `run_robustness_benchmark.py`, `run_count_benchmark.py`, `run_nuisance_stress.py`, `run_loss_aware_benchmark.py`, `run_window_power_benchmark.py` |
| Calibration and restricted geometry | `run_calibration_pilot.py`, `run_allocation_frontier.py`, `run_exact_comparison.py`, `run_best_response_stability.py`, `run_off_grid_geometry.py`, `run_detector_power.py` |
| Drift comparisons | `run_drift_pilot.py`, `run_direct_drift_pilot.py`, `run_classical_drift_audit.py` |
| Independent checks | `audit_*.py`, `check_identifiability.py`, `diagnose_robustness.py`, `verify_classical_drift_audit.py` |
| Notebook companions | `build_drift_notebook.py`, `build_direct_drift_notebook.py` |

The quick saved-evidence check does not generate new observations:

```bash
python quantum-ai-lab/scripts/verify_classical_drift_audit.py \
  quantum-ai-lab/results/benchmarks/classical-drift-audit-v1
```

Full runners require fresh output directories. Saved summaries, compressed replay
files and their manifests belong together; do not overwrite or rename them to
reuse an example command. See the [results guide](../results/README.md) and each
experiment's [protocol](../docs/README.md).
