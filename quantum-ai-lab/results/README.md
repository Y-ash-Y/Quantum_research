# Saved synthetic evidence

[benchmarks/](benchmarks/README.md) contains reviewed experiment summaries,
protocols, audits and selected compressed replay tables. These are evidence
supporting the research record, not disposable cache files.

The Git allowlist deliberately publishes selected files only. Larger `.npz`/
`.npy` arrays and incomplete or superseded local runs remain ignored. Do not
delete a compressed replay table because it resembles a generated artifact:
the release tests and independent verifier read those exact observations.

From the repository root:

```bash
python quantum-ai-lab/scripts/verify_classical_drift_audit.py \
  quantum-ai-lab/results/benchmarks/classical-drift-audit-v1
```

Use a new directory when generating another campaign. Runtime fields and gzip
metadata may differ across reproductions; compare the declared observations,
decisions and assumptions rather than requiring arbitrary binary identity.
