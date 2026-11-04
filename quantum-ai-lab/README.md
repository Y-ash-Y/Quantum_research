# Quantum AI Lab: code and evidence guide

Run installation and tests from the [repository root](../README.md):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

| Folder | Purpose |
|---|---|
| [src/](src/README.md) | Simulator, detectors, research methods and exploratory models |
| [scripts/](scripts/README.md) | Experiment runners, audits and notebook builders |
| [tests/](tests/README.md) | Physical, statistical, replay and packaging checks |
| [docs/](docs/README.md) | Assumptions, frozen protocols and interpretation |
| [notebooks/](notebooks/README.md) | Saved drift-study companions |
| [results/](results/README.md) | Reviewed summaries and compressed synthetic replay evidence |

The primary contribution is a model-specific finite-sample monitoring study,
not a proof of BB84 insecurity or quantum computational advantage. The
`quantum_ml/` and `sequence_ml/` components are exploratory; they are not the
established comparator behind the main claims.

Existing result paths are deliberate. Runners require new output directories;
do not rename evidence folders or overwrite completed studies while reproducing
the work. Large ignored arrays and personal working material are not required
for the published compressed replay check.
