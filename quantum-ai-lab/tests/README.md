# Verification suite

From the repository root:

```bash
python -m pytest -q
```

The root `pytest.ini` selects this directory and adds `quantum-ai-lab/src` to the
import path. Tests use synthetic observations and temporary output directories;
no hardware or downloaded private dataset is required.

- `test_physics.py`: real-Aer probabilities, density-matrix references and analytical limits.
- `test_adversarial.py`: restricted policies, constraints and separated search/evaluation bookkeeping.
- Count, nuisance, loss and robustness tests: calibration and honest negative controls.
- Calibration, geometry, power and drift tests: model-specific inference and replay checks.
- `test_release_reproduction.py`: the published compressed evidence and explicit source-path contract.
- `test_repository_docs.py`: the single requirements file and public navigation links.

Passing tests is not hardware validation, a new security proof or independent
confirmation of every research claim. The [documentation index](../docs/README.md)
links the assumptions and limitations for each campaign. Full notebook/campaign
reruns are separate from installation checks.
