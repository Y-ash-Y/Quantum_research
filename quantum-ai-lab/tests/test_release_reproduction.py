"""Release packaging supports replay evidence outside the default directory."""
import importlib.util
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        'release_verifier', ROOT / 'scripts/verify_classical_drift_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_verifier_accepts_explicit_source_directory(tmp_path):
    source = ROOT / 'results/benchmarks/direct-drift-pilot-v1'
    alternate = tmp_path / 'direct-pilot'
    alternate.mkdir()
    shutil.copy2(source / 'public_counts.csv.gz', alternate / 'public_counts.csv.gz')
    result = load_verifier().verify(
        ROOT / 'results/benchmarks/classical-drift-audit-v1', source=alternate)
    assert result['status'] == 'passed'
    assert result['selected_original_pairs'] == 12288


def test_replay_verifier_does_not_silently_fall_back_to_default_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_verifier().verify(
            ROOT / 'results/benchmarks/classical-drift-audit-v1', source=tmp_path)
