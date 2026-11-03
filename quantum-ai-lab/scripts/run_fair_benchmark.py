"""Run from repository root using the pinned regression-test environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from classical_ml.fair_benchmark import main

if __name__ == '__main__':
    main()
