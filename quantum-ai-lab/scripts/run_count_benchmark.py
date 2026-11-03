"""Repository-root entry point for count-aware detectors."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from classical_ml.count_benchmark import main

if __name__ == '__main__':
    main()
