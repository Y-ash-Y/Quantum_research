"""Run from any directory, without overwriting previous evidence."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from classical_ml.window_power_benchmark import main

if __name__ == '__main__':
    main()
