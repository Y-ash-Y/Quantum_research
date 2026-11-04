# Classical learning, monitoring and benchmark audits

This package contains both ML baselines and explicit statistical monitoring
rules. They answer different questions and are evaluated separately.

- [models.py](models.py) and [evaluate.py](evaluate.py): exploratory classifier fitting and evaluation.
- [fair_benchmark.py](fair_benchmark.py): fixed-policy physical/statistical versus ML comparison.
- [count_detectors.py](count_detectors.py) and [loss_aware_detectors.py](loss_aware_detectors.py): count-based and gain-aware rules.
- `*_benchmark.py`: nuisance, robustness, window-length and loss-aware campaigns.
- `*_audit.py`: replay, rates, bookkeeping and interpretation checks.
- [power_numerics.py](power_numerics.py): numerical support for power calculations.

Use the corresponding [runner](../../scripts/README.md) rather than manually
combining a detector with thresholds from another experiment. A high classifier
score is not an equal-size comparison, a physical security proof or a population
false-alarm guarantee.

Start with the [fixed-policy protocol](../../docs/fair-benchmark.md), then the
[robustness study](../../docs/robustness-benchmark.md) for the failure of a frozen
detector under allowed honest variation.
