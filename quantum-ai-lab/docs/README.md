# Research documentation

Use the root [README](../../README.md) for the main results and installation.
These versioned notes preserve the assumptions and interpretation behind them.

| Start with | Purpose |
|---|---|
| [Physical model](physics-model.md) | Noise channels, observations and simulator checks |
| [Fixed-policy benchmark](fair-benchmark.md) | Common calibration target and physical-versus-ML comparison |
| [Robustness](robustness-benchmark.md), [nuisance controls](nuisance-stress.md) | Honest variation and detector failure modes |
| [Count-aware](count-aware-benchmark.md), [loss-aware](loss-aware-benchmark.md), [window power](window-power-benchmark.md) | Counts, gains and finite exposure |
| [Calibration](calibration-protocol.md), [allocation](allocation-frontier.md) | Trusted reference resources and monitoring budgets |
| [Bounded drift](bounded-drift-model.md), [direct test](direct-drift-test.md) | Model-specific overlap and inference constructions |
| [Classical audit](classical-baseline-audit.md) | Established comparator and retrospective interpretation |

The adjacent protocol/result pairs provide the full settings. All observations
are synthetic. Calibration is assumed attack-free, attack families are restricted,
and ordinary floating point is not a certified numerical proof. Keep these
limitations attached to any figures or numerical summaries reused elsewhere.

All current install commands use the single root [requirements.txt](../../requirements.txt).
The published PDF and historical outputs remain unchanged records of their runs.
