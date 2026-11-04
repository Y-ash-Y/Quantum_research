# Restricted adversarial policies

[eavesdropper_game.py](eavesdropper_game.py) implements restricted Eve policies,
session simulation and adaptive-search bookkeeping. It is not an unrestricted
quantum adversary or a live attack tool.

Policy selection, calibration and final evaluation must use separate streams.
Infeasible information constraints, rejected sessions and empty searches remain
explicit rather than silently changing labels or denominators.

The [adaptive evaluation notes](../../docs/adaptive-evaluation.md) explain what
was corrected, what was tested and which older exploratory claims remain
unestablished. [test_adversarial.py](../../tests/test_adversarial.py) includes
physical and small actual-search checks.

Adaptive search results do not replace fixed-policy baselines, a worst-case
proof over all attacks, or a complete QKD security analysis.
