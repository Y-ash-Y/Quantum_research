# Exploratory quantum-kernel SVM

[kernel.py](kernel.py) encodes classical features with a Qiskit feature map,
computes simulated statevector fidelities and fits an SVM with a precomputed
kernel. This is classical simulation of a quantum feature representation, not
a hardware speedup demonstration.

Encoding scale and circuit repetitions affect kernel concentration. Any use
must be compared with appropriately calibrated classical and physical baselines
on separated data. The exploratory results do not establish quantum advantage
or superiority over the maintained finite-sample monitoring methods.

Dependencies are included in the root [requirements.txt](../../../requirements.txt).
For the project's supported research claims, start with the
[main README](../../../README.md) and [classical benchmarks](../classical_ml/README.md).
