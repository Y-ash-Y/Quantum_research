# Exploratory temporal CNN

[sequences.py](sequences.py) prepares temporal error sequences;
[cnn.py](cnn.py) defines a small one-dimensional convolutional classifier and
training helper. Torch is included in the consolidated requirements.

This module explores temporal patterns. It is not the validated comparator
behind the main public-count/drift results, and no advantage over corrected
physical or statistical baselines is established here.

Any experiment must declare its sequence construction, split, calibration and
sample exposure. More training is not a substitute for testing honest-noise
and device controls. For the maintained benchmark path, use
[classical monitoring](../classical_ml/README.md) and the
[experiment runners](../../scripts/README.md).
