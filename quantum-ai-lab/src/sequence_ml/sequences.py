"""
Raw sifted-error-sequence datasets for temporally-structured eavesdropping.

The continuous and bursty intercept-resend attacks are generated at the *same*
per-sample strength, so their mean QBER (and basis symmetry) are matched: only
the time-ordering of the errors differs. This is what makes the task invisible
to any scalar/aggregate detector.
"""
import random
import numpy as np
from bb84.simulator import simulate_key_sequence


def generate_sequence_dataset(n_per_class=500, length=256, channel_noise=0.03,
                              strength_range=(0.2, 0.4), burst_len=16, seed=None):
    """
    Build a balanced dataset of raw error sequences for the
    continuous-IR (label 0) vs bursty-IR (label 1) task.

    Both classes draw strength from the same range, so their aggregate QBER
    distributions are identical by construction. Returns (X, y, strengths) with
    X of shape (2*n_per_class, length).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    X, y, strengths = [], [], []
    i = 0
    for label, attack in [(0, 'cont_ir'), (1, 'bursty_ir')]:
        for _ in range(n_per_class):
            s = random.uniform(*strength_range)
            seq = simulate_key_sequence(
                length=length, channel_noise=channel_noise, attack=attack,
                strength=s, burst_len=burst_len,
                seed=(seed + i) if seed is not None else None)
            X.append(seq)
            y.append(label)
            strengths.append(s)
            i += 1

    return np.array(X), np.array(y), np.array(strengths)


def temporal_features(seq):
    """
    Hand-crafted temporal descriptors of one error sequence: lag
    autocorrelations, error density variance across windows, and run-length
    statistics. These capture clustering that the mean error rate cannot.
    """
    s = seq.astype(float)
    feats = [s.mean()]                      # mean QBER (the scalar a threshold sees)
    s0 = s - s.mean()
    denom = np.sum(s0 * s0)
    for k in (1, 2, 4, 8, 16, 32):         # autocorrelation at several lags
        feats.append(np.sum(s0[:-k] * s0[k:]) / denom if denom > 0 else 0.0)
    # variance of error density across 16 equal windows (burstiness)
    nwin = 16
    trim = (len(s) // nwin) * nwin
    feats.append(s[:trim].reshape(nwin, -1).mean(1).var())
    # run lengths of consecutive errors
    runs, c = [], 0
    for v in seq:
        if v:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)
    feats.append(np.max(runs) if runs else 0.0)
    feats.append(np.mean(runs) if runs else 0.0)
    return feats


def temporal_feature_matrix(X):
    return np.array([temporal_features(s) for s in X])


TEMPORAL_FEATURE_NAMES = (['Mean QBER'] +
                          [f'Autocorr lag {k}' for k in (1, 2, 4, 8, 16, 32)] +
                          ['Window-density var', 'Max run length', 'Mean run length'])
