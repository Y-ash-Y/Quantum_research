"""
Dataset generation for sub-threshold eavesdropping detection.

Every channel here keeps the total QBER below the BB84 security threshold, so a
classifier built on QBER alone is forced to fail. The discriminating signal
lives in the *other* statistics — per-basis QBER asymmetry and the decoy/signal
gain ratio — which is what the rich feature set exposes.
"""
import random
import numpy as np
from bb84.simulator import run_bb84

# Rich, physically-motivated feature set.
FEATURE_NAMES = ['QBER', 'QBER Asymmetry |Z-X|', 'Signal Gain',
                 'Decoy/Signal Ratio', 'Sift Ratio']

# Attack families used for the multi-class "which attack" analysis.
ATTACK_FAMILIES = ['secure', 'biased_ir', 'pns']


def extract_features(stats):
    """Map raw channel statistics to the model feature vector."""
    return [
        stats['qber'],
        stats['qber_asymmetry'],
        stats['gain_signal'],
        stats['decoy_ratio'],
        stats['sift_ratio'],
    ]


def generate_sample(n_bits=2500, channel_noise=0.03, seed=None):
    """
    Generate ONE labelled sample.

    label 0 = secure, label 1 = under attack (basis-biased IR or PNS).
    The basis-biased attack strength is restricted so the total QBER stays
    below the 11% threshold — the whole point of the sub-threshold regime.
    Returns (features, label, family).
    """
    label = random.randint(0, 1)

    if label == 0:
        family = 'secure'
        stats = run_bb84(n_bits, channel_noise=channel_noise, attack='none', seed=seed)
    else:
        family = random.choice(['biased_ir', 'pns'])
        if family == 'biased_ir':
            strength = random.uniform(0.08, 0.22)   # keeps total QBER < 11%
            stats = run_bb84(n_bits, channel_noise=channel_noise,
                             attack='biased_ir', attack_strength=strength, seed=seed)
        else:
            stats = run_bb84(n_bits, channel_noise=channel_noise, attack='pns', seed=seed)

    if stats is None:
        return None, None, None
    return extract_features(stats), label, family


def generate_dataset(n_samples=800, n_bits=2500, channel_noise=0.03,
                     return_family=False, seed=None):
    """
    Generate a full dataset of sub-threshold secure/attack samples.

    Pass `seed` for a fully reproducible dataset: the global RNG is seeded once
    and each session gets a distinct simulator seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    X, y, fam = [], [], []
    i = 0
    while len(X) < n_samples:
        sample_seed = (seed + i) if seed is not None else None
        features, label, family = generate_sample(n_bits, channel_noise, seed=sample_seed)
        i += 1
        if features is not None:
            X.append(features)
            y.append(label)
            fam.append(family)

    X = np.array(X)
    y = np.array(y)
    if return_family:
        return X, y, np.array(fam)
    return X, y
