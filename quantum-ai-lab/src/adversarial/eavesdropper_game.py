"""
Adversarial eavesdropper vs. ML detector for BB84.

Eve chooses an attack *policy* over a palette of channels parameterised so that
they share one information axis -- the fraction of the sifted key Eve learns:

  * intercept-resend (IR)        rate `ir_rate`        -> raises QBER
  * basis-biased IR              `basis_bias`          -> raises per-basis asymmetry
  * photon-number splitting (PNS) `pns_frac`           -> collapses decoy/signal gain

We then ask how well detectors of increasing *feature coverage* withstand Eve's
*best-response* evasion. The qubit error process uses the same Qiskit-derived
outcome probabilities as the main simulator (`bb84.simulator._outcome_table`);
loss, intensities and PNS are layered as a fast vectorised Monte-Carlo.

Two failure modes emerge:
  * coverage failure  -- a detector blind to a channel's signature is evaded at
                         every information level (qualitative, unfixable by data)
  * finite-key stealth -- a feature-complete detector misses low-information
                         attacks only because of sampling noise; the gap closes
                         as the key length grows.
"""
import numpy as np
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from bb84.simulator import _outcome_table

# High-loss channel: the regime where gain-matched PNS is a genuine threat.
MU_S, MU_D, ETA, DECOY_PROB = 0.5, 0.1, 0.15, 0.5

FEATURE_NAMES = ['QBER', 'QBER Asymmetry', 'Signal Gain', 'Decoy/Signal Ratio']

# Detectors of increasing feature coverage (column indices into the feature vector).
DETECTORS = {
    'QBER-only':       [0],
    'Asymmetry-aware': [0, 1],
    'Decoy-aware':     [0, 3],
    'Combined':        [0, 1, 2, 3],
}


@lru_cache(maxsize=None)
def _lut(channel_noise):
    """Flatten the Qiskit-derived P(Bob=1 | t_bit, t_basis, b_basis) into an array
    indexed by t_bit*4 + t_basis*2 + b_basis (Z=0, X=1)."""
    p1 = _outcome_table(channel_noise)
    lut = np.zeros(8)
    for tb in (0, 1):
        for t in (0, 1):
            for b in (0, 1):
                lut[tb * 4 + t * 2 + b] = p1[(tb, 'Z' if t == 0 else 'X',
                                              'Z' if b == 0 else 'X')]
    return lut


def pns_forward_prob():
    """Eve's multi-photon forwarding probability for a gain-matched PNS attack
    (so the observed signal gain equals the legitimate 1 - e^{-mu_s*eta})."""
    return (1 - np.exp(-MU_S * ETA)) / (1 - np.exp(-MU_S) * (1 + MU_S))


def simulate_session(ir_rate=0.0, basis_bias=0.5, pns_frac=0.0,
                     n_pulses=6000, channel_noise=0.03, rng=None):
    """
    Simulate one BB84 session under Eve's policy and return
    ([QBER, asymmetry, signal_gain, decoy_ratio], info_stolen) or None if the
    sifted key is too short.

    `info_stolen` is the fraction of sifted-key bits Eve learns (PNS pulses fully;
    IR pulses when her random basis matched Alice's).
    """
    rng = rng or np.random.default_rng()
    lut = _lut(channel_noise)
    q_pns = pns_forward_prob()
    n = n_pulses

    decoy = rng.random(n) < DECOY_PROB
    mu = np.where(decoy, MU_D, MU_S)
    k = rng.poisson(mu)

    is_pns = rng.random(n) < pns_frac
    is_ir = (~is_pns) & (rng.random(n) < ir_rate)

    a_bit = rng.integers(0, 2, n)
    a_basis = rng.integers(0, 2, n)        # 0 = Z, 1 = X
    b_basis = rng.integers(0, 2, n)

    detected = np.zeros(n, bool)
    p_loss = 1 - (1 - ETA) ** k
    honest_or_ir = ~is_pns
    detected[honest_or_ir] = (k[honest_or_ir] >= 1) & \
        (rng.random(honest_or_ir.sum()) < p_loss[honest_or_ir])
    detected[is_pns] = (k[is_pns] >= 2) & (rng.random(is_pns.sum()) < q_pns)

    eve_basis = np.where(rng.random(n) < basis_bias, 0, 1)
    t_bit, t_basis = a_bit.copy(), a_basis.copy()
    wrong = is_ir & (eve_basis != a_basis)
    t_bit[wrong] = rng.integers(0, 2, wrong.sum())
    t_basis[is_ir] = eve_basis[is_ir]

    sift = detected & (a_basis == b_basis)
    if sift.sum() < 20:
        return None

    idx = t_bit * 4 + t_basis * 2 + b_basis
    bob = (rng.random(n) < lut[idx]).astype(int)
    err = (a_bit != bob).astype(int)

    known = np.zeros(n, bool)
    known[is_pns] = True
    known[is_ir] = (eve_basis[is_ir] == a_basis[is_ir])

    z = sift & (a_basis == 0)
    x = sift & (a_basis == 1)
    qber = err[sift].mean()
    qz = err[z].mean() if z.any() else 0.0
    qx = err[x].mean() if x.any() else 0.0
    signal_gain = detected[~decoy].mean()
    decoy_gain = detected[decoy].mean()
    decoy_ratio = decoy_gain / signal_gain if signal_gain > 0 else 0.0

    features = [qber, abs(qz - qx), signal_gain, decoy_ratio]
    return features, float(known[sift].mean())


def _sample(policy, reps, n_pulses, channel_noise, rng):
    feats, infos = [], []
    for _ in range(reps):
        out = simulate_session(*policy, n_pulses=n_pulses,
                               channel_noise=channel_noise, rng=rng)
        if out:
            feats.append(out[0]); infos.append(out[1])
    return np.array(feats), np.array(infos)


def build_training_set(n_secure=700, n_attack=1400, n_pulses=6000,
                       channel_noise=0.03, rng=None):
    """Secure sessions + a broad random mix of single-channel and mixed attacks."""
    rng = rng or np.random.default_rng(0)
    X, y = [], []
    for _ in range(n_secure):
        out = simulate_session(n_pulses=n_pulses, channel_noise=channel_noise, rng=rng)
        if out:
            X.append(out[0]); y.append(0)
    for _ in range(n_attack):
        r = rng.uniform(0, 0.6); be = rng.uniform(0.5, 1.0); ro = rng.uniform(0, 0.6)
        if rng.random() < 0.5: ro = 0.0          # ensure pure-IR coverage
        if rng.random() < 0.5: r = 0.0           # ensure pure-PNS coverage
        out = simulate_session(r, be, ro, n_pulses=n_pulses,
                               channel_noise=channel_noise, rng=rng)
        if out:
            X.append(out[0]); y.append(1)
    return np.array(X), np.array(y)


def train_detectors(X, y, n_pulses=6000, channel_noise=0.03, fpr=0.05,
                    n_secure_cal=500, rng=None):
    """Train each coverage-level detector and calibrate a threshold at `fpr`."""
    rng = rng or np.random.default_rng(1)
    secure, _ = _sample((0.0, 0.5, 0.0), n_secure_cal, n_pulses, channel_noise, rng)
    dets = {}
    for name, cols in DETECTORS.items():
        model = RandomForestClassifier(n_estimators=300, random_state=0).fit(X[:, cols], y)
        scores = model.predict_proba(secure[:, cols])[:, 1]
        dets[name] = {'model': model, 'cols': cols,
                      'thr': float(np.quantile(scores, 1 - fpr))}
    return dets


def detection_rate(det, feats):
    """Fraction of `feats` flagged by detector `det` at its calibrated threshold."""
    s = det['model'].predict_proba(feats[:, det['cols']])[:, 1]
    return float((s > det['thr']).mean())


def best_response(dets, target_info, n_pulses=6000, channel_noise=0.03,
                  reps=200, n_candidates=40, rng=None):
    """
    Eve's best response at a target information level: search attack policies
    (pure IR, biased IR, pure PNS, and IR/PNS mixes) that achieve ~`target_info`
    and return, per detector, the minimum detection rate she can force.
    """
    rng = rng or np.random.default_rng(2)
    candidates = [(min(1.0, 2 * target_info), 0.5, 0.0),     # pure IR
                  (min(1.0, 2 * target_info), 0.95, 0.0),    # biased IR
                  (0.0, 0.5, target_info)]                   # pure PNS
    for _ in range(n_candidates):                            # random IR/PNS mixes
        f = rng.random()
        candidates.append((2 * (1 - f) * target_info, rng.uniform(0.5, 1.0),
                           f * target_info))
    best = {name: (1.0, None, 0.0) for name in dets}         # (detection, policy, info)
    for pol in candidates:
        feats, infos = _sample(pol, reps, n_pulses, channel_noise, rng)
        if len(feats) == 0:
            continue
        achieved = float(infos.mean())
        for name, det in dets.items():
            dr = detection_rate(det, feats)
            if dr < best[name][0]:
                best[name] = (dr, pol, achieved)
    return best


def frontier_curves(dets, target_infos, rng=None, **kw):
    """Eve best-response detection vs information, per detector.
    Returns (achieved_infos, {detector_name: [detection rates]})."""
    rng = rng or np.random.default_rng(3)
    curves = {name: [] for name in dets}
    achieved = []
    for it in target_infos:
        br = best_response(dets, it, rng=rng, **kw)
        achieved.append(float(np.mean([br[n][2] for n in dets])))
        for name in dets:
            curves[name].append(br[name][0])
    return achieved, curves


def stealth_floor_curve(key_lengths, pns_frac=0.4, channel_noise=0.03,
                        n_secure=300, n_attack=600, reps=200, rng=None):
    """Combined-detector detection of a fixed pure-PNS attack vs key length.
    Shows the stealth floor is finite-sample: detection -> 1 as the key grows."""
    rng = rng or np.random.default_rng(4)
    rows = []
    for n in key_lengths:
        X, y = build_training_set(n_secure, n_attack, n_pulses=n,
                                  channel_noise=channel_noise, rng=rng)
        dets = train_detectors(X, y, n_pulses=n, channel_noise=channel_noise,
                               n_secure_cal=n_secure, rng=rng)
        feats, infos = _sample((0.0, 0.5, pns_frac), reps, n, channel_noise, rng)
        rows.append((n, float(infos.mean()),
                     detection_rate(dets['Combined'], feats)))
    return rows
