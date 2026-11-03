"""Restricted BB84 attack-policy search with independent final evaluation.

Information means Eve's raw Alice-bit knowledge fraction among *signal and decoy*
sifted observations. It is not mutual information, secret-key leakage, or a
security proof. Policies satisfy a long-run expectation constraint; finite
evaluation samples fluctuate. Searches report the best policy found, not a
global optimum. All alarm rates are conditional on usable telemetry sessions.
"""
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from bb84.telemetry import TelemetryCounts
from bb84.simulator import (
    _count, _probability, _outcome_table, _pns_forward_prob,
    _multiphoton_probability,
)

MU_S, MU_D, ETA, DECOY_PROB = 0.5, 0.1, 0.15, 0.5
FEATURE_NAMES = ['QBER', 'QBER Asymmetry', 'Signal Gain', 'Decoy/Signal Ratio']
DETECTORS = {
    'QBER-only': [0], 'Asymmetry-aware': [0, 1],
    'Decoy-aware': [0, 3], 'Combined': [0, 1, 2, 3],
}


@dataclass(frozen=True)
class AttackPolicy:
    ir_rate: float = 0.0
    basis_bias: float = 0.5
    pns_frac: float = 0.0

    def __post_init__(self):
        for name in ('ir_rate', 'basis_bias', 'pns_frac'):
            object.__setattr__(self, name, _probability(getattr(self, name), name))

    def as_tuple(self):
        return self.ir_rate, self.basis_bias, self.pns_frac


@dataclass
class Session:
    features: list
    known_bits: int
    sifted_bits: int
    counts: TelemetryCounts | None = None

    @property
    def info(self):
        return self.known_bits / self.sifted_bits


@dataclass
class SampleBatch:
    features: np.ndarray
    infos: np.ndarray
    known_bits: int
    sifted_bits: int
    attempted: int
    rejected: int

    @property
    def pooled_info(self):
        if not self.sifted_bits:
            raise ValueError('No usable sifted observations')
        return self.known_bits / self.sifted_bits


@dataclass(frozen=True)
class RateEstimate:
    flagged: int
    accepted: int
    attempted: int
    rejected: int
    rate: float
    interval: tuple


@dataclass
class TrainingData:
    features: np.ndarray
    labels: np.ndarray
    policies: np.ndarray
    attempted: int
    rejected: int


@dataclass
class CandidateResult:
    policy: AttackPolicy
    expected_info: float
    selection_info: float | None
    selection_rates: dict
    attempted: int
    rejected: int


@dataclass
class BestResponse:
    policy: AttackPolicy
    expected_info: float
    selection: RateEstimate
    evaluation: RateEstimate
    achieved_info: float
    known_bits: int
    sifted_bits: int


@dataclass
class SearchResult:
    target_info: float
    responses: dict
    candidates: list
    streams: dict


@dataclass
class FrontierResult:
    target_infos: list
    searches: list

    @property
    def detection_rates(self):
        return {n: [s.responses[n].evaluation.rate for s in self.searches]
                for n in self.searches[0].responses}

    @property
    def achieved_infos(self):
        # Each detector selected its own policy; never average their x-values.
        return {n: [s.responses[n].achieved_info for s in self.searches]
                for n in self.searches[0].responses}

    @property
    def detection_intervals(self):
        return {n: [s.responses[n].evaluation.interval for s in self.searches]
                for n in self.searches[0].responses}


@dataclass
class PulseSweepResult:
    emitted_pulses: int
    expected_info: float
    achieved_info: float
    evaluation: RateEstimate
    honest_test: RateEstimate


def _rng(rng, seed=None):
    if rng is None:
        return np.random.default_rng(seed)
    if not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator')
    return rng


def _streams(rng, count):
    entropy = [int(x) for x in rng.integers(0, 2**32, size=4, dtype=np.uint32)]
    children = np.random.SeedSequence(entropy).spawn(count)
    return children, {'entropy': entropy,
                      'spawn_keys': [list(s.spawn_key) for s in children]}


def _settings(n_pulses, channel_noise):
    return (_count(n_pulses, 'n_pulses'),
            _probability(channel_noise, 'channel_noise'))


@lru_cache(maxsize=64, typed=True)
def _lut(channel_noise):
    p1 = _outcome_table(_probability(channel_noise, 'channel_noise'))
    lut = np.array([p1[(bit, 'ZX'[prep], 'ZX'[meas])]
                    for bit in (0, 1) for prep in (0, 1) for meas in (0, 1)])
    lut.setflags(write=False)
    return lut


def pns_forward_prob():
    """Use the shared simulator's physically feasible gain-matching rule."""
    return _pns_forward_prob(MU_S, ETA)


def _gains():
    qs, qd = -np.expm1(-MU_S * ETA), -np.expm1(-MU_D * ETA)
    honest = (1 - DECOY_PROB) * qs + DECOY_PROB * qd
    pns = ((1 - DECOY_PROB) * qs
           + DECOY_PROB * pns_forward_prob() * _multiphoton_probability(MU_D))
    return float(honest), float(pns)


def expected_information(policy):
    """Long-run raw knowledge fraction, including decoy observations.

    Basis sifting contributes 1/2 to numerator and denominator and cancels.
    Uniform Alice bases make IR knowledge probability r/2, regardless of beta.
    """
    policy = policy if isinstance(policy, AttackPolicy) else AttackPolicy(*policy)
    a, b = _gains()
    r, _, rho = policy.as_tuple()
    return float((rho * b + (1 - rho) * a * r / 2)
                 / (rho * b + (1 - rho) * a))


def policy_for_target(target_info, ir_rate=0.0, basis_bias=0.5):
    """Solve rho = A(I-r/2)/[(1-I)B + A(I-r/2)]."""
    target = _probability(target_info, 'target_info')
    r = _probability(ir_rate, 'ir_rate')
    beta = _probability(basis_bias, 'basis_bias')
    if r / 2 > target + 1e-12:
        raise ValueError('IR alone already exceeds the requested information')
    if target == 1:
        return AttackPolicy(0, beta, 1)
    a, b = _gains()
    delta = max(0.0, target - r / 2)
    rho = a * delta / ((1 - target) * b + a * delta)
    return AttackPolicy(r, beta, rho)


def candidate_policies(target_info, n_candidates, rng):
    target = _probability(target_info, 'target_info')
    count = _count(n_candidates, 'n_candidates', minimum=0)
    rng = _rng(rng)
    policies = [policy_for_target(target)]
    if target <= 0.5:
        policies += [policy_for_target(target, 2 * target, beta)
                     for beta in (0.5, 0.95)]
    if 0 < target < 1:
        policies += [policy_for_target(target, rng.uniform(0, min(1, 2 * target)),
                                      rng.uniform(0.5, 1)) for _ in range(count)]
    # Bias has no effect for a null policy, so target zero has just one candidate.
    if target == 0:
        return [AttackPolicy()]
    return list(dict.fromkeys(policies))


def simulate_session(ir_rate=0.0, basis_bias=0.5, pns_frac=0.0,
                     n_pulses=6000, channel_noise=0.03, rng=None, *,
                     return_details=False):
    """Vectorized Monte Carlo, using exact Aer-derived single-qubit outcomes.

    PNS takes precedence; ir_rate is conditional on a pulse not being assigned
    PNS. Reject insufficient/undefined telemetry rather than invent zero QBER.
    Default return is (features, raw knowledge fraction); details add counts.
    """
    policy = AttackPolicy(ir_rate, basis_bias, pns_frac)
    n, noise = _settings(n_pulses, channel_noise)
    rng = _rng(rng)
    lut, q_pns = _lut(noise), pns_forward_prob()
    decoy = rng.random(n) < DECOY_PROB
    k = rng.poisson(np.where(decoy, MU_D, MU_S))
    is_pns = rng.random(n) < policy.pns_frac
    is_ir = (~is_pns) & (rng.random(n) < policy.ir_rate)
    a_bit, a_basis, b_basis = [rng.integers(0, 2, n) for _ in range(3)]
    detected = np.zeros(n, bool)
    normal = ~is_pns
    detected[normal] = ((k[normal] >= 1)
                        & (rng.random(normal.sum()) < 1 - (1 - ETA)**k[normal]))
    detected[is_pns] = ((k[is_pns] >= 2)
                        & (rng.random(is_pns.sum()) < q_pns))
    eve_basis = np.where(rng.random(n) < policy.basis_bias, 0, 1)
    t_bit, t_basis = a_bit.copy(), a_basis.copy()
    wrong = is_ir & (eve_basis != a_basis)
    t_bit[wrong] = rng.integers(0, 2, wrong.sum())
    t_basis[is_ir] = eve_basis[is_ir]
    sift = detected & (a_basis == b_basis)
    z, x = sift & (a_basis == 0), sift & (a_basis == 1)
    if (sift.sum() < 20 or not z.any() or not x.any() or not decoy.any()
            or decoy.all() or not detected[~decoy].any()):
        return None
    bob = (rng.random(n) < lut[t_bit * 4 + t_basis * 2 + b_basis]).astype(int)
    errors = a_bit != bob
    known = is_pns | (is_ir & (eve_basis == a_basis))
    signal_gain = float(detected[~decoy].mean())
    features = [float(errors[sift].mean()),
                float(abs(errors[z].mean() - errors[x].mean())),
                signal_gain, float(detected[decoy].mean() / signal_gain)]
    counts = TelemetryCounts(
        int(z.sum()), int(x.sum()), int(errors[z].sum()), int(errors[x].sum()),
        int((~decoy).sum()), int(decoy.sum()),
        int(detected[~decoy].sum()), int(detected[decoy].sum()))
    result = Session(features, int(known[sift].sum()), int(sift.sum()), counts)
    return result if return_details else (features, result.info)


def _sample(policy, reps, n_pulses, channel_noise, rng):
    policy = policy if isinstance(policy, AttackPolicy) else AttackPolicy(*policy)
    reps = _count(reps, 'reps')
    n_pulses, channel_noise = _settings(n_pulses, channel_noise)
    rng = _rng(rng)
    feats, infos, known, sifted = [], [], 0, 0
    for _ in range(reps):
        out = simulate_session(*policy.as_tuple(), n_pulses=n_pulses,
                               channel_noise=channel_noise, rng=rng,
                               return_details=True)
        if out is not None:
            feats.append(out.features)
            infos.append(out.info)
            known += out.known_bits
            sifted += out.sifted_bits
    return SampleBatch(np.asarray(feats, dtype=float).reshape(-1, 4),
                       np.asarray(infos, dtype=float), known, sifted,
                       reps, reps - len(feats))


def build_training_set(n_secure=700, n_attack=1400, n_pulses=6000,
                       channel_noise=0.03, rng=None, *, max_attempts=None,
                       return_details=False):
    """Exactly requested usable rows, with nonzero IR/PNS/mixed attack labels.

    Retries are bounded. Attack presence is a policy label, not a guarantee that
    a finite session contains an observed intercepted photon.
    """
    counts = [_count(n_secure, 'n_secure', 0), _count(n_attack, 'n_attack', 0)]
    n_pulses, channel_noise = _settings(n_pulses, channel_noise)
    total = sum(counts)
    if not total:
        raise ValueError('At least one training session is required')
    limit = _count(10 * total if max_attempts is None else max_attempts, 'max_attempts')
    rng = _rng(rng, 0)
    feats, labels, policies, attempted = [], [], [], 0
    for label, count in enumerate(counts):
        accepted = 0
        while accepted < count and attempted < limit:
            policy = AttackPolicy()
            if label:
                family = int(rng.integers(0, 3))
                r = rng.uniform(0.02, 0.6) if family != 1 else 0.0
                rho = rng.uniform(0.02, 0.6) if family != 0 else 0.0
                policy = AttackPolicy(r, rng.uniform(0.5, 1), rho)
            attempted += 1
            out = simulate_session(*policy.as_tuple(), n_pulses=n_pulses,
                                   channel_noise=channel_noise, rng=rng)
            if out is not None:
                feats.append(out[0]); labels.append(label)
                policies.append(policy.as_tuple()); accepted += 1
        if accepted < count:
            raise ValueError('Unable to collect requested usable training sessions '
                             'within max_attempts; increase pulse count or retry budget')
    result = TrainingData(np.asarray(feats, dtype=float), np.asarray(labels, dtype=int),
                          np.asarray(policies), attempted, attempted - total)
    return result if return_details else (result.features, result.labels)


def wilson_interval(successes, trials, confidence=0.95):
    trials = _count(trials, 'trials')
    successes = _count(successes, 'successes', 0)
    if successes > trials:
        raise ValueError('successes cannot exceed trials')
    confidence = _probability(confidence, 'confidence')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between 0 and 1')
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, den = successes / trials, 1 + z*z / trials
    center = (p + z*z / (2*trials)) / den
    radius = z * np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    # Floating-point arithmetic can otherwise put an all-success upper endpoint
    # a few ulps below 1 (or an all-failure lower endpoint above 0).
    low = 0.0 if successes == 0 else float(max(0, center-radius))
    high = 1.0 if successes == trials else float(min(1, center+radius))
    return low, high


def _features(feats):
    feats = np.asarray(feats, dtype=float)
    if feats.ndim != 2 or feats.shape[1] != 4 or not len(feats) or not np.isfinite(feats).all():
        raise ValueError('features must be a nonempty finite matrix with four columns')
    return feats


def _scores(det, feats):
    feats = _features(feats)
    classes = np.asarray(det['model'].classes_)
    index = np.flatnonzero(classes == 1)
    if len(index) != 1:
        raise ValueError('Detector must contain the positive class 1')
    cols = det['cols']
    if not cols or any(c not in range(4) for c in cols):
        raise ValueError('Detector feature columns must be nonempty indices 0..3')
    scores = np.asarray(det['model'].predict_proba(feats[:, cols]), dtype=float)
    if scores.shape != (len(feats), len(classes)) or not np.isfinite(scores).all():
        raise ValueError('Invalid detector probability output')
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError('Detector probabilities must lie in [0, 1]')
    return scores[:, index[0]]


def detection_rate(det, feats):
    threshold = _probability(det['thr'], 'threshold')
    return float((_scores(det, feats) > threshold).mean())


def _estimate(det, batch):
    threshold = _probability(det['thr'], 'threshold')
    flagged = int((_scores(det, batch.features) > threshold).sum())
    accepted = len(batch.features)
    return RateEstimate(flagged, accepted, batch.attempted, batch.rejected,
                        flagged / accepted, wilson_interval(flagged, accepted))


def train_detectors(X, y, n_pulses=6000, channel_noise=0.03, fpr=0.05,
                    n_secure_cal=500, rng=None, *, n_secure_test=500,
                    n_estimators=300):
    """Fit training rows, calibrate on honest rows, audit on fresh honest rows.

    Quantile calibration with score ties need not attain the nominal fpr.
    Honest-test rates/intervals are measured, never used to tune the threshold.
    """
    X = _features(X)
    y = np.asarray(y)
    if y.shape != (len(X),) or set(np.unique(y)) != {0, 1}:
        raise ValueError('y must match X and include exactly classes 0 and 1')
    fpr = _probability(fpr, 'fpr')
    if not 0 < fpr < 1:
        raise ValueError('fpr must lie strictly between zero and one')
    n_pulses, channel_noise = _settings(n_pulses, channel_noise)
    n_estimators = _count(n_estimators, 'n_estimators')
    children, metadata = _streams(_rng(rng, 1), 2)
    cal = _sample(AttackPolicy(), n_secure_cal, n_pulses, channel_noise,
                  np.random.default_rng(children[0]))
    audit = _sample(AttackPolicy(), n_secure_test, n_pulses, channel_noise,
                    np.random.default_rng(children[1]))
    _features(cal.features); _features(audit.features)
    dets = {}
    for name, cols in DETECTORS.items():
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=0).fit(X[:, cols], y)
        det = {'model': model, 'cols': cols, 'thr': 0.0,
               'nominal_fpr': fpr, 'streams': metadata}
        det['thr'] = float(np.quantile(_scores(det, cal.features), 1-fpr))
        det['calibration'] = _estimate(det, cal)
        det['honest_test'] = _estimate(det, audit)
        dets[name] = det
    return dets


def best_response(dets, target_info, n_pulses=6000, channel_noise=0.03,
                  reps=200, n_candidates=40, rng=None, *, eval_reps=None):
    """Search expected-information-feasible policies, freeze, independently test.

    The final evaluation is never minimized. A result contains selection and
    test rates separately, counts, Wilson intervals, and reproducible stream
    metadata. A failed/empty search raises rather than returning an empty policy.
    """
    if not dets:
        raise ValueError('At least one detector is required')
    target_info = _probability(target_info, 'target_info')
    n_pulses, channel_noise = _settings(n_pulses, channel_noise)
    reps = _count(reps, 'reps')
    eval_reps = _count(reps if eval_reps is None else eval_reps, 'eval_reps')
    children, metadata = _streams(_rng(rng, 2), 3)
    candidate_rng, select_rng = [np.random.default_rng(s) for s in children[:2]]
    policies = candidate_policies(target_info, n_candidates, candidate_rng)
    # Policy-specific final streams keep evaluation independent of detector order.
    test_streams = children[2].spawn(len(policies))
    metadata['evaluation_policy_spawn_keys'] = [list(s.spawn_key) for s in test_streams]
    candidates, chosen = [], {}
    for index, policy in enumerate(policies):
        expected = expected_information(policy)
        if abs(expected - target_info) > 1e-10:
            continue
        batch = _sample(policy, reps, n_pulses, channel_noise, select_rng)
        if not len(batch.features):
            candidates.append(CandidateResult(policy, expected, None, {},
                                              batch.attempted, batch.rejected))
            continue
        rates = {name: _estimate(det, batch) for name, det in dets.items()}
        candidates.append(CandidateResult(policy, expected, batch.pooled_info,
                                          rates, batch.attempted, batch.rejected))
        for name, rate in rates.items():
            if name not in chosen or rate.rate < chosen[name][1].rate:
                chosen[name] = (index, rate, policy, expected)
    if len(chosen) != len(dets):
        raise ValueError('No feasible policy produced usable selection sessions')
    test_batches, responses = {}, {}
    for name, (index, selection, policy, expected) in chosen.items():
        if index not in test_batches:
            test_batches[index] = _sample(policy, eval_reps, n_pulses, channel_noise,
                                           np.random.default_rng(test_streams[index]))
        batch = test_batches[index]
        evaluation = _estimate(dets[name], batch)  # fails explicitly if all rejected
        responses[name] = BestResponse(policy, expected, selection, evaluation,
                                       batch.pooled_info, batch.known_bits, batch.sifted_bits)
    return SearchResult(target_info, responses, candidates, metadata)


def frontier_curves(dets, target_infos, rng=None, **kw):
    """Return detector-specific measured information coordinates and test rates."""
    targets = [_probability(t, 'target_info') for t in target_infos]
    if not targets:
        raise ValueError('At least one information target is required')
    rng = _rng(rng, 3)
    return FrontierResult(targets, [best_response(dets, t, rng=rng, **kw) for t in targets])


def stealth_floor_curve(key_lengths, pns_frac=0.4, channel_noise=0.03,
                        n_secure=300, n_attack=600, reps=200, rng=None, *,
                        return_details=False):
    """Legacy tuple API: empirical emitted-pulse sweep, retraining at each size.

    This is NOT a finite-key bound, a theorem of convergence to detection 1, or
    an equal-operating-FPR comparison. Each model's nominal calibration is 5%.
    """
    policy, rng = AttackPolicy(0, 0.5, pns_frac), _rng(rng, 4)
    rows = []
    for n in key_lengths:
        X, y = build_training_set(n_secure, n_attack, n_pulses=n,
                                  channel_noise=channel_noise, rng=rng)
        dets = train_detectors(X, y, n_pulses=n, channel_noise=channel_noise,
                               n_secure_cal=n_secure, n_secure_test=n_secure, rng=rng)
        batch = _sample(policy, reps, n, channel_noise, rng)
        estimate = _estimate(dets['Combined'], batch)
        row = PulseSweepResult(int(n), expected_information(policy), batch.pooled_info,
                               estimate, dets['Combined']['honest_test'])
        rows.append(row if return_details else
                    (row.emitted_pulses, row.achieved_info, estimate.rate))
    return rows
