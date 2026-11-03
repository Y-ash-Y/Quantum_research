"""Simplified BB84 photon/decoy simulator with an explicit transmission channel.

Model version 2: ideal preparation/measurement gates by default, one depolarizing
channel per transmitted qubit, optional separate gate errors. Photon number,
loss and gain-matched PNS are Monte-Carlo abstractions, not hardware simulation
or a finite-key security proof. Historical ML results used the previous model
and must be regenerated before being attributed to this implementation.
"""
import math
import random
from functools import lru_cache
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def _probability(value, name):
    """Validate the supported physical mixing-probability range [0, 1]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real probability in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite probability in [0, 1]')
    return value


def _count(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _intensity(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a nonnegative finite intensity')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a nonnegative finite intensity')
    return value


def _seed(value):
    return None if value is None else _count(value, 'seed', minimum=0)


def build_noise_model(*, gate_noise=0.0):
    """Optional local H/X gate errors only; transmission is explicit in circuits.

    Gate errors may produce device-dependent asymmetry. They are zero in the
    reference model and must not be confused with the transmission parameter.
    """
    gate_noise = _probability(gate_noise, 'gate_noise')
    noise_model = NoiseModel()
    if gate_noise:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(gate_noise, 1), ['h', 'x'])
    return noise_model


@lru_cache(maxsize=64, typed=True)
def _transmission_instruction(channel_noise):
    return depolarizing_error(_probability(channel_noise, 'channel_noise'), 1).to_instruction()


def build_qubit_circuit(bit, basis, meas_basis, *, channel_noise=0.0):
    """Prepare, pass through one explicit channel, and measure a BB84 state.

    Barriers keep the channel boundary explicit during circuit optimization.
    Noise applies even to Z-basis bit zero, whose ideal preparation has no gates.
    A non-unitary QuantumError instruction avoids relying on an identity gate
    that an optimizer could delete. channel_noise is lambda, not QBER:
    E(rho)=(1-lambda)rho+lambda*I/2 implies matched-basis QBER=lambda/2.
    """
    bit = _count(bit, 'bit', minimum=0)
    if bit not in (0, 1) or basis not in ('Z', 'X') or meas_basis not in ('Z', 'X'):
        raise ValueError('bit must be 0/1 and bases must be Z/X')
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    if basis == 'X':
        qc.h(0)  # X then H prepares |-> for bit 1, not H then X.
    qc.barrier(0)
    qc.append(_transmission_instruction(channel_noise), [0])
    qc.barrier(0)
    if meas_basis == 'X':
        qc.h(0)
    qc.measure(0, 0)
    return qc


def _multiphoton_probability(mu):
    """Poisson P(k>=2), avoiding cancellation for weak intensities."""
    mu = _intensity(mu, 'mu')
    if mu < 1e-3:
        term = mu * mu / 2
        total = term
        for k in range(3, 20):
            term *= mu / k
            total += term
        return math.exp(-mu) * total
    return -math.expm1(-mu) - mu * math.exp(-mu)


def _pns_forward_prob(mu_signal, eta):
    """Forwarding probability for feasible, signal-gain-matched PNS.

    Eve blocks k<=1 and forwards k>=2 over an ideal lossless line with an
    intensity-independent probability q. q*P(k>=2|mu_s)=1-exp(-mu_s*eta).
    If q>1 the requested match is physically impossible in this model.
    """
    mu_signal = _intensity(mu_signal, 'mu_signal')
    eta = _probability(eta, 'eta')
    p_multi = _multiphoton_probability(mu_signal)
    if p_multi <= 0:
        raise ValueError('gain-matched PNS requires a positive multiphoton probability')
    q_target = -math.expm1(-mu_signal * eta)
    q = q_target / p_multi
    if q > 1 + 1e-12:
        raise ValueError('gain-matched PNS is infeasible: forwarding probability exceeds 1')
    return min(1.0, q)  # Floating-point boundary tolerance, not policy clipping.


def _pulse_detected(mu, eta, pns, pns_q=1.0, *, rng=None):
    """Draw photon count and detection using a caller-owned NumPy generator.

    Session entry points validate inputs; the inner loop avoids repeating that
    work. Normal detection is 1-(1-eta)^k; PNS requires k>=2 and forwarding q.
    """
    rng = np.random.default_rng() if rng is None else rng
    k = int(rng.poisson(mu))
    if pns:
        return (k >= 2 and rng.random() < pns_q), k
    if k == 0:
        return False, k
    return rng.random() < 1 - (1 - eta) ** k, k


def run_bb84(n_bits=500, channel_noise=0.03, attack='none', attack_strength=1.0,
             mu_signal=0.5, mu_decoy=0.1, eta=0.15, decoy_prob=0.5,
             eve_basis='Z', seed=None, *, gate_noise=0.0):
    """Simulate n_bits EMITTED PULSES (legacy name), not secret key bits.

    none: honest channel; biased_ir: fixed-Eve-basis interception with probability
    attack_strength; pns: gain-matched idealized photon-number splitting.
    Expected sub-threshold QBER does not guarantee every finite sample stays
    below a threshold. A QBER value by itself is not a security proof.

    seed controls all pulse/bit/basis/interception/Aer sampling through local RNGs.
    QBER and key_length include signal AND decoy sifted observations, not a
    distilled secret key. Raw counts disambiguate missing denominators; legacy
    scalar statistics retain zero for a missing denominator.
    Returns None if no matched-basis detections occur.
    """
    n_bits = _count(n_bits, 'n_bits')
    channel_noise = _probability(channel_noise, 'channel_noise')
    gate_noise = _probability(gate_noise, 'gate_noise')
    attack_strength = _probability(attack_strength, 'attack_strength')
    eta = _probability(eta, 'eta')
    decoy_prob = _probability(decoy_prob, 'decoy_prob')
    mu_signal = _intensity(mu_signal, 'mu_signal')
    mu_decoy = _intensity(mu_decoy, 'mu_decoy')
    seed = _seed(seed)
    if attack not in ('none', 'biased_ir', 'pns'):
        raise ValueError(f'unknown attack: {attack}')
    if eve_basis not in ('Z', 'X'):
        raise ValueError('eve_basis must be Z or X')
    rng = np.random.default_rng(seed)
    sim_seed = int(rng.integers(0, 2**31 - 1))
    pns = attack == 'pns'
    pns_q = _pns_forward_prob(mu_signal, eta) if pns else 1.0
    signal_sent = signal_det = decoy_sent = decoy_det = 0
    recs = []
    for _ in range(n_bits):
        is_decoy = rng.random() < decoy_prob
        mu = mu_decoy if is_decoy else mu_signal
        if is_decoy:
            decoy_sent += 1
        else:
            signal_sent += 1
        detected, _k = _pulse_detected(mu, eta, pns, pns_q, rng=rng)
        if not detected:
            continue
        if is_decoy:
            decoy_det += 1
        else:
            signal_det += 1
        a_bit = int(rng.integers(0, 2))
        a_basis = 'Z' if rng.integers(0, 2) == 0 else 'X'
        b_basis = 'Z' if rng.integers(0, 2) == 0 else 'X'
        t_bit, t_basis = a_bit, a_basis
        if attack == 'biased_ir' and rng.random() < attack_strength:
            if eve_basis != a_basis:
                t_bit = int(rng.integers(0, 2))
            t_basis = eve_basis
        recs.append((a_bit, a_basis, b_basis, t_bit, t_basis))
    if not recs:
        return None

    sim = AerSimulator(method='density_matrix',
                       noise_model=build_noise_model(gate_noise=gate_noise),
                       seed_simulator=sim_seed)
    circuits = [build_qubit_circuit(t_bit, t_basis, b_basis, channel_noise=channel_noise)
                for (_, _, b_basis, t_bit, t_basis) in recs]
    result = sim.run(circuits, shots=1).result()
    if not result.success:
        raise RuntimeError(f'Aer session failed: {result.status}')
    bob_bits = [int(next(iter(result.get_counts(i)))) for i in range(len(recs))]
    z_total = z_err = x_total = x_err = 0
    for (a_bit, a_basis, b_basis, _t_bit, _t_basis), bob in zip(recs, bob_bits):
        if a_basis != b_basis:
            continue
        err = int(a_bit != bob)
        if a_basis == 'Z':
            z_total += 1
            z_err += err
        else:
            x_total += 1
            x_err += err
    sifted = z_total + x_total
    if not sifted:
        return None
    qber_z = z_err / z_total if z_total else 0.0
    qber_x = x_err / x_total if x_total else 0.0
    gain_signal = signal_det / signal_sent if signal_sent else 0.0
    gain_decoy = decoy_det / decoy_sent if decoy_sent else 0.0
    return {
        'qber': (z_err + x_err) / sifted,
        'qber_z': qber_z, 'qber_x': qber_x,
        'qber_asymmetry': abs(qber_z - qber_x),
        'sift_ratio': sifted / len(recs),
        'gain_signal': gain_signal, 'gain_decoy': gain_decoy,
        'decoy_ratio': gain_decoy / gain_signal if gain_signal else 0.0,
        'key_length': sifted, 'n_detected': len(recs),
        'signal_sent': signal_sent, 'signal_detected': signal_det,
        'decoy_sent': decoy_sent, 'decoy_detected': decoy_det,
        'z_sifted': z_total, 'z_errors': z_err,
        'x_sifted': x_total, 'x_errors': x_err,
        'model_version': 2,
    }


@lru_cache(maxsize=64, typed=True)
def _outcome_table(channel_noise, shots=None, *, gate_noise=0.0):
    """Exact Aer density-matrix P(Bob=1) for all eight configurations.

    save_probabilities returns probabilities BEFORE measurement, without shot
    estimates. shots is an ignored legacy compatibility argument (validated if
    supplied). An immutable bounded cache avoids accidental caller mutation.
    Both channel and device noise enter its cache key.
    """
    channel_noise = _probability(channel_noise, 'channel_noise')
    gate_noise = _probability(gate_noise, 'gate_noise')
    if shots is not None:
        _count(shots, 'shots')
    keys = [(bit, basis, meas) for bit in (0, 1)
            for basis in ('Z', 'X') for meas in ('Z', 'X')]
    circuits = []
    for bit, basis, meas in keys:
        qc = build_qubit_circuit(bit, basis, meas, channel_noise=channel_noise)
        qc = qc.remove_final_measurements(inplace=False)
        qc.save_probabilities()
        circuits.append(qc)
    sim = AerSimulator(method='density_matrix',
                       noise_model=build_noise_model(gate_noise=gate_noise))
    result = sim.run(circuits, shots=1, seed_simulator=0).result()
    if not result.success:
        raise RuntimeError(f'Aer probability calculation failed: {result.status}')
    return MappingProxyType({key: float(np.clip(result.data(i)['probabilities'][1], 0, 1))
                             for i, key in enumerate(keys)})


def simulate_key_sequence(length=256, channel_noise=0.03, attack='none',
                          strength=0.3, burst_len=16, seed=None, *, gate_noise=0.0):
    """Emit a fixed number of sifted-error observations over a lossless channel.

    cont_ir intercepts independently; bursty_ir uses a stationary two-state
    Markov process whose ON runs average burst_len EMITTED PULSES, before sifting.
    The two share an expected interception duty cycle, not identical sample-mean
    QBER distributions. Noise probabilities come from the exact circuit table.
    End-point duty cycles 0 and 1 use absorbing OFF/ON states.

    This is simulator-observable telemetry, not a claim that all error bits of a
    retained secret key are safely available to an operational detector.
    """
    length = _count(length, 'length')
    channel_noise = _probability(channel_noise, 'channel_noise')
    gate_noise = _probability(gate_noise, 'gate_noise')
    strength = _probability(strength, 'strength')
    seed = _seed(seed)
    if attack not in ('none', 'cont_ir', 'bursty_ir'):
        raise ValueError(f'unknown sequence attack: {attack}')
    if (isinstance(burst_len, (bool, np.bool_)) or not isinstance(burst_len, Real)
            or not math.isfinite(burst_len) or burst_len < 1):
        raise ValueError('burst_len must be finite and >= 1')
    p_on_to_off = 1.0 / burst_len
    p_off_to_on = strength / (1 - strength) / burst_len if strength < 1 else 0.0
    if attack == 'bursty_ir' and 0 < strength < 1 and p_off_to_on > 1:
        raise ValueError('strength and burst_len imply an invalid Markov transition probability')
    rng = random.Random(seed)
    p1 = _outcome_table(channel_noise, gate_noise=gate_noise)
    eve_on = attack == 'bursty_ir' and rng.random() < strength
    errors = []
    while len(errors) < length:
        a_bit = rng.randint(0, 1)
        a_basis = rng.choice(['Z', 'X'])
        b_basis = rng.choice(['Z', 'X'])
        t_bit, t_basis = a_bit, a_basis
        if attack == 'cont_ir':
            intercept = rng.random() < strength
        elif attack == 'bursty_ir':
            if 0 < strength < 1:
                eve_on = (rng.random() > p_on_to_off) if eve_on else (rng.random() < p_off_to_on)
            intercept = eve_on
        else:
            intercept = False
        if intercept:
            eve_basis = rng.choice(['Z', 'X'])
            if eve_basis != a_basis:
                t_bit = rng.randint(0, 1)
            t_basis = eve_basis
        if a_basis == b_basis:
            bob = int(rng.random() < p1[(t_bit, t_basis, b_basis)])
            errors.append(int(a_bit != bob))
    return np.array(errors, dtype=np.int8)
