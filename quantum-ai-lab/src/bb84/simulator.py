"""
BB84 simulator with a photon-number / decoy-state channel model.

The point of this module is to support *sub-threshold* eavesdropping: attacks
that keep the total QBER below the BB84 security threshold (~11%) yet leave a
statistical fingerprint that the standard QBER test cannot see. Two such
attacks are modelled:

  * basis-biased intercept-resend  -> per-basis QBER asymmetry (qber_z != qber_x)
  * photon-number-splitting (PNS)  -> decoy-state gain collapse at ~zero added QBER

The single-qubit transmission/measurement still uses a real Qiskit-Aer circuit
with a depolarizing channel; the photon statistics (loss, decoy intensities,
PNS blocking) are layered on top as a Monte-Carlo model.
"""
import random
from functools import lru_cache
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_noise_model(error_rate=0.02):
    """Realistic quantum channel noise using depolarizing error."""
    noise_model = NoiseModel()
    error = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error, ['h', 'x', 'z'])
    return noise_model


def build_qubit_circuit(bit, basis, meas_basis):
    """Build the single-qubit prepare/transmit/measure circuit for one qubit."""
    qc = QuantumCircuit(1, 1)

    # Alice prepares: set the bit value first, then rotate into the basis.
    # Order matters — applying H before X prepares the wrong X-basis state
    # (bit=1 in the X basis must be |->, i.e. X then H, not H then X).
    if bit == 1:
        qc.x(0)
    if basis == 'X':
        qc.h(0)

    # Bob measures
    if meas_basis == 'X':
        qc.h(0)
    qc.measure(0, 0)
    return qc


def _pns_forward_prob(mu_signal, eta):
    """
    Forwarding probability Eve uses in a gain-matched PNS attack.

    Eve blocks single-photon pulses and forwards multi-photon ones through a
    lossless line, but only with probability q so that Bob's *signal* gain still
    equals the legitimate value 1 - e^{-mu_signal*eta}. This masks the loss she
    introduces, so the signal gain (and QBER) look normal. Because she cannot
    tell signal from decoy pulses, the same q is applied to decoy pulses, whose
    multi-photon fraction is far smaller -- so the decoy gain collapses. That
    residual is the only detectable signature, exactly as in decoy-state theory.
    Requires the high-loss regime (q <= 1), which is where PNS is a real threat.
    """
    p_multi = 1 - np.exp(-mu_signal) * (1 + mu_signal)   # P(k >= 2 | mu_signal)
    q_target = 1 - np.exp(-mu_signal * eta)              # legitimate signal gain
    return min(1.0, q_target / p_multi)


def _pulse_detected(mu, eta, pns, pns_q=1.0):
    """
    Decide whether Bob registers a detection for one weak-coherent pulse.

    Photon number k ~ Poisson(mu).
      * normal channel: detection prob = 1 - (1 - eta)^k  (loss eta per photon)
      * PNS: Eve blocks k <= 1 and forwards k >= 2 with probability `pns_q`
        (gain matching, see _pns_forward_prob).
    Returns (detected: bool, n_photons: int).
    """
    k = np.random.poisson(mu)
    if pns:
        if k < 2:
            return False, k
        return (random.random() < pns_q), k
    if k == 0:
        return False, k
    return (random.random() < (1 - (1 - eta) ** k)), k


def run_bb84(n_bits=500, channel_noise=0.03, attack='none', attack_strength=1.0,
             mu_signal=0.5, mu_decoy=0.1, eta=0.15, decoy_prob=0.5,
             eve_basis='Z', seed=None):
    """
    Simulate one BB84 session and return channel statistics.

    attack:
      'none'       secure channel (loss + depolarizing noise only)
      'biased_ir'  basis-biased intercept-resend (Eve always measures in
                   `eve_basis`); intercepts a fraction `attack_strength` of
                   detected pulses -> asymmetric per-basis QBER, low total QBER
      'pns'        gain-matched photon-number-splitting -> signal gain and QBER
                   look normal, only the decoy/signal gain ratio collapses

    The default channel is high-loss (eta=0.15), the regime where PNS is a real
    threat and Eve can mask the loss. Pass `seed` to make Bob's measurement
    sampling reproducible (the photon Monte-Carlo uses the global numpy/random
    state, which the caller seeds once).

    Returns a dict with total/per-basis QBER, sift ratio, and signal/decoy gains,
    or None if no key could be sifted.
    """
    noise_model = build_noise_model(channel_noise)
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    pns = (attack == 'pns')
    pns_q = _pns_forward_prob(mu_signal, eta) if pns else 1.0

    signal_sent = signal_det = decoy_sent = decoy_det = 0
    recs = []  # detected pulses: (alice_bit, alice_basis, bob_basis, t_bit, t_basis)

    for _ in range(n_bits):
        is_decoy = random.random() < decoy_prob
        mu = mu_decoy if is_decoy else mu_signal
        if is_decoy:
            decoy_sent += 1
        else:
            signal_sent += 1

        detected, _k = _pulse_detected(mu, eta, pns, pns_q)
        if not detected:
            continue
        if is_decoy:
            decoy_det += 1
        else:
            signal_det += 1

        a_bit = random.randint(0, 1)
        a_basis = random.choice(['Z', 'X'])
        b_basis = random.choice(['Z', 'X'])
        t_bit, t_basis = a_bit, a_basis

        # Qubit-level eavesdropping (PNS forwards faithfully, so no change here).
        if attack == 'biased_ir' and random.random() < attack_strength:
            # Eve measures in her fixed basis and resends in that basis.
            if eve_basis != a_basis:
                t_bit = random.randint(0, 1)   # wrong-basis measurement randomises
            t_basis = eve_basis

        recs.append((a_bit, a_basis, b_basis, t_bit, t_basis))

    if not recs:
        return None

    # Run all detected qubits through the noisy simulator in one batch.
    circuits = [build_qubit_circuit(t_bit, t_basis, b_basis)
                for (_, _, b_basis, t_bit, t_basis) in recs]
    result = sim.run(circuits, shots=1).result()
    bob_bits = [int(list(result.get_counts(i).keys())[0]) for i in range(len(recs))]

    # Sift (keep matched bases) and split errors by basis.
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
    if sifted == 0:
        return None

    qber_z = z_err / z_total if z_total else 0.0
    qber_x = x_err / x_total if x_total else 0.0
    qber = (z_err + x_err) / sifted

    gain_signal = signal_det / signal_sent if signal_sent else 0.0
    gain_decoy = decoy_det / decoy_sent if decoy_sent else 0.0
    decoy_ratio = gain_decoy / gain_signal if gain_signal else 0.0

    return {
        'qber': qber,
        'qber_z': qber_z,
        'qber_x': qber_x,
        'qber_asymmetry': abs(qber_z - qber_x),
        'sift_ratio': sifted / len(recs),
        'gain_signal': gain_signal,
        'gain_decoy': gain_decoy,
        'decoy_ratio': decoy_ratio,
        'key_length': sifted,
        'n_detected': len(recs),
    }


@lru_cache(maxsize=None)
def _outcome_table(channel_noise, shots=40000):
    """
    P(Bob measures 1) for each of the 8 single-qubit configurations
    (t_bit, t_basis, b_basis) under the depolarizing channel.

    There are only 8 distinct circuits, so we evaluate each once with real
    noisy Qiskit-Aer circuits and cache the result. Sampling sequences from
    this table is statistically identical to running a fresh shots=1 circuit
    per pulse, but ~50x faster (it removes hundreds of thousands of circuit
    builds). Cached per noise level.
    """
    sim = AerSimulator(noise_model=build_noise_model(channel_noise))
    keys = [(tb, tba, bba) for tb in (0, 1)
            for tba in ('Z', 'X') for bba in ('Z', 'X')]
    circuits = [build_qubit_circuit(tb, tba, bba) for (tb, tba, bba) in keys]
    result = sim.run(circuits, shots=shots).result()
    return {k: result.get_counts(i).get('1', 0) / shots for i, k in enumerate(keys)}


def simulate_key_sequence(length=256, channel_noise=0.03, attack='none',
                          strength=0.3, burst_len=16, seed=None):
    """
    Emit a fixed-length sequence of sifted-key error bits (1 = error, 0 = match)
    over a lossless channel (every pulse detected). Bob's per-qubit outcome
    probabilities come from real noisy Qiskit circuits via `_outcome_table`.

    This supports *temporally-structured* attacks whose aggregate statistics
    (mean QBER, basis symmetry) match a weak continuous attacker, so only the
    time-ordering of the errors distinguishes them:

      'none'       noise only (errors i.i.d. at the noise floor)
      'cont_ir'    continuous symmetric intercept-resend: each pulse intercepted
                   i.i.d. with probability `strength` (errors uniform, elevated)
      'bursty_ir'  duty-cycled symmetric intercept-resend driven by a two-state
                   (telegraph) Markov process with mean ON length `burst_len` and
                   stationary ON fraction `strength`. Same mean QBER as 'cont_ir',
                   but interception (hence errors) is clustered into random bursts.

    Returns a length-`length` numpy int array of sifted error bits.
    """
    if seed is not None:
        random.seed(seed)
    p1 = _outcome_table(channel_noise)

    # Telegraph transition probabilities for the bursty attacker:
    # stationary P(ON) = strength, mean ON run = burst_len.
    p_off_to_on = (strength / max(1e-6, 1 - strength)) / burst_len
    p_on_to_off = 1.0 / burst_len
    eve_on = (attack == 'bursty_ir') and (random.random() < strength)  # stationary init

    errors = []
    while len(errors) < length:
        a_bit = random.randint(0, 1)
        a_basis = random.choice(['Z', 'X'])
        b_basis = random.choice(['Z', 'X'])
        t_bit, t_basis = a_bit, a_basis

        if attack == 'cont_ir':
            intercept = random.random() < strength
        elif attack == 'bursty_ir':
            eve_on = (random.random() > p_on_to_off) if eve_on \
                else (random.random() < p_off_to_on)
            intercept = eve_on
        else:
            intercept = False

        if intercept:
            eve_basis = random.choice(['Z', 'X'])  # symmetric: no asymmetry
            if eve_basis != a_basis:
                t_bit = random.randint(0, 1)
            t_basis = eve_basis

        if a_basis == b_basis:                     # sift
            bob = 1 if random.random() < p1[(t_bit, t_basis, b_basis)] else 0
            errors.append(int(a_bit != bob))

    return np.array(errors[:length], dtype=np.int8)
