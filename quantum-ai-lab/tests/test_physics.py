"""Independent theory checks and real-Aer regression tests (no simulator mocks)."""
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from bb84.simulator import (
    _multiphoton_probability,
    _outcome_table,
    _pns_forward_prob,
    _pulse_detected,
    build_noise_model,
    build_qubit_circuit,
    run_bb84,
    simulate_key_sequence,
)
from bb84.dataset import generate_dataset


def reference_probability(bit, basis, meas, channel_noise, gate_noise=0.0):
    """Independent 2x2 density-matrix calculation; no production helpers used."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2)
    rho = np.array([[1.0, 0.0], [0.0, 0.0]])
    for gate in ([x] if bit else []) + ([h] if basis == 'X' else []):
        rho = gate @ rho @ gate.T
        rho = (1-gate_noise) * rho + gate_noise * np.eye(2)/2
    rho = (1-channel_noise) * rho + channel_noise * np.eye(2)/2
    if meas == 'X':
        rho = h @ rho @ h.T
        rho = (1-gate_noise) * rho + gate_noise * np.eye(2)/2
    return rho[1, 1]


@pytest.mark.parametrize('channel_noise', [0.0, 0.03, 0.4, 1.0])
@pytest.mark.parametrize('gate_noise', [0.0, 0.02])
def test_exact_aer_probabilities_match_independent_density_matrix(channel_noise, gate_noise):
    table = _outcome_table(channel_noise, gate_noise=gate_noise)
    for bit, basis, meas in itertools.product((0, 1), ('Z', 'X'), ('Z', 'X')):
        assert table[bit, basis, meas] == pytest.approx(
            reference_probability(bit, basis, meas, channel_noise, gate_noise), abs=1e-12)


def test_correct_noiseless_bb84_states_and_wrong_basis_randomness():
    table = _outcome_table(0.0)
    for bit, basis in itertools.product((0, 1), ('Z', 'X')):
        assert table[bit, basis, basis] == pytest.approx(bit, abs=1e-12)
        other = 'X' if basis == 'Z' else 'Z'
        assert table[bit, basis, other] == pytest.approx(0.5, abs=1e-12)


def test_channel_noise_has_no_artificial_bit_or_basis_bias():
    table = _outcome_table(0.03)
    for basis in ('Z', 'X'):
        assert table[0, basis, basis] == pytest.approx(0.015, abs=1e-12)
        assert 1-table[1, basis, basis] == pytest.approx(0.015, abs=1e-12)
    qz = (table[0, 'Z', 'Z'] + 1-table[1, 'Z', 'Z'])/2
    qx = (table[0, 'X', 'X'] + 1-table[1, 'X', 'X'])/2
    assert abs(qz-qx) < 1e-12


def test_local_gate_errors_are_explicitly_separate_from_transmission():
    assert build_noise_model().is_ideal()
    assert set(build_noise_model(gate_noise=0.03).noise_instructions) == {'h', 'x'}
    table = _outcome_table(0.0, gate_noise=0.03)
    assert table[0, 'Z', 'Z'] == pytest.approx(0.0, abs=1e-12)
    assert table[1, 'Z', 'Z'] == pytest.approx(1-0.03/2, abs=1e-12)
    assert table[0, 'X', 'X'] > table[0, 'Z', 'Z']


@pytest.mark.parametrize('bit,basis,meas', list(itertools.product((0, 1), ('Z', 'X'), ('Z', 'X'))))
def test_transmission_occurs_once_and_survives_transpilation(bit, basis, meas):
    qc = build_qubit_circuit(bit, basis, meas, channel_noise=0.2)
    assert qc.count_ops()['quantum_channel'] == 1
    qc.remove_final_measurements()
    qc.save_probabilities()
    backend = AerSimulator(method='density_matrix')
    for level in (0, 3):
        compiled = transpile(qc, backend, optimization_level=level, seed_transpiler=0)
        result = backend.run(compiled, shots=1).result()
        assert result.success
        assert result.data(0)['probabilities'][1] == pytest.approx(
            reference_probability(bit, basis, meas, 0.2), abs=1e-12)


def enumerated_ir_errors(rate, eve_z_probability, channel_noise):
    """Enumerate Alice bits/bases, Eve bases and wrong-basis resend outcomes."""
    table = _outcome_table(channel_noise)
    rates = {}
    for basis in ('Z', 'X'):
        error = 0.0
        for bit in (0, 1):
            def error_prob(t_bit, t_basis):
                p_one = table[t_bit, t_basis, basis]
                return p_one if bit == 0 else 1-p_one
            error += 0.5 * (1-rate) * error_prob(bit, basis)
            for eve, prob in [('Z', eve_z_probability), ('X', 1-eve_z_probability)]:
                resent = [(bit, 1.0)] if eve == basis else [(0, 0.5), (1, 0.5)]
                error += 0.5 * rate * prob * sum(w*error_prob(tb, eve) for tb, w in resent)
        rates[basis] = error
    return rates


@pytest.mark.parametrize('rate,bias,noise', [(1, 0.5, 0), (1, 1, 0), (0.2, 1, 0.03), (0.4, 0.7, 0.1)])
def test_intercept_resend_theory_including_basis_biased_attack(rate, bias, noise):
    rates = enumerated_ir_errors(rate, bias, noise)
    assert rates['Z'] == pytest.approx(noise/2 + rate*(1-bias)*(1-noise)/2, abs=1e-12)
    assert rates['X'] == pytest.approx(noise/2 + rate*bias*(1-noise)/2, abs=1e-12)
    assert (rates['Z']+rates['X'])/2 == pytest.approx(noise/2 + rate*(1-noise)/4, abs=1e-12)


def test_pns_closed_form_gain_matching_and_decoy_signature():
    # Compute theory independently with math, not the production tail helper.
    mu_s, mu_d, eta = 0.5, 0.1, 0.15
    multi_s = 1-math.exp(-mu_s)*(1+mu_s)
    multi_d = 1-math.exp(-mu_d)*(1+mu_d)
    signal = 1-math.exp(-mu_s*eta)
    decoy = 1-math.exp(-mu_d*eta)
    q = _pns_forward_prob(mu_s, eta)
    assert q == pytest.approx(0.8010343811340691)
    assert q*multi_s == pytest.approx(signal, abs=1e-14)
    assert signal == pytest.approx(0.07225651367144714)
    assert decoy/signal == pytest.approx(0.2060445438128094)
    assert q*multi_d/signal == pytest.approx(0.051869535934001613)
    assert q*multi_d < decoy


@pytest.mark.parametrize('mu', [0, 1e-8, 1e-5, 0.1, 0.5, 3.0])
def test_multiphoton_probability_against_poisson_series(mu):
    expected = math.exp(-mu) * sum(mu**k/math.factorial(k) for k in range(2, 80))
    assert _multiphoton_probability(mu) == pytest.approx(expected, rel=1e-10, abs=1e-24)


def test_weak_intensity_gain_matching_is_numerically_stable():
    q = _pns_forward_prob(1e-8, 1e-10)
    assert math.isfinite(q)
    assert q == pytest.approx(0.0200000001333333, rel=1e-8)


def assert_binomial_rate(observed, expected, n):
    # Six-sigma envelope plus one-count resolution; fixed seeds avoid flaky tests.
    tolerance = 6*math.sqrt(expected*(1-expected)/n) + 1/n
    assert abs(observed-expected) <= tolerance


@pytest.mark.parametrize('mu,pns', [(0.5, False), (0.1, False), (0.5, True), (0.1, True)])
def test_monte_carlo_photon_detection_matches_theory(mu, pns):
    n, eta = 100000, 0.15
    q = _pns_forward_prob(0.5, eta)
    rng = np.random.default_rng(100)
    detections = 0
    for _ in range(n):
        detected, k = _pulse_detected(mu, eta, pns, q, rng=rng)
        if pns and k < 2:
            assert not detected
        detections += detected
    expected = q*(1-math.exp(-mu)*(1+mu)) if pns else 1-math.exp(-mu*eta)
    assert_binomial_rate(detections/n, expected, n)


@pytest.mark.parametrize('pns', [False, True])
def test_zero_loss_transmittance_and_vacuum_emit_no_detections(pns):
    rng = np.random.default_rng(1)
    for _ in range(100):
        assert not _pulse_detected(0.0, 0.15, pns, 0.8, rng=rng)[0]
        assert not _pulse_detected(0.5, 0.0, pns, 0.0, rng=rng)[0]


def test_seeded_session_repeatability_counts_and_no_global_rng_changes():
    py_state = random.getstate()
    np_state = np.random.get_state()
    params = dict(n_bits=3000, channel_noise=0.03, seed=12)
    first = run_bb84(**params)
    second = run_bb84(**params)
    assert first is not None and first == second
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert np_state[0] == after[0]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    assert first['model_version'] == 2
    assert first['signal_sent'] + first['decoy_sent'] == params['n_bits']
    assert first['signal_detected'] + first['decoy_detected'] == first['n_detected']
    assert first['z_sifted'] + first['x_sifted'] == first['key_length']
    assert first['qber'] == (first['z_errors']+first['x_errors'])/first['key_length']
    assert first['gain_signal'] == first['signal_detected']/first['signal_sent']
    assert first['gain_decoy'] == first['decoy_detected']/first['decoy_sent']


@pytest.mark.parametrize('attack', ['none', 'pns'])
def test_noiseless_session_pns_adds_no_bit_errors(attack):
    stats = run_bb84(n_bits=6000, attack=attack, channel_noise=0, seed=42)
    assert stats is not None
    assert stats['qber'] == stats['qber_z'] == stats['qber_x'] == 0
    assert_binomial_rate(stats['gain_signal'], 1-math.exp(-0.5*0.15), stats['signal_sent'])


def test_zero_transmittance_returns_no_session():
    assert run_bb84(n_bits=100, eta=0, seed=0) is None


def test_full_basis_biased_intercept_resend_session():
    stats = run_bb84(n_bits=5000, mu_signal=1, eta=1, decoy_prob=0,
                     attack='biased_ir', attack_strength=1, channel_noise=0, seed=1)
    assert stats['qber_z'] == 0
    assert_binomial_rate(stats['qber_x'], 0.5, stats['x_sifted'])
    assert_binomial_rate(stats['qber'], 0.25, stats['key_length'])


@pytest.mark.parametrize('attack', ['none', 'pns'])
def test_noisy_session_rates_and_gains_match_reference_model(attack):
    stats = run_bb84(n_bits=60000, attack=attack, channel_noise=0.03, seed=4)
    for basis in ('z', 'x'):
        assert_binomial_rate(stats[f'qber_{basis}'], 0.015, stats[f'{basis}_sifted'])
    signal = 1-math.exp(-0.5*0.15)
    decoy = 1-math.exp(-0.1*0.15)
    if attack == 'pns':
        decoy = (signal/(1-math.exp(-0.5)*1.5)) * (1-math.exp(-0.1)*1.1)
    assert_binomial_rate(stats['gain_signal'], signal, stats['signal_sent'])
    assert_binomial_rate(stats['gain_decoy'], decoy, stats['decoy_sent'])


def test_dataset_caller_retains_seeded_reproducibility():
    first = generate_dataset(n_samples=6, n_bits=1000, seed=9, return_family=True)
    second = generate_dataset(n_samples=6, n_bits=1000, seed=9, return_family=True)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('attack,strength,expected', [('none', 0.3, 0.015), ('cont_ir', 1, 0.25),
                                                     ('cont_ir', 0.3, 0.08775), ('bursty_ir', 0, 0)])
def test_sequence_sampling_expected_qber(attack, strength, expected):
    noise = 0 if strength in (0, 1) else 0.03
    seq = simulate_key_sequence(length=10000, attack=attack, strength=strength,
                                channel_noise=noise, seed=44)
    assert seq.dtype == np.int8
    assert set(seq) <= {0, 1}
    assert_binomial_rate(seq.mean(), expected, len(seq))


@pytest.mark.parametrize('attack', ['none', 'cont_ir', 'bursty_ir'])
def test_sequence_repeatability_and_no_global_rng_changes(attack):
    state = random.getstate()
    first = simulate_key_sequence(length=256, attack=attack, seed=456)
    _outcome_table.cache_clear()
    second = simulate_key_sequence(length=256, attack=attack, seed=456)
    np.testing.assert_array_equal(first, second)
    assert random.getstate() == state


def test_full_bursty_duty_cycle_has_absorbing_on_state():
    seq = simulate_key_sequence(length=10000, attack='bursty_ir', strength=1,
                                burst_len=16, channel_noise=0, seed=44)
    assert_binomial_rate(seq.mean(), 0.25, len(seq))


def test_bursty_and_continuous_share_expected_mean_not_variance():
    n, length = 400, 256
    means = {}
    for attack in ('cont_ir', 'bursty_ir'):
        means[attack] = np.array([simulate_key_sequence(length=length, attack=attack,
                                  strength=0.3, channel_noise=0, seed=i).mean() for i in range(n)])
        standard_error = means[attack].std(ddof=1)/math.sqrt(n)
        assert abs(means[attack].mean()-0.075) < 6*standard_error
    assert means['bursty_ir'].var() > means['cont_ir'].var()


def test_lookup_is_immutable_and_not_a_shot_estimate():
    table = _outcome_table(0.03)
    with pytest.raises(TypeError):
        table[0, 'Z', 'Z'] = 0.99
    assert dict(table) == dict(_outcome_table(0.03, shots=1))
    assert dict(table) == dict(_outcome_table(0.03, shots=40000))


@pytest.mark.parametrize('value', [True, np.bool_(True), -0.1, float('nan')])
def test_invalid_channel_inputs_cannot_bypass_validation_through_a_warm_cache(value):
    # bool True compares/hashes like float 1.0; cache types must remain distinct.
    _outcome_table(1.0)
    build_qubit_circuit(0, 'Z', 'Z', channel_noise=1.0)
    with pytest.raises(ValueError):
        _outcome_table(value)
    with pytest.raises(ValueError):
        build_qubit_circuit(0, 'Z', 'Z', channel_noise=value)


def test_seeded_sequence_is_identical_in_two_fresh_processes():
    source = str(Path(__file__).resolve().parents[1]/'src')
    # Do not pass unrelated host credentials/configuration to diagnostic children.
    env = {'PYTHONPATH': source, 'OMP_NUM_THREADS': '1'}
    code = ('import json; from bb84.simulator import simulate_key_sequence, run_bb84; '
            'print(json.dumps({"sequence": simulate_key_sequence(128, attack="bursty_ir", '
            'seed=99).tolist(), "session": run_bb84(n_bits=3000, seed=99)}))')
    first = subprocess.check_output([sys.executable, '-c', code], env=env, timeout=30)
    second = subprocess.check_output([sys.executable, '-c', code], env=env, timeout=30)
    assert json.loads(first) == json.loads(second)


@pytest.mark.parametrize('kwargs', [
    {'n_bits': 0}, {'n_bits': -1}, {'n_bits': 1.5}, {'n_bits': True},
    {'channel_noise': -0.1}, {'channel_noise': 1.1}, {'channel_noise': float('nan')},
    {'channel_noise': float('inf')}, {'gate_noise': -0.1},
    {'eta': -0.1}, {'eta': 1.1}, {'decoy_prob': 2}, {'attack_strength': -0.1},
    {'mu_signal': -1}, {'mu_decoy': float('nan')}, {'attack': 'typo'},
    {'eve_basis': 'Y'}, {'seed': -1}, {'seed': True},
])
def test_session_rejects_invalid_inputs_before_sampling(kwargs):
    with pytest.raises(ValueError):
        run_bb84(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'length': 0}, {'length': 2.5}, {'attack': 'typo'}, {'strength': 1.1},
    {'burst_len': 0}, {'burst_len': True}, {'burst_len': float('nan')},
    {'attack': 'bursty_ir', 'strength': 0.9, 'burst_len': 1},
])
def test_sequence_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate_key_sequence(**kwargs)


@pytest.mark.parametrize('mu,eta', [(0, 0.15), (0.5, 0.5), (0.5, 1)])
def test_infeasible_pns_is_rejected_not_silently_clipped(mu, eta):
    with pytest.raises(ValueError):
        _pns_forward_prob(mu, eta)
    with pytest.raises(ValueError):
        run_bb84(attack='pns', mu_signal=mu, eta=eta)


def test_pns_feasible_boundary_and_zero_transmittance():
    mu = 0.5
    multi = 1-math.exp(-mu)*(1+mu)
    eta_boundary = -math.log1p(-multi)/mu
    assert _pns_forward_prob(mu, eta_boundary) == pytest.approx(1)
    assert _pns_forward_prob(mu, 0) == 0
