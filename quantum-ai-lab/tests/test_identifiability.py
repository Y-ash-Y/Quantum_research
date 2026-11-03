"""Feasibility derivations checked with real Aer and independent matrices."""
import itertools
import math

import numpy as np
import pytest

from bb84.simulator import _outcome_table
from research.identifiability import (
    aggregate_kl_per_pulse, bernoulli_kl, fixed_z_hidden_limit, gate_mimic,
    honest_basis_errors, honest_bit_errors, ir_basis_errors, ir_pauli_weights,
    power_ceiling,
)


@pytest.mark.parametrize('noise,gate', list(itertools.product([0, .03, .4, 1], [0, .02, .2, 1])))
def test_bit_specific_gate_counts_match_real_aer(noise, gate):
    table = _outcome_table(noise, gate_noise=gate)
    expected = honest_bit_errors(noise, gate)
    actual = tuple(table[0, b, b] if bit == 0 else 1-table[1, b, b]
                   for b, bit in [('Z', 0), ('Z', 1), ('X', 0), ('X', 1)])
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('noise,gate', list(itertools.product([0, .03, .4, 1], [0, .02, .2, 1])))
def test_inverse_round_trip_and_nonunique_boundary(noise, gate):
    pair = honest_basis_errors(noise, gate)
    inverse = gate_mimic(*pair)
    assert inverse is not None
    assert honest_basis_errors(inverse.channel_noise, inverse.gate_noise) == pytest.approx(pair)
    if noise == 1:
        assert inverse.nonunique and inverse.channel_noise == 1
    else:
        assert (inverse.channel_noise, inverse.gate_noise) == pytest.approx((noise, gate))


@pytest.mark.parametrize('pair', [(0, .5), (.2, .1), (.6, .6), (.5, .1)])
def test_inverse_rejects_nonphysical_gate_family(pair):
    assert gate_mimic(*pair) is None


@pytest.mark.parametrize('noise,cap', [(0, .02), (.03, .02), (.03, .2), (.4, .5), (.6, 1), (1, 0)])
def test_closed_form_fixed_z_boundary(noise, cap):
    limit = fixed_z_hidden_limit(noise, cap)
    for r in (0, limit/2, limit):
        mimic = gate_mimic(*ir_basis_errors(noise, r, 1))
        assert mimic is not None and mimic.gate_noise <= cap+1e-12
    if limit < 1:
        above = min(1, limit+1e-5)
        mimic = gate_mimic(*ir_basis_errors(noise, above, 1))
        assert mimic is None or mimic.gate_noise > cap


def test_two_percent_gate_cap_exact_hidden_attack():
    assert fixed_z_hidden_limit(.03, .02) == pytest.approx(.0396)
    attack = ir_basis_errors(.03, .0396, 1)
    mimic = gate_mimic(*attack)
    assert mimic.gate_noise == pytest.approx(.02)
    assert mimic.channel_noise == pytest.approx(.02020202020202022)
    assert attack == pytest.approx((.015, .034206))
    # Stronger bit-conditioned telemetry separates THIS narrow honest model.
    z0, z1, x0, x1 = honest_bit_errors(mimic.channel_noise, mimic.gate_noise)
    assert z0 != z1 and x0 != x1


def test_unbiased_ir_not_resolved_by_bit_conditioning():
    attack = ir_basis_errors(.03, .1, .5)
    mimic = gate_mimic(*attack)
    assert mimic.gate_noise == pytest.approx(0)
    assert mimic.channel_noise == pytest.approx(.0785)
    assert honest_bit_errors(mimic.channel_noise, 0) == pytest.approx(attack*2)


@pytest.mark.parametrize('noise,rate,bias', [(0, 1, 1), (.03, .1, .95), (.2, .7, .2), (1, 1, .5)])
def test_whole_channel_ir_equals_honest_pauli_for_arbitrary_states(noise, rate, bias):
    ident = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    weights = ir_pauli_weights(noise, rate, bias)
    assert sum(weights.values()) == pytest.approx(1)
    assert all(v >= 0 for v in weights.values())
    # Pure states along X,Y,Z and one mixed state; matrix channel calculation
    # does not call outcome-table, gate-noise, or detector-scoring helpers.
    vectors = (np.array([1, 0]), np.array([1, 1])/math.sqrt(2),
               np.array([1, 1j])/math.sqrt(2), np.array([math.sqrt(.3), math.sqrt(.7)]))
    states = [np.outer(vec, vec.conj()) for vec in vectors]
    states.append(.3*ident/2+.7*states[-1])
    for rho in states:
        measured_z = (rho+z@rho@z)/2
        measured_x = (rho+x@rho@x)/2
        intercepted = (1-rate)*rho+rate*bias*measured_z+rate*(1-bias)*measured_x
        attacked = (1-noise)*intercepted+noise*ident/2
        honest = sum(weights[n]*p@rho@p.conj().T
                     for n, p in [('I', ident), ('X', x), ('Y', y), ('Z', z)])
        assert np.allclose(attacked, honest, rtol=0, atol=1e-12)


@pytest.mark.parametrize('noise,gate', [(0, 0), (.03, .02), (.1, 1)])
def test_mismatched_bases_do_not_break_this_model_ambiguity(noise, gate):
    table = _outcome_table(noise, gate_noise=gate)
    for bit, prep, meas in itertools.product((0, 1), ('Z', 'X'), ('Z', 'X')):
        if prep != meas:
            assert table[bit, prep, meas] == pytest.approx(.5, abs=1e-12)


def test_aggregate_kl_equals_explicit_categorical_enumeration():
    gain = .06
    p, q = (.015, .04), (.012, .032)
    def cells(pair):
        return [1-gain, gain/2, gain*(1-pair[0])/4, gain*pair[0]/4,
                gain*(1-pair[1])/4, gain*pair[1]/4]
    raw = sum(a*math.log(a/b) for a, b in zip(cells(p), cells(q)))
    assert aggregate_kl_per_pulse(p, q, gain) == pytest.approx(raw, abs=1e-14)


def test_information_bounds_and_zero_loss_edges():
    assert bernoulli_kl(0, .5) == pytest.approx(math.log(2))
    assert bernoulli_kl(1, .5) == pytest.approx(math.log(2))
    assert math.isinf(bernoulli_kl(.1, 0))
    assert aggregate_kl_per_pulse((.1, .2), (.1, .2), .06) == 0
    assert aggregate_kl_per_pulse((.1, .2), (0, 0), 0) == 0
    assert power_ceiling(.05, 120000, 0) == .05
    assert power_ceiling(.05, 0, math.inf) == .05
    assert power_ceiling(.05, 10, .001) == pytest.approx(.05+math.sqrt(.005))


@pytest.mark.parametrize('bad', [True, -1, 1.1, float('nan'), float('inf'), '0.1'])
def test_invalid_physics_inputs_rejected(bad):
    for func, args in [(honest_bit_errors, (bad, 0)), (ir_basis_errors, (0, bad, .5)),
                       (fixed_z_hidden_limit, (.03, bad)), (gate_mimic, (bad, .1)),
                       (ir_pauli_weights, (0, .1, bad))]:
        with pytest.raises(ValueError):
            func(*args)


@pytest.mark.parametrize('n,k', [(True, 0), (-1, 0), (1.5, 0), (1, -1), (1, math.nan), (1, True)])
def test_invalid_bound_inputs_rejected(n, k):
    with pytest.raises(ValueError):
        power_ceiling(.05, n, k)
