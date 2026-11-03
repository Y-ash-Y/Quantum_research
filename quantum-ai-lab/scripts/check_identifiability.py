"""Print deterministic theory examples; no fitting, large simulation or writes."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.identifiability import (
    aggregate_kl_per_pulse, fixed_z_hidden_limit, gate_mimic,
    honest_basis_errors, honest_bit_errors, ir_basis_errors, ir_pauli_weights,
    power_ceiling,
)
from bb84.simulator import _outcome_table


def main():
    cases = []
    for name, r, beta in [('biased_ir_10pct', .1, .95),
                          ('fixed_z_gate_cap_boundary', .0396, 1),
                          ('unbiased_ir_10pct', .1, .5)]:
        attack = ir_basis_errors(.03, r, beta)
        mimic = gate_mimic(*attack)
        if mimic is None:
            raise AssertionError('Expected a physical honest counterpart')
        honest = honest_basis_errors(mimic.channel_noise, mimic.gate_noise)
        table = _outcome_table(mimic.channel_noise, gate_noise=mimic.gate_noise)
        aer = tuple((table[0, b, b]+1-table[1, b, b])/2 for b in ('Z', 'X'))
        error = max(abs(p-q) for p, q in zip(aer, attack))
        if error > 1e-12:
            raise AssertionError('Aer and analytical overlap disagree')
        gain = (-math.expm1(-.5*.15)-math.expm1(-.1*.15))/2
        kl = aggregate_kl_per_pulse(attack, honest, gain)
        calibration_kl = aggregate_kl_per_pulse(honest_basis_errors(.03, 0), honest, gain)
        cases.append({'name': name, 'attack': {'lambda': .03, 'r': r, 'beta': beta},
                      'attack_qz_qx': attack, 'honest_mimic': asdict(mimic),
                      'honest_bit_errors_z0_z1_x0_x1': honest_bit_errors(
                          mimic.channel_noise, mimic.gate_noise),
                      'max_aer_mean_residual': error,
                      'aggregate_kl_per_emitted_pulse': kl,
                      'power_ceiling_alpha05_120k': power_ceiling(.05, 120000, kl),
                      'trusted_stable_calibration_kl_per_pulse': calibration_kl,
                      'necessary_calibration_pulses_for_80pct_power': (
                          2*(.8-.05)**2/calibration_kl if calibration_kl else None),
                      'equivalent_honest_pauli_weights': ir_pauli_weights(.03, r, beta)})
    print(json.dumps({'kind': 'deterministic_theory_not_benchmark',
                      'lambda': .03, 'gate_cap': .02,
                      'fixed_z_overlap_max_ir_rate': fixed_z_hidden_limit(.03, .02),
                      'cases': cases}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
