"""Frozen small validation study, not training or an adaptive-Eve benchmark.

Default: 400 independent repetitions per cell. Writes only to a new directory
when --output is supplied; existing output directories are never overwritten.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research.calibration_protocol import (
    COORDINATES, ObservationLaw, change_alarm, gate_law, pauli_law,
    sample_public_batch,
)
from research.identifiability import ir_basis_errors, ir_pauli_weights

RESOURCES = ((2000, 2000000), (200000, 200000), (2000000, 2000000))
METHODS = ('hoeffding', 'clopper-pearson')
SEED = 20261005
ALPHA, DISCLOSURE = .05, .25


def scenarios():
    nominal = gate_law()
    weights = ir_pauli_weights(.03, .0396, 1)
    pauli = pauli_law(tuple(weights[k] for k in ('I', 'X', 'Y', 'Z')))
    honest = [('ideal', gate_law(0)), ('nominal', nominal),
              ('gate_error', gate_law(.03, .02)), ('ir_equivalent_pauli', pauli),
              ('high_error_pauli', pauli_law((0, .2, .7, .1))),
              ('fully_mixed', gate_law(1)), ('zero_loss', gate_law(eta=0))]
    result = [(name, 'stable_honest', law, law) for name, law in honest]
    for name, rate in [('weak_fixed_z_ir', .0396), ('full_fixed_z_ir', 1.)]:
        qz, qx = ir_basis_errors(.03, rate, 1)
        result.append((name, 'attack_shift', nominal, ObservationLaw((qz, qz, qx, qx))))
    # Demonstrates attribution limits: an honest shift can also raise an alarm.
    result.append(('honest_noise_drift', 'outside_stability_assumption', nominal, gate_law(.3)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=400)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.trials <= 2000:
        parser.error('This small pilot permits 1–2000 repetitions per cell')
    if args.output and args.output.exists():
        parser.error('Output already exists; use a new directory to preserve results')
    rows, cells = [], []
    cases = scenarios()
    for resource_id, (m, n) in enumerate(RESOURCES):
        for case_id, (name, category, cal_law, mon_law) in enumerate(cases):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, resource_id, case_id]))
            cal = sample_public_batch(args.trials, m, cal_law, 1., rng)
            mon = sample_public_batch(args.trials, n, mon_law, DISCLOSURE, rng)
            alarms = {method: change_alarm(cal, mon, ALPHA, method) for method in METHODS}
            for method, alarm in alarms.items():
                count = int(alarm.sum())
                ci = binomtest(count, args.trials).proportion_ci(.95, method='exact')
                cells.append({'calibration_emitted': m, 'monitoring_emitted': n,
                              'scenario': name, 'category': category, 'method': method,
                              'alarms': count, 'trials': args.trials, 'alarm_rate': count/args.trials,
                              'alarm_rate_ci95': [ci.low, ci.high],
                              'mean_calibration_sacrificed': float(cal.disclosed.mean()),
                              'mean_monitoring_disclosed': float(mon.disclosed.mean()),
                              'mean_remaining_candidates': float(mon.remaining_candidates.mean())})
            for trial in range(args.trials):
                row = {'resource_id': resource_id, 'scenario': name, 'trial': trial,
                       'calibration_emitted': m, 'monitoring_emitted': n}
                for stage, batch in [('cal', cal), ('mon', mon)]:
                    for j, coord in enumerate(COORDINATES):
                        row[f'{stage}_{coord}_numerator'] = int(batch.numerators[trial, j])
                        row[f'{stage}_{coord}_denominator'] = int(batch.denominators[trial, j])
                    for j, coord in enumerate(COORDINATES[:4]):
                        row[f'{stage}_{coord}_sifted'] = int(batch.sifted[trial, j])
                row.update({method: int(alarms[method][trial]) for method in METHODS})
                rows.append(row)
    summary = {'schema_version': 1, 'kind': 'small_fixed_protocol_validation_not_novelty_proof',
               'seed': SEED, 'alpha_per_rule_per_pair': ALPHA,
               'monitoring_disclosure_fraction': DISCLOSURE, 'calibration_disclosure_fraction': 1.,
               'trials_per_cell': args.trials, 'paired_transcripts': len(rows),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__},
               'scenario_laws': [{'name': name, 'category': category,
                                  'calibration': asdict(cal), 'monitoring': asdict(mon)}
                                 for name, category, cal, mon in cases],
               'source_sha256': {str(path.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__).resolve(),
                                              Path(__file__).resolve().parents[1]/'src/research/calibration_protocol.py',
                                              Path(__file__).resolve().parents[1]/'src/research/identifiability.py']},
               'limitations': ['Simulation only; attack-free stable calibration is assumed.',
                               'Monte Carlo intervals are marginal per cell, not simultaneous.',
                               'Two rule outputs are not combined; alpha is not a budget for their union.',
                               'No repeated peeking, real hardware, adaptive Eve or secret-key proof.',
                               'All rows are included; no usability filtering or tuning.'], 'cells': cells}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        audit = args.output/'public_counts.csv'
        with audit.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary['public_counts_sha256'] = hashlib.sha256(audit.read_bytes()).hexdigest()
        (args.output/'pilot.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
