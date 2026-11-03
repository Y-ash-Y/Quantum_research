"""Post-hoc explanation of a saved SVM failure; never changes any fitted rule.

Refits from the recorded training rows only and reproduces the saved scores.
The seed/case is explicitly selected after evaluation, so this is diagnostic
evidence, not another predeclared performance or superiority test.
"""
import argparse
import csv
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from classical_ml.fair_benchmark import FEATURES, fit_scorers
from classical_ml.robustness_audit import audit_results


def diagnose(destination, seed=101, *, check_current_sources=True):
    destination = Path(destination)
    audit_results(destination, check_current_sources=check_current_sources)
    report_file = destination/'robustness.json'
    report = json.loads(report_file.read_text())
    if version('scikit-learn') != report['provenance']['packages']['scikit-learn']:
        raise ValueError('Use the recorded sklearn version for exact replay')
    run = next((r for r in report['runs'] if r['seed'] == seed), None)
    if run is None:
        raise ValueError('Requested seed is not in the recorded run')
    with gzip.open(destination/'session_audit.csv.gz', 'rt') as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if int(r['seed']) == seed]
    train = [r for r in selected if r['cohort'] == 'training']
    matrix = lambda group: np.array([[float(r[f]) for f in FEATURES] for r in group])
    pipeline = fit_scorers(matrix(train), np.array([int(r['label']) for r in train]),
                           seed, report['config']['n_estimators'])['RBF SVM'].model
    scaler, svm = pipeline.steps[0][1], pipeline.steps[1][1]
    tested = [r for r in selected if r['cohort'] == 'nominal::IR above support']
    X = matrix(tested)
    saved = np.array([float(r['RBF SVM::score']) for r in tested])
    replay = pipeline.decision_function(X)
    # Explicit RBF expansion: f(x)=sum_i a_i exp(-gamma||x-s_i||^2)+b.
    standardized = scaler.transform(X)
    squared = ((standardized[:, None, :]-svm.support_vectors_[None, :, :])**2).sum(axis=2)
    kernel = np.exp(-svm._gamma*squared)
    manual = kernel @ svm.dual_coef_[0] + svm.intercept_[0]
    replay_error = float(np.max(abs(replay-saved)))
    manual_error = float(np.max(abs(manual-replay)))
    if replay_error > 1e-10 or manual_error > 1e-10:
        raise ValueError('Saved SVM scores do not replay within numerical tolerance')
    physics = []
    for record in report['runs']:
        for condition, settings in report['conditions'].items():
            group = [r for r in rows if int(r['seed']) == record['seed']
                     and r['cohort'] == f'{condition}::honest']
            sifted = sum(int(r['sifted_bits']) for r in group)
            errors = sum(round(float(r['qber'])*int(r['sifted_bits'])) for r in group)
            expected = settings['channel_noise']/2
            observed = errors/sifted
            z = (observed-expected)/np.sqrt(expected*(1-expected)/sifted)
            physics.append({'seed': record['seed'], 'condition': condition,
                            'error_bits': errors, 'sifted_bits': sifted,
                            'observed_pooled_qber': observed,
                            'expected_qber': expected, 'binomial_z': float(z)})
    return {
        'scope': 'Post-hoc seed-specific mechanism diagnostic, not an independent benchmark. '
                 'No thresholds or training rows are changed.',
        'benchmark_sha256': hashlib.sha256(report_file.read_bytes()).hexdigest(),
        'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': seed, 'cohort': 'nominal::IR above support',
        'score_replay_max_absolute_error': replay_error,
        'manual_rbf_max_absolute_error': manual_error,
        'intercept': float(svm.intercept_[0]), 'gamma': float(svm._gamma),
        'cutoff': run['methods']['RBF SVM']['cutoff']['value'],
        'accepted': len(tested), 'flagged': int(sum(int(r['RBF SVM::flag']) for r in tested)),
        'score_min': float(saved.min()), 'score_mean': float(saved.mean()),
        'score_max': float(saved.max()),
        'fraction_within_0_01_of_intercept': float((abs(saved-svm.intercept_[0]) < 0.01).mean()),
        'nearest_support_kernel_median': float(np.median(kernel.max(axis=1))),
        'honest_physics_checks': physics,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical-sources', action='store_true',
                        help='Replay saved training rows without asserting current generator source hashes match')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; refusing to overwrite')
    result = diagnose(args.destination, args.seed, check_current_sources=not args.historical_sources)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'honest_physics_checks'}, indent=2))
    print('Maximum honest pooled-QBER |z|:', max(abs(r['binomial_z']) for r in result['honest_physics_checks']))
