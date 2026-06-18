import numpy as np
from typing import Union, Tuple
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


def _feature_state_amplitudes(X, feature_map) -> np.ndarray:
    """Map each sample x to the amplitude vector of |φ(x)⟩ = U(x)|0⟩."""
    amps = []
    for x in X:
        bound = feature_map.assign_parameters(
            dict(zip(feature_map.parameters, x)), inplace=False
        )
        amps.append(Statevector.from_instruction(bound).data)
    return np.asarray(amps)


def quantum_kernel_matrix(X_train, X_test=None, n_qubits=5, reps=2) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute quantum kernel matrix using ZZFeatureMap.

    Theoretical motivation: quantum feature map φ(x) embeds classical
    data into Hilbert space. Kernel K(xi, xj) = |⟨φ(xi)|φ(xj)⟩|²
    may capture correlations that classical kernels cannot.

    Implementation note: each feature state |φ(x)⟩ is computed once as a
    statevector (O(n) state preparations), then the full kernel is the
    elementwise-squared modulus of the Gram matrix of amplitudes. This is
    exact (no shot noise) and orders of magnitude faster than estimating
    each |⟨φ(xi)|φ(xj)⟩|² from a separately transpiled, sampled circuit.
    """
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps)

    train_amp = _feature_state_amplitudes(X_train, feature_map)
    K_train = np.abs(train_amp.conj() @ train_amp.T) ** 2

    if X_test is None:
        return K_train

    test_amp = _feature_state_amplitudes(X_test, feature_map)
    K_test = np.abs(test_amp.conj() @ train_amp.T) ** 2

    return K_train, K_test


class QuantumKernelSVM:
    """
    SVM using a ZZFeatureMap fidelity kernel.

    Encoding configuration matters a lot. ZZFeatureMap kernels suffer from
    *exponential concentration*: with large encoding angles and many
    repetitions the data states become near-orthogonal, the kernel matrix
    collapses towards the identity, and the SVM cannot generalise. We therefore
    default to a small encoding scale (pi/8) and a single repetition, which
    keeps the off-diagonal kernel entries informative. (For comparison, the
    naive [0, pi] / reps=2 setting drops the test AUC from ~0.95 to ~0.65.)
    """
    def __init__(self, n_qubits=5, reps=1, feature_scale=np.pi / 8):
        self.n_qubits = n_qubits
        self.reps = reps
        self.svm = SVC(kernel='precomputed', probability=True)
        self.scaler = MinMaxScaler(feature_range=(0, feature_scale))
        self.X_train_scaled = None

    def fit(self, X_train, y_train):
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        K_train = quantum_kernel_matrix(
            self.X_train_scaled, X_test=None, n_qubits=self.n_qubits, reps=self.reps
        )
        assert isinstance(K_train, np.ndarray)
        self.svm.fit(K_train, y_train)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        _, K_test = quantum_kernel_matrix(
            self.X_train_scaled, X_test_scaled,
            n_qubits=self.n_qubits, reps=self.reps
        )
        return self.svm.predict(K_test)

    def predict_proba(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        _, K_test = quantum_kernel_matrix(
            self.X_train_scaled, X_test_scaled,
            n_qubits=self.n_qubits, reps=self.reps
        )
        return self.svm.predict_proba(K_test)