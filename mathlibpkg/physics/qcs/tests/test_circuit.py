import numpy as np
import pytest
from mathlibpkg.physics.qcs import QubitSystem
from mathlibpkg.physics.qcs import Gates
from mathlibpkg.physics.qcs import QubitCircuits
from functools import reduce
from mathlibpkg.physics.qcs import Instruction

# Creating many qubit systems and circuits for testing

@pytest.fixture(params=[1, 2, 3, 4, 5, 6])
def qubit_circuit(request):
    qs = QubitSystem(n=request.param)
    return QubitCircuits(qs=qs)


def test_embed_one_gate_correct_shape():
    gate = np.array([[0, 1], [1, 0]], dtype=complex)
    embedded = QubitCircuits._embed_one(gate, 0, 2)
    assert embedded.shape == (4, 4)
    assert np.allclose(embedded.toarray(), np.kron(gate, np.eye(2)))

def test_random_circuit_output_shape(qubit_circuit):
    n = qubit_circuit.qs.n  # oder qubit_circuit.qs.num_qubits, je nach Definition
    state = qubit_circuit.random_circuit(n, 5, seed=42)
    assert state.shape == (2**n, 1) or state.shape == (2**n,)

def test_random_circuit_normalization(qubit_circuit):
    n = qubit_circuit.qs.n  # oder qubit_circuit.qs.num_qubits, je nach Definition
    state = qubit_circuit.random_circuit(n, 5, seed=42)
    assert np.isclose(np.linalg.norm(state), 1.0)

def test_haar_random_state_is_pure(qubit_circuit):
    qubit_circuit.haar_random_state()
    rho = qubit_circuit.qs.density_matrix
    assert np.isclose(np.trace(rho @ rho), 1.0)

def test_hilbert_schmidt_random_state(qubit_circuit):
    qubit_circuit.hilbert_schmidt_random_state()
    rho = qubit_circuit.qs.density_matrix
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-12)

def test_bures_random_state(qubit_circuit):
    qubit_circuit.bures_random_state()
    rho = qubit_circuit.qs.density_matrix
    assert np.isclose(np.trace(rho), 1.0)
    assert np.all(np.linalg.eigvalsh(rho) >= -1e-12)

# testing the underlying distributions of the random states
# purity expectation value of the bures distribution is 5/(2N), N is the dimension of the density matrix
# definition  of purity: Tr(rho^2)
# mean value of many density matricies is 1/N \sum_i Tr(rho_i^2) -> should be around 5/(2N)
# purity expectation value of the Hilbert Schmidt distribution is 2/(N), N is the dimension of the density matrix

def test_bures_random_state_purity(qubit_circuit):

    N = qubit_circuit.qs.dim
    num_samples = 5000
    purities = []

    for _ in range(num_samples):
        qubit_circuit.bures_random_state()
        rho = qubit_circuit.qs.density_matrix
        purity = np.trace(rho @ rho)
        purities.append(purity)

    mean_purity = np.mean(purities)
    expected_purity = ((5 * (N**2)) + 1) / (2 * N * ((N**2) + 2))                # for high N: 5 / (2 * N)
    # making interval, because the purity of the Bures distribution is not exactly 5 / (2 * N), but around it
    # due to the random nature of the state generation
    # we want a tolerance of 0.1 to account for this
    tolerance = 0.005
    # check if the mean purity is within the expected range
    assert np.isclose(mean_purity.real ,expected_purity, atol=tolerance* expected_purity)  # using atol to allow for small numerical errors
    # assert expected_purity_ar[0] <= mean_purity <= expected_purity_ar[1]


def test_hilbert_schmidt_random_state_purity(qubit_circuit):
    N = qubit_circuit.qs.dim
    num_samples = 5000
    purities = []

    for _ in range(num_samples):
        qubit_circuit.hilbert_schmidt_random_state()
        rho = qubit_circuit.qs.density_matrix
        purity = np.trace(rho @ rho)
        purities.append(purity)

    mean_purity = np.mean(purities)
    expected_purity =  (2 * N) / (N**2 + 1)                 # for high N: 2 / N
    # making interval, because the purity of the Hilbert-Schmidt distribution is not exactly 2/N, but around it
    # due to the random nature of the state generation
    # we want a tolerance of 0.1 to account for this
    variance_theory = 0.005* expected_purity
    expected_purity = (expected_purity - variance_theory, expected_purity + variance_theory)
    # check if the mean purity is within the expected range
    assert expected_purity[0] <= mean_purity <= expected_purity[1]

def test_haar_random_state_purity(qubit_circuit):
    N = qubit_circuit.qs.dim
    num_samples = 5000
    purities = []

    for _ in range(num_samples):
        qubit_circuit.haar_random_state()
        rho = qubit_circuit.qs.density_matrix
        purity = np.trace(rho @ rho)
        purities.append(purity)

    mean_purity = np.mean(purities)
    expected_purity = 1
    
    expected_purity = expected_purity 
    
    assert np.isclose(mean_purity, expected_purity)