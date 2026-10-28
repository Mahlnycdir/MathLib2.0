import numpy as np
import pytest
from mathlibpkg.physics.qcs import Gates
from mathlibpkg.physics.qcs import Measurement
from mathlibpkg.physics.qcs import QubitSystem

@pytest.fixture
def two_qubit_system():
    return QubitSystem(n=5)

def test_measurement(two_qubit_system):
    m = Measurement()
    g = Gates(two_qubit_system)

    # Prepare a state |00...0>
    state = np.zeros(2**5, dtype=complex)
    state[0] = 1.0

    # Measure the state
    outcomes = m.measure(state, num_shots=1000, return_binary=True)

    # Check that the outcome is always '00000'
    assert all(outcome == '00000' for outcome in outcomes)

    # Apply a Hadamard gate to the first qubit
    H = g.gate("H", ["q[0]"])
    state = H @ state

    # Measure the new state
    outcomes = m.measure(state, num_shots=1000, return_binary=True)

    # Check that the outcomes are uniformly distributed between '00000' and '10000'
    assert sum(outcome == '00000' for outcome in outcomes) > 400
    assert sum(outcome == '10000' for outcome in outcomes) > 400

def test_measurement_mixed_state(two_qubit_system):
    m = Measurement()
    g = Gates(two_qubit_system)

    # Prepare a mixed state (density matrix)
    state_vector = np.array([1, 0, 0, 0, 0], dtype=complex)  # |00000>
    density_matrix = np.outer(state_vector, state_vector.conjugate())

    # Measure the mixed state
    outcomes = m.measure(density_matrix, num_shots=1000, return_binary=True)

    # Check that the outcome is always '00000'
    assert all(outcome == '00' for outcome in outcomes)

def test_measurement_zero_state(two_qubit_system):
    m = Measurement()

    # Test with a zero state vector
    zero_state = np.zeros(2**5, dtype=complex)
    zero_state[0] = 1.0  # |00000>
    outcomes = m.measure(zero_state, num_shots=1000, return_binary=True)

    # Check that the outcomes are all '00000' since the state is zero
    assert all(outcome == '00000' for outcome in outcomes)

def test_measurement_single_shot(two_qubit_system):
    m = Measurement()
    g = Gates(two_qubit_system)

    # Prepare a state |00...0>
    state = np.zeros(2**5, dtype=complex)
    state[0] = 1.0

    # Measure the state with a single shot
    outcome = m.measure(state, num_shots=1, return_binary=True)

    # Check that the outcome is '00000'
    assert outcome[0] == '00000'

    # Apply a Hadamard gate to the first qubit
    H = g.gate("H", ["q[0]"])
    state = H @ state

    # Measure the new state with a single shot
    outcome = m.measure(state, num_shots=1, return_binary=True)

    # Check that the outcome is either '00000' or '10000'
    assert outcome[0] in ['00000', '10000']

def test_measurement_state(two_qubit_system):
    m = Measurement()
    g = Gates(two_qubit_system)

    # Prepare a state |00...0>
    state = np.zeros(2**5, dtype=complex)
    state[0] = 1.0

    # Measure the state
    outcomes = m.measure(state, num_shots=1000, return_binary=True)

    # Check that the outcome is always '00000'
    assert all(outcome == '00000' for outcome in outcomes)

    # Apply a Hadamard gate to the first qubit
    H = g.gate("H", ["q[0]"])
    state = H @ state

    # Measure the new state
    outcomes = m.measure(state, num_shots=1000, return_binary=True)

    # Check that the outcomes are uniformly distributed between '00000' and '10000'
    assert sum(outcome == '00000' for outcome in outcomes) > 400
    assert sum(outcome == '10000' for outcome in outcomes) > 400

def test_measurement_invalid_state(two_qubit_system):
    m = Measurement()

    # Test with an invalid state (not a 1D or 2D array)
    invalid_state = np.array([[1, 0], [0, 1], [0, 0]], dtype=complex)

    with pytest.raises(ValueError, match="Density matrix must be square."):
        m.measure(invalid_state, num_shots=1000, return_binary=True)
def test_measurement_invalid_dimension(two_qubit_system):
    m = Measurement()

    # Test with a state that is not 1D or 2D
    invalid_state = np.array([[[1, 0], [0, 1]]], dtype=complex)  # 3D array

    with pytest.raises(ValueError, match="state must be a 1D or 2D array."):
        m.measure(invalid_state, num_shots=1000, return_binary=True)

def test_measurement_invalid_num_shots(two_qubit_system):
    m = Measurement()

    # Test with an invalid number of shots
    state = np.zeros(2**5, dtype=complex)
    state[0] = 1.0  # |00000>

    with pytest.raises(ValueError, match="num_shots must be a positive integer."):
        m.measure(state, num_shots=0, return_binary=True)

def test_measurement_invalid_return_binary(two_qubit_system):
    m = Measurement()

    # Prepare a state |00...0>
    state = np.zeros(2**5, dtype=complex)
    state[0] = 1.0  # |00000>

    # Measure the state with an invalid return_binary value
    with pytest.raises(ValueError, match="return_binary must be a boolean."):
        m.measure(state, num_shots=1000, return_binary="invalid")


    