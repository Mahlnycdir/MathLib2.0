import numpy as np
import pytest
from mathlibpkg.physics.qcs import QubitSystem
from mathlibpkg.physics.qcs import Gates
from mathlibpkg.physics.qcs import Instruction
from functools import reduce

@pytest.fixture
def two_qubit_system():
    return QubitSystem(n=4)

def test_qubit_system_initialization(two_qubit_system):
    # Check if the system is initialized with the correct number of qubits
    assert two_qubit_system.n == 4
    assert two_qubit_system.state_vector.shape == (16,)

    # Check if the initial state is |0000>
    expected_state = np.zeros(16, dtype=complex)
    expected_state[0] = 1.0
    assert np.allclose(two_qubit_system.state_vector, expected_state)

def test_apply_single_qubit_gate(two_qubit_system):
    # Apply a Hadamard gate to the first qubit
    g = Gates(two_qubit_system)
    two_qubit_system.reset_state()  # Reset to |0000>
    #print ("Initial state:", two_qubit_system.state_vector)
    ins = Instruction(two_qubit_system)
    ins.apply_instruction(["H", ["q[0]"]])
    #print ("State after applying H gate:", two_qubit_system.state_vector)
    # Startzustand |00⟩ → Basiszustand 0 im 2-Qubit-System (Länge 16 bei 4 Qubits?)
    init = np.zeros(16, dtype=complex)
    init[0] = 1.0

    # Hole Hadamard-Gate für Qubit 0
    H = g.gate("H", ["q[0]"])

    # Erwarteter Zustand: Hadamard auf Qubit 0 angewendet auf |0000>
    expected_state = H @ init
    #print ("Expected state after applying H gate:", expected_state)
    assert np.allclose(two_qubit_system.state_vector, expected_state)

def test_apply_two_qubit_gate(two_qubit_system):
    # Apply a CNOT gate with q[0] (in state 0) as control and q[1] (in state 0) as target 
    g = Gates(two_qubit_system)
    ins = Instruction(two_qubit_system)
    instruction1 = ["CNOT", ["q[0]", "q[1]"]]
    ins.apply_instruction(instruction1)

    assert np.allclose(two_qubit_system.state_vector, np.array([1, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0], dtype=complex))
    
    two_qubit_system.reset_state()  # Reset to |0000>
    init = np.zeros(16, dtype=complex)
    init[0] = 1.0

    # Hole Hadamard-Gate für Qubit 0
    H = g.gate("H", ["q[0]"])
    # Erwarteter Zustand: Hadamard auf Qubit 0 angewendet auf |0000>
    expected_state = H @ init
    instruction2 = ["H", ["q[0]"]]
    ins.apply_instruction(instruction2)
    
    assert np.allclose(two_qubit_system.state_vector, expected_state)

def test_apply_instruction_sequence(two_qubit_system):
    ins = Instruction(two_qubit_system)
    g = Gates(two_qubit_system)
    two_qubit_system.reset_state()  # Reset to |0000>
    instructions = [
        ("H", ["q[0]"]),
        ("CNOT", ["q[0]", "q[1]"]),
        ("X", ["q[2]"]),
        ("CNOT", ["q[1]", "q[2]"])
    ]
    
    # Apply the sequence of instructions
    reduce(ins.apply_instruction, instructions, two_qubit_system)
    init = np.zeros(16, dtype=complex)
    init[0] = 1.0

    H = g.gate("H", ["q[0]"])
    CNOT_01 = g.gate("CNOT", ["q[0]", "q[1]"])
    X_q2 = g.gate("X", ["q[2]"])
    CNOT_12 = g.gate("CNOT", ["q[1]", "q[2]"])
    # Apply the gates to the initial state
    state_after_H = H @ init
    state_after_CNOT_01 = CNOT_01 @ state_after_H
    state_after_X_q2 = X_q2 @ state_after_CNOT_01
    state_after_CNOT_12 = CNOT_12 @ state_after_X_q2

    # Expected state after applying the sequence
    expected_state = state_after_CNOT_12
    
    assert np.allclose(two_qubit_system.state_vector, expected_state)

def test_init_with_qubit_number():
    qs = QubitSystem(n=2)
    assert qs.n == 2
    assert qs.dim == 4
    assert np.allclose(qs.state_vector, np.array([1, 0, 0, 0], dtype=complex))

def test_init_with_valid_state_vector():
    psi = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)
    qs = QubitSystem(state=psi)
    assert np.allclose(qs.state_vector, psi)
    assert np.allclose(qs.density_matrix, np.outer(psi, psi.conj()))

def test_init_with_density_matrix():
    rho = np.eye(4) / 4
    qs = QubitSystem(state=rho)
    assert np.allclose(qs.density_matrix, rho)
    assert qs.state_vector is None

def test_invalid_state_shape_raises():
    with pytest.raises(ValueError):
        QubitSystem(state=np.array([1, 2, 3]))  # not a power of 2 length

def test_change_single_qubit_changes_state():
    qs = QubitSystem(n=2)
    qs.change_single_qubit("q[0]", np.array([0, 1], dtype=complex))
    new_state = qs.state_vector
    expected = np.kron([0, 1], [1, 0])
    assert np.allclose(new_state, expected)

def test_reset_state_resets_to_zero():
    qs = QubitSystem(n=2)
    qs.change_single_qubit("q[1]", np.array([0, 1], dtype=complex))
    qs.reset_state()
    assert np.allclose(qs.state_vector, np.array([1, 0, 0, 0], dtype=complex))

def test_vec_as_dict_for_vector():
    psi = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)
    qs = QubitSystem(state=psi)
    d = qs.vec_as_dict(psi)
    assert isinstance(d, dict)
    assert "00" in d and np.isclose(d["00"], 1/np.sqrt(2))
    assert "10" in d and np.isclose(d["10"], 1/np.sqrt(2))

def test_vec_as_dict_for_density_matrix():
    rho = np.eye(4) / 4
    qs = QubitSystem(state=rho)
    d = qs.vec_as_dict(rho)
    assert "0000" in d and np.isclose(d["0000"], 0.25)
    assert "1111" in d and np.isclose(d["1111"], 0.25)

def test_state_dict_property_matches_state_vector():
    qs = QubitSystem(n=2)
    d = qs.state_dict
    assert d["00"] == 1 + 0j
    assert np.isclose(sum(abs(v)**2 for v in d.values()), 1.0)

def test_n_basis_returns_correct_vector():
    qs = QubitSystem(n=3)
    vec = qs.n_basis(5)
    expected = np.zeros(8, dtype=complex)
    expected[5] = 1.0
    assert np.allclose(vec, expected)

def test_n_basis_out_of_range_raises():
    qs = QubitSystem(n=3)
    with pytest.raises(ValueError):
        qs.n_basis(8)
