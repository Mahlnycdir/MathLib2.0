import numpy as np
import pytest
from functools import reduce
from mathlibpkg.physics.qcs import Gates
from mathlibpkg.physics.qcs import Instruction
from mathlibpkg.physics.qcs import QubitSystem

@pytest.fixture
def pure_qubit_system():
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0  # |00⟩
    return QubitSystem(state=state)

@pytest.fixture
def mixed_qubit_system():
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    return QubitSystem(state=rho)

def test_is_mixed_state_true(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    assert instr._is_mixed_state() is True

def test_is_mixed_state_false(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    assert instr._is_mixed_state() is False

def test_run_h_on_pure_state(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    instr._run("H", ["q[0]"])
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 1, 0], dtype=complex)
    assert np.allclose(pure_qubit_system.state_vector, expected)

def test_run_h_on_mixed_state(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    original_rho = mixed_qubit_system.density_matrix.copy()
    instr._run("H", ["q[0]"])
    U = instr.g.gate("H", ["q[0]"])
    expected = U @ original_rho @ np.conjugate(U.T)
    assert np.allclose(mixed_qubit_system.density_matrix, expected)

def test_apply_instruction_single(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    acc = ["H", ["q[0]"]]
    instr.apply_instruction(acc)
    expected = (1 / np.sqrt(2)) * np.array([1, 0, 1, 0], dtype=complex)
    assert np.allclose(pure_qubit_system.state_vector, expected)

def test_apply_instruction_sequence(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    instructions = [
        ("H", ["q[0]"]),
        ("CNOT", ["q[0]", "q[1]"])
    ]
    reduce(instr.apply_instruction, instructions)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    assert np.allclose(pure_qubit_system.state_vector, expected)

def test_invalid_gate_name(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    with pytest.raises(KeyError):
        instr._run("DOES_NOT_EXIST", ["q[0]"])

def test_invalid_qubit_label(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    with pytest.raises(KeyError):
        instr._run("H", ["q[99]"])

def test_gate_with_parameters():
    qsys = QubitSystem(n = 2)
    instr = Instruction(qsys)
    g = Gates(qsys)
    instr._run("RX", ["q[0]"], [np.pi])
    print()
    RX = g.gate("RX", ["q[0]"], [np.pi])
    state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    print(RX @ state)
    print("___________________________________________")
    print(RX @ qsys.state_vector)
    expected = np.array([0, 0, -1j, 0], dtype=complex)
    assert np.allclose(qsys.state_vector, expected, atol=1e-8)

def test_gate_getH(pure_qubit_system):
    instr = Instruction(pure_qubit_system)
    H = instr.g.gate("H", ["q[0]"])
    H_dagger = instr.g.getH("H", ["q[0]"])
    identity = H_dagger @ H
    assert np.allclose(identity, np.eye(4, dtype=complex))

def test_gate_getH_mixed_state(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    H = instr.g.gate("H", ["q[0]"])
    H_dagger = instr.g.getH("H", ["q[0]"])
    identity = H_dagger @ H
    assert np.allclose(identity, np.eye(4, dtype=complex))
    # Check that the mixed state is still valid after applying the gate
    assert mixed_qubit_system.density_matrix.shape == (4, 4)

def test_apply_instruction_mixed_state(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    g = Gates(mixed_qubit_system)
    
    mixed_qubit_system.reset_state()  # Reset to initial mixed state
    initial_rho = mixed_qubit_system.density_matrix.copy()  # <- Save BEFORE applying the instruction

    acc = ["H", ["q[0]"]]
    instr.apply_instruction(acc)

    U = g.gate("H", ["q[0]"])
    expected = U @ initial_rho @ U.conj().T

    assert np.allclose(mixed_qubit_system.density_matrix, expected)
def test_apply_instruction_mixed_state_sequence(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    g = Gates(mixed_qubit_system)

    mixed_qubit_system.reset_state()  # Reset to initial mixed state
    initial_rho = mixed_qubit_system.density_matrix.copy()  # <- Save BEFORE applying the instruction

    instructions = [
        ("H", ["q[0]"]),
        ("CNOT", ["q[0]", "q[1]"])
    ]
    reduce(instr.apply_instruction, instructions)

    U1 = g.gate("H", ["q[0]"])
    U2 = g.gate("CNOT", ["q[0]", "q[1]"])
    expected = U2 @ (U1 @ initial_rho @ U1.conj().T) @ U2.conj().T

    assert np.allclose(mixed_qubit_system.density_matrix, expected)

def test_apply_instruction_mixed_state_invalid_gate(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    with pytest.raises(KeyError):
        instr._run("DOES_NOT_EXIST", ["q[0]"])

def test_apply_instruction_mixed_state_invalid_qubit(mixed_qubit_system):
    instr = Instruction(mixed_qubit_system)
    with pytest.raises(KeyError):
        instr._run("H", ["q[99]"])

def test_apply_instruction_mixed_state_gate_with_parameters():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)
    g = Gates(qsys)
    instr._run("RX", ["q[0]"], [np.pi])
    RX = g.gate("RX", ["q[0]"], [np.pi])
    state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    expected = np.array([0, 0, -1j, 0], dtype=complex)
    assert np.allclose(RX @ state, expected, atol=1e-8)

def test_apply_instruction_mixed_state_gate_getH():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)
    H = instr.g.gate("H", ["q[0]"])
    H_dagger = instr.g.getH("H", ["q[0]"])
    identity = H_dagger @ H
    assert np.allclose(identity, np.eye(4, dtype=complex))
    # Check that the mixed state is still valid after applying the gate
    assert qsys.density_matrix.shape == (4, 4)

def test_apply_instruction_mixed_state_gate_getH_mixed_state():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)
    H = instr.g.gate("H", ["q[0]"])
    H_dagger = instr.g.getH("H", ["q[0]"])
    identity = H_dagger @ H
    assert np.allclose(identity, np.eye(4, dtype=complex))
    # Check that the mixed state is still valid after applying the gate
    assert qsys.density_matrix.shape == (4, 4)

def test_apply_instruction_mixed_state_gate_getH_invalid_gate():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)
    with pytest.raises(KeyError):
        instr.g.getH("DOES_NOT_EXIST", ["q[0]"])

def test_apply_instruction_mixed_state_gate_getH_invalid_qubit():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)
    with pytest.raises(KeyError):
        instr.g.getH("H", ["q[99]"])

def test_nonparametric_gate_with_params_raises():
    qsys = QubitSystem(n=2)
    instr = Instruction(qsys)

    with pytest.raises(ValueError, match=f"Gate 'H' has no parameters."):
        instr.g.gate("H", ["q[0]"], [np.pi])
