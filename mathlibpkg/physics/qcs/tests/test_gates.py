import numpy as np
import pytest
from mathlibpkg.physics.qcs import Gates, Channels
from mathlibpkg.physics.qcs import QubitSystem

@pytest.fixture
def two_qubit_system():
    return QubitSystem(n=2)

def test_I_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    I = np.eye(4, dtype=complex)
    assert np.allclose(g.gate("I", ["q[0]"]), I)

def test_Z_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    Z = np.array([[1, 0], [0, -1]], complex)
    expected = np.kron(Z, np.eye(2))
    assert np.allclose(g.gate("Z", ["q[0]"]), expected)

def test_Y_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    expected = np.kron(Y, np.eye(2))
    assert np.allclose(g.gate("Y", ["q[0]"]), expected)

def test_X_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    expected = np.kron(np.array([[0, 1], [1, 0]], complex), np.eye(2))
    assert np.allclose(g.gate("X", ["q[0]"]), expected)

def test_H_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], complex)
    expected = np.kron(H, np.eye(2))
    assert np.allclose(g.gate("H", ["q[0]"]), expected)

def test_RX_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    theta = np.pi / 2
    RX = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                   [-1j*np.sin(theta/2), np.cos(theta/2)]], complex)
    expected = np.kron(RX, np.eye(2))
    assert np.allclose(g.gate("RX", ["q[0]"], [theta]), expected)

def test_RZ_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    theta = np.pi
    RZ = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]], complex)
    expected = np.kron(RZ, np.eye(2))
    assert np.allclose(g.gate("RZ", ["q[0]"], [theta]), expected)

def test_U_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    a, b, c, d = 0.1, 0.2, 0.3, 0.4
    U = g.gate("U", ["q[0]"], [a, b, c, d])
    assert U.shape == (4, 4)
    assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-12)

def test_CNOT_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("CNOT", ["q[0]", "q[1]"])
    assert np.allclose(U @ U.conj().T, np.eye(4))

def test_SWAP_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("SWAP", ["q[0]", "q[1]"])
    assert np.allclose(U @ U, np.eye(4))

def test_ISWAP_gate(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    assert np.allclose(U @ U.conj().T, np.eye(4))

def test_getH(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("H", ["q[0]"])
    H = g.getH("H", ["q[0]"])
    assert np.allclose(H, U.conj().T)

def test_hadamard_unitarity(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("H", ["q[0]"])
    I = np.eye(4, dtype=complex)
    assert np.allclose(U @ U.conj().T, I)

def test_x_squared_identity(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("X", ["q[1]"])
    I = np.eye(4, dtype=complex)
    assert np.allclose(U @ U, I)

def test_swap_squared_identity(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("SWAP", ["q[0]", "q[1]"])
    I = np.eye(4, dtype=complex)
    assert np.allclose(U @ U, I)

def test_rx_gate_unitarity(two_qubit_system):
    g = Gates(two_qubit_system)
    theta = np.pi / 3
    U = g.gate("RX", ["q[0]"], [theta])
    assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex))

def test_getH_matches_manual_conjugate(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("H", ["q[1]"])
    UH = g.getH("H", ["q[1]"])
    assert np.allclose(UH, U.conj().T)

def test_two_qubit_gate_dimension(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("CNOT", ["q[0]", "q[1]"])
    assert U.shape == (4, 4)

def test_invalid_gate_name(two_qubit_system):
    g = Gates(two_qubit_system)
    with pytest.raises(KeyError):
        g.gate("INVALID", ["q[0]"])

def test_wrong_qubit_label(two_qubit_system):
    g = Gates(two_qubit_system)
    with pytest.raises(KeyError):
        g.gate("X", ["q[3]"])

def test_ry_gate_rotation(two_qubit_system):
    g = Gates(two_qubit_system)
    theta = np.pi / 2
    RY = g.gate("RY", ["q[0]"], [theta])
    assert np.allclose(RY @ RY.conj().T, np.eye(4, dtype=complex))

def test_general_u_gate_unitarity(two_qubit_system):
    g = Gates(two_qubit_system)
    alpha, beta, gamma, delta = np.pi/2, np.pi/3, np.pi/4, np.pi/5
    U = g.gate("U", ["q[0]"], [alpha, beta, gamma, delta])
    assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex))

def test_iswap_gate_unitarity(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex))
    
def test_iswap_gate_properties(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    assert U.shape == (4, 4)
    assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex)) 


def test_iswap_gate_truth_table(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    basis = np.eye(4, dtype=complex)
    outs = U @ basis
    truth = np.array([[1, 0, 0, 0],   # |00>
                      [0, 0, 1j, 0], # |01> → i|10>
                      [0, 1j, 0, 0], # |10> → i|01>
                      [0, 0, 0, 1]]) # |11>
    assert np.allclose(outs, truth)

def test_iswap_gate_phase(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    ket01 = np.array([0, 1, 0, 0], complex)
    ket10 = np.array([0, 0, 1, 0], complex)
    assert np.allclose(U @ ket01, 1j * ket10)
    assert np.allclose(U @ ket10, 1j * ket01)

def test_iswap_gate_inverse(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    U_inv = g.getH("ISWAP", ["q[0]", "q[1]"])
    assert np.allclose(U @ U_inv, np.eye(4, dtype=complex))
    assert np.allclose(U_inv @ U, np.eye(4, dtype=complex))

def test_iswap_gate_inverse_properties(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    U_inv = g.getH("ISWAP", ["q[0]", "q[1]"])
    
    # Check if the inverse gate is correct
    assert np.allclose(U @ U_inv, np.eye(4, dtype=complex))
    assert np.allclose(U_inv @ U, np.eye(4, dtype=complex))
    
    # Check if the inverse gate has the same shape
    assert U_inv.shape == (4, 4)
def test_iswap_gate_inverse_unitarity(two_qubit_system):
    g = Gates(two_qubit_system)
    U = g.gate("ISWAP", ["q[0]", "q[1]"])
    U_inv = g.getH("ISWAP", ["q[0]", "q[1]"])
    
    # Check if the inverse gate is unitary
    assert np.allclose(U_inv @ U_inv.conj().T, np.eye(4, dtype=complex))
    assert np.allclose(U @ U.conj().T, np.eye(4, dtype=complex)) 

# Test Channels
def test_bit_flip_channel():
    rho = np.array([[1, 0], [0, 0]], complex)  # |0><0|
    qs = QubitSystem(state=rho)
    g = Channels(qs)
    p = 0.5
    new_rho = g.bit_flip_channel(p)
    X = np.array([[0, 1], [1, 0]], complex)  # Pauli-X gate
    expected_rho = (1 - p) * rho + p * X @ rho @ X
    assert np.allclose(new_rho, expected_rho)

def test_phase_flip_channel():
    rho = np.array([[1, 0], [0, 0]], complex)  # |0><0|
    qs = QubitSystem(state=rho)
    g = Channels(qs)
    p = 0.5
    new_rho = g.phase_flip_channel(place = 0, p=p)
    Z = np.array([[1, 0], [0, -1]], complex)  # Pauli-Z gate
    expected_rho = (1 - p) * rho + p * Z @ rho @ Z
    assert np.allclose(new_rho, expected_rho)

def test_amplitude_damping_channel():
    rho = np.array([[1, 0], [0, 0]], complex)  # |0><0|
    qs = QubitSystem(state=rho)
    g = Channels(qs)
    gamma = 0.5
    new_rho = g.amplitude_damping_channel(gamma)
    expected_rho = rho
    assert np.allclose(new_rho, expected_rho)

def test_depolarizing_channel():
    rho = np.array([[1, 0], [0, 0]], complex)  # |0><0|
    qs = QubitSystem(state=rho)
    g = Channels(qs)
    p = 0.5
    new_rho = g.depolarizing_channel(p)
    expected_rho = (1 - p) * rho + p * np.eye(2, dtype=complex) / 2
    assert np.allclose(new_rho, expected_rho)

def test_depolarizing_channel_for_high_qubit_number():
    rho = np.eye(4, dtype=complex) / 4  # Mixed state
    qs = QubitSystem(state=rho)
    g = Channels(qs)
    p = 0.5
    new_rho = g.depolarizing_channel(p)
    expected_rho = (1 - p) * rho + p * np.eye(4, dtype=complex) / 4
    assert np.allclose(new_rho, expected_rho)

def test_depolarizing_channel_for_high_qubit_number_2():
    qs = QubitSystem(n=4) # Initialize a 4-qubit system with 0000 state
    g = Channels(qs)
    p = 0.5
    expected_rho = (1 - p) * qs.density_matrix + p * np.eye(16, dtype=complex) / 16
    g.depolarizing_channel(p)
    assert np.allclose(qs.density_matrix, expected_rho)