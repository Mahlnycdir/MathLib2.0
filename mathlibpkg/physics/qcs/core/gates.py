import numpy as np
from functools import reduce
from typing import Dict, Callable, Sequence, Any
import numpy as np
import re
from scipy.linalg import expm
from .system import QubitSystem



class Gates:
    """
    Provides 1-qubit, parametric 1-qubit, and 2-qubit gates.
    Generates a (2**n × 2**n) matrix for the specified qubit
    or qubit pair of the associated QubitSystem object upon request.
    """

    # ------------------------------------------------------------------
    # Constructor: Gate dictionaries 
    # ------------------------------------------------------------------
    def __init__(self, qubit_system : QubitSystem):
        self.qubit_system = qubit_system

        self.one_qubit_gates: Dict[str, np.ndarray] = {}
        self.one_parametric_gates: Dict[str, Callable[..., np.ndarray]] = {}
        self.two_qubit_gates: Dict[str, Callable[..., np.ndarray]] = {}

        self._build_one_qubit_gates()
        self._build_one_parametric_gates()
        self._build_two_qubit_gates()

    # ------------------------------------------------------------------
    # Helpfunctions
    # ------------------------------------------------------------------
    @staticmethod
    def _kron_all(ops):
        '''
        Kroneckerprodukt aller Matrizen in ops.
        '''
        return reduce(np.kron, ops)

    @staticmethod
    def _parse_label(label: str):
        '''q[3]  →  3   : Value error for wrong format.'''
        m = re.fullmatch(r"q\[(\d+)]", label.replace(" ", ""))
        if not m:
            raise ValueError(
                f"Invalid qubit identifier '{label}' (expected 'q[<int>]')"
            )
        return int(m.group(1))

    # ------------------------------------------------------------------
    # Filling Gate-Dictionaries
    # ------------------------------------------------------------------
    def _build_one_qubit_gates(self):
        I = np.array([[1, 0], [0, 1]], complex)
        X = np.array([[0, 1], [1, 0]], complex)
        Y = np.array([[0, -1j], [1j, 0]], complex)
        Z = np.array([[1, 0], [0, -1]], complex)
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], complex)
        S = np.array([[1, 0], [0, 1j]], complex)
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], complex)

        self.one_qubit_gates.update(
            {"I": I, "X": X, "Y": Y, "Z": Z, "H": H, "S": S, "T": T}
        )

    def _build_one_parametric_gates(self):
        def RX(theta: float):
            return np.array(
                [[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                 [-1j * np.sin(theta / 2), np.cos(theta / 2)]],
                complex,
            )

        def RY(theta: float):
            return np.array(
                [[np.cos(theta / 2), -np.sin(theta / 2)],
                 [np.sin(theta / 2), np.cos(theta / 2)]],
                complex,
            )

        def RZ(theta: float):
            return np.array(
                [[np.exp(-1j * theta / 2), 0],
                 [0, np.exp(1j * theta / 2)]],
                complex,
            )

        def U(alpha, beta, gamma, delta = 0.0):
            return np.exp(1j * delta) * RZ(gamma) @ RY(beta) @ RX(alpha)

        self.one_parametric_gates.update({"RX": RX, "RY": RY, "RZ": RZ, "U": U})

    def _build_two_qubit_gates(self):
        # -------- Projections and Pauli-X --------
        P0 = np.array([[1, 0], [0, 0]], complex)
        P1 = np.array([[0, 0], [0, 1]], complex)
        X = np.array([[0, 1], [1, 0]], complex)
        I = np.eye(2, dtype=complex)

        def CNOT(ctrl: str, tgt: str):
            ops0, ops1 = [], []
            for label in self.qubit_system.single_qubit_states.keys():
                if label == ctrl:
                    ops0.append(P0)
                    ops1.append(P1)
                elif label == tgt:
                    ops0.append(I)
                    ops1.append(X)
                else:
                    ops0.append(I)
                    ops1.append(I)
            return self._kron_all(ops0) + self._kron_all(ops1)

        def ISWAP(q1: str, q2: str):
            n = self.qubit_system.n
            dim = 2 ** n
            labels = list(self.qubit_system.single_qubit_states.keys())
            i1, i2 = labels.index(q1), labels.index(q2)

            mat = np.zeros((dim, dim), complex)
            for i in range(dim):
                bits = list(map(int, f"{i:0{n}b}"))
                jbits = bits.copy()
                jbits[i1], jbits[i2] = jbits[i2], jbits[i1]
                j = int("".join(map(str, jbits)), 2)
                if bits[i1] != bits[i2]:
                    mat[j, i] = 1j
                else:
                    mat[i, i] = 1
            return mat

        def SWAP(q1: str, q2: str):
            return CNOT(q1, q2) @ CNOT(q2, q1) @ CNOT(q1, q2)

        self.two_qubit_gates.update({"CNOT": CNOT, "ISWAP": ISWAP, "SWAP": SWAP})

    # ------------------------------------------------------------------
    # Central factory-methode
    # ------------------------------------------------------------------
    def gate(
        self,
        name: str,
        qubits: Sequence[str],
        params: Sequence[Any] | None = None,
    ):
        '''Returns the full (2**n × 2**n) matrix of the desired gate.'''
        n = self.qubit_system.n
        dim = self.qubit_system.dim
        I2 = np.eye(2, dtype=complex)

        # -------------------- 1-Qubit-gates --------------------
        if name in self.one_qubit_gates:
            if len(qubits) != 1:
                raise ValueError(f"Gate '{name}' provides exactly 1 Qubit, got {qubits}")
            if params:
                raise ValueError(f"Gate '{name}' has no parameters.")

            target = qubits[0]
            if target not in self.qubit_system.single_qubit_states:
                raise KeyError(f"{target} don't exist in the system")

            ops = [
                self.one_qubit_gates[name] if label == target else I2
                for label in self.qubit_system.single_qubit_states.keys()
            ]
            U = self._kron_all(ops)
            assert U.shape == (dim, dim)
            return U

        # ------------ Parametric 1-Qubit-gates --------------
        if name in self.one_parametric_gates:
            if len(qubits) != 1:
                raise ValueError(f"Gate '{name}' requires exactly 1 qubit, got {qubits}")

            needed = 4 if name == "U" else 1
            if not params or len(params) != needed:
                raise ValueError(
                    f"Gate '{name}' expects {needed} parameters, got {params}"
                )

            target = qubits[0]
            if target not in self.qubit_system.single_qubit_states:
                raise KeyError(f"{target} does not exist in the system")

            base = self.one_parametric_gates[name](*params)
            ops = [
                base if label == target else I2
                for label in self.qubit_system.single_qubit_states.keys()
            ]
            U = self._kron_all(ops)
            assert U.shape == (dim, dim)
            return U

        # ------------------ 2-Qubit-gates ----------------------
        if name in self.two_qubit_gates:
            if len(qubits) != 2:
                raise ValueError(f"Gate '{name}' requires 2 qubits, got {qubits}")

            q1, q2 = qubits
            if (
                q1 not in self.qubit_system.single_qubit_states
                or q2 not in self.qubit_system.single_qubit_states
            ):
                raise KeyError(f"{q1} or {q2} does not exist in the system")

            U = self.two_qubit_gates[name](q1, q2)
            if U.shape != (dim, dim):
                raise ValueError(
                    f"2-qubit gate '{name}' returned a matrix of shape {U.shape}, "
                    f"expected ({dim}, {dim})"
                )
            return U

        # ------------------ Unkown gate -------------------
        raise KeyError(f"Unkown gate '{name}'")
    
    def getH(self, name: str, qubits: Sequence[str]):
        """
        Returns the conjugate transpose (Hermitian adjoint) of the gate.
        """
        U = self.gate(name, qubits)
        return np.conjugate(U.T)
    
class Channels:
    """
    Provides quantum channels for the QubitSystem.
    Generates a (2**n × 2**n) matrix for the specified qubit
    or qubit pair of the associated QubitSystem object upon request.
    """

    def __init__(self, qubit_system : QubitSystem):
        self.qubit_system = qubit_system
    
    def _kron_all(self, ops):
        """
        Computes the Kronecker product of all matrices in ops.
        """
        return reduce(np.kron, ops)

    # ------------------------------------------------------------------
    # Channels 
    # ------------------------------------------------------------------
    def bit_flip_channel(self, p, place = 0):
        """
        Applies a bit flip channel to the qubit system's density matrix.
        The bit flip channel applies the Pauli-X gate with probability p.
        Args:
            p (float): Probability of applying the bit flip (0 <= p <= 1).
        """
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be in the range [0, 1].")
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        E0 = np.sqrt(1 - p) * I
        E1 = np.sqrt(p) * X

        num_qubits = int(np.log2(self.qubit_system.density_matrix.shape[0]))

        if num_qubits == 1 and place != 0:
            raise ValueError("For a 1-qubit system, place must be 0.")
        if place < 0 or place >= num_qubits:
            raise ValueError(f"Invalid qubit index {place} for {num_qubits}-qubit system.")

        E0_dim = self._kron_all([E0 if i == place else I for i in range(num_qubits)])
        E1_dim = self._kron_all([E1 if i == place else I for i in range(num_qubits)])

        new_density = (E0_dim @ self.qubit_system.density_matrix @ E0_dim.conj().T +
                    E1_dim @ self.qubit_system.density_matrix @ E1_dim.conj().T)
        
        self.qubit_system.update_system_state(new_density)
        return new_density
        # for n-th -> IxIx...xE_o x I .... rho x .... 

    def phase_flip_channel(self, place : int = 0, p: float = 0):
        """
        Applies a phase flip channel to the qubit system's density matrix.
        The phase flip channel applies the Pauli-Z gate with probability p.
        Args:
            place (int): Qubit index to apply the noise to.
            p (float): Probability of applying the phase flip (0 <= p <= 1).
        """
        if not (0 <= p <= 1):
            raise ValueError("Noise level p must be in the range [0, 1].")
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        E0 = np.sqrt(1 - p) * I
        E1 = np.sqrt(p) * Z

        num_qubits = int(np.log2(self.qubit_system.density_matrix.shape[0]))

        if num_qubits == 1 and place != 0:
            raise ValueError("For a 1-qubit system, place must be 0.")
        if place < 0 or place >= num_qubits:
            raise ValueError(f"Invalid qubit index {place} for {num_qubits}-qubit system.")

        E0_dim = self._kron_all([E0 if i == place else I for i in range(num_qubits)])
        E1_dim = self._kron_all([E1 if i == place else I for i in range(num_qubits)])

        new_density = (E0_dim @ self.qubit_system.density_matrix @ E0_dim.conj().T +
                    E1_dim @ self.qubit_system.density_matrix @ E1_dim.conj().T)
        
        self.qubit_system.update_system_state(new_density)
        return new_density

    
    def amplitude_damping_channel(self, p, place = 0):
        if not (0 <= p <= 1):
            raise ValueError("Noise level p must be in the range [0, 1].")
        I = np.eye(2, dtype=complex)
        E0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
        E1 = np.array([[0, np.sqrt(p)], [0, 0]])
        num_qubits = int(np.log2(self.qubit_system.density_matrix.shape[0]))

        if num_qubits == 1 and place != 0:
            raise ValueError("For a 1-qubit system, place must be 0.")
        if place < 0 or place >= num_qubits:
            raise ValueError(f"Invalid qubit index {place} for {num_qubits}-qubit system.")

        E0_dim = self._kron_all([E0 if i == place else I for i in range(num_qubits)])
        E1_dim = self._kron_all([E1 if i == place else I for i in range(num_qubits)])

        new_density = (E0_dim @ self.qubit_system.density_matrix @ E0_dim.conj().T +
                    E1_dim @ self.qubit_system.density_matrix @ E1_dim.conj().T)
        
        self.qubit_system.update_system_state(new_density)
        return new_density
    
    def depolarizing_channel(self, p):
        """
        Applies a depolarizing channel to the qubit system's density matrix.
        The depolarizing channel replaces the density matrix with a uniform mixture
        of the identity matrix with probability p.
        Args:
            p (float): Probability of applying the depolarizing channel (0 <= p <= 1).
        """
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be in the range [0, 1].")
        d = self.qubit_system.density_matrix.shape[0]
        identity = np.eye(d)
        new_density =  (1 - p) * self.qubit_system.density_matrix + p * identity / d
        self.qubit_system.update_system_state(new_density)
        return new_density
    
    def apply_channel(self, channel_name: str, noise_level: float, place = 0):
        """
        Applies the specified quantum channel to the qubit system's density matrix.
        Args:
            channel_name (str): Name of the quantum channel to apply.
            *args: Additional arguments for the channel function.
        Returns:
            np.ndarray: Updated density matrix after applying the channel.
        """
        if channel_name == "bit_flip":
            self.bit_flip_channel(p = noise_level, place = place)
        elif channel_name == "phase_flip":
            self.phase_flip_channel(place = place,  p = noise_level)
        elif channel_name == "amplitude_damping":
            self.amplitude_damping_channel(place = place,  p = noise_level)
        elif channel_name == "depolarizing":
            self.depolarizing_channel(noise_level)
        else:
            raise ValueError(f"Unknown channel: {channel_name}")
    
    