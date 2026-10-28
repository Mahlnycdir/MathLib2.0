import numpy as np
from typing import Dict, Callable, Any, Sequence
from functools import reduce
from ..core.system import QubitSystem
from ..core.gates import Gates, Channels
from scipy.sparse import csr_matrix
from scipy.stats import unitary_group
from numpy.linalg import qr, norm, svd
from ..utils.black_box import black_box

class QubitCircuits:
    def __init__(self, qs = None):
        """
        Initializes the QubitCircuits class with a given QubitSystem.
        """
        self.qs = qs
        
    # ---------------------------------------------------------------
    # Helping methods: 1-Qubit-Gate in 2**n × 2**n
    # ---------------------------------------------------------------
    @staticmethod
    def _embed_one(gate2: np.ndarray, q_index: int, n: int):
        eyes = [np.eye(2, dtype=complex) for _ in range(n)]
        eyes[q_index] = gate2
        return csr_matrix(reduce(np.kron, eyes))

    # ---------------------------------------------------------------
    # Main-Routine
    # ---------------------------------------------------------------
    def random_circuit(
        self,
        number_qubits:   int,
        number_columns:  int,
        p_two:           float = 0.30,   # Wahrscheinlichkeit für ein 2-Qubit-Gate
        seed:            int | None = None
    ):

        # ---------- Error-handling ----------
        if number_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        if number_columns < 1:
            raise ValueError("Number of columns must be at least 1.")
        if seed is not None:
            np.random.seed(seed)
        if self.qs is None:
            qsys = QubitSystem(number_qubits)
            g = Gates(qsys)                         # Gates-Objekt für dieses QubitSystem
        else:
            qsys = self.qs
            g  = Gates(qsys)                         # Gates-Objekt für dieses QubitSystem
        # qs.single_qubits()                    # sorgt für Namen "q[0]", "q[1]", …

        state = qsys.state_vector.copy()
        dim   = qsys.dim

        # Pools vorbereiten
        one_simple   = list(g.one_qubit_gates.keys())
        one_param    = list(g.one_parametric_gates.keys())
        two_qubit    = ["CNOT", "SWAP", "ISWAP"]

        self._columns: list[np.ndarray] = []  # optional für Tests / Debug

        # ---------- Spaltenweise ----------
        for _ in range(number_columns):
            free = list(range(number_qubits))       # noch unverplante Qubits
            col  = csr_matrix(np.eye(dim, dtype=complex))       # Startmatrix dieser Spalte

            while free:
                use_two = len(free) >= 2 and np.random.rand() < p_two

                # ===== 2-Qubit-Gate ========================================
                if use_two:
                    q1, q2 = np.random.choice(free, size=2, replace=False)
                    g_name  = np.random.choice(two_qubit)

                    if g_name == "CNOT":
                        full = csr_matrix(g.gate("CNOT",[f"q[{q1}]", f"q[{q2}]"]))
                    elif g_name == "SWAP":
                        full = csr_matrix(g.gate("SWAP",[f"q[{q1}]", f"q[{q2}]"]))
                    else:  # ISWAP
                        full = csr_matrix(g.gate("ISWAP",[f"q[{q1}]", f"q[{q2}]"]))

                    free.remove(q1); free.remove(q2)

                # ===== 1-Qubit-Gate ========================================
                else:
                    q      = np.random.choice(free)
                    g_name = np.random.choice(one_simple + one_param)

                    # Parametrische Gates -> Zufallswinkel
                    if g_name in one_param:
                        nargs   = g.one_parametric_gates[g_name].__code__.co_argcount
                        angles  = np.random.uniform(0, 2*np.pi, nargs)
                        gate2   = g.one_parametric_gates[g_name](*angles)
                    else:
                        gate2   = g.one_qubit_gates[g_name]

                    full   = self._embed_one(gate2, q, number_qubits)
                    free.remove(q)

                # Gate an den Anfang der Spalte „kleben“
                col = full @ col

            # Spalte fertig – anwenden & speichern
            self._columns.append(csr_matrix(col))
            state = csr_matrix(col) @ state

        return state
    
    def ginibre_matrix(self):
        """Generate complex square Ginibre matrix with N rows, K columns."""
        N = self.qs.dim
        real = np.random.normal(0, 1, (N, N))
        imag = np.random.normal(0, 1, (N, N))
        return real + 1j * imag
    
    def haar_random_state(self):
        """Generate a Haar random pure state (density matrix form)."""
        N = self.qs.dim 
        psi = np.random.randn(N) + 1j * np.random.randn(N)
        psi /= norm(psi)
        self.qs.update_system_state(np.outer(psi, psi.conj()))    # new rho
    
    def hilbert_schmidt_random_state(self):
        """Generate Hilbert-Schmidt random mixed state."""
        N = self.qs.dim
        G = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        self.qs.update_system_state(rho)

    def bures_random_state(self):
        """Generate Bures random mixed state."""
        N = self.qs.dim
        U = unitary_group.rvs(N)
        G = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
        rho = (np.eye(N) + U) @ G @ G.conj().T @ (np.eye(N) + U.conj().T)
        rho /= np.trace(rho)
        self.qs.update_system_state(rho)

    # ---------------------------------------------------------------
    # Deutsch-Josza algorithm
    # ---------------------------------------------------------------

    # Helper functions
    def indToState(self, n, k):
        num = bin(k)[2:].zfill(n)
        return np.array([int(x) for x in str(num)])

    # not necessarily needed:
    def stateToInd(self, state):
        return int("".join(str(x) for x in state),2)
    

    def Deutsch_Josza_algorithm(self, f, number_qubits, noise_level = None, type = None, place = None):
        '''
        Generalized Deutsch-Josza algorithm implementation, with uses density matrices.
        Args:
            f: A callable that represents the black box function -> a function of x
            number_qubits: The number of qubits in the quantum system -> n+1 qubits, where n is the number of bits in the input x.
            noise_level: Optional noise level to apply to the quantum system.
            type: Optional type of noise to apply to the quantum system. (input: "depolarizing", "amplitude_damping", "bit_flip", "phase_flip")
            place: Optional place to apply the noise to the quantum system. (input: 1,2,3,4)
            
        Returns:
            Bool: True if the function is constant, False if it is balanced.
        '''
        zero = np.array([1,0], dtype=complex)  # |0>
        state_vec = reduce(np.kron, [zero] * number_qubits)
        state = np.outer(state_vec, np.conjugate(state_vec.T)) # |0...0> ⊗ <0...0| -> density matrix form

        qusys = QubitSystem(state=state) # Initialize the quantum system with the state as density matrix

        '''Noise handling'''        
        Noise_array = [noise_level, type, place]  # Store noise parameters for later use

        '''Place 1 of the noise'''
        if Noise_array[0] is not None and Noise_array[1] is not None and Noise_array[2] is not None and Noise_array[2] == 1:
            # Apply noise to the quantum system
            Channels(qusys).apply_channel(channel_name = Noise_array[1], noise_level = Noise_array[0])

        dim = qusys.dim

        # Apply Hadamard gates to all qubits
        # gate for Hadamard gate for n qubits

        g = Gates(qusys)  # Gates object for this QubitSystem
        H_array = []
        for i in range(qusys.n):
            H_array.append(csr_matrix(g.gate("H", [f"q[{i}]"])))
        Hn = reduce(np.kron, H_array)  # Kronecker product to get the Hadamard gate for n qubits

        # H^n @ rho @ H^n†
        qusys.update_system_state(Hn @ qusys.density_matrix @ Hn.conjugate().T)
        '''Vector form of the state after applying Hadamard gates -> check up'''
        # new_state = Hn @ state_vec 

        '''Place 2 of the noise'''
        if Noise_array[0] is not None and Noise_array[1] is not None and Noise_array[2] is not None and Noise_array[2] == 2:
            # Apply noise to the quantum system
            Channels(qusys).apply_channel(noise_level = Noise_array[0], channel_name = Noise_array[1])

        # Apply to each row and each column their appropriate phase (which means that for row one,  each entry gets a factor (-1)^(f(1)) and in column 3 each entry gets a factor 1^(f(3))        
        # This is equivalent to multiplying each element of the density matrix by (-1)^(f(bitstring))
        # where bitstring is the binary representation of the row index.
        density = qusys.density_matrix.copy()  # Copy the density matrix to avoid modifying the original
        for row in range(dim):
            bitstring_row = self.indToState(number_qubits, row)
            phase_row = (-1) ** f(bitstring_row)
            for col in range(dim):
                bitstring_col = self.indToState(number_qubits, col)
                phase_col = (-1) ** f(bitstring_col)
                density[row, col] *= phase_col * phase_row  # Apply the phase to the density matrix
        
        qusys.update_system_state(density)  # Update the quantum system with the new density matrix
        '''Now for the state vector -> Check up'''
        """ for x in range(dim):
            bitstring = self.indToState(number_qubits, x)
            fx = f(bitstring)
            new_state[x] *= (-1) ** fx """

        '''Place 3 of the noise'''
        if Noise_array[0] is not None and Noise_array[1] is not None and Noise_array[2] is not None and Noise_array[2] == 3:
            # Apply noise to the quantum system
            Channels(qusys).apply_channel(noise_level = Noise_array[0], channel_name = Noise_array[1])

        # H⊗n again
        qusys.update_system_state(Hn @ qusys.density_matrix @ Hn.conjugate().T)
        '''Vector form of the state after applying Hadamard gates again -> check up'''
        """ new_state = Hn @ new_state
        new_state_density = np.outer(new_state, np.conjugate(new_state.T))
        print("Density matrix formalism = ", qusys.density_matrix, '\n')
        print("Vector formalism (Checkup) = ", new_state_density)
        if not np.allclose(qusys.density_matrix, new_state_density):
            raise ValueError("Density matrix and vector form do not match.") """
        
        '''Place 4 of the noise'''
        if Noise_array[0] is not None and Noise_array[1] is not None and Noise_array[2] is not None and Noise_array[2] == 4:
            # Apply noise to the quantum system
            Channels(qusys).apply_channel(noise_level = Noise_array[0], channel_name = Noise_array[1])

        # Measure amplitude at |0...0>
        prob_0 = np.abs(qusys.density_matrix[0, 0]) ** 2

        return (True if np.isclose(prob_0, 1.0) else False, qusys.density_matrix, prob_0)
    
