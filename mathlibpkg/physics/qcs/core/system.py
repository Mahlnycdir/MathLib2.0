import numpy as np
from typing import Dict, Callable, Any, Sequence
from functools import reduce

class QubitSystem:
    def __init__(self, n = None, state = None):
        """
        Initialize a quantum system either from a number of qubits or a given state
        (pure state vector or density matrix).

        Args:
            n (int): Number of qubits. Optional if `state` is provided.
            state (np.ndarray): Initial state (1D for pure state, 2D for density matrix).
        """
        # Validate input
        if state is not None and not isinstance(state, np.ndarray):
            raise ValueError("State must be a NumPy array.")

        # Infer number of qubits if state is provided
        if state is not None:
            if state.ndim == 1:
                dim = state.shape[0]
            elif state.ndim == 2:
                if state.shape[0] != state.shape[1]:
                    raise ValueError("Density matrix must be square.")
                dim = state.shape[0]
            else:
                raise ValueError("State must be either 1D (vector) or 2D (density matrix).")
            inferred_n = int(np.log2(dim))
            if 2 ** inferred_n != dim:
                raise ValueError("State dimension is not a power of 2.")
            if n is not None and n != inferred_n:
                raise ValueError("Provided n does not match state dimension.")
            self.n = inferred_n
        elif n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive integer.")
            self.n = n
        else:
            raise ValueError("Either n or state must be provided.")

        self.dim = 2 ** self.n
        self.qubit_labels = [f"q[{i}]" for i in range(self.n)]  # fixed order for gate construction

        # Initialize internal states
        self.single_qubit_states: Dict[str, np.ndarray] = {}
        self._init_single_qubits()

        self._state_vector: np.ndarray | None = None
        self._density_matrix: np.ndarray | None = None

        if state is None:
            # Default to |00...0⟩
            self._state_vector = np.zeros(self.dim, dtype=complex)
            self._state_vector[0] = 1
            self._density_matrix = np.outer(self._state_vector, self._state_vector.conj())
        elif state.ndim == 1:
            # Pure state
            self._state_vector = state
            self._density_matrix = np.outer(state, state.conj())
        elif state.ndim == 2:
            # Mixed state (density matrix)
            self._density_matrix = state
            self._state_vector = None

    #---------------------------------------------------------------------
    #  Helper functions
    #---------------------------------------------------------------------

    def _inner_product(self, a: np.ndarray, b: np.ndarray):
        '''
        Return the inner product <a|b> = conj(a)·b.
        '''
        a = np.asarray(a, dtype=complex)
        b = np.asarray(b, dtype=complex)
        if a.shape != b.shape:
            raise ValueError("Both vectors must have same shape.")
        return np.vdot(a, b)

    def _tensor_power(self, gate: np.ndarray) -> np.ndarray:
        '''
        Return gate ⊗ gate ⊗ ... n times.
        '''
        result = gate
        for _ in range(self.n - 1):
            result = np.kron(result, gate)
        return result
    
    def _kron_n(self, *matrices):
        return reduce(np.kron, matrices)
 
    def _combine_states(self, single_states: Sequence[np.ndarray]):
        '''
        Tensor‑product all single‑qubit states in the list.
        Each state must be a (2,) array.
        '''
        if len(single_states) != self.n:
            raise ValueError(f"Expected {self.n} states, got {len(single_states)}.")

        # Start from the first qubit
        combined_state = np.asarray(single_states[0], dtype=complex)
        if combined_state.shape != (2,):
            raise ValueError(f"State 0 has shape {combined_state.shape}, expected (2,)")

        # Tensor with the rest
        for i, state in enumerate(single_states[1:], start=1):
            state = np.asarray(state, dtype=complex)
            if state.shape != (2,):
                raise ValueError(f"State {i} has shape {state.shape}, expected (2,)")
            combined_state = np.kron(combined_state, state)

        return combined_state
    
    #---------------------------------------------------------------------
    # Qubit system functions
    #---------------------------------------------------------------------

    def _init_single_qubits(self):
        '''
        Initialize all qubits in |0> state and store in dict.
        '''
        zero = np.array([1, 0], complex)
        for i in range(self.n):
           self.single_qubit_states[f"q[{i}]"] = zero
        return self.single_qubit_states

    @property
    def state_vector(self) -> np.ndarray:
        """
        Returns the pure state vector of the quantum system.

        The state vector represents the quantum state of the system in a 
        2^n-dimensional complex vector space, where n is the number of qubits.

        Returns:
            np.ndarray: A NumPy array representing the pure state vector of the system -> Pure state vector (dimension 2**n).
        """
    
        return self._state_vector
    
    @property
    def density_matrix(self):
        """
        Returns the density matrix of the quantum system.

        The density matrix is a 2^n x 2^n complex matrix that represents the 
        quantum state of the system, including both pure and mixed states.

        Returns:
            np.ndarray: A NumPy array representing the density matrix of the system.
        """
        """Density matrix (dimension 2**n x 2**n)."""
        return self._density_matrix                          # np.outer(self._state_vector, self._state_vector.conjugate())
    
    def vec_as_dict(self, state = None):
        '''
        Return the state of the system as a dictionary.
        The keys are the basis states and the values are the amplitudes.
        '''
        if state.ndim == 1:             # state is given as vector
            if state is None:
                state = self.system_state
            if state.shape != (self.dim,):
                raise ValueError(f"State vector must have shape ({self.dim},)")
            result = {}
            for index in range(self.dim):
                basis_label = format(index, f'0{self.n}b' )    # e.g. '00', '01', '10', ...
                amplitude = state[index]             # Coefficients of the basis state
                result[f"{basis_label}"] = complex(amplitude)
            return result
        elif state.ndim == 2:           # state is given as density matrix
            if state.shape != (self.dim, self.dim):
                raise ValueError(f"Density matrix must have shape ({self.dim}, {self.dim}).")
            result = {}
            for i in range(self.dim):
                for j in range(self.dim):
                    basis_label = format(i, f'0{self.n}b') + format(j, f'0{self.n}b')
                    amplitude = state[i, j]             # Coefficients of the basis state
                    result[f"{basis_label}"] = complex(amplitude)
            return result

    @property
    def state_dict(self):
        """Dictionary {'|000>': α, '|001>': β, …} """
        if self._state_vector is None and self._density_matrix is None:
            raise ValueError("State vector is not initialized.")
        elif self._state_vector is not None:
            return self.vec_as_dict(self._state_vector)
        elif self._density_matrix is not None:
            return self.vec_as_dict(self._density_matrix)


    def update_system_state(self, new_state: np.ndarray):
        """
        Update the system state. Accepts either a pure state vector (1D) or a density matrix (2D).
        """
        if new_state.ndim == 1:
            # Pure state vector
            self._state_vector = new_state
            self._density_matrix = np.outer(new_state, new_state.conjugate())
        elif new_state.ndim == 2:
            # Density matrix: do NOT attempt to extract state vector!
            if new_state.shape != (self.dim, self.dim):
                raise ValueError(f"Density matrix must have shape ({self.dim}, {self.dim}).")
            self._density_matrix = new_state
            self._state_vector = None  # <- Explicitly signal: we're in mixed state


    def change_single_qubit(self, qubit: str, state: np.ndarray):
        '''
        Change the state of one qubit.
        '''
        if not self.single_qubit_states:
            self._init_single_qubits()
        if qubit not in self.single_qubit_states:
            raise KeyError(f"Qubit {qubit} not found.")
        state = np.asarray(state, dtype=complex)
        if state.shape != (2,):
            raise ValueError("State must be a vector of size 2.")
        self.single_qubit_states[qubit] = state
        self.update_system_state(self._combine_states(list(self.single_qubit_states.values())))

    def reset_state(self):
        '''
        Reset the state of the system to |0>^n.
        '''
        update_state = np.zeros(self.dim, complex)
        update_state[0] = 1
        self.update_system_state(update_state)

    def n_basis(self, index: int):
        '''
        input: index of the basis vector (0, 1, ..., 2^n-1)
        output: the corresponding basis vector |index> in the n-qubit system
        (Maybe Outdated)
        '''
        if not (0 <= index < self.dim):
            raise ValueError("Basis index out of range.")
        vec = np.zeros(self.dim, complex)
        vec[index] = 1
        return vec