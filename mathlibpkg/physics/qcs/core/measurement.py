import numpy as np
from .system import QubitSystem
from .gates import Gates
from typing import Dict, Callable, Any, Sequence
from functools import reduce
from typing import List

class Measurement:
    """
    Provides measurement functions for a quantum system.
    Generates a (2**n × 2**n) matrix for the specified qubit
    or qubit pair of the associated QubitSystem object upon request.
    """    

    def measure(self, state: np.ndarray, num_shots : int = 1, return_binary : bool = True):
        '''
        Measure the state of a quantum system.
        Args:
            state: The state vector or density matrix to be measured.
            num_shots: The number of measurement shots.
            return_binary: If True, return the outcomes as binary strings.
            Returns: A list of measurement outcomes.
        '''
        if not isinstance(state, np.ndarray) or state.ndim not in (1, 2):
            raise ValueError("state must be a 1D or 2D array.")
        elif not isinstance(num_shots, int) or num_shots <= 0:
            raise ValueError("num_shots must be a positive integer.")
        elif not isinstance(return_binary, bool):
            raise ValueError("return_binary must be a boolean value.")
        else:
            pass

        if state.ndim == 1:
            density_matrix = np.outer(state, state.conjugate())  # |ψ⟩⟨ψ|
        elif state.ndim == 2:
            if state.shape[0] != state.shape[1]:
                raise ValueError("Density matrix must be square.")
            density_matrix = state
        
        dim = density_matrix.shape[0]
        num_qubits = int(np.log2(dim))

        # Define the projectors for the measurement
        computational_basis = [np.eye(dim)[i] for i in range(dim)]
        projectors = [np.outer(b, b.conj()) for b in computational_basis]

        # Initialize the results
        probs = np.array([np.trace(density_matrix @ Pi).real for Pi in projectors])

         # Inverse transform sampling with np.cumsum
        cum_probs = np.cumsum(probs)
        random_values = np.random.rand(num_shots)
        outcomes = np.searchsorted(cum_probs, random_values)

        if return_binary:
            return [format(i, f'0{num_qubits}b') for i in outcomes]
        else:
            return outcomes.tolist()