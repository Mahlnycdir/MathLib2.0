from typing import Any, Sequence, Tuple, List
import numpy as np
from scipy.sparse import csr_matrix
from .gates import Gates
from .system import QubitSystem

InstructionPkg = Tuple[str, List[str], Sequence[Any] | None]

class Instruction:
    def __init__(self, qs: QubitSystem):
        self.qs = qs
        self.g  = Gates(qs)

    def _is_mixed_state(self):
        """
        Checks whether the system is currently in a mixed state representation.
        """
        if self.qs.state_vector is None and self.qs.density_matrix is not None:
            return True
        elif self.qs.state_vector is not None:
            return False

    def _run(self, name: str,
             qubits: Sequence[str],
             params: Sequence[Any] | None = None):
        """
        Apply exactly one quantum gate to the system and update its state.
        Handles both pure states and mixed states.
        """
        op = self.g.gate(name, qubits, params)  # Construct full gate matrix in sparse format
        print(op)
        if self._is_mixed_state():
            # For mixed states: ρ → UρU†
            rho_new = op @ self.qs.density_matrix @ np.conjugate(op.T)
            self.qs.update_system_state(rho_new)
        else:
            # For pure states: |ψ⟩ → U|ψ⟩
            psi_new = op @ self.qs.state_vector
            self.qs.update_system_state(psi_new)

    def apply_instruction(self, acc, instr=None):
        """
        Allows reduce(ins.apply_instruction, instruction_list) for sequential application.
        'acc' is the accumulator (first or previous state).
        'instr' is the next instruction tuple.
        """
        def run(obj):
            if isinstance(obj, (list, tuple)) and len(obj) >= 2 and isinstance(obj[0], str):
                name   = obj[0]                              # Gate name (e.g., 'H', 'CNOT')
                qubits = obj[1]                              # List of qubit labels
                params = obj[2] if len(obj) > 2 else None    # Optional parameters
                self._run(name, qubits, params)
        run(acc)  # Apply first instruction (or initial state)
        if instr is not None:
            run(instr)  # Apply next instruction

        # Return updated state vector (for compatibility with reduce)
        if self._is_mixed_state():
            return self.qs.density_matrix
        else: return self.qs.state_vector
