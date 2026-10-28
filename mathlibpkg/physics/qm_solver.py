import numpy as np

class SEQ_1D:
    """
    
    """
    def __init__(self, potential=None, mass=1.0, N=100, a=0.1):
        if self.N < 2 or not isinstance(self.N, int): 
            raise ValueError(f"A simulation on less then two ({self.N}) or non-integer lattice points is not possible!")
        if potential is None:
            self.potential = lambda x: 0 
        else:
            self.potential = potential
        self.mass = mass
        self.N = N
        self.hamiltonian = self.free_hamiltonian()


    def free_hamiltonian(self):   
        N = self.N
        self.hamiltonian = ((2 * np.eye(N)) + - np.eye(N, k=1) - np.eye(N, k=-1)) / self.a

    
    def set_potential(self, funcObj=None):
        if funcObj is None:
           return
        

    

            