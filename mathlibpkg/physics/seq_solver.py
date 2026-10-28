import numpy as np
import matplotlib.pyplot as plt
import datetime

class SEQ_1D:
    """
    Finite-difference solver for the stationary 1D Schroedinger equation.

    Discretizes

        H = -hbar^2/(2m) d^2/dx^2 + V(x)

    on N equidistant lattice points x_i = x0 + i*a (i = 0, ..., N-1), using the
    three-point central difference for the second derivative. The result is a
    real, symmetric, tridiagonal N x N matrix:

        H_ii      =  hbar^2/(m a^2) + V(x_i)
        H_i,i+-1  = -hbar^2/(2 m a^2)

    Boundary conditions: truncating the stencil at the edges imposes psi = 0 at
    x0 - a and x0 + N*a, i.e. the system sits in an infinite square well of
    width (N+1)*a. Choose the lattice so that the states of interest have
    decayed well before the walls.

    Args:
        potential (callable, optional): V(x), called with a single float x and
                                returning a float. Defaults to V(x) = 0 (particle in a box).
        mass (float): Particle mass, m > 0. Defaults to 1.0.
        N (int): Number of lattice points, N >= 2. Defaults to 100.
        a (float): Lattice spacing, a > 0. Defaults to 0.1.
        x0 (float): Position of the first lattice point. Defaults to 0.0.
        hbar (float): Reduced Planck constant, hbar > 0. Defaults to 1.0
                      (natural units).

    Attributes:
        lattice (np.ndarray): Lattice points x_i, shape (N,).
        hamiltonian (np.ndarray): Discretized Hamiltonian, shape (N, N).

    Raises:
        TypeError: If N is not an int, or a, mass, hbar or x0 is not an
            int or float.
        ValueError: If N < 2, or a, mass or hbar is not positive.

    Note:
        The Hamiltonian is built on construction. After changing any of
        N, a, x0, mass, hbar or potential, call build_hamiltonian() to update it.

    Example:
        >>> sim = SEQ_1D(potential=lambda x: 0.5 * x**2, N=1000, a=0.01, x0=-5.0)
        >>> sim.energy_spectrum()[:3]   # approx. [0.5, 1.5, 2.5]
    """
    def __init__(self, potential=None, mass=1.0, N=100, a=0.1, x0 = 0.0, hbar = 1.0):
        if not isinstance(N, int):
            raise TypeError("A simulation with a non-integer N is not possible!")
        if N < 2:
            raise ValueError("A simulation with N < 2 is not possible!")
        
        for val in [a, mass, hbar, x0]:
            if not(isinstance(val, float) or isinstance(val,int)):
                raise TypeError("A value of (a, mass, hbar, x0) has to be of Type float or int!")
            
        for val in [a, mass, hbar]:
            if val <= 0:
                raise ValueError("A value of (a, mass, hbar) has to be greater than 0!")
        
        if potential is None:
            self.potential = lambda x: 0 
        else:
            self.potential = potential   
            
        self.N = N              
        self.mass = mass        
        self.a = a              
        self.x0 = x0           
        self.hbar = hbar 
        self.energy_spectrum = None
        
        # setting up the Hamiltonian
        self.build_hamiltonian()
        
        
    def build_hamiltonian(self):
        self._build_lattice()
        self._free_hamiltonian()
        self._set_potential()


    def _build_lattice(self):
        self.lattice = np.array([self.x0 + self.a*i for i in range(self.N)])
        

    def _free_hamiltonian(self):   
        N = self.N
        factor = ((self.hbar**2) / (2 * self.mass * (self.a**2)))
        self.hamiltonian = factor * ((2 * np.eye(N)) + -np.eye(N, k=1) - np.eye(N, k=-1))
 
    
    def _set_potential(self):
        """
        Creating a diagonal matrix containing the potential on lattice point (i,V_i), depending on the lattice 
        generated.
        """
        self.hamiltonian += np.diag([self.potential(lattice_points) for lattice_points in self.lattice])
        
    
    def show_hamiltonian(self):
       print(self.hamiltonian)
       
    
    def calculate_energy_spectrum(self):
        """
        Calculates the eigenvalue spectrum of the set up Hamiltonian. This is the simple 
        solution of the stationary Schroedinger equation H psi = E psi. It's stored class wide
        in `self.energy_spectrum`.
        """            
        self.energy_spectrum = np.linalg.eigvalsh(self.hamiltonian)
        return self.energy_spectrum
    
    
    def plot_energy_spectrum(self, n_max = 3):
        if self.energy_spectrum is None:
            self.calculate_energy_spectrum()
    
    
    def energy_spectrum_convergence(self, 
                                    n_max = 3, 
                                    N_array = [10,50,100,500,1000], 
                                    show_plot = True,
                                    save_figure = False,
                                    file_type = "pdf",
                                    file_name = f"en_spec_con_{datetime.datetime.now()}"):          
        """
        Convergence simulation of the model for different values of N in `N_array` and plots
        to the convergence for the 0,...,`n_max` energy values.

        Args:
            n_max (int, optional): Maximum energy band. Defaults to 3.
            N_array (list, optional): Array of numbers of lattice points. Defaults to [10,50,100,500,1000].

        Returns:
            list: All energy convergences from the calculations.
        """
        sims = [SEQ_1D(potential=self.potential, N=N_val, a=self.a, x0=self.x0) for N_val in N_array]
        e_spectrum =[sim.calculate_energy_spectrum() for sim in sims]
        e_spectra_nmax = [e_band[:n_max] for e_band in e_spectrum]         # approx. [0.5, 1.5, 2.5] (only the eigenvalues of n=0,1,2)
        
        # nearly square grid with enough cells for all n_max bands
        n_cols = int(np.ceil(np.sqrt(n_max)))
        n_rows = int(np.ceil(n_max / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                                 figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for j in range(n_max):
            e_n = [e_spectra_nmax[i][j] for i in range(len(N_array))]
            axes[j].plot(N_array, e_n, marker="o", label=r"$E_{conv} = $" + f"{e_n[-1]:.4f}")
            axes[j].set_title(f"Convergence for n = {j}")
            axes[j].set_xscale("log")
            axes[j].legend()
            axes[j].grid()

        # hide the unused cells of the grid
        for ax in axes[n_max:]:
            ax.set_visible(False)

        # x-labels on the lowest visible plot of every column
        for ax in axes[max(n_max - n_cols, 0):n_max]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("N")
        for ax in axes[::n_cols]:
            ax.set_ylabel("E")

        fig.tight_layout()
        
        if save_figure:
            plt.savefig(file_name + "." + file_type)
        
        if show_plot:
            plt.show()
        
        return e_spectrum