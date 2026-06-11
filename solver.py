import numpy as np
import scipy.linalg as la
from typing import Callable, Tuple

class FDMSolver:
    """
    1-D Time-Independent Schrodinger Equation Solver using Matrix Finite-Difference.
    """
    def __init__(self, x_min: float, x_max: float, N: int):
        # Set up the spatial grid
        self.x = np.linspace(x_min, x_max, N)
        self.dx = self.x[1] - self.x[0]
        self.N = N

    def build_hamiltonian(self, V_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Constructs the tridiagonal Hamiltonian matrix."""
        # Using atomic units: hbar = 1, m = 1
        kinetic_coeff = -1.0 / (2.0 * self.dx**2)
        
        main_diag = -2.0 * kinetic_coeff * np.ones(self.N)
        off_diag = kinetic_coeff * np.ones(self.N - 1)
        
        # Evaluate the user-defined potential across the grid
        V_array = V_func(self.x)
        
        # H = Kinetic (T) + Potential (V)
        H = np.diag(main_diag + V_array) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return H

    def solve(self, V_func: Callable[[np.ndarray], np.ndarray], num_states: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the lowest energy eigenvalues and normalized wavefunctions."""
        H = self.build_hamiltonian(V_func)
        
        # la.eigh is heavily optimized for Hermitian/symmetric matrices
        eigenvalues, eigenvectors = la.eigh(H, subset_by_index=[0, num_states - 1])
        
        # Normalize wavefunctions over the spatial grid
        wavefunctions = eigenvectors.T / np.sqrt(self.dx)
        return eigenvalues, wavefunctions