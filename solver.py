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

import scipy.integrate as integrate
import scipy.optimize as optimize
from typing import List

class ShootingSolver:
    """
    1-D Time-Independent Schrodinger Equation Solver using the Shooting Method
    with adaptive step-size integration and bisection eigenvalue search.
    """
    def __init__(self, x_min: float, x_max: float):
        self.x_min = x_min
        self.x_max = x_max

    def _schrodinger_ivp(self, x: float, y: np.ndarray, E: float, V_func: Callable) -> np.ndarray:
        """Defines the system of 1st order ODEs for the Schrodinger equation."""
        psi, dpsi_dx = y
        # Atomic units: hbar = 1, m = 1
        # psi'' = 2 * (V(x) - E) * psi
        d2psi_dx2 = 2.0 * (V_func(np.array([x]))[0] - E) * psi
        return np.array([dpsi_dx, d2psi_dx2])

    def _shoot(self, E: float, V_func: Callable) -> float:
        """Integrates the wavefunction and returns its value at the right boundary."""
        # Initial conditions at x_min: psi = 0, psi' = small non-zero slope
        y0 = np.array([0.0, 1e-3])
        
        # solve_ivp uses RK45 by default, which is an adaptive step-size algorithm
        sol = integrate.solve_ivp(
            fun=lambda x, y: self._schrodinger_ivp(x, y, E, V_func),
            t_span=(self.x_min, self.x_max),
            y0=y0,
            method='RK45', 
            rtol=1e-6, atol=1e-9
        )
        # Return the boundary mismatch (we want this to be 0)
        return sol.y[0, -1]

    def solve(self, V_func: Callable, energy_brackets: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds eigenvalues using bisection search across provided energy brackets.
        """
        eigenvalues = []
        wavefunctions = []
        x_eval = np.linspace(self.x_min, self.x_max, 1000)

        for bracket in energy_brackets:
            # Run Bisection Search to find the exact energy that makes boundary = 0
            res = optimize.root_scalar(self._shoot, args=(V_func,), bracket=bracket, method='bisect', xtol=1e-8)
            E_opt = res.root
            eigenvalues.append(E_opt)

            # Re-integrate with the correct energy to get the full wavefunction plot
            sol = integrate.solve_ivp(
                fun=lambda x, y: self._schrodinger_ivp(x, y, E_opt, V_func),
                t_span=(self.x_min, self.x_max),
                y0=np.array([0.0, 1e-3]),
                t_eval=x_eval,
                method='RK45',
                rtol=1e-6, atol=1e-9
            )
            
            # Normalize the wavefunction numerically
            psi = sol.y[0]
            norm = np.sqrt(integrate.simpson(y=psi**2, x=x_eval))
            wavefunctions.append(psi / norm)

        return np.array(eigenvalues), np.array(wavefunctions), x_eval