# 1-D Computational Quantum Mechanics Solver

**Project 11 Submission**
**Author:** Shreyash Krishna

https://drive.google.com/file/d/17Vn-PP4G3T0TPNa9CAVHL23yaPdjeLnt/view?usp=sharing

A robust, dual-method computational engine designed to numerically solve the 1-D Time-Independent Schrodinger Equation for arbitrary potential profiles. This project features high-precision eigenvalue extraction and interactive visualizations of quantum phenomena, including quantum tunneling and solid-state band gap formation.

---

## Core Capabilities
This library implements two distinct numerical approaches to ensure solver generality across any piecewise smooth potential:

1. **Matrix Finite-Difference Method (FDM):** Utilizes a central difference approximation to construct the Hamiltonian matrix, solved via optimized eigenvalue decomposition (`scipy.linalg.eigh`).
2. **The Shooting Method:** Formulates the Schrodinger equation as an initial value problem, utilizing adaptive step-size integration (Runge-Kutta 45) combined with a bisection root-finding algorithm to precisely satisfy boundary conditions.

---

## Architectural Reasoning & Method Selection

A core objective of this project was to implement and cross-validate two distinct numerical methods, as each excels in different computational scenarios.

### 1. When to use the Finite-Difference Method (FDM)
The FDM was strictly utilized for the interactive exploration notebooks (Double-Well and Kronig-Penney Lattice). 
* **Reasoning:** FDM casts the differential equation as an eigenvalue problem, allowing `scipy.linalg.eigh` to solve for all requested energy states simultaneously. This matrix diagonalization is highly optimized and exceptionally fast for static grids, making it the only viable method for driving interactive widgets where multiple energy bands must update continuously in real-time without computational lag.
* **Limitations:** The mathematical truncation error scales with the grid spacing squared. Achieving higher precision requires exponentially larger matrices, which can bottleneck system memory.

### 2. When to use the Shooting Method
The Shooting Method was developed for targeted, high-precision eigenvalue extraction.
* **Reasoning:** By treating the boundary value problem as an initial value problem, we leverage the adaptive step-size capabilities of the Runge-Kutta 45 algorithm. Instead of relying on a rigid grid, the solver dynamically adjusts its resolution when it encounters steep potential gradients. This provides superior precision for specific, individual energy levels without the memory overhead of massive matrix operations.
* **Limitations:** It is a root-finding algorithm that must hunt for one state at a time. It is computationally inefficient for visualizing full systems or multiple bands simultaneously.

### 3. Grid Convergence and Boundary Logic
For textbook models (Harmonic Oscillator, Square Well), accuracy was verified against exact analytical solutions. However, for arbitrary potentials like the Double-Well, exact solutions do not exist. In these cases, accuracy is secured via **Grid Convergence**—increasing the grid resolution until the calculated eigenvalues stabilize to a fixed floating-point precision. Additionally, all grids were carefully dimensioned to act as an "artificial box," ensuring boundaries were placed far enough away to capture the complete natural decay of the wavefunctions without artificially raising the ground state energy.

---

## Verification & Accuracy
Both mathematical engines were rigorously calibrated against textbook analytical models to ensure maximum precision. 
* **Infinite Square Well Ground State Error:** 0.0000% (Grid boundaries implicitly aligned to exact well dimensions).
* **Quantum Harmonic Oscillator Ground State Error:** < 0.0002%
* **Excited State Accuracy:** The first 5 eigenvalues for both test potentials fall comfortably within the strict < 0.1% tolerance threshold.

---

## Interactive Physics Explorations
The included `02_exploration.ipynb` notebook provides interactive `ipywidgets` to visualize non-relativistic quantum systems dynamically:
* **The Double-Well Potential:** Adjust barrier height and well separation to observe real-time wavefunction overlap and the resulting energy level splitting between symmetric and antisymmetric states.
* **The Kronig-Penney Lattice:** Modulate periodic barrier widths to visualize the hybridization of discrete energy states into continuous allowed bands and forbidden band gaps.

---

## How to Run
This project is designed to run locally or in a cloud Jupyter environment (e.g., Lightning AI, Google Colab).

**1. Install Dependencies:**
```bash
pip install numpy scipy matplotlib ipywidgets