# Thermodynamically Admissible Neural Solvers

Code implementation for solving stiff 1D coupled electro-elastodynamics using geometrically constrained Physics-Informed Neural Networks (PINNs).

## Architecture Overview
This repository contains a neural architecture that bypasses standard soft-penalty gradient pathologies by enforcing exact Dirichlet boundaries and 2nd-order time initial conditions directly within the network's functional topology via analytical distance constraints. 

## Repository Structure
- `model.py`: PINN architecture and topological distance manifolds.
- `physics.py`: Normalized IEEE electro-elastodynamic constants and PDE residuals.
- `train.py`: The main execution script applying the 3-stage optimization pipeline (Adam -> AdamW -> L-BFGS).

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
