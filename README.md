# SDR: Selected Decomposition Routines
SDR pack several selected decompositions routines and linear system solvers for severals types of structured block-sparse matrices.

## Decompositions routines
### Block cholesky factorization
A block cholesky factorization routines for symmetric, positive-definite matrices.
### Block LU factorization
A block LU factorization routines for general, non-singular, matrices.

## Supported structured block-sparse matrices
### Block tridiagonal
### Arrowhead block tridiagonal
### Block ndiagonals
### Arrowhead block ndiagonals

## How to install
    # Recommended: Create a new conda environment with python version above 3.9
    conda create --name sdr_env python=3.11

    # Activate the created environment
    conda activate sdr_env

    # Move to the root of the repository
    cd /path/to/SDR/

    # Install the package in editable mode
    pip install -e .

