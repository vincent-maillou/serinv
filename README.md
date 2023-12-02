# SDR: Selected Decomposition Routines
SDR pack several selected decompositions routines and linear system solvers for severals types of structured block-sparse matrices.

## Decompositions routines
### Block cholesky factorization
A block cholesky factorization routines for symmetric, positive-definite matrices.
### Block LU factorization
A block LU factorization routines for general, non-singular, matrices.

## Supported structured block-sparse matrices
### Block tridiagonal sparsity patterns

Such matrices presente a block tridiagonal sparsity pattern that can be associated with an arrowhead. In the case of an arrowhead matrix the sizes of the blocks of
the last column/last row can be differents.

![Block tridiagonal sparsity pattern](/doc/images/structured_sparsity_patterns/tridiag_white.png)

### Block n-diagonals sparsity patterns

Such matrices presente a block n-diagonals sparsity pattern that can be associated with an arrowhead. In the case of an arrowhead matrix the sizes of the blocks of
the last column/last row can be differents.

![Block tridiagonal sparsity pattern](/doc/images/structured_sparsity_patterns/ndiags_white.png)

## How to install
    # Recommended: Create a new conda environment with python version above 3.9
    conda create --name sdr_env python=3.11

    # Activate the created environment
    conda activate sdr_env

    # Move to the root of the repository
    cd /path/to/SDR/

    # Install the package in editable mode
    pip install -e .

## TODO/States of the implementations:
### Decomposition routines
- Cholesky
   1. chol_dcmp_tridiag() [x]
   2. chol_dcmp_tridiag_arrowhead() [x]
   3. chol_dcmp_ndiags [x]
   4. chol_dcmp_ndiags_arrowhead [x]
- LU
   1. lu_dcmp_tridiag() [x]
   1. lu_dcmp_tridia_arrowhead() [x]
   2. lu_dcmp_ndiags() [x]
   3. lu_dcmp_ndiags_arrowhead() [x]
### Solvers
- Cholesky
   1. chol_slv_tridiag() [x]
   2. chol_slv_tridiag_arrowhead() []
   3. chol_slv_ndiags() []
   4. chol_slv_ndiags_arrowhead() []
- LU
   1. lu_slv_tridiag() []
   2. lu_slv_tridiag_arrowhead() []
   3. lu_slv_ndiags() []
   4. lu_slv_ndiags_arrowhead() []
### Selected Inversion
- Cholesky
   1. chol_sinv_tridiag() [x]
   2. chol_sinv_tridiag_arrowhead() []
   3. chol_sinv_ndiags() []
   4. chol_sinv_ndiags_arrowhead() []
- LU
   1. lu_sinv_tridiag() []
   2. lu_sinv_tridiag_arrowhead() []
   3. lu_sinv_ndiags() []
   4. lu_sinv_ndiags_arrowhead() []


