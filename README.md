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
   1. blk_tridiag [x]
   2. blk_tridiag_arrowhead [x]
   3. blk_ndiags [x]
   4. blk_ndiags_arrowhead [x]
- LU
   1. blk_tridiag [x]
   1. blk_tridiag_arrowhead [x]
   2. blk_ndiags [x]
   3. blk_ndiags_arrowhead [x]
### Solvers
- Cholesky
   1. blk_tridiag []
   2. blk_tridiag_arrowhead []
   3. blk_ndiags []
   4. blk_ndiags_arrowhead []
- LU
   1. blk_tridiag []
   1. blk_tridiag_arrowhead []
   2. blk_ndiags []
   3. blk_ndiags_arrowhead []



