<table>
  <tr>
    <td><img src="doc/images/logo_noback.png" style="width: 100%;" /></td>
    <td><h1>SerinV: Selected Inversion, factorization and solver for structured sparse matrices</h1></td>
  </tr>
</table>

[![codecov](https://codecov.io/gh/vincent-maillou/SDR/graph/badge.svg?token=VZTGAUW2NW)](https://codecov.io/gh/vincent-maillou/SDR)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

SerinV bundle implementations of several factorization, selected-inversion and solver for severals types of block-structured sparse matrices. It implement sequential and distributed algorithm and support GPUs backends.

# Routine Naming Conventions
## Computation scheme:
	D_: Distributed nested dissection scheme
## Type of matrix:
	PO: Symmetric or Hermitian positive definite
	DD: General, square, diagonoally dominante matrix
## Sparsity pattern:
	BT: Block Tridiagonal
	BTA: Block Tridiagonal Arrowhead
	BB: Block Banded
	BBA: Block Banded Arrowhead
## operations performed:
	F: Perform matrix Factorization
	S: Solve a linear system given a factorization
	SI: Compute a Selected Inversion given a factorization.
    SSI: Compute a Schur Selected Inversion from a matrix. Do not explicit the factorization step.

## Examples:
  - pobtaf: Perform the factorization of a block tridiagonal arrowhead, symmetric positive definite, matrix.
  - d_ddbtsi: Compute the selected inversion of a block tridiagonal, diagonally dominant, matrix given its LU factorization.

# How to install
    # Recommended: Create a new conda environment with python version above 3.9
    conda create --name serinv_env python=3.11

    # Activate the created environment
    conda activate serinv_env

    # Move to the root of the repository
    cd /path/to/serinv/

    # Install the package in editable mode
    pip install -e .



