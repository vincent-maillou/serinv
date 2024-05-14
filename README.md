<table>
  <tr>
    <td><img src="doc/images/logo_noback.png" style="width: 100%;" /></td>
    <td><h1>SerinV: Selected Inversion, factorization and solver for structured sparse matrices</h1></td>
  </tr>
</table>

[![codecov](https://codecov.io/gh/vincent-maillou/SDR/graph/badge.svg?token=VZTGAUW2NW)](https://codecov.io/gh/vincent-maillou/SDR)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

SerinV bundle implementations of several selected factorization, inversion and solver algorithms for severals types of block-structured sparse matrices.

# Routine Naming Conventions
## Computation scheme:
	S: Sequential
	D: Distributed nested dissection scheme
## Backend:
	C: CPU
	G: GPU
## Type of matrix:
	PO: Symmetric or Hermitian positive definite
	DD: General, square, diagonoally dominante matrix
## Sparsity pattern:
	BT: Block Tridiagonal
	BTA: Block Tridiagonal Arrowhead
	BB: Block Banded
	BBA: Block Banded Arrowhead
## operations performed:
	F: Perform matrix factorization
	S: Solve the linear system with factored matrix
	SI: Compute the selected inverse matrix using the factorization

Examples:
  - scpobtaf: Perform the factorization of a symmetric positive definite block tridiagonal arrowhead matrix using a sequential CPU backend.
  - dgddbtsi: Compute the selected inverse of a general diagonally dominant block tridiagonal matrix using a distributed GPU backend.

# How to install
    # Recommended: Create a new conda environment with python version above 3.9
    conda create --name serinv_env python=3.11

    # Activate the created environment
    conda activate serinv_env

    # Move to the root of the repository
    cd /path/to/serinv/

    # Install the package in editable mode
    pip install -e .



