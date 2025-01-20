<table>
  <tr>
    <td><img src="doc/images/logo_noback.png" style="width: 100%;" /></td>
    <td><h1>SerinV: Selected Inversion, factorization and solver for structured sparse matrices</h1></td>
  </tr>
</table>

[![codecov](https://codecov.io/gh/vincent-maillou/SDR/graph/badge.svg?token=VZTGAUW2NW)](https://codecov.io/gh/vincent-maillou/SDR)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

SerinV bundle implementations of several factorization, selected-inversion and solver for severals types of structured sparse matrices. It implements sequential and distributed algorithm and support GPUs backends.

# Routine Naming Conventions
We have adopted a LAPACK-like naming convention for the routines. The naming convention is as follows:

## Computation scheme:
	P: Distributed algorithm using a nested dissection scheme
## Types of matrices:
	PO: Symmetric or Hermitian positive definite matrix
	DD: General, square, diagonally dominante matrix
## Sparsity pattern:
	BT: Block-Tridiagonal
	BTA: Block-Tridiagonal with Arrowhead, by convention the arrowhead is pointing down.
	BB: Block-Banded
	BBA: Block-Banded with Arrowhead, by convention the arrowhead is pointing down.
## Operations performed:
	F: Factorization
    SC: Schur-complement
	SI: Selected Inversion
    S: Solve
## Others:
    RS: Reduced system

## Examples:
  - pobtaf: Perform the factorization of a block-tridiagonal with arrowhead, symmetric positive definite, matrix.
  - ppobtasi: Compute the selected inversion of a block-tridiagonal with arrowhead, matrix given its Cholesky factorization, using a distributed algorithm.

# How to install
First, you'll need to instal the project in your current environment. You can do this by running the following commands:

    # Recommended: Create a new conda environment with python version above 3.9
    conda create --name serinv_env python=3.11

    # Activate the created environment
    conda activate serinv_env

    # Move to the root of the repository
    cd /path/to/serinv/

    # Install the package in editable mode
    pip install -e .

To use the distributed version of the algorithms, you'll need to install mpi4py and have a working MPI implementation. You can find all the relevant information on the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html).

To use the GPU version of the algorithms, you'll need to install CuPy. You can find all the relevant information on the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).