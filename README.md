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
	D_: Distributed algorithm using a nested dissection scheme
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
	S: Solve a linear system given a decomposition of the system matrix and a right-hand side.
	SI: Compute a Selected Inversion given a decomposition of the system matrix.
    SSI: Compute a Schur-complement based Selected Inversion. Do not explicit the factorization but perform a Schur-complement and a selected inversion.
## Others:
    _RSS: reduced-system solve. Solve a reduced system constructed from the distributed factorization of a matrix.

## Examples:
  - pobtaf: Perform the factorization of a block-tridiagonal with arrowhead, symmetric positive definite, matrix.
  - d_pobtasi: Compute the selected inversion of a block-tridiagonal with arrowhead, matrix given its Cholesky factorization, using a distributed algorithm.

# How to install
First, you'll need to instal the project in your current environment. You can do this by running the following commands:

```
1. Create a conda environment from the environment.yml file
    $ conda env create -f environment.yml
    $ conda activate serinv

2. Install mpi4py : (Optional)
    $ conda install -c conda-forge mpi4py mpich

    Please refere to https://mpi4py.readthedocs.io/en/stable/install.html for more details.

3. Install CuPy : (Optional)
    # conda install -c conda-forge cupy cuda-version=xx.x

    Please refere to https://docs.cupy.dev/en/stable/install.html for more details.

4. Install SerinV
    $ cd path/to/serinv
    $ pip install --no-dependencies -e .
```

## Alternative procedure
```
  $ conda create -n serinv python=3.11
  $ conda activate serinv
  $ conda install conda-forge::numpy conda-forge::scipy conda-forge::matplotlib conda-forge::pydantic conda-forge::pytest conda-forge::pytest-mpi conda-forge::pytest-cov conda-forge::coverage conda-forge::black conda-forge::isort conda-forge::ruff conda-forge::pre_commit conda-forge::pytest-xdist
  $ conda install anaconda::sqlite
  $ conda install -c conda-forge mpi4py mpich
  $ conda install -c conda-forge cupy cuda-version=xx.x
```
