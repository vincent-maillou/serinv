<table>
  <tr>
    <td><img src="doc/images/logo_noback.png" style="width: 100%;" /></td>
    <td><h1>Serinv: Selected Inversion, factorization and solver for structured sparse matrices</h1></td>
  </tr>
</table>

[![codecov](https://codecov.io/gh/vincent-maillou/SDR/graph/badge.svg?token=VZTGAUW2NW)](https://codecov.io/gh/vincent-maillou/SDR)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Serinv is a selected solver library for structured sparse matrices. 
It bundle implementations of several factorization, selected-inversion and solver routines for severals types of structured sparse matrices. 
It implements sequential and distributed algorithm and support GPUs backends.

# Routine Naming Conventions
We have adopted a LAPACK-like naming convention for the routines. The naming convention is as follows:

## Computation prefix:
	P: Parallel implementation (distributed memory)

## Types of matrices:
	PO: Symmetric or Hermitian positive definite matrix
	DD: Diagonally dominante (or block-diagonally dominante) matrix

### Note:
The routines that applies to `DD` matrices ca be used on any general matrices, however stability and accuracy are not guaranteed.
In details:
- Strictly diagonally dominant matrices: accurate + stable.
- Block-diagonally dominant matrices: stable, accuracy may vary.
- General matrices: stability and accuracy are not guaranteed.

## Sparsity pattern:
	BT: Block-Tridiagonal
	BTA: Block-Tridiagonal with Arrowhead
	BB: Block-Banded
	BBA: Block-Banded with Arrowhead

### Note:
By convention arrowheads matrices are pointing downwards.

## Operations performed:
	F: Factorization
	SI: Selected Inversion
    S: Solve
    SC: Schur-complement
    SCI: Schur-complement Inversion (selected inversion of the Schur complement)

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

## Dev-note
Here are some installation guidelines to install the project on the Fau; Alex and Fritz clusters.
We recommend to test any development in 3 separated environments:
- Bare: The environment without any MPI or GPU support
- Fritz: The environment with MPI support, CPU backend
- Alex: The environment with MPI support, GPU backend (optional: NCCL)

This ensure compatibility no matter the available backend.

```bash
# --- Alex-env ---
module load python
module load openmpi/4.1.6-nvhpc23.7-cuda
module load cuda/12.6.1

conda create -n alex
conda activate alex

CFLAGS=-noswitcherror MPICC=$(which mpicc) pip install --no-cache-dir mpi4py

salloc --partition=a40 --nodes=1 --gres=gpu:a40:1 --time 01:00:00
conda activate alex

conda install -c conda-forge cupy-core
conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y
cd /path/to/serinv/
python -m pip install -e .
```

```bash
# --- Fritz-env ---
module load python
module load openmpi/4.1.2-gcc11.2.0

conda create -n fritz
conda activate fritz

MPICC=$(which mpicc) pip install --no-cache-dir mpi4py

salloc -N 4 --time 01:00:00
conda activate fritz

conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y
cd /path/to/serinv/
python -m pip install -e .
```

```bash
# --- Bare-env ---
module load python
conda create -n bare

salloc -N 4 --time 01:00:00
conda activate bare

conda install blas=*=*mkl
conda install libblas=*=*mkl
conda install numpy scipy
conda install -c conda-forge pytest pytest-mpi pytest-cov coverage black isort ruff just pre-commit matplotlib numba -y
cd /path/to/serinv/
python -m pip install -e .
```