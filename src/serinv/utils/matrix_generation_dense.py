# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
from serinv.utils.matrix_transformation_dense import (
    make_dense_matrix_diagonally_dominante,
    make_dense_matrix_symmetric,
)


def generate_block_tridiagonal_dense(
    nblocks: int,
    blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """Generate a block tridiagonal matrix, returned as dense matrix.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal dense matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = nblocks * blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            np.random.rand(blocksize, blocksize)
        )
        if i > 0:
            A[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ] = np.random.rand(blocksize, blocksize)
            A[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ] = np.random.rand(blocksize, blocksize)

    if symmetric:
        A = make_dense_matrix_symmetric(A)

    if diagonal_dominant:
        A = make_dense_matrix_diagonally_dominante(A)

    return A


def generate_block_tridiagonal_arrowhead_dense(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """Generate a block tridiagonal arrowhead matrix, returned as dense.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes:
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = (nblocks - 1) * diag_blocksize + arrow_blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks - 1:
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = np.random.rand(diag_blocksize, diag_blocksize)
            if i > 0:
                A[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                ] = np.random.rand(diag_blocksize, diag_blocksize)
                A[
                    (i - 1) * diag_blocksize : i * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ] = np.random.rand(diag_blocksize, diag_blocksize)

            A[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = np.random.rand(arrow_blocksize, diag_blocksize)
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
            ] = np.random.rand(diag_blocksize, arrow_blocksize)

        else:
            A[
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
            ] = np.random.rand(arrow_blocksize, arrow_blocksize)

    if symmetric:
        A = make_dense_matrix_symmetric(A)

    if diagonal_dominant:
        A = make_dense_matrix_diagonally_dominante(A)

    return A


def generate_blocks_banded_dense(
    nblocks: int,
    ndiags: int,
    blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """Generate a block n-diagonals matrix, returned as dense.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block n-diagonals matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    if (ndiags + 1) / 2 > nblocks:
        raise ValueError("(ndiags+1)/2 must be smaller or equal to nblocks")

    if ndiags % 2 == 0:
        raise ValueError("ndiags must be odd")

    matrice_size = nblocks * blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            np.random.rand(blocksize, blocksize)
        )
        for j in range(1, (ndiags + 1) // 2):
            if i + j < nblocks:
                A[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ] = np.random.rand(blocksize, blocksize)
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] = np.random.rand(blocksize, blocksize)
            if i - j >= 0:
                A[
                    i * blocksize : (i + 1) * blocksize,
                    (i - j) * blocksize : (i - j + 1) * blocksize,
                ] = np.random.rand(blocksize, blocksize)
                A[
                    (i - j) * blocksize : (i - j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] = np.random.rand(blocksize, blocksize)

    if symmetric:
        A = make_dense_matrix_symmetric(A)

    if diagonal_dominant:
        A = make_dense_matrix_diagonally_dominante(A)

    return A


def generate_blocks_banded_arrowhead_dense(
    nblocks: int,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> np.ndarray:
    """Generate a block tridiagonal arrowhead matrix, returned as dense.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes:
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A : np.ndarray
        Block tridiagonal arrowhead matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    matrice_size = (nblocks - 1) * diag_blocksize + arrow_blocksize

    A = np.zeros((matrice_size, matrice_size))

    for i in range(nblocks):
        if i < nblocks - 1:
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = np.random.rand(diag_blocksize, diag_blocksize)
            for j in range(1, (ndiags + 1) // 2):
                if i + j < nblocks - 1:
                    A[
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    ] = np.random.rand(diag_blocksize, diag_blocksize)
                    A[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ] = np.random.rand(diag_blocksize, diag_blocksize)
                if i - j >= 0:
                    A[
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                        (i - j) * diag_blocksize : (i - j + 1) * diag_blocksize,
                    ] = np.random.rand(diag_blocksize, diag_blocksize)
                    A[
                        (i - j) * diag_blocksize : (i - j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ] = np.random.rand(diag_blocksize, diag_blocksize)

            A[
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = np.random.rand(arrow_blocksize, diag_blocksize)
            A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (nblocks - 1) * diag_blocksize : (nblocks - 1) * diag_blocksize
                + arrow_blocksize,
            ] = np.random.rand(diag_blocksize, arrow_blocksize)

        else:
            A[
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
                i * diag_blocksize : i * diag_blocksize + arrow_blocksize,
            ] = np.random.rand(arrow_blocksize, arrow_blocksize)

    if symmetric:
        A = make_dense_matrix_symmetric(A)

    if diagonal_dominant:
        A = make_dense_matrix_diagonally_dominante(A)

    return A
