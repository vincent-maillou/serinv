# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbbf(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Perform the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a CPU backend.

    The matrix is assumed to be block-diagonally dominant.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    L : np.ndarray
        Lower factor of the LU factorization of the matrix.
    U : np.ndarray
        Upper factor of the LU factorization of the matrix.
    """

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    L_inv_temp = np.zeros((blocksize, blocksize))
    U_inv_temp = np.zeros((blocksize, blocksize))

    n_offdiags_blk = ndiags // 2

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
        ) = la.lu(
            A[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )

        # Invert L_{i, i} and U_{i, i} for later use
        L_inv_temp = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        )
        U_inv_temp = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        for j in range(1, min(n_offdiags_blk + 1, nblocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ U{i, i}^{-1}
            L[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ] = (
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ U_inv_temp
            )

            # U_{i, i+j} = L{i, i}^{-1} @ A_{i, i+j}
            U[
                i * blocksize : (i + 1) * blocksize,
                (i + j) * blocksize : (i + j + 1) * blocksize,
            ] = (
                L_inv_temp
                @ A[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
            )

            for k in range(1, j):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ U_{i, i+k}
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                ] = (
                    A[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                    - L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ U[
                        i * blocksize : (i + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                )

                # A_{i+k, i+j} = A_{i+k, i+j} - L_{i+k, i} @ U_{i, i+j}
                A[
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ] = (
                    A[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                    ]
                    - L[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ U[
                        i * blocksize : (i + 1) * blocksize,
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                    ]
                )

            # A_{i+j, i+j} = A_{i+j, i+j} - L_{i+j, i} @ U_{i, i+j}
            A[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                (i + j) * blocksize : (i + j + 1) * blocksize,
            ] = (
                A[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
                - L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ U[
                    i * blocksize : (i + 1) * blocksize,
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                ]
            )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    L[-blocksize:, -blocksize:], U[-blocksize:, -blocksize:] = la.lu(
        A[-blocksize:, -blocksize:], permute_l=True
    )

    return L, U
