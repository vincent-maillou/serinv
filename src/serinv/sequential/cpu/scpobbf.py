# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import numpy.linalg as npla
import scipy.linalg as la
import scipy.linalg as scla


def scpobbf(
    A: np.ndarray,
    ndiags: int,
    blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals matrix. The
    matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        Size of the blocks of the matrix.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L = A
    else:
        L = np.copy(A)

    L_inv_temp = np.zeros((blocksize, blocksize))

    n_offdiags_blk = ndiags // 2

    nblocks = A.shape[0] // blocksize
    for i in range(0, nblocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize] = (
            la.cholesky(
                L[
                    i * blocksize : (i + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
            ).T
        )

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=True,
        ).T

        for j in range(1, min(n_offdiags_blk + 1, nblocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L[
                (i + j) * blocksize : (i + j + 1) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ] = (
                L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                @ L_inv_temp
            )

            for k in range(1, j + 1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L[
                    (i + j) * blocksize : (i + j + 1) * blocksize,
                    (i + k) * blocksize : (i + k + 1) * blocksize,
                ] = (
                    L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                    ]
                    - L[
                        (i + j) * blocksize : (i + j + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ]
                    @ L[
                        (i + k) * blocksize : (i + k + 1) * blocksize,
                        i * blocksize : (i + 1) * blocksize,
                    ].T
                )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[-blocksize:, -blocksize:] = la.cholesky(L[-blocksize:, -blocksize:]).T

    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)

    return L
