# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobbs(
    L: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Solve a linear system using a cholesky factorization of a block bidiagonal matrix
    using a sequential algorithm on CPU backend.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    B : np.ndarray
        The right hand side.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        The blocksize of the matrix.
    overwrite: bool
        If True, the rhs B is overwritten with the solution X. Default is False.


    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    if overwrite:
        X = B
    else:
        X = np.copy(B)

    # temporary buffer for sum
    B_temp = np.zeros_like(B[:blocksize])

    n_offdiags_blk = ndiags // 2

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    X[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_{k=max(0,i-offdiags)}^{i-1} L_{i,k} Y_{k})
        B_temp[:blocksize] = B[i * blocksize : (i + 1) * blocksize]
        for k in range(max(0, i - n_offdiags_blk), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
            lower=True,
        )

    # ----- Backward substitution -----

    X[-blocksize:] = la.solve_triangular(
        L[-blocksize:, -blocksize:], X[-blocksize:], lower=True, trans="T"
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - sum_{k=i+1}^{i+n_offdiags_blk} L^T_{k,i} X_{k})
        B_temp[:blocksize] = X[i * blocksize : (i + 1) * blocksize]
        for k in range(i + 1, min(i + 1 + n_offdiags_blk, n_blocks)):
            B_temp = (
                B_temp
                - L[
                    k * blocksize : (k + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ].T
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
            lower=True,
            trans="T",
        )

    return X
