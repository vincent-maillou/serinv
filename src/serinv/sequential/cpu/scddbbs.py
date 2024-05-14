# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbbs(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """Solve a linear system given the LU factorization of a block banded matrix using
    a sequential algotithm on a CPU backend.

    Parameters
    ----------
    L : np.ndarray
        The LU factorization of the matrix.
    U : np.ndarray
        The LU factorization of the matrix.
    B : np.ndarray
        The right hand side.
    ndiags : int
        Number of diagonals of the matrix.
    blocksize : int
        The blocksize of the matrix.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    # number of lower (or upper) off-diagonal blocks
    n_offdiags = int((ndiags - 1) / 2)

    Y = np.zeros_like(B)
    X = np.zeros_like(B)

    # ----- Forward substitution -----
    n_blocks = L.shape[0] // blocksize
    Y[0:blocksize] = la.solve_triangular(
        L[0:blocksize, 0:blocksize], B[0:blocksize], lower=True
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_k=max(0,i-offdiags)^i-1 L_{i,k} Y_{k})
        B_temp = B[i * blocksize : (i + 1) * blocksize]
        for k in range(max(0, i - n_offdiags), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ Y[k * blocksize : (k + 1) * blocksize]
            )

        Y[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            B_temp,
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(
        U[-blocksize:, -blocksize:], Y[-blocksize:], lower=False
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k})
        Y_temp = Y[i * blocksize : (i + 1) * blocksize]
        for k in range(i + 1, min(i + 1 + n_offdiags, n_blocks)):
            Y_temp = (
                Y_temp
                - U[
                    i * blocksize : (i + 1) * blocksize,
                    k * blocksize : (k + 1) * blocksize,
                ]
                @ X[k * blocksize : (k + 1) * blocksize]
            )

        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            U[i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize],
            Y_temp,
            lower=False,
        )

    return X
