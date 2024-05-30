# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbts(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    U_diagonal_blocks: np.ndarray,
    U_upper_diagonal_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a linear system given the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor.
    B : np.ndarray
        The right hand side.

    Returns
    -------
    X : np.ndarray
        The solution of the system.
    """

    Y = np.zeros_like(B, dtype=B.dtype)
    X = np.zeros_like(B, dtype=B.dtype)

    # ----- Forward substitution -----
    blocksize = L_diagonal_blocks.shape[0]
    n_blocks = L_diagonal_blocks.shape[1] // blocksize

    Y[0:blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, 0:blocksize],
        B[0:blocksize],
        lower=True,
    )
    for i in range(1, n_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        Y[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            B[i * blocksize : (i + 1) * blocksize, :]
            - L_lower_diagonal_blocks[
                :,
                (i - 1) * blocksize : i * blocksize,
            ]
            @ Y[(i - 1) * blocksize : (i) * blocksize, :],
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:, :] = la.solve_triangular(
        U_diagonal_blocks[:, -blocksize:], Y[-blocksize:], lower=False
    )
    for i in range(n_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1})
        X[i * blocksize : (i + 1) * blocksize, :] = la.solve_triangular(
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            Y[i * blocksize : (i + 1) * blocksize, :]
            - U_upper_diagonal_blocks[
                :,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ X[(i + 1) * blocksize : (i + 2) * blocksize, :],
            lower=False,
        )

    return X
