# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scpobts(
    L_diagonal_blocks: np.ndarray,
    L_lower_diagonal_blocks: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Solve a linear system using a cholesky factorization of a block tridiagonal matrix
    using a sequential algorithm on CPU backend.


    Parameters
    ----------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the cholesky factorization.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the cholesky factorization.
    B : np.ndarray
        Right-hand side of the linear system.

    Returns
    -------
    X : np.ndarray
        Solution of the linear system.
    """

    # ----- Forward substitution -----
    blocksize = L_diagonal_blocks.shape[0]
    nblocks = L_diagonal_blocks.shape[1] // blocksize

    X = np.zeros_like(B)

    X[0:blocksize] = la.solve_triangular(
        L_diagonal_blocks[:, 0:blocksize], B[0:blocksize], lower=True
    )

    for i in range(1, nblocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            B[i * blocksize : (i + 1) * blocksize]
            - L_lower_diagonal_blocks[:, (i - 1) * blocksize : i * blocksize]
            @ X[(i - 1) * blocksize : (i) * blocksize],
            lower=True,
        )

    # ----- Backward substitution -----
    X[-blocksize:] = la.solve_triangular(
        L_diagonal_blocks[:, -blocksize:], X[-blocksize:], lower=True, trans="T"
    )

    for i in range(nblocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i} X_{i+1})
        X[i * blocksize : (i + 1) * blocksize] = la.solve_triangular(
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            X[i * blocksize : (i + 1) * blocksize]
            - L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T
            @ X[(i + 1) * blocksize : (i + 2) * blocksize],
            lower=True,
            trans="T",
        )

    return X
