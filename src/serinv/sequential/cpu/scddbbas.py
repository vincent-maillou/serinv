# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbbas(
    L: np.ndarray,
    U: np.ndarray,
    B: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
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
    diag_blocksize : int
        The blocksize of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

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
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize

    Y[0:diag_blocksize] = la.solve_triangular(
        L[0:diag_blocksize, 0:diag_blocksize], B[0:diag_blocksize], lower=True
    )
    for i in range(1, n_diag_blocks, 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - sum_k=max(0,i-offdiags)^i-1 L_{i,k} Y_{k})
        B_temp = B[i * diag_blocksize : (i + 1) * diag_blocksize]
        for k in range(max(0, i - n_offdiags), i, 1):
            B_temp = (
                B_temp
                - L[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ Y[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        Y[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            B_temp,
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_temp = B[-arrow_blocksize:]
    for i in range(n_diag_blocks):
        B_temp = (
            B_temp
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ Y[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    Y[-arrow_blocksize:] = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], B_temp, lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:] = la.solve_triangular(
        U[-arrow_blocksize:, -arrow_blocksize:], Y[-arrow_blocksize:], lower=False
    )

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb, ndb+1} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        U[
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
            -arrow_blocksize - diag_blocksize : -arrow_blocksize,
        ],
        Y[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - U[-arrow_blocksize - diag_blocksize : -arrow_blocksize, -arrow_blocksize:]
        @ X[-arrow_blocksize:],
        lower=False,
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k} - U_{i,ndb+1} X_{ndb+1})
        # Y_tmp = Y_{i} - sum_k=i+1^i+n_offdiags U_{i,k} X_{k}
        Y_temp = Y[i * diag_blocksize : (i + 1) * diag_blocksize]

        for k in range(i + 1, min(i + 1 + n_offdiags, n_diag_blocks)):
            Y_temp = (
                Y_temp
                - U[
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                    k * diag_blocksize : (k + 1) * diag_blocksize,
                ]
                @ X[k * diag_blocksize : (k + 1) * diag_blocksize]
            )

        # Y_tmp = Y_tmp - U_{i,ndb+1} X_{ndb+1}
        Y_temp = (
            Y_temp
            - U[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
            @ X[-arrow_blocksize:]
        )

        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            U[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            Y_temp,
            lower=False,
        )

    return X
