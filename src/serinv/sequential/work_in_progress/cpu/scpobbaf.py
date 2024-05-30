# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import numpy.linalg as npla
import scipy.linalg as la
import scipy.linalg as scla


def scpobbaf(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
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

    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_offdiags_blk = ndiags // 2

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = la.cholesky(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
        ).T

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L[
                (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ] = (
                L[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ]
                @ L_inv_temp
            )

            for k in range(1, j + 1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L[
                    (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ] = (
                    L[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    ]
                    - L[
                        (i + j) * diag_blocksize : (i + j + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ]
                    @ L[
                        (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                        i * diag_blocksize : (i + 1) * diag_blocksize,
                    ].T
                )

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_inv_temp
        )

        for k in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # A_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            L[
                -arrow_blocksize:,
                (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
            ] = (
                L[
                    -arrow_blocksize:,
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                ]
                - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
                @ L[
                    (i + k) * diag_blocksize : (i + k + 1) * diag_blocksize,
                    i * diag_blocksize : (i + 1) * diag_blocksize,
                ].T
            )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        L[-arrow_blocksize:, -arrow_blocksize:] = (
            L[-arrow_blocksize:, -arrow_blocksize:]
            - L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L[-arrow_blocksize:, i * diag_blocksize : (i + 1) * diag_blocksize].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        -diag_blocksize - arrow_blocksize : -arrow_blocksize,
    ] = la.cholesky(
        L[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            -diag_blocksize - arrow_blocksize : -arrow_blocksize,
        ]
    ).T

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize] = (
        L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ la.solve_triangular(
            L[
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
                -diag_blocksize - arrow_blocksize : -arrow_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L[-arrow_blocksize:, -arrow_blocksize:] = (
        L[-arrow_blocksize:, -arrow_blocksize:]
        - L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize]
        @ L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize : -arrow_blocksize].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L[-arrow_blocksize:, -arrow_blocksize:] = la.cholesky(
        L[-arrow_blocksize:, -arrow_blocksize:]
    ).T

    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)

    return L
