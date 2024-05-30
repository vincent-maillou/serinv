# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import numpy.linalg as npla
import scipy.linalg as la
import scipy.linalg as scla


def pobtaf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the Cholesky factorization of a block tridiagonal matrix using
    a sequential algorithm on CPU backend.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the matrix.
    A_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the matrix.
    A_arrow_tip_block : np.ndarray
        Arrow tip block of the matrix.

    Returns
    -------
    L_diagonal_blocks : np.ndarray
    L_lower_diagonal_blocks : np.ndarray
    L_arrow_bottom_blocks : np.ndarray
    L_arrow_tip_block : np.ndarray
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    L_diagonal_blocks = np.zeros_like(A_diagonal_blocks)
    L_lower_diagonal_blocks = np.zeros_like(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks = np.zeros_like(A_arrow_bottom_blocks)
    L_arrow_tip_block = np.zeros_like(A_arrow_tip_block)

    L_inv_temp = np.zeros(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = npla.cholesky(
            A_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
        )

        # Temporary storage of used twice lower triangular solving
        L_inv_temp[:, :] = la.solve_triangular(
            L_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            A_lower_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_inv_temp
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_inv_temp
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A_diagonal_blocks[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_diagonal_blocks[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_lower_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A_arrow_bottom_blocks[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_arrow_bottom_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_arrow_bottom_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks[:, -diag_blocksize:] = npla.cholesky(
        A_diagonal_blocks[:, -diag_blocksize:]
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks[:, -diag_blocksize:]
        @ la.solve_triangular(
            L_diagonal_blocks[:, -diag_blocksize:],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[:, -diag_blocksize:]
        @ L_arrow_bottom_blocks[:, -diag_blocksize:].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block[:, :] = npla.cholesky(A_arrow_tip_block[:, :])

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )
