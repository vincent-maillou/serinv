# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import numpy.linalg as npla
import scipy.linalg as scla


def scpobtf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform the Cholesky factorization of a block tridiagonal matrix using
    a sequential algorithm on CPU backend.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the matrix.

    Returns
    -------
    L_diagonal_blocks : np.ndarray
    L_lower_diagonal_blocks : np.ndarray
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    L_diagonal_blocks = np.zeros_like(A_diagonal_blocks)
    L_lower_diagonal_blocks = np.zeros_like(A_lower_diagonal_blocks)

    for i in range(0, nblocks - 1, 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = npla.cholesky(
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
        )

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            A_lower_diagonal_blocks[
                :,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ scla.solve_triangular(
                L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
                np.eye(blocksize),
                lower=True,
            ).T
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}^{T}
        A_diagonal_blocks[
            :,
            (i + 1) * blocksize : (i + 2) * blocksize,
        ] = (
            A_diagonal_blocks[
                :,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            - L_lower_diagonal_blocks[
                :,
                i * blocksize : (i + 1) * blocksize,
            ]
            @ L_lower_diagonal_blocks[
                :,
                i * blocksize : (i + 1) * blocksize,
            ].T
        )

    L_diagonal_blocks[:, -blocksize:] = npla.cholesky(A_diagonal_blocks[:, -blocksize:])

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
    )
