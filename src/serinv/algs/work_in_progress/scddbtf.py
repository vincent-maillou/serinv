# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la


def scddbtf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a CPU backend.

    The matrix is assumed to be block-diagonally dominant.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray
        The blocks on the upper diagonal of the matrix.

    Returns
    -------
    L_diagonal_blocks : np.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the lower factor.
    U_diagonal_blocks : np.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the upper factor
    """
    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    L_diagonal_blocks = np.empty(
        (blocksize, nblocks * blocksize), dtype=A_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks = np.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=A_diagonal_blocks.dtype
    )

    U_diagonal_blocks = np.empty(
        (blocksize, nblocks * blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_upper_diagonal_blocks = np.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=A_diagonal_blocks.dtype
    )

    for i in range(0, nblocks - 1, 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
        ) = la.lu(
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = A_lower_diagonal_blocks[
            :, i * blocksize : (i + 1) * blocksize
        ] @ la.solve_triangular(
            U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
            np.eye(blocksize),
            lower=False,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            la.solve_triangular(
                L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
                np.eye(blocksize),
                lower=True,
            )
            @ A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize] = (
            A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
            - L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
            @ U_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks[:, -blocksize:],
        U_diagonal_blocks[:, -blocksize:],
    ) = la.lu(A_diagonal_blocks[:, -blocksize:], permute_l=True)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )
