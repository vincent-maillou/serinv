# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import cupy as cp
import cupyx as cpx
import cupyx.scipy.linalg as cpla
import numpy as np


def sgddbtf(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the LU factorization of a block tridiagonal matrix using
    a sequential algotithm on a GPU backend.

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

    A_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_diagonal_blocks)
    A_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_lower_diagonal_blocks)
    A_upper_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_upper_diagonal_blocks)

    # Host side arrays
    L_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_gpu)
    L_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        A_lower_diagonal_blocks_gpu
    )
    U_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_gpu)
    U_upper_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        A_upper_diagonal_blocks_gpu
    )

    # Device side arrays
    L_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_lower_diagonal_blocks)

    U_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_upper_diagonal_blocks)

    for i in range(0, nblocks - 1, 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            U_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
        ) = cpla.lu(
            A_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )

        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_gpu[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = A_lower_diagonal_blocks_gpu[
            :, i * blocksize : (i + 1) * blocksize
        ] @ cpla.solve_triangular(
            U_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            cp.eye(blocksize),
            trans=0,
            lower=False,
            unit_diagonal=False,
        )

        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_gpu[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            cpla.solve_triangular(
                L_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
                cp.eye(blocksize),
                trans=0,
                lower=True,
                unit_diagonal=True,
            )
            @ A_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
        )

        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize] = (
            A_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize]
            - L_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ U_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_gpu[:, -blocksize:],
        U_diagonal_blocks_gpu[:, -blocksize:],
    ) = cpla.lu(A_diagonal_blocks_gpu[:, -blocksize:], permute_l=True)

    L_diagonal_blocks_gpu.get(out=L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu.get(out=L_lower_diagonal_blocks)
    U_diagonal_blocks_gpu.get(out=U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu.get(out=U_upper_diagonal_blocks)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )
