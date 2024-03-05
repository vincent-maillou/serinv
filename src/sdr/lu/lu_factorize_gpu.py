"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-02

Contains the lu selected factorization routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import scipy.linalg as la

import cupy as cp

def lu_factorize_tridiag_gpu(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the non-pivoted LU factorization of a block tridiagonal matrix.
    The matrix is assumed to be non-singular and blocks are assumed to be of the
    same size given in a sequential array.

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

    L_diagonal_blocks_gpu = cp.empty(
        (blocksize, nblocks * blocksize), dtype=A_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_gpu = cp.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=A_diagonal_blocks.dtype
    )

    U_diagonal_blocks_gpu = cp.empty(
        (blocksize, nblocks * blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_upper_diagonal_blocks_gpu = cp.empty(
        (blocksize, (nblocks - 1) * blocksize), dtype=A_diagonal_blocks.dtype
    )

    # for i in range(0, nblocks - 1, 1):
    #     # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
    #     (
    #         L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
    #         U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
    #     ) = la.lu(
    #         A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
    #         permute_l=True,
    #     )

    #     # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
    #     L_lower_diagonal_blocks[
    #         :,
    #         i * blocksize : (i + 1) * blocksize,
    #     ] = A_lower_diagonal_blocks[
    #         :, i * blocksize : (i + 1) * blocksize
    #     ] @ la.solve_triangular(
    #         U_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
    #         np.eye(blocksize),
    #         lower=False,
    #     )

    #     # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
    #     U_upper_diagonal_blocks[
    #         :,
    #         i * blocksize : (i + 1) * blocksize,
    #     ] = (
    #         la.solve_triangular(
    #             L_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize],
    #             np.eye(blocksize),
    #             lower=True,
    #         )
    #         @ A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
    #     )

    #     # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
    #     A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize] = (
    #         A_diagonal_blocks[:, (i + 1) * blocksize : (i + 2) * blocksize]
    #         - L_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
    #         @ U_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
    #     )

    # # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    # (
    #     L_diagonal_blocks[:, -blocksize:],
    #     U_diagonal_blocks[:, -blocksize:],
    # ) = la.lu(A_diagonal_blocks[:, -blocksize:], permute_l=True)

    L_diagonal_blocks: np.ndarray = cp.asnumpy(L_diagonal_blocks_gpu)
    L_lower_diagonal_blocks: np.ndarray = cp.asnumpy(L_lower_diagonal_blocks_gpu)
    U_diagonal_blocks: np.ndarray = cp.asnumpy(U_diagonal_blocks_gpu)
    U_upper_diagonal_blocks: np.ndarray = cp.asnumpy(U_upper_diagonal_blocks_gpu)

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
    )


