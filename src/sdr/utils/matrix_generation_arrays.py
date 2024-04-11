"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Matrix generations routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
from sdr.utils.matrix_transformation import (
    make_symmetric_tridiagonal_arrays,
    make_diagonally_dominante_tridiagonal_arrays,
    make_diagonally_dominante_tridiagonal_arrowhead_arrays,
)


def generate_tridiag_array(
    nblocks: int,
    blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Generate a block tridiagonal matrix returned as three arrays.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    blocksize : int
        Size of the blocks.
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the block tridiagonal matrix.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the block tridiagonal matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the block tridiagonal matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    A_diagonal_blocks = np.empty((blocksize, nblocks * blocksize))
    A_upper_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))
    A_lower_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))

    for i in range(nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = np.random.rand(
            blocksize, blocksize
        )
        if i > 0:
            A_upper_diagonal_blocks[
                :, (i - 1) * blocksize : i * blocksize
            ] = np.random.rand(blocksize, blocksize)
        if i < nblocks - 1:
            A_lower_diagonal_blocks[
                :, i * blocksize : (i + 1) * blocksize
            ] = np.random.rand(blocksize, blocksize)

    if symmetric:
        (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
        ) = make_symmetric_tridiagonal_arrays(
            A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
        )

    if diagonal_dominant:
        A_diagonal_blocks = make_diagonally_dominante_tridiagonal_arrays(
            A_diagonal_blocks, A_upper_diagonal_blocks, A_lower_diagonal_blocks
        )

    return (A_diagonal_blocks, A_upper_diagonal_blocks, A_lower_diagonal_blocks)


def generate_tridiag_arrowhead_arrays(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    symmetric: bool = False,
    diagonal_dominant: bool = False,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a block tridiagonal arrowhead matrix, returned as arrays.

    Parameters
    ----------
    nblocks : int
        Number of diagonal blocks.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrowhead blocks. These blocks will be of sizes:
        (arrow_blocksize*diag_blocksize).
    symmetric : bool, optional, default=False
        If True, the matrix will be symmetric.
    diagonal_dominant : bool, optional, default=False
        If True, the matrix will be diagonally dominant.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the block tridiagonal arrowhead matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the block tridiagonal arrowhead matrix.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the block tridiagonal arrowhead matrix.
    A_arrow_bottom_blocks : np.ndarray
        Bottom arrowhead blocks of the block tridiagonal arrowhead matrix.
    A_arrow_right_blocks : np.ndarray
        Right arrowhead blocks of the block tridiagonal arrowhead matrix.
    A_arrow_tip_block : np.ndarray
        Tip arrowhead block of the block tridiagonal arrowhead matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    n_diag_blocks = nblocks - 1

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = generate_tridiag_array(
        n_diag_blocks, diag_blocksize, symmetric, diagonal_dominant, seed
    )

    A_arrow_bottom_blocks = np.random.rand(
        arrow_blocksize, n_diag_blocks * diag_blocksize
    )
    A_arrow_right_blocks = np.random.rand(
        arrow_blocksize, n_diag_blocks * diag_blocksize
    )

    A_arrow_tip_block = np.random.rand(arrow_blocksize, arrow_blocksize)

    if symmetric:
        (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
        ) = make_symmetric_tridiagonal_arrays(
            A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
        )

        for i in range(n_diag_blocks):
            A_arrow_bottom_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] = A_arrow_right_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize].T

        A_arrow_tip_block += A_arrow_tip_block.T

    if diagonal_dominant:
        (
            A_diagonal_blocks
        ) = make_diagonally_dominante_tridiagonal_arrowhead_arrays(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )
