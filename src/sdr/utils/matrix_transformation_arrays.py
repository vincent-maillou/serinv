"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Matrix transformations routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np


# -----------------------------------------------
# Section: Make matrix symmetric
# -----------------------------------------------

def make_arrays_block_tridiagonal_symmetric(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a tridiagonal matrix, passed as arrays, symmetric.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made symmetric.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks made symmetric.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks made symmetric.
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    for i in range(nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += A_diagonal_blocks[
            :, i * blocksize : (i + 1) * blocksize
        ].T
        if i < nblocks - 1:
            A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = (
                A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize].T
            )

    return (A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)


def make_arrays_block_tridiagonal_arrowhead_symmetric(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
    ) = make_arrays_block_tridiagonal_symmetric(
        A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
    )

    diagonal_blocksize = A_diagonal_blocks.shape[0]
    n_diag_blocks = A_arrow_bottom_blocks.shape[1] // diagonal_blocksize

    for i in range(n_diag_blocks):
        A_arrow_bottom_blocks[
            :, i * diagonal_blocksize : (i + 1) * diagonal_blocksize
        ] = A_arrow_right_blocks[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize, :
        ].T

    A_arrow_tip_block += A_arrow_tip_block.T

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )


def make_arrays_blocks_banded_symmetric():
    pass


def make_arrays_blocks_banded_arrowhead_symmetric():
    pass


# -----------------------------------------------
# Section: Make diagonally dominante
# -----------------------------------------------


def make_arrays_block_tridiagonal_diagonally_dominante(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> np.ndarray:
    """Make a tridiagonal matrix, passed as arrays, diagonally dominant.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made diagonally dominant.
    """

    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    for i in range(0, nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
            np.sum(
                np.abs(A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]),
                axis=1,
            )
        )
        if i > 0:
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
                np.sum(
                    np.abs(
                        A_lower_diagonal_blocks[:, (i - 1) * blocksize : i * blocksize]
                    ),
                    axis=1,
                )
            )
        if i < nblocks - 1:
            A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] += np.diag(
                np.sum(
                    np.abs(
                        A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize]
                    ),
                    axis=1,
                )
            )

    return A_diagonal_blocks


def make_arrays_block_tridiagonal_arrowhead_diagonally_dominante(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> np.ndarray:
    """Make a tridiagonal arrowhead matrix, passed as arrays, diagonally dominant.

    Parameters
    ----------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the input matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the input matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the input matrix.
    A_arrow_bottom_blocks: np.ndarray
        Arrow bottom blocks of the input matrix.
    A_arrow_right_blocks: np.ndarray
        Arrow right blocks of the input matrix.
    A_arrow_tip_block: np.ndarray
        Arrow tip block of the input matrix.

    Returns
    -------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks made diagonally dominant.
    """

    diag_blocksize = A_diagonal_blocks.shape[0]
    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    (A_diagonal_blocks) = make_arrays_block_tridiagonal_diagonally_dominante(
        A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] += np.diag(
            np.sum(
                np.abs(
                    A_arrow_right_blocks[
                        i * diag_blocksize : (i + 1) * diag_blocksize, :
                    ]
                ),
                axis=1,
            )
        )

    A_arrow_tip_block += np.diag(np.sum(np.abs(A_arrow_tip_block), axis=1))
    A_arrow_tip_block += np.diag(np.sum(np.abs(A_arrow_bottom_blocks), axis=1))

    return (A_diagonal_blocks, A_arrow_tip_block)


def make_arrays_blocks_banded_diagonally_dominante():
    pass


def make_arrays_blocks_banded_arrowhead_diagonally_dominante():
    pass


# -----------------------------------------------
# Section: Convert from dense to arrays
# -----------------------------------------------

def convert_block_tridiagonal_dense_to_arrays(
    A: np.ndarray,
    blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the tridiagonal blocks from a dense matrix.

    Parameters
    ----------
    A : np.ndarray
        Dense matrix.
    blocksize : int
        Size of the blocks.

    Returns
    -------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the tridiagonal matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the tridiagonal matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the tridiagonal matrix.
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    A_diagonal_blocks = np.empty((blocksize, nblocks * blocksize))
    A_upper_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))
    A_lower_diagonal_blocks = np.empty((blocksize, (nblocks - 1) * blocksize))

    for i in range(nblocks):
        A_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ]
        if i < (nblocks - 1):
            A_upper_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
                i * blocksize : (i + 1) * blocksize,
                (i + 1) * blocksize : (i + 2) * blocksize,
            ]
            A_lower_diagonal_blocks[:, i * blocksize : (i + 1) * blocksize] = A[
                (i + 1) * blocksize : (i + 2) * blocksize,
                i * blocksize : (i + 1) * blocksize,
            ]

    return (A_diagonal_blocks, A_lower_diagonal_blocks, A_upper_diagonal_blocks)


def convert_block_tridiagonal_arrowhead_dense_to_arrays(
    A: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the arrowhead blocks from a dense matrix.

    Parameters
    ----------
    A : np.ndarray
        Dense matrix.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    A_diagonal_blocks: np.ndarray
        Diagonal blocks of the tridiagonal matrix.
    A_lower_diagonal_blocks: np.ndarray
        Lower diagonal blocks of the tridiagonal matrix.
    A_upper_diagonal_blocks: np.ndarray
        Upper diagonal blocks of the tridiagonal matrix.
    A_arrow_bottom_blocks: np.ndarray
        Arrow bottom blocks of the tridiagonal matrix.
    A_arrow_right_blocks: np.ndarray
        Arrow right blocks of the tridiagonal matrix.
    A_arrow_tip_block: np.ndarray
        Arrow tip block of the tridiagonal matrix.
    """

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize

    A_diagonal_blocks = np.empty((diag_blocksize, n_diag_blocks * diag_blocksize))
    A_lower_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize)
    )
    A_upper_diagonal_blocks = np.empty(
        (diag_blocksize, (n_diag_blocks - 1) * diag_blocksize)
    )
    A_arrow_bottom_blocks = np.empty((arrow_blocksize, n_diag_blocks * diag_blocksize))
    A_arrow_right_blocks = np.empty((n_diag_blocks * diag_blocksize, arrow_blocksize))
    A_arrow_tip_block = np.empty((arrow_blocksize, arrow_blocksize))

    for i in range(0, n_diag_blocks):
        A_diagonal_blocks[:, i * diag_blocksize : (i + 1) * diag_blocksize] = A[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ]
        if i < n_diag_blocks - 1:
            A_lower_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] = A[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            A_upper_diagonal_blocks[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ] = A[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]

    A_arrow_bottom_blocks = A[-arrow_blocksize:, :-arrow_blocksize]
    A_arrow_right_blocks = A[:-arrow_blocksize, -arrow_blocksize:]
    A_arrow_tip_block = A[-arrow_blocksize:, -arrow_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )


def convert_blocks_banded_dense_to_arrays():
    pass


def convert_blocks_banded_arrowhead_dense_to_arrays():
    pass
