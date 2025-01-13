# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import ArrayLike, CUPY_AVAIL, _get_module_from_array

import numpy as np

SEED = 63

np.random.seed(SEED)

if CUPY_AVAIL:
    import cupy as cp

    cp.random.seed(cp.uint64(63))


def bta_dense_to_arrays(
    A: ArrayLike,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks.

    Parameters
    ----------
    A : ArrayLike
        Dense representation of the block tridiagonal arrowhead matrix.
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.

    Returns
    -------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_right_blocks : ArrayLike
        The arrow right blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Notes
    -----
    - The BTA matrix in array representation will be returned according
    to the array module of the input matrix, A.
    """
    xp, _ = _get_module_from_array(A)

    A_diagonal_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
        dtype=A.dtype,
    )

    A_lower_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=A.dtype,
    )
    A_upper_diagonal_blocks = xp.zeros(
        (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
        dtype=A.dtype,
    )

    A_arrow_bottom_blocks = xp.zeros(
        (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
        dtype=A.dtype,
    )

    A_arrow_right_blocks = xp.zeros(
        (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
        dtype=A.dtype,
    )

    A_arrow_tip_block = xp.zeros(
        (arrowhead_blocksize, arrowhead_blocksize),
        dtype=A.dtype,
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[i] = A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]
        if i > 0:
            A_lower_diagonal_blocks[i - 1] = A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ]
        if i < n_diag_blocks - 1:
            A_upper_diagonal_blocks[i] = A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ]

        A_arrow_bottom_blocks[i] = A[
            -arrowhead_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ]

        A_arrow_right_blocks[i] = A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            -arrowhead_blocksize:,
        ]

    A_arrow_tip_block[:] = A[-arrowhead_blocksize:, -arrowhead_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )


def bta_arrays_to_dense(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_right_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
):
    """Converts arrays of blocks to a block tridiagonal arrowhead matrix in a dense representation.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_right_blocks : ArrayLike
        The arrow right blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Returns
    -------
    A : ArrayLike
        Dense representation of the block tridiagonal arrowhead matrix.

    Notes
    -----
    - The BTA matrix in array representation will be returned according
    to the array module of the input matrix, A_diagonal_blocks.
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    diagonal_blocksize = A_diagonal_blocks.shape[1]
    arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=A_diagonal_blocks.dtype,
    )

    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = A_diagonal_blocks[i]
        if i > 0:
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = A_lower_diagonal_blocks[i - 1]
        if i < n_diag_blocks - 1:
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = A_upper_diagonal_blocks[i]

        A[
            -arrowhead_blocksize:,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = A_arrow_bottom_blocks[i]

        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            -arrowhead_blocksize:,
        ] = A_arrow_right_blocks[i]

    A[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block[:]

    return A


def symmetrize(
    A: ArrayLike,
):
    """Symmetrizes the given matrix.

    Parameters
    ----------
    A : ArrayLike
        The matrix to symmetrize.
    """

    A[:] = (A + A.conj().T) / 2


def dd_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix.

    Parameters
    ----------
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.
    device_array : bool
        Whether to return a device (CuPy) array or not (NumPy).
    dtype : np.dtype
        Data type of the matrix. Either np.float64 or np.complex128.

    Returns
    -------
    A : ArrayLike
        Random, diagonaly dominant general, block tridiagonal arrowhead matrix.
    """
    if device_array:
        xp = cp
    else:
        xp = np

    A = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    A[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    A[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    A[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        A[
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            A[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(A.shape[0]):
        A[i, i] = 1 + xp.sum(A[i, :])

    return A


def rhs(
    n_rhs: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random right-hand side.

    Parameters
    ----------
    n_rhs : int
        Number of right-hand sides.
    diagonal_blocksize : int
        Size of the diagonal blocks.
    arrowhead_blocksize : int
        Size of the arrowhead blocks.
    n_diag_blocks : int
        Number of diagonal blocks.
    device_array : bool
        Whether to return a device (CuPy) array or not (NumPy).
    dtype : np.dtype
        Data type of the matrix. Either np.float64 or np.complex128.

    Returns
    -------
    B : ArrayLike
        Random right-hand side.
    """
    if device_array:
        xp = cp
    else:
        xp = np

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    B = rc * xp.random.rand(
        diagonal_blocksize * n_diag_blocks + arrowhead_blocksize, n_rhs
    )

    return B
