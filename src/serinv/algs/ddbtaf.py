# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike


def ddbtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_right_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the LU factorization of a block tridiagonal arrowhead matrix using
    a sequential block algorithm.

    Note:
    -----
    - The matrix is assumed to be block-diagonally dominant.
    - The given matrix will be overwritten.
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : ArrayLike
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : ArrayLike
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : ArrayLike
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : ArrayLike
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : ArrayLike
        The block at the tip of the arrowhead.

    Returns
    -------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the lower factor.
    L_arrow_bottom_blocks : ArrayLike
        Bottom arrow blocks of the lower factor.
    L_arrow_tip_block : ArrayLike
        Tip arrow block of the lower factor.
    U_diagonal_blocks : ArrayLike
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : ArrayLike
        Upper diagonal blocks of the upper factor
    U_arrow_right_blocks : ArrayLike
        Right arrow blocks of the upper factor
    U_arrow_tip_block : ArrayLike
        Tip arrow block of the upper factor
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = A_diagonal_blocks.shape[1]
    n_diag_blocks = A_diagonal_blocks.shape[0]

    L_diagonal_blocks = xp.empty_like(A_diagonal_blocks)
    L_lower_diagonal_blocks = xp.empty_like(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks = xp.empty_like(A_arrow_bottom_blocks)
    L_arrow_tip_block = xp.empty_like(A_arrow_tip_block)

    U_diagonal_blocks = xp.empty_like(A_diagonal_blocks)
    U_upper_diagonal_blocks = xp.empty_like(A_upper_diagonal_blocks)
    U_arrow_right_blocks = xp.empty_like(A_arrow_right_blocks)
    U_arrow_tip_block = xp.empty_like(A_arrow_tip_block)

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks[i, :, :],
            U_diagonal_blocks[i, :, :],
        ) = la.lu(
            A_diagonal_blocks[i, :, :],
            permute_l=True,
        )

        # Compute lower factors
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks[i, :, :] = (
            la.solve_triangular(
                U_diagonal_blocks[i, :, :],
                A_lower_diagonal_blocks[i, :, :].conj().T,
                lower=False,
                trans="T" if L_diagonal_blocks.dtype.char == "f" else "C",
            )
            .conj()
            .T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks[i, :, :] = (
            la.solve_triangular(
                U_diagonal_blocks[i, :, :],
                A_arrow_bottom_blocks[i, :, :].conj().T,
                lower=False,
                trans="T" if L_diagonal_blocks.dtype.char == "f" else "C",
            )
            .conj()
            .T
        )

        # Compute upper factors
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks[i, :, :] = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            A_upper_diagonal_blocks[i, :, :],
            lower=True,
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks[i, :, :] = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            A_arrow_right_blocks[i, :, :],
            lower=True,
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks[i + 1, :, :] = (
            A_diagonal_blocks[i + 1, :, :]
            - L_lower_diagonal_blocks[i, :, :] @ U_upper_diagonal_blocks[i, :, :]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks[i + 1, :, :] = (
            A_arrow_bottom_blocks[i + 1, :, :]
            - L_arrow_bottom_blocks[i, :, :] @ U_upper_diagonal_blocks[i, :, :]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks[i + 1, :, :] = (
            A_arrow_right_blocks[i + 1, :, :]
            - L_lower_diagonal_blocks[i, :, :] @ U_arrow_right_blocks[i, :, :]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[i, :, :] @ U_arrow_right_blocks[i, :, :]
        )

    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        L_diagonal_blocks[-1, :, :],
        U_diagonal_blocks[-1, :, :],
    ) = la.lu(
        A_diagonal_blocks[-1, :, :],
        permute_l=True,
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ U_{ndb, ndb}^{-1}
    L_arrow_bottom_blocks[-1, :, :] = (
        la.solve_triangular(
            U_diagonal_blocks[-1, :, :],
            A_arrow_bottom_blocks[-1, :, :].conj().T,
            lower=False,
            trans="T" if L_diagonal_blocks.dtype.char == "f" else "C",
        )
        .conj()
        .T
    )

    # U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} @ A_{ndb, ndb+1}
    U_arrow_right_blocks[-1, :, :] = la.solve_triangular(
        L_diagonal_blocks[-1, :, :],
        A_arrow_right_blocks[-1, :, :],
        lower=True,
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[-1, :, :] @ U_arrow_right_blocks[-1, :, :]
    )

    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (L_arrow_tip_block[:, :], U_arrow_tip_block[:, :]) = la.lu(
        A_arrow_tip_block[:, :], permute_l=True
    )

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        U_arrow_tip_block,
    )
