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
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
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
    LU_diagonal_blocks : ArrayLike
        LU factors of the diagonal blocks.
    LU_lower_diagonal_blocks : ArrayLike
        LU factors of the lower diagonal blocks.
    LU_upper_diagonal_blocks : ArrayLike
        LU factors of the upper diagonal blocks.
    LU_arrow_bottom_blocks : ArrayLike
        LU factors of the bottom arrow blocks.
    LU_arrow_right_blocks : ArrayLike
        LU factors of the right arrow blocks.
    LU_arrow_tip_block : ArrayLike
        LU factors of the tip block of the arrowhead.
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    n_diag_blocks = A_diagonal_blocks.shape[0]

    # LU aliases
    LU_diagonal_blocks = A_diagonal_blocks
    LU_lower_diagonal_blocks = A_lower_diagonal_blocks
    LU_upper_diagonal_blocks = A_upper_diagonal_blocks
    LU_arrow_bottom_blocks = A_arrow_bottom_blocks
    LU_arrow_right_blocks = A_arrow_right_blocks
    LU_arrow_tip_block = A_arrow_tip_block

    # Pivots array
    P_diag = xp.zeros((n_diag_blocks, A_diagonal_blocks.shape[1]), dtype=xp.int32)
    P_tip = xp.zeros((A_arrow_tip_block.shape[1]), dtype=xp.int32)

    for i in range(0, n_diag_blocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            LU_diagonal_blocks[i, :, :],
            P_diag[i, :],
        ) = la.lu_factor(
            A_diagonal_blocks[i, :, :],
        )

        # Compute lower factors
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        LU_lower_diagonal_blocks[i, :, :] = (
            la.solve_triangular(
                LU_diagonal_blocks[i, :, :],
                A_lower_diagonal_blocks[i, :, :].conj().T,
                lower=False,
                trans="T" if LU_diagonal_blocks.dtype.char == "f" else "C",
            )
            .conj()
            .T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        LU_arrow_bottom_blocks[i, :, :] = (
            la.solve_triangular(
                LU_diagonal_blocks[i, :, :],
                A_arrow_bottom_blocks[i, :, :].conj().T,
                lower=False,
                trans="T" if LU_diagonal_blocks.dtype.char == "f" else "C",
            )
            .conj()
            .T
        )

        # Compute upper factors
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        LU_upper_diagonal_blocks[i, :, :] = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            A_upper_diagonal_blocks[i, :, :],
            lower=True,
            unit_diagonal=True,
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        LU_arrow_right_blocks[i, :, :] = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            A_arrow_right_blocks[i, :, :],
            lower=True,
            unit_diagonal=True,
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks[i + 1, :, :] = (
            A_diagonal_blocks[i + 1, :, :]
            - LU_lower_diagonal_blocks[i, :, :] @ LU_upper_diagonal_blocks[i, :, :]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks[i + 1, :, :] = (
            A_arrow_bottom_blocks[i + 1, :, :]
            - LU_arrow_bottom_blocks[i, :, :] @ LU_upper_diagonal_blocks[i, :, :]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks[i + 1, :, :] = (
            A_arrow_right_blocks[i + 1, :, :]
            - LU_lower_diagonal_blocks[i, :, :] @ LU_arrow_right_blocks[i, :, :]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - LU_arrow_bottom_blocks[i, :, :] @ LU_arrow_right_blocks[i, :, :]
        )

    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        LU_diagonal_blocks[-1, :, :],
        P_diag[-1, :],
    ) = la.lu_factor(
        A_diagonal_blocks[-1, :, :],
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ U_{ndb, ndb}^{-1}
    LU_arrow_bottom_blocks[-1, :, :] = (
        la.solve_triangular(
            LU_diagonal_blocks[-1, :, :],
            A_arrow_bottom_blocks[-1, :, :].conj().T,
            lower=False,
            trans="T" if LU_diagonal_blocks.dtype.char == "f" else "C",
        )
        .conj()
        .T
    )

    # U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} @ A_{ndb, ndb+1}
    LU_arrow_right_blocks[-1, :, :] = la.solve_triangular(
        LU_diagonal_blocks[-1, :, :],
        A_arrow_right_blocks[-1, :, :],
        lower=True,
        unit_diagonal=True,
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - LU_arrow_bottom_blocks[-1, :, :] @ LU_arrow_right_blocks[-1, :, :]
    )

    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (
        LU_arrow_tip_block[:, :],
        P_tip[:],
    ) = la.lu_factor(
        A_arrow_tip_block[:, :],
    )

    return (
        LU_diagonal_blocks,
        LU_lower_diagonal_blocks,
        LU_upper_diagonal_blocks,
        LU_arrow_bottom_blocks,
        LU_arrow_right_blocks,
        LU_arrow_tip_block,
        P_diag,
        P_tip,
    )
