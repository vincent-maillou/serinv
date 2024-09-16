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


def ddbtas(
    LU_diagonal_blocks: ArrayLike,
    LU_lower_diagonal_blocks: ArrayLike,
    LU_upper_diagonal_blocks: ArrayLike,
    LU_arrow_bottom_blocks: ArrayLike,
    LU_arrow_right_blocks: ArrayLike,
    LU_arrow_tip_block: ArrayLike,
    B: ArrayLike,
) -> ArrayLike:
    """Solve a block tridiagonal arrowhead linear system given its LU factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
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
    B : ArrayLike
        The right hand side.

    Returns
    -------
    X : ArrayLike
        The solution of the system.
    """

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(LU_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    X = xp.zeros_like(B)

    # ----- Forward substitution -----
    diag_blocksize = LU_diagonal_blocks.shape[1]
    arrow_blocksize = LU_arrow_bottom_blocks.shape[1]
    n_diag_blocks = LU_diagonal_blocks.shape[0]

    X[0:diag_blocksize, :] = la.solve_triangular(
        LU_diagonal_blocks[0, :, :],
        B[0:diag_blocksize, :],
        lower=True,
        unit_diagonal=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * diag_blocksize : (i + 1) * diag_blocksize, :] = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            B[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - LU_lower_diagonal_blocks[i - 1, :, :]
            @ X[(i - 1) * diag_blocksize : i * diag_blocksize, :],
            lower=True,
            unit_diagonal=True,
        )

    # Accumulation of the arrowhead blocks
    B_tip_rhs = B[-arrow_blocksize:, :]
    for i in range(0, n_diag_blocks):
        B_tip_rhs -= (
            LU_arrow_bottom_blocks[i, :, :]
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    X[-arrow_blocksize:, :] = la.solve_triangular(
        LU_arrow_tip_block[:, :],
        B_tip_rhs[:, :],
        lower=True,
        unit_diagonal=True,
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = U_{ndb+1,ndb+1}^{-1} (Y_{ndb+1})
    X[-arrow_blocksize:, :] = la.solve_triangular(
        LU_arrow_tip_block[:, :],
        X[-arrow_blocksize:, :],
        lower=False,
    )

    # X_{ndb} = U_{ndb,ndb}^{-1} (Y_{ndb} - U_{ndb,ndb+1} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :] = la.solve_triangular(
        LU_diagonal_blocks[-1, :, :],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :]
        - LU_arrow_right_blocks[-1, :, :] @ X[-arrow_blocksize:, :],
        lower=False,
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = U_{i,i}^{-1} (Y_{i} - U_{i,i+1} X_{i+1}) - U_{i,ndb+1} X_{ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            LU_diagonal_blocks[i, :, :],
            X[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - LU_upper_diagonal_blocks[i, :, :]
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - LU_arrow_right_blocks[i, :, :] @ X[-arrow_blocksize:, :],
            lower=False,
        )

    return X
