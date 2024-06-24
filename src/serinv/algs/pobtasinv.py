# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
from numpy.typing import ArrayLike


def pobtasinv(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the selected inversion of a block tridiagonal arrowhead matrix. Do
    not explicitly factorize the matrix, but rather compute the selected inverse
    using a schur complement approach.

    Note:
    -----
    - By convention takes and produce lower triangular parts of the matrix.
    - If a device array is given, the algorithm will run on the GPU.
    - Will overwrite the inputs and store the results in them (in-place).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : ArrayLike
        The blocks on the lower diagonal of the matrix.
    A_arrow_bottom_blocks : ArrayLike
        The blocks on the bottom arrow of the matrix.
    A_arrow_tip_block : ArrayLike
        The block at the tip of the arrowhead.

    Returns
    -------
    X_diagonal_blocks : ArrayLike
        Diagonal blocks of the selected inversion of the matrix.
    X_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the selected inversion of the matrix.
    X_arrow_bottom_blocks : ArrayLike
        Bottom arrow blocks of the selected inversion of the matrix.
    X_arrow_tip_block : ArrayLike
        Tip arrow block of the selected inversion of the matrix.
    """

    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
    else:
        xp = np

    n_diag_blocks = A_diagonal_blocks.shape[0]

    # Dual assignments for naming convention and readiness of the code
    X_diagonal_blocks = A_diagonal_blocks
    X_lower_diagonal_blocks = A_lower_diagonal_blocks
    X_arrow_bottom_blocks = A_arrow_bottom_blocks
    X_arrow_tip_block = A_arrow_tip_block

    # Buffers for the intermediate results of the forward pass
    D0 = xp.zeros_like(A_diagonal_blocks[0, :, :])

    # Buffers for the intermediate results of the backward pass
    A_lower_diagonal_blocks_i = xp.zeros_like(A_diagonal_blocks[0, :, :])
    A_arrow_bottom_blocks_i = xp.zeros_like(A_arrow_bottom_blocks[0, :, :])

    C1 = xp.zeros_like(X_diagonal_blocks[0, :, :])
    C2 = xp.zeros_like(X_arrow_bottom_blocks[0, :, :])

    # Forward pass
    X_diagonal_blocks[0, :, :] = xp.linalg.inv(A_diagonal_blocks[0, :, :])

    for i in range(1, n_diag_blocks):
        # Precompute reused matmul: D0 = X_{i-1,i-1} @ A_{i,i-1}^{\dagger}
        D0[:, :] = (
            X_diagonal_blocks[i - 1, :, :]
            @ A_lower_diagonal_blocks[i - 1, :, :].conj().T
        )

        # X_{ii} = (A_{ii} - A_{i,i-1} @ X_{i-1,i-1} @ A_{i,i-1}^{\dagger})^{-1}
        X_diagonal_blocks[i, :, :] = xp.linalg.inv(
            A_diagonal_blocks[i, :, :] - A_lower_diagonal_blocks[i - 1, :, :] @ D0[:, :]
        )

        # A_{ndb+1,i} = A_{ndb+1,i} - A_{ndb+1,i-1} @ X_{i-1,i-1} @ A_{i,i-1}^{\dagger}
        A_arrow_bottom_blocks[i, :, :] = (
            A_arrow_bottom_blocks[i, :, :]
            - A_arrow_bottom_blocks[i - 1, :, :] @ D0[:, :]
        )

        # A_{ndb+1,ndb+1} = A_{ndb+1,ndb+1} - A_{ndb+1,i-1} @ X_{i-1,i-1} @ A_{i-1,ndb+1}
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - A_arrow_bottom_blocks[i - 1, :, :]
            @ X_diagonal_blocks[i - 1, :, :]
            @ A_arrow_bottom_blocks[i - 1, :, :].conj().T
        )

    # X_{ndb+1, ndb+1} = (A_{ndb+1, ndb+1} - A_{ndb+1,ndb} @ X_{ndb,ndb} @ A_{ndb,ndb+1})^{-1}
    X_arrow_tip_block[:, :] = xp.linalg.inv(
        A_arrow_tip_block[:, :]
        - A_arrow_bottom_blocks[-1, :, :]
        @ X_diagonal_blocks[-1, :, :]
        @ A_arrow_bottom_blocks[-1, :, :].conj().T
    )

    # Backward pass
    A_arrow_bottom_blocks_i[:, :] = A_arrow_bottom_blocks[-1, :, :]

    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :]
        @ A_arrow_bottom_blocks[-1, :, :]
        @ X_diagonal_blocks[-1, :, :]
    )

    X_diagonal_blocks[-1, :, :] = (
        X_diagonal_blocks[-1, :, :]
        + X_diagonal_blocks[-1, :, :]
        @ A_arrow_bottom_blocks_i[:, :].conj().T
        @ X_arrow_tip_block[:, :]
        @ A_arrow_bottom_blocks_i[:, :]
        @ X_diagonal_blocks[-1, :, :]
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        A_lower_diagonal_blocks_i[:, :] = A_lower_diagonal_blocks[i, :, :]
        A_arrow_bottom_blocks_i[:, :] = A_arrow_bottom_blocks[i, :, :]

        # X_{i+1,i} = - (X_{i+1,i+1} @ A_{i+1,i} + X_{i+1,ndb+1} @ A_{ndb+1,i}) @ X_{ii}
        # C1 = (X_{i+1,i+1} @ A_{i+1,i} + X_{ndb+1,i+1}^{\dagger} @ A_{ndb+1,i})
        C1[:, :] = (
            X_diagonal_blocks[i + 1, :, :] @ A_lower_diagonal_blocks_i[:, :]
            + X_arrow_bottom_blocks[i + 1, :, :].conj().T
            @ A_arrow_bottom_blocks_i[:, :]
        )
        # X_{i+1,i} = - C1 @ X_{ii}
        X_lower_diagonal_blocks[i, :, :] = -C1[:, :] @ X_diagonal_blocks[i, :, :]

        # X_{ndb+1,i} = - (X_{ndb+1,i+1} @ A_{i+1,i} + X_{ndb+1,ndb+1} @ A_{ndb+1,i}) @ X_{ii}
        # C2 = (X_{ndb+1,i+1} @ A_{i+1,i} + X_{ndb+1,ndb+1} @ A_{ndb+1,i})
        C2[:, :] = (
            X_arrow_bottom_blocks[i + 1, :, :] @ A_lower_diagonal_blocks_i[:, :]
            + X_arrow_tip_block[:, :] @ A_arrow_bottom_blocks_i[:, :]
        )
        # X_{ndb+1,i} = - C2 @ X_{ii}
        X_arrow_bottom_blocks[i, :, :] = -C2[:, :] @ X_diagonal_blocks[i, :, :]

        X_diagonal_blocks[i, :, :] = (
            X_diagonal_blocks[i, :, :]
            + X_diagonal_blocks[i, :, :]
            @ (
                A_lower_diagonal_blocks_i[:, :].conj().T @ C1[:, :]
                + A_arrow_bottom_blocks_i[:, :].conj().T @ C2[:, :]
            )
            @ X_diagonal_blocks[i, :, :]
        )

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )
