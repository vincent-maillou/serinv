# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np


def ddbtasinv(
    A_diagonal_blocks: np.ndarray | cp.ndarray,
    A_lower_diagonal_blocks: np.ndarray | cp.ndarray,
    A_upper_diagonal_blocks: np.ndarray | cp.ndarray,
    A_arrow_bottom_blocks: np.ndarray | cp.ndarray,
    A_arrow_right_blocks: np.ndarray | cp.ndarray,
    A_arrow_tip_block: np.ndarray | cp.ndarray,
) -> tuple[
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
    np.ndarray | cp.ndarray,
]:
    """Perform the selected inversion of a block tridiagonal arrowhead matrix. Do
    not explicitly factorize the matrix, but rather compute the selected inverse
    using a schur complement approach.

    Note:
    -----
    - This implementation will handle pivoting of the diagonal blocks, hence strict
        diagonal dominance is not required. Nevertheless, the matrix must be invertible
        and the diagonal blocks must be invertibles.
    - If strict diagonal dominance is not satisfied, full accuracy cannot be guaranteed
        since no pivoting between the blocks will be performed.
    - If a device array is given, the algorithm will run on the GPU.
    - Will overwrite the input, A, matrix.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray | cp.ndarray
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : np.ndarray | cp.ndarray
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : np.ndarray | cp.ndarray
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : np.ndarray | cp.ndarray
        The block at the tip of the arrowhead.

    Returns
    -------
    X_diagonal_blocks : np.ndarray | cp.ndarray
        Diagonal blocks of the selected inversion of the matrix.
    X_lower_diagonal_blocks : np.ndarray | cp.ndarray
        Lower diagonal blocks of the selected inversion of the matrix.
    X_upper_diagonal_blocks : np.ndarray | cp.ndarray
        Upper diagonal blocks of the selected inversion of the matrix.
    X_arrow_bottom_blocks : np.ndarray | cp.ndarray
        Bottom arrow blocks of the selected inversion of the matrix.
    X_arrow_right_blocks : np.ndarray | cp.ndarray
        Right arrow blocks of the selected inversion of the matrix.
    X_arrow_tip_block : np.ndarray | cp.ndarray
        Tip arrow block of the selected inversion of the matrix.
    """

    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
    else:
        xp = np

    n_diag_blocks = A_diagonal_blocks.shape[0]

    X_diagonal_blocks = xp.zeros_like(A_diagonal_blocks)
    X_lower_diagonal_blocks = xp.zeros_like(A_lower_diagonal_blocks)
    X_upper_diagonal_blocks = xp.zeros_like(A_upper_diagonal_blocks)
    X_arrow_bottom_blocks = xp.zeros_like(A_arrow_bottom_blocks)
    X_arrow_right_blocks = xp.zeros_like(A_arrow_right_blocks)
    X_arrow_tip_block = xp.zeros_like(A_arrow_tip_block)

    # Buffers for the intermediate results of the backward pass
    B1 = xp.zeros_like(X_diagonal_blocks[0, :, :])
    B2 = xp.zeros_like(X_arrow_right_blocks[0, :, :])
    C1 = xp.zeros_like(X_diagonal_blocks[0, :, :])
    C2 = xp.zeros_like(X_arrow_bottom_blocks[0, :, :])

    # Forward pass
    X_diagonal_blocks[0, :, :] = xp.linalg.inv(A_diagonal_blocks[0, :, :])

    for i in range(1, n_diag_blocks):
        # X_{ii} = (A_{ii} - A_{i,i-1} @ X_{i-1,i-1} @ A_{i-1,i})^{-1}
        X_diagonal_blocks[i, :, :] = xp.linalg.inv(
            A_diagonal_blocks[i, :, :]
            - A_lower_diagonal_blocks[i - 1, :, :]
            @ X_diagonal_blocks[i - 1, :, :]
            @ A_upper_diagonal_blocks[i - 1, :, :]
        )

        # A_{i,ndb+1} = A_{i,ndb+1} - A_{i,i-1} @ X_{i-1,i-1} @ A_{i-1,ndb+1}
        A_arrow_right_blocks[i, :, :] = (
            A_arrow_right_blocks[i, :, :]
            - A_lower_diagonal_blocks[i - 1, :, :]
            @ X_diagonal_blocks[i - 1, :, :]
            @ A_arrow_right_blocks[i - 1, :, :]
        )

        # A_{ndb+1,i} = A_{ndb+1,i} - A_{ndb+1,i-1} @ X_{i-1,i-1} @ A_{i-1,i}
        A_arrow_bottom_blocks[i, :, :] = (
            A_arrow_bottom_blocks[i, :, :]
            - A_arrow_bottom_blocks[i - 1, :, :]
            @ X_diagonal_blocks[i - 1, :, :]
            @ A_upper_diagonal_blocks[i - 1, :, :]
        )

        # A_{ndb+1,ndb+1} = A_{ndb+1,ndb+1} - A_{ndb+1,i-1} @ X_{i-1,i-1} @ A_{i-1,ndb+1}
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - A_arrow_bottom_blocks[i - 1, :, :]
            @ X_diagonal_blocks[i - 1, :, :]
            @ A_arrow_right_blocks[i - 1, :, :]
        )

    # X_{ndb+1, ndb+1} = (A_{ndb+1, ndb+1} - A_{ndb+1,ndb} @ X_{ndb,ndb} @ A_{ndb,ndb+1})^{-1}
    X_arrow_tip_block[:, :] = xp.linalg.inv(
        A_arrow_tip_block[:, :]
        - A_arrow_bottom_blocks[-1, :, :]
        @ X_diagonal_blocks[-1, :, :]
        @ A_arrow_right_blocks[-1, :, :]
    )

    # Backward pass
    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :]
        @ A_arrow_bottom_blocks[-1, :, :]
        @ X_diagonal_blocks[-1, :, :]
    )

    X_arrow_right_blocks[-1, :, :] = (
        -X_diagonal_blocks[-1, :, :]
        @ A_arrow_right_blocks[-1, :, :]
        @ X_arrow_tip_block[:, :]
    )

    X_diagonal_blocks[-1, :, :] = (
        X_diagonal_blocks[-1, :, :]
        + X_diagonal_blocks[-1, :, :]
        @ A_arrow_right_blocks[-1, :, :]
        @ X_arrow_tip_block[:, :]
        @ A_arrow_bottom_blocks[-1, :, :]
        @ X_diagonal_blocks[-1, :, :]
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i,i+1} = - X_{ii} @ (A_{i,i+1} @ X_{i+1,i+1} + A_{i,ndb+1} @ X_{ndb+1,i+1})
        # B1 = (A_{i,i+1} @ X_{i+1,i+1} + A_{i,ndb+1} @ X_{ndb+1,i+1})
        B1[:, :] = (
            A_upper_diagonal_blocks[i, :, :] @ X_diagonal_blocks[i + 1, :, :]
            + A_arrow_right_blocks[i, :, :] @ X_arrow_bottom_blocks[i + 1, :, :]
        )
        # X_{i,i+1} = - X_{ii} @ B1
        X_upper_diagonal_blocks[i, :, :] = -X_diagonal_blocks[i, :, :] @ B1[:, :]

        # X_{i,ndb+1} = - X_{ii} @ (A_{i,i+1} @ X_{i+1,ndb+1} + A_{i,ndb+1} @ X_{ndb+1,ndb+1})
        # B2 = (A_{i,i+1} @ X_{i+1,ndb+1} + A_{i,ndb+1} @ X_{ndb+1,ndb+1})
        B2[:, :] = (
            A_upper_diagonal_blocks[i, :, :] @ X_arrow_right_blocks[i + 1, :, :]
            + A_arrow_right_blocks[i, :, :] @ X_arrow_tip_block[:, :]
        )
        # X_{i,ndb+1} = - X_{ii} @ B2
        X_arrow_right_blocks[i, :, :] = -X_diagonal_blocks[i, :, :] @ B2[:, :]

        # X_{i+1,i} = - (X_{i+1,i+1} @ A_{i+1,i} + X_{i+1,ndb+1} @ A_{ndb+1,i}) @ X_{ii}
        # C1 = (X_{i+1,i+1} @ A_{i+1,i} + X_{i+1,ndb+1} @ A_{ndb+1,i})
        C1[:, :] = (
            X_diagonal_blocks[i + 1, :, :] @ A_lower_diagonal_blocks[i, :, :]
            + X_arrow_right_blocks[i + 1, :, :] @ A_arrow_bottom_blocks[i, :, :]
        )
        # X_{i+1,i} = - C1 @ X_{ii}
        X_lower_diagonal_blocks[i, :, :] = -C1[:, :] @ X_diagonal_blocks[i, :, :]

        # X_{ndb+1,i} = - (X_{ndb+1,i+1} @ A_{i+1,i} + X_{ndb+1,ndb+1} @ A_{ndb+1,i}) @ X_{ii}
        # C2 = (X_{ndb+1,i+1} @ A_{i+1,i} + X_{ndb+1,ndb+1} @ A_{ndb+1,i})
        C2[:, :] = (
            X_arrow_bottom_blocks[i + 1, :, :] @ A_lower_diagonal_blocks[i, :, :]
            + X_arrow_tip_block[:, :] @ A_arrow_bottom_blocks[i, :, :]
        )
        # X_{ndb+1,i} = - C2 @ X_{ii}
        X_arrow_bottom_blocks[i, :, :] = -C2[:, :] @ X_diagonal_blocks[i, :, :]

        # X_{i,i} = X_{i,i} + X_{i,i} @ (B1 @ C1 + B2 @ C2) @ X_{i,i}
        X_diagonal_blocks[i, :, :] = (
            X_diagonal_blocks[i, :, :]
            + X_diagonal_blocks[i, :, :]
            @ (
                B1[:, :] @ A_lower_diagonal_blocks[i, :, :]
                + B2[:, :] @ A_arrow_bottom_blocks[i, :, :]
            )
            @ X_diagonal_blocks[i, :, :]
        )

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_upper_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_right_blocks,
        X_arrow_tip_block,
    )
