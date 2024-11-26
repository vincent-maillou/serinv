# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np

from serinv import (
    ArrayLike,
    CUPY_AVAIL,
    DEVICE_STREAMING,
    _get_array_module,
)


def pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    arrow_direction: str = "downward",
    buffers: tuple = None,
) -> ArrayLike:
    """Solve a block tridiagonal arrowhead linear system given its Cholesky factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.
    """
    xp, _ = _get_array_module(L_diagonal_blocks)

    if CUPY_AVAIL and xp == np and DEVICE_STREAMING:
        # Call streaming codes
        raise NotImplementedError("H2D streaming not implemented.")
    else:
        # Call direct-array (non-streaming) codes
        if arrow_direction == "downward":
            return _pobtas_downward(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_arrow_bottom_blocks,
                L_arrow_tip_block,
                B,
            )
        elif arrow_direction == "upward":
            raise NotImplementedError(
                "Upward arrowhead, H2D streaming not implemented."
            )
        elif arrow_direction == "downward-permuted":
            raise NotImplementedError(
                "downward-Permuted arrowhead, H2D streaming not implemented."
            )
        elif arrow_direction == "upward-permuted":
            raise NotImplementedError(
                "upward-Permuted arrowhead, H2D streaming not implemented."
            )
        else:
            raise ValueError(f"Unknown arrow direction: {arrow_direction}")


def _pobtas_downward(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
) -> ArrayLike:

    xp, la = _get_array_module(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_bottom_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    X = xp.zeros_like(B)

    # ----- Forward substitution -----
    X[0:diag_blocksize, :] = la.solve_triangular(
        L_diagonal_blocks[0, :, :],
        B[0:diag_blocksize],
        lower=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        X[i * diag_blocksize : (i + 1) * diag_blocksize, :] = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            B[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            - L_lower_diagonal_blocks[i - 1, :, :]
            @ X[(i - 1) * diag_blocksize : (i) * diag_blocksize, :],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_tip_rhs = B[-arrow_blocksize:, :]
    for i in range(0, n_diag_blocks):
        B_tip_rhs -= (
            L_arrow_bottom_blocks[i, :, :]
            @ X[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    X[-arrow_blocksize:, :] = la.solve_triangular(
        L_arrow_tip_block[:, :], B_tip_rhs[:, :], lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    X[-arrow_blocksize:, :] = la.solve_triangular(
        L_arrow_tip_block[:, :],
        X[-arrow_blocksize:, :],
        lower=True,
        trans="C",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :] = la.solve_triangular(
        L_diagonal_blocks[-1, :, :],
        X[-arrow_blocksize - diag_blocksize : -arrow_blocksize, :]
        - L_arrow_bottom_blocks[-1, :, :].conj().T @ X[-arrow_blocksize:, :],
        lower=True,
        trans="C",
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        X[i * diag_blocksize : (i + 1) * diag_blocksize, :] = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            X[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i, :, :].conj().T
            @ X[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_arrow_bottom_blocks[i, :, :].conj().T @ X[-arrow_blocksize:, :],
            lower=True,
            trans="C",
        )

    return X
