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


def pobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    device_streaming: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix
    using a sequential block algorithm.

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of L.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of L.
    L_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of L.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of L.
    device_streaming : bool
        Whether to use streamed GPU computation.

    Returns
    -------
    X_diagonal_blocks : ArrayLike
        Diagonal blocks of X.
    X_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of X.
    X_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of X.
    X_arrow_tip_block : ArrayLike
        Arrow tip block of X.
    """
    if CUPY_AVAIL and cp.get_array_module(L_diagonal_blocks) == np and device_streaming:
        return _streaming_pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
        )

    return _pobtasi(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )


def _pobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[0]

    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks
    X_arrow_bottom_blocks = L_arrow_bottom_blocks
    X_arrow_tip_block = L_arrow_tip_block
    L_last_blk_inv = L_arrow_tip_block

    L_lower_diagonal_blocks_i = xp.zeros_like(L_diagonal_blocks[0, :, :])
    L_arrow_bottom_blocks_i = xp.zeros_like(L_arrow_bottom_blocks[0, :, :])

    L_blk_inv = xp.zeros_like(L_diagonal_blocks[0, :, :])

    L_last_blk_inv = la.solve_triangular(
        L_arrow_tip_block[:, :], xp.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block[:, :] = L_last_blk_inv.conj().T @ L_last_blk_inv

    # Backward block-selected inversion
    L_arrow_bottom_blocks_i[:, :] = L_arrow_bottom_blocks[-1, :, :]

    L_blk_inv = la.solve_triangular(
        L_diagonal_blocks[-1, :, :],
        xp.eye(diag_blocksize),
        lower=True,
    )

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X_arrow_bottom_blocks[-1, :, :] = (
        -X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks_i[:, :] @ L_blk_inv
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks[-1, :, :] = (
        L_blk_inv.conj().T
        - X_arrow_bottom_blocks[-1, :, :].conj().T @ L_arrow_bottom_blocks_i[:, :]
    ) @ L_blk_inv

    n_diag_blocks = L_diagonal_blocks.shape[0]
    for i in range(n_diag_blocks - 2, -1, -1):
        L_lower_diagonal_blocks_i[:, :] = L_lower_diagonal_blocks[i, :, :]
        L_arrow_bottom_blocks_i[:, :] = L_arrow_bottom_blocks[i, :, :]

        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            xp.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -X_diagonal_blocks[i + 1, :, :] @ L_lower_diagonal_blocks_i[:, :]
            - X_arrow_bottom_blocks[i + 1, :, :].conj().T
            @ L_arrow_bottom_blocks_i[:, :]
        ) @ L_blk_inv

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks[i, :, :] = (
            -X_arrow_bottom_blocks[i + 1, :, :] @ L_lower_diagonal_blocks_i[:, :]
            - X_arrow_tip_block[:, :] @ L_arrow_bottom_blocks_i[:, :]
        ) @ L_blk_inv

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            L_blk_inv.conj().T
            - X_lower_diagonal_blocks[i, :, :].conj().T
            @ L_lower_diagonal_blocks_i[:, :]
            - X_arrow_bottom_blocks[i, :, :].conj().T @ L_arrow_bottom_blocks_i[:, :]
        ) @ L_blk_inv

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )


def _streaming_pobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    cp.cuda.nvtx.RangePush("_streaming_pobtasi:mem_init")
    n_diag_blocks = L_diagonal_blocks.shape[0]
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[0]

    # X hosts arrays pointers
    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks
    X_arrow_bottom_blocks = L_arrow_bottom_blocks
    X_arrow_tip_block = L_arrow_tip_block

    # Device buffers
    L_diagonal_blocks_d = cp.zeros(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.zeros_like(L_diagonal_blocks[0])
    L_arrow_bottom_blocks_d = cp.zeros(
        (2, *L_arrow_bottom_blocks.shape[1:]), dtype=L_arrow_bottom_blocks.dtype
    )
    L_arrow_tip_block_d = cp.zeros_like(L_arrow_tip_block)

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d
    X_arrow_bottom_blocks_d = L_arrow_bottom_blocks_d
    X_arrow_tip_block_d = L_arrow_tip_block_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_blk_inv_d = cp.zeros_like(L_diagonal_blocks[0, :, :])
    L_last_blk_inv_d = cp.zeros_like(L_arrow_tip_block)

    L_lower_diagonal_blocks_d_i = cp.zeros_like(L_diagonal_blocks[0, :, :])
    L_arrow_bottom_blocks_d_i = cp.zeros_like(L_arrow_bottom_blocks[0, :, :])
    cp.cuda.nvtx.RangePop()

    # Backward block-selected inversion
    cp.cuda.nvtx.RangePush("_streaming_pobtasi:bwd_sinv")
    # --- Host 2 Device transfers ---
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_diagonal_blocks[-1, :, :]
    )
    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_arrow_bottom_blocks[-1, :, :]
    )
    L_arrow_tip_block_d.set(arr=L_arrow_tip_block)

    # --- Computations ---
    # X_{ndb+1, ndb+1} = L_{ndb+1, ndb}^{-T} L_{ndb+1, ndb}^{-1}
    L_last_blk_inv_d = cu_la.solve_triangular(
        L_arrow_tip_block_d[:, :], cp.eye(arrow_blocksize), lower=True
    )

    X_arrow_tip_block_d[:, :] = L_last_blk_inv_d.conj().T @ L_last_blk_inv_d

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[
        (n_diag_blocks - 1) % 2, :, :
    ]

    L_blk_inv_d = cu_la.solve_triangular(
        L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
        cp.eye(diag_blocksize),
        lower=True,
    )

    X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
        -X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :] @ L_blk_inv_d
    )

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
        L_blk_inv_d.conj().T
        - X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
        @ L_arrow_bottom_blocks_d_i[:, :]
    ) @ L_blk_inv_d

    # --- Device 2 Host transfers ---
    X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_diagonal_blocks[-1, :, :]
    )
    X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_arrow_bottom_blocks[-1, :, :]
    )
    X_arrow_tip_block_d[:, :].get(out=X_arrow_tip_block)

    for i in range(n_diag_blocks - 2, -1, -1):
        # --- Host 2 Device transfers ---
        L_diagonal_blocks_d[i % 2, :, :].set(arr=L_diagonal_blocks[i, :, :])
        L_lower_diagonal_blocks_d[:, :].set(arr=L_lower_diagonal_blocks[i, :, :])
        L_arrow_bottom_blocks_d[i % 2, :, :].set(arr=L_arrow_bottom_blocks[i, :, :])

        L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[:, :]
        L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i, :, :]

        # --- Computations ---
        L_blk_inv_d = cu_la.solve_triangular(
            L_diagonal_blocks_d[i % 2, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_d[:, :] = (
            -X_diagonal_blocks_d[(i + 1) % 2, :, :] @ L_lower_diagonal_blocks_d_i[:, :]
            - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
            @ L_arrow_bottom_blocks_d_i[:, :]
        ) @ L_blk_inv_d

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_d[i % 2, :, :] = (
            -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
            @ L_lower_diagonal_blocks_d_i[:, :]
            - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
        ) @ L_blk_inv_d

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_d[i % 2, :, :] = (
            L_blk_inv_d.conj().T
            - X_lower_diagonal_blocks_d[:, :].conj().T
            @ L_lower_diagonal_blocks_d_i[:, :]
            - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
            @ L_arrow_bottom_blocks_d_i[:, :]
        ) @ L_blk_inv_d

        # --- Device 2 Host transfers ---
        L_diagonal_blocks_d[i % 2, :, :].get(out=X_diagonal_blocks[i, :, :])
        X_lower_diagonal_blocks_d[:, :].get(out=X_lower_diagonal_blocks[i, :, :])
        X_arrow_bottom_blocks_d[i % 2, :, :].get(out=X_arrow_bottom_blocks[i, :, :])

    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePop()

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
    )
