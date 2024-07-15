# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike


def pobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    device_streaming: bool = False,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the Cholesky factorization of a block tridiagonal with arrowhead
    matrix using a sequential block algorithm.

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        Diagonal blocks of A.
    A_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of A.
    A_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of A.
    A_arrow_tip_block : ArrayLike
        Arrow tip block of A.
    device_streaming : bool
        Whether to use streamed GPU computation.

    Returns
    -------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of L.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of L.
    L_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of L.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of L.
    """

    if CUPY_AVAIL and cp.get_array_module(A_diagonal_blocks) == np and device_streaming:
        return _streaming_pobtaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
        )

    return _pobtaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )


def _pobtaf(
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
    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks)
        if xp == cp:
            la = cu_la
            cholesky = cholesky_lowerfill
        else:
            cholesky = np.linalg.cholesky
    else:
        xp = np
        cholesky = np.linalg.cholesky

    n_diag_blocks = A_diagonal_blocks.shape[0]

    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_arrow_bottom_blocks = A_arrow_bottom_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Forward block-Cholesky
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = cholesky(A_diagonal_blocks[i, :, :])

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[i, :, :] = (
            la.solve_triangular(
                L_diagonal_blocks[i, :, :],
                A_lower_diagonal_blocks[i, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks[i, :, :] = (
            la.solve_triangular(
                L_diagonal_blocks[i, :, :],
                A_arrow_bottom_blocks[i, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        A_diagonal_blocks[i + 1, :, :] = (
            A_diagonal_blocks[i + 1, :, :]
            - L_lower_diagonal_blocks[i, :, :]
            @ L_lower_diagonal_blocks[i, :, :].conj().T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        A_arrow_bottom_blocks[i + 1, :, :] = (
            A_arrow_bottom_blocks[i + 1, :, :]
            - L_arrow_bottom_blocks[i, :, :] @ L_lower_diagonal_blocks[i, :, :].conj().T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        A_arrow_tip_block[:, :] = (
            A_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[i, :, :] @ L_arrow_bottom_blocks[i, :, :].conj().T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks[-1, :, :] = cholesky(A_diagonal_blocks[-1, :, :])

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[-1, :, :] = (
        la.solve_triangular(
            L_diagonal_blocks[-1, :, :],
            A_arrow_bottom_blocks[-1, :, :].conj().T,
            lower=True,
        )
        .conj()
        .T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block[:, :] = (
        A_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[-1, :, :] @ L_arrow_bottom_blocks[-1, :, :].conj().T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block[:, :] = cholesky(A_arrow_tip_block[:, :])

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )


def _streaming_pobtaf(
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
    cp.cuda.nvtx.RangePush("_streaming_pobtaf:mem_init")
    n_diag_blocks = A_diagonal_blocks.shape[0]

    # A/L hosts arrays pointers
    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_arrow_bottom_blocks = A_arrow_bottom_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Device buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty_like(A_diagonal_blocks[0])
    A_arrow_bottom_blocks_d = cp.empty(
        (2, *A_arrow_bottom_blocks.shape[1:]), dtype=A_arrow_bottom_blocks.dtype
    )
    A_arrow_tip_block_d = cp.empty_like(A_arrow_tip_block)

    # X Device buffers arrays pointers
    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_arrow_bottom_blocks_d = A_arrow_bottom_blocks_d
    L_arrow_tip_block_d = A_arrow_tip_block_d

    # Buffers for the intermediate results of the forward cholesky
    cp.cuda.nvtx.RangePop()

    # Forward pass
    cp.cuda.nvtx.RangePush("_streaming_pobtaf:fwd_cholesky")

    # --- Host 2 Device transfers ---
    A_diagonal_blocks_d[0, :, :].set(arr=A_diagonal_blocks[0, :, :])
    A_arrow_bottom_blocks_d[0, :, :].set(arr=A_arrow_bottom_blocks[0, :, :])
    A_arrow_tip_block_d.set(arr=A_arrow_tip_block)

    for i in range(0, n_diag_blocks - 1):
        # --- Host 2 Device transfers ---
        A_diagonal_blocks_d[(i + 1) % 2, :, :].set(arr=A_diagonal_blocks[i + 1, :, :])
        A_lower_diagonal_blocks_d[:, :].set(arr=A_lower_diagonal_blocks[i, :, :])
        A_arrow_bottom_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_arrow_bottom_blocks[i + 1, :, :]
        )

        # --- Computations ---
        # L_{i, i} = chol(A_{i, i})
        cp.cuda.nvtx.RangePush("_streaming_pobtaf:cholesky")
        L_diagonal_blocks_d[i % 2, :, :] = cholesky_lowerfill(
            A_diagonal_blocks_d[i % 2, :, :]
        )
        cp.cuda.nvtx.RangePop()

        cp.cuda.nvtx.RangePush("_streaming_pobtaf:solve_triangular")
        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks_d[:, :] = (
            cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                A_lower_diagonal_blocks_d[:, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        # L_arrow_bottom_blocks_d[i % 2, :, :] = (
        #     A_arrow_bottom_blocks_d[i % 2, :, :] @ L_inv_temp_d[:, :]
        # )
        L_arrow_bottom_blocks_d[i % 2, :, :] = (
            cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                A_arrow_bottom_blocks_d[i % 2, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )
        cp.cuda.nvtx.RangePop()

        # Update next diagonal block
        cp.cuda.nvtx.RangePush("_streaming_pobtaf:A_update")
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        A_diagonal_blocks_d[(i + 1) % 2, :, :] = (
            A_diagonal_blocks_d[(i + 1) % 2, :, :]
            - L_lower_diagonal_blocks_d[:, :] @ L_lower_diagonal_blocks_d[:, :].conj().T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        A_arrow_bottom_blocks_d[(i + 1) % 2, :, :] = (
            A_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
            - L_arrow_bottom_blocks_d[i % 2, :, :]
            @ L_lower_diagonal_blocks_d[:, :].conj().T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        A_arrow_tip_block_d[:, :] = (
            A_arrow_tip_block_d[:, :]
            - L_arrow_bottom_blocks_d[i % 2, :, :]
            @ L_arrow_bottom_blocks_d[i % 2, :, :].conj().T
        )
        cp.cuda.nvtx.RangePop()

        # --- Device 2 Host transfers ---
        L_diagonal_blocks_d[i % 2, :, :].get(out=L_diagonal_blocks[i, :, :])
        L_lower_diagonal_blocks_d[:, :].get(out=L_lower_diagonal_blocks[i, :, :])
        L_arrow_bottom_blocks_d[i % 2, :, :].get(out=L_arrow_bottom_blocks[i, :, :])

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = cholesky_lowerfill(
        A_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :]
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
        cu_la.solve_triangular(
            L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
            A_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T,
            lower=True,
        )
        .conj()
        .T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    A_arrow_tip_block_d[:, :] = (
        A_arrow_tip_block_d[:, :]
        - L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :]
        @ L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block_d[:, :] = cholesky_lowerfill(A_arrow_tip_block_d[:, :])

    # --- Device 2 Host transfers ---
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_diagonal_blocks[-1, :, :]
    )
    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_arrow_bottom_blocks[-1, :, :]
    )
    L_arrow_tip_block_d[:, :].get(out=L_arrow_tip_block[:, :])

    cp.cuda.Device().synchronize()
    cp.cuda.nvtx.RangePop()

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )
