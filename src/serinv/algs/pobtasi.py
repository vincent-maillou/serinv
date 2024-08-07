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
import time


def pobtasi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    device_streaming: bool = False,
    timing: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix
    using a sequential block algorithm.

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.

    Complexity analysis:
        Parameters:
            n : number of diagonal blocks
            b : diagonal block size
            a : arrow block size

        FLOPS count:
            GEMM_b^3 : (n-1) * 8 * b^3 + 2 * b^3
            GEMM_b^2_a : (n-1) * 10 * b^2 * a + 4 * b^2 * a
            GEMM_a^2_b : (n-1) * 2 * a^2 * b + 2 * a^2 * b
            GEMM_a^3 : 2 * a^3
            TRSM_b^3 : (n-1) * b^3
            TRSM_a_b^2 : a * b^2

        Total FLOPS:
            T_{flops_{POBTASI}} = (n-1) * (9*b^3 + 10*b^2 + 2*a^2*b) + 2*b^3 + 5*a*b^2 + 2*a^2*b + 2*a^3

        Complexity:
            By making the assumption that b >> a, the complexity of the POBTASI
            algorithm is O(n * b^3).

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
        if timing:
            return _streaming_pobtasi_timed(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_arrow_bottom_blocks,
                L_arrow_tip_block,
            )
        else:
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

    L_lower_diagonal_blocks_i = xp.empty_like(L_diagonal_blocks[0, :, :])
    L_arrow_bottom_blocks_i = xp.empty_like(L_arrow_bottom_blocks[0, :, :])

    L_blk_inv = xp.empty_like(L_diagonal_blocks[0, :, :])

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
    dict_timings = {}
    dict_timings["trsm"] = 0.
    dict_timings["gemm"] = 0.

    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_diagonal_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_tip_event = cp.cuda.Event()

    n_diag_blocks = L_diagonal_blocks.shape[0]
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[0]

    # X hosts arrays pointers
    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks
    X_arrow_bottom_blocks = L_arrow_bottom_blocks
    X_arrow_tip_block = L_arrow_tip_block

    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_arrow_bottom_blocks_d = cp.empty(
        (2, *L_arrow_bottom_blocks.shape[1:]), dtype=L_arrow_bottom_blocks.dtype
    )
    L_arrow_tip_block_d = cp.empty_like(L_arrow_tip_block)

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d
    X_arrow_bottom_blocks_d = L_arrow_bottom_blocks_d
    X_arrow_tip_block_d = L_arrow_tip_block_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_blk_inv_d = cp.empty_like(L_diagonal_blocks[0, :, :])
    L_last_blk_inv_d = cp.empty_like(L_arrow_tip_block)

    L_lower_diagonal_blocks_d_i = cp.empty_like(L_diagonal_blocks[0, :, :])
    L_arrow_bottom_blocks_d_i = cp.empty_like(L_arrow_bottom_blocks[0, :, :])

    # Backward block-selected inversion
    # --- C: events + transfers---
    compute_diagonal_h2d_events[(n_diag_blocks - 2) % 2].record(stream=compute_stream)
    compute_arrow_h2d_events[(n_diag_blocks - 2) % 2].record(stream=compute_stream)
    L_arrow_tip_block_d.set(arr=L_arrow_tip_block[:, :], stream=compute_stream)

    with compute_stream:
        # X_{ndb+1, ndb+1} = L_{ndb+1, ndb}^{-T} L_{ndb+1, ndb}^{-1}
        L_last_blk_inv_d = cu_la.solve_triangular(
            L_arrow_tip_block_d[:, :], cp.eye(arrow_blocksize), lower=True
        )

        X_arrow_tip_block_d[:, :] = L_last_blk_inv_d.conj().T @ L_last_blk_inv_d
        compute_arrow_tip_event.record(stream=compute_stream)

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(compute_arrow_tip_event)
    X_arrow_tip_block_d[:, :].get(
        out=X_arrow_tip_block,
        stream=d2h_stream,
        blocking=False,
    )

    # --- Host 2 Device transfers ---
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_diagonal_blocks[-1, :, :], stream=h2d_stream
    )
    h2d_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_arrow_bottom_blocks[-1, :, :], stream=h2d_stream
    )
    h2d_arrow_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
        # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
        L_blk_inv_d = cu_la.solve_triangular(
            L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        )

        compute_stream.wait_event(h2d_arrow_events[(n_diag_blocks - 1) % 2])
        L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[
            (n_diag_blocks - 1) % 2, :, :
        ]

        X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            -X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :] @ L_blk_inv_d
        )
        compute_arrow_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

        # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
        X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            L_blk_inv_d.conj().T
            - X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
            @ L_arrow_bottom_blocks_d_i[:, :]
        ) @ L_blk_inv_d
        compute_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(compute_arrow_events[(n_diag_blocks - 1) % 2])
    X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_arrow_bottom_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    d2h_stream.wait_event(compute_diagonal_events[(n_diag_blocks - 1) % 2])
    X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_diagonal_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    d2h_lower_events[(n_diag_blocks - 2) % 2].record(stream=d2h_stream)

    for i in range(n_diag_blocks - 2, -1, -1):
        h2d_stream.wait_event(compute_diagonal_h2d_events[i % 2])
        L_diagonal_blocks_d[i % 2, :, :].set(
            arr=L_diagonal_blocks[i, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[i % 2].record(stream=h2d_stream)

        h2d_stream.wait_event(d2h_lower_events[i % 2])
        L_lower_diagonal_blocks_d[i % 2, :, :].set(
            arr=L_lower_diagonal_blocks[i, :, :], stream=h2d_stream
        )
        h2d_lower_events[i % 2].record(stream=h2d_stream)

        h2d_stream.wait_event(compute_arrow_h2d_events[i % 2])
        L_arrow_bottom_blocks_d[i % 2, :, :].set(
            arr=L_arrow_bottom_blocks[i, :, :], stream=h2d_stream
        )
        h2d_arrow_events[i % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_blk_inv_d = cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )

            # --- Off-diagonal block part ---
            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[i % 2, :, :]
            compute_stream.wait_event(h2d_arrow_events[i % 2])
            L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i % 2, :, :]
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_d[i % 2, :, :] = (
                -X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_diagonal_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_lower_events[i % 2].record(stream=compute_stream)

            # --- Arrowhead part ---
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_d[i % 2, :, :] = (
                -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_arrow_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_arrow_events[i % 2].record(stream=compute_stream)

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_blk_inv_d.conj().T
                - X_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(compute_lower_events[i % 2])
        X_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=X_lower_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )
        d2h_lower_events[i % 2].record(stream=d2h_stream)

        d2h_stream.wait_event(compute_arrow_events[i % 2])
        X_arrow_bottom_blocks_d[i % 2, :, :].get(
            out=X_arrow_bottom_blocks[i, :, :], stream=d2h_stream, blocking=False
        )

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        X_diagonal_blocks_d[i % 2, :, :].get(
            out=X_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )

    cp.cuda.Device().synchronize()

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
        dict_timings,
    )

def _streaming_pobtasi_timed(
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
    dict_timings = {}
    dict_timings["trsm"] = 0.
    dict_timings["gemm"] = 0.
    tic_gpu = cp.cuda.Event()
    toc_gpu = cp.cuda.Event()

    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_diagonal_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_tip_event = cp.cuda.Event()

    n_diag_blocks = L_diagonal_blocks.shape[0]
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_tip_block.shape[0]

    # X hosts arrays pointers
    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks
    X_arrow_bottom_blocks = L_arrow_bottom_blocks
    X_arrow_tip_block = L_arrow_tip_block

    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_arrow_bottom_blocks_d = cp.empty(
        (2, *L_arrow_bottom_blocks.shape[1:]), dtype=L_arrow_bottom_blocks.dtype
    )
    L_arrow_tip_block_d = cp.empty_like(L_arrow_tip_block)

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d
    X_arrow_bottom_blocks_d = L_arrow_bottom_blocks_d
    X_arrow_tip_block_d = L_arrow_tip_block_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_blk_inv_d = cp.empty_like(L_diagonal_blocks[0, :, :])
    L_last_blk_inv_d = cp.empty_like(L_arrow_tip_block)

    L_lower_diagonal_blocks_d_i = cp.empty_like(L_diagonal_blocks[0, :, :])
    L_arrow_bottom_blocks_d_i = cp.empty_like(L_arrow_bottom_blocks[0, :, :])

    # Backward block-selected inversion
    # --- C: events + transfers---
    compute_diagonal_h2d_events[(n_diag_blocks - 2) % 2].record(stream=compute_stream)
    compute_arrow_h2d_events[(n_diag_blocks - 2) % 2].record(stream=compute_stream)
    L_arrow_tip_block_d.set(arr=L_arrow_tip_block[:, :], stream=compute_stream)

    with compute_stream:
        # X_{ndb+1, ndb+1} = L_{ndb+1, ndb}^{-T} L_{ndb+1, ndb}^{-1}
        tic_gpu.record(stream=compute_stream)
        L_last_blk_inv_d = cu_la.solve_triangular(
            L_arrow_tip_block_d[:, :], cp.eye(arrow_blocksize), lower=True
        )
        toc_gpu.record(stream=compute_stream)
        compute_stream.synchronize()
        dict_timings["trsm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)

        tic_gpu.record(stream=compute_stream)
        X_arrow_tip_block_d[:, :] = L_last_blk_inv_d.conj().T @ L_last_blk_inv_d
        toc_gpu.record(stream=compute_stream)
        compute_stream.synchronize()
        dict_timings["gemm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)
        compute_arrow_tip_event.record(stream=compute_stream)

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(compute_arrow_tip_event)
    X_arrow_tip_block_d[:, :].get(
        out=X_arrow_tip_block,
        stream=d2h_stream,
        blocking=False,
    )

    # --- Host 2 Device transfers ---
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_diagonal_blocks[-1, :, :], stream=h2d_stream
    )
    h2d_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_arrow_bottom_blocks[-1, :, :], stream=h2d_stream
    )
    h2d_arrow_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
        tic_gpu.record(stream=compute_stream)
        # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
        L_blk_inv_d = cu_la.solve_triangular(
            L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
            cp.eye(diag_blocksize),
            lower=True,
        )
        toc_gpu.record(stream=compute_stream)
        compute_stream.synchronize()
        dict_timings["trsm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)

        compute_stream.wait_event(h2d_arrow_events[(n_diag_blocks - 1) % 2])
        tic_gpu.record(stream=compute_stream)
        L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[
            (n_diag_blocks - 1) % 2, :, :
        ]

        X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            -X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :] @ L_blk_inv_d
        )
        compute_arrow_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

        # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
        X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            L_blk_inv_d.conj().T
            - X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
            @ L_arrow_bottom_blocks_d_i[:, :]
        ) @ L_blk_inv_d
        toc_gpu.record(stream=compute_stream)
        compute_stream.synchronize()
        dict_timings["gemm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)
        compute_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(compute_arrow_events[(n_diag_blocks - 1) % 2])
    X_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_arrow_bottom_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    d2h_stream.wait_event(compute_diagonal_events[(n_diag_blocks - 1) % 2])
    X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=X_diagonal_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    d2h_lower_events[(n_diag_blocks - 2) % 2].record(stream=d2h_stream)

    for i in range(n_diag_blocks - 2, -1, -1):
        h2d_stream.wait_event(compute_diagonal_h2d_events[i % 2])
        L_diagonal_blocks_d[i % 2, :, :].set(
            arr=L_diagonal_blocks[i, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[i % 2].record(stream=h2d_stream)

        h2d_stream.wait_event(d2h_lower_events[i % 2])
        L_lower_diagonal_blocks_d[i % 2, :, :].set(
            arr=L_lower_diagonal_blocks[i, :, :], stream=h2d_stream
        )
        h2d_lower_events[i % 2].record(stream=h2d_stream)

        h2d_stream.wait_event(compute_arrow_h2d_events[i % 2])
        L_arrow_bottom_blocks_d[i % 2, :, :].set(
            arr=L_arrow_bottom_blocks[i, :, :], stream=h2d_stream
        )
        h2d_arrow_events[i % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            tic_gpu.record(stream=compute_stream)
            L_blk_inv_d = cu_la.solve_triangular(
                L_diagonal_blocks_d[i % 2, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )
            toc_gpu.record(stream=compute_stream)
            compute_stream.synchronize()
            dict_timings["trsm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)

            # --- Off-diagonal block part ---
            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[i % 2, :, :]
            compute_stream.wait_event(h2d_arrow_events[i % 2])
            L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i % 2, :, :]
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
            tic_gpu.record(stream=compute_stream)
            X_lower_diagonal_blocks_d[i % 2, :, :] = (
                -X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_diagonal_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_lower_events[i % 2].record(stream=compute_stream)

            # --- Arrowhead part ---
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_d[i % 2, :, :] = (
                -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_arrow_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_arrow_events[i % 2].record(stream=compute_stream)

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_blk_inv_d.conj().T
                - X_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                @ L_arrow_bottom_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            toc_gpu.record(stream=compute_stream)
            compute_stream.synchronize()
            dict_timings["gemm"] += cp.cuda.get_elapsed_time(tic_gpu, toc_gpu)
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(compute_lower_events[i % 2])
        X_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=X_lower_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )
        d2h_lower_events[i % 2].record(stream=d2h_stream)

        d2h_stream.wait_event(compute_arrow_events[i % 2])
        X_arrow_bottom_blocks_d[i % 2, :, :].get(
            out=X_arrow_bottom_blocks[i, :, :], stream=d2h_stream, blocking=False
        )

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        X_diagonal_blocks_d[i % 2, :, :].get(
            out=X_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )

    cp.cuda.Device().synchronize()

    return (
        X_diagonal_blocks,
        X_lower_diagonal_blocks,
        X_arrow_bottom_blocks,
        X_arrow_tip_block,
        dict_timings,
    )