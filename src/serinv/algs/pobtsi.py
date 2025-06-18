# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
    _get_module_from_str,
)

from serinv.block_primitive import trsm

def pobtsi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    **kwargs,
):
    """Perform a selected inversion of a block tridiagonal with arrowhead matrix (pointing downward by convention)
    using a sequential block algorithm.

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor of the matrix.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor of the matrix.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will run on the GPU. (default: False)
    buffer : ArrayLike, optional
        Buffer array for the permuted arrowhead. (default: None)

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.
    - If a buffer is provided, the algorithm will perform the factorization of a permuted arrowhead matrix.

    """
    expected_kwargs = {"device_streaming", "buffer", "invert_last_block"}
    unexpected_kwargs = set(kwargs) - expected_kwargs
    if unexpected_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

    device_streaming: bool = kwargs.get("device_streaming", False)
    buffer = kwargs.get("buffer", None)
    invert_last_block = kwargs.get("invert_last_block", True)

    if buffer is not None:
        # Permuted arrowhead
        if device_streaming:
            return _pobtsi_permuted_streaming(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                buffer,
            )
        else:
            return _pobtsi_permuted(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                buffer,
            )
    else:
        # Natural arrowhead
        if device_streaming:
            return _pobtsi_streaming(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                invert_last_block,
            )
        else:
            return _pobtsi(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                invert_last_block,
            )


def _pobtsi(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    xp, la = _get_module_from_array(L_diagonal_blocks)

    n_diag_blocks: int = L_diagonal_blocks.shape[0]

    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks

    L_lower_diagonal_blocks_i = xp.empty_like(L_diagonal_blocks[0, :, :])
    L_blk_inv = xp.empty_like(L_diagonal_blocks[0, :, :])
    Identity = xp.eye(L_diagonal_blocks.shape[1])

    if invert_last_block:
        L_blk_inv = trsm(
            L_diagonal_blocks[-1, :, :],
            Identity,
            lower=True,
        )

        # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
        X_diagonal_blocks[-1, :, :] = L_blk_inv.conj().T @ L_blk_inv

    for i in range(n_diag_blocks - 2, -1, -1):
        L_lower_diagonal_blocks_i[:, :] = L_lower_diagonal_blocks[i, :, :]

        L_blk_inv = trsm(
            L_diagonal_blocks[i, :, :],
            Identity,
            lower=True,
        )

        # --- Off-diagonal block part ---
        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -X_diagonal_blocks[i + 1, :, :] @ L_lower_diagonal_blocks_i[:, :]
        ) @ L_blk_inv

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            L_blk_inv.conj().T
            - X_lower_diagonal_blocks[i, :, :].conj().T
            @ L_lower_diagonal_blocks_i[:, :]
        ) @ L_blk_inv


def _pobtsi_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    buffer: ArrayLike,
):
    xp, la = _get_module_from_array(arr=L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks

    # Backward selected-inversion
    L_inv_temp = xp.empty_like(L_diagonal_blocks[0])
    L_lower_diagonal_blocks_temp = xp.empty_like(L_lower_diagonal_blocks[0])

    buffer_temp = xp.empty_like(buffer[0, :, :])

    for i in range(n_diag_blocks - 2, 0, -1):
        L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks[i, :, :]
        buffer_temp[:, :] = buffer[i, :, :]

        L_inv_temp[:, :] = trsm(
            L_diagonal_blocks[i, :, :],
            xp.eye(diag_blocksize),
            lower=True,
        )

        # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks[i, :, :] = (
            -buffer[i + 1, :, :].conj().T @ buffer_temp[:, :]
            - X_diagonal_blocks[i + 1, :, :] @ L_lower_diagonal_blocks_temp[:, :]
        ) @ L_inv_temp[:, :]

        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i}) L_{i, i}^{-1}
        buffer[i, :, :] = (
            -buffer[i + 1, :, :] @ L_lower_diagonal_blocks_temp[:, :]
            - X_diagonal_blocks[0, :, :] @ buffer_temp[:, :]
        ) @ L_inv_temp[:, :]

        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i}) L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = (
            L_inv_temp[:, :].conj().T
            - X_lower_diagonal_blocks[i, :, :].conj().T
            @ L_lower_diagonal_blocks_temp[:, :]
            - buffer[i, :, :].conj().T @ buffer_temp[:, :]
        ) @ L_inv_temp[:, :]

    # Copy back the 2 first blocks that have been produced in the 2-sided pattern
    # to the tridiagonal storage.
    X_lower_diagonal_blocks[0, :, :] = buffer[1, :, :].conj().T


def _pobtsi_streaming(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    arr_module, _ = _get_module_from_array(arr=L_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")

    # Streams and events
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_diagonal_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    n_diag_blocks = L_diagonal_blocks.shape[0]

    # X hosts arrays pointers
    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks

    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )

    Identity = cp.eye(L_diagonal_blocks.shape[1])

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_blk_inv_d = cp.empty_like(L_diagonal_blocks[0, :, :])

    L_lower_diagonal_blocks_d_i = cp.empty_like(L_diagonal_blocks[0, :, :])

    # Backward block-selected inversion
    # --- C: events + transfers---
    compute_diagonal_h2d_events[(n_diag_blocks - 2) % 2].record(stream=compute_stream)

    # --- Host 2 Device transfers ---
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_diagonal_blocks[-1, :, :], stream=h2d_stream
    )
    h2d_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
        if invert_last_block:
            # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
            L_blk_inv_d = trsm(
                L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                Identity,
                lower=True,
            )

        if invert_last_block:
            # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
            X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
                L_blk_inv_d.conj().T @ L_blk_inv_d
            )
        compute_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(compute_diagonal_events[(n_diag_blocks - 1) % 2])
    if invert_last_block:
        X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
            out=X_diagonal_blocks[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )
    else:
        X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
            arr=X_diagonal_blocks[-1, :, :],
            stream=h2d_stream,
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

        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_blk_inv_d = trsm(
                L_diagonal_blocks_d[i % 2, :, :],
                Identity,
                lower=True,
            )

            # --- Off-diagonal block part ---
            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[i % 2, :, :]
            # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
            X_lower_diagonal_blocks_d[i % 2, :, :] = (
                -X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_diagonal_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_lower_events[i % 2].record(stream=compute_stream)

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_blk_inv_d.conj().T
                - X_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
            ) @ L_blk_inv_d
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(compute_lower_events[i % 2])
        X_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=X_lower_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )
        d2h_lower_events[i % 2].record(stream=d2h_stream)

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        X_diagonal_blocks_d[i % 2, :, :].get(
            out=X_diagonal_blocks[i, :, :], stream=d2h_stream, blocking=False
        )

    cp.cuda.Device().synchronize()


def _pobtsi_permuted_streaming(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    buffer: ArrayLike,
):
    arr_module, _ = _get_module_from_array(arr=L_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")

    # Streams and events
    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    X_diagonal_blocks = L_diagonal_blocks
    X_lower_diagonal_blocks = L_lower_diagonal_blocks

    # Backward selected-inversion
    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_inv_temp_d = cp.empty_like(L_diagonal_blocks[0])
    L_lower_diagonal_blocks_d_i = cp.empty_like(L_lower_diagonal_blocks[0])

    # Copy/Compute overlap strems
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_diagonal_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    # Device aliases & buffers specific to the middle process
    X_diagonal_top_block_d = cp.empty_like(X_diagonal_blocks[0])
    buffer_d = cp.empty(
        (2, *buffer.shape[1:]),
        dtype=buffer.dtype,
    )

    X_upper_nested_dissection_buffer_d = buffer_d

    buffer_d_i = cp.empty_like(buffer[0, :, :])

    h2d_upper_nested_dissection_buffer_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_upper_nested_dissection_buffer_events = [
        cp.cuda.Event(),
        cp.cuda.Event(),
    ]
    compute_upper_nested_dissection_buffer_h2d_events = [
        cp.cuda.Event(),
        cp.cuda.Event(),
    ]

    # --- Host 2 Device transfers ---
    X_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].set(
        arr=L_diagonal_blocks[-1, :, :], stream=h2d_stream
    )
    X_diagonal_top_block_d.set(arr=X_diagonal_blocks[0, :, :], stream=h2d_stream)
    h2d_diagonal_events[(n_diag_blocks - 1) % 2].record(h2d_stream)

    buffer_d[(n_diag_blocks - 1) % 2, :, :].set(arr=buffer[-1, :, :], stream=h2d_stream)
    h2d_upper_nested_dissection_buffer_events[(n_diag_blocks - 1) % 2].record(
        stream=h2d_stream
    )

    for i in range(n_diag_blocks - 2, 0, -1):
        # --- Host 2 Device transfers ---
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

        h2d_stream.wait_event(compute_upper_nested_dissection_buffer_h2d_events[i % 2])
        buffer_d[i % 2, :, :].set(arr=buffer[i, :, :], stream=h2d_stream)
        h2d_upper_nested_dissection_buffer_events[i % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_inv_temp_d[:, :] = trsm(
                L_diagonal_blocks_d[i % 2, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )

            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[i % 2, :, :]

            compute_stream.wait_event(h2d_upper_nested_dissection_buffer_events[i % 2])
            buffer_d_i[:, :] = buffer_d[i % 2, :, :]

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_d[i % 2, :, :] = (
                -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :].conj().T
                @ buffer_d_i[:, :]
                - X_diagonal_blocks_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
            ) @ L_inv_temp_d[:, :]
            compute_diagonal_h2d_events[(i + 1) % 2].record(stream=compute_stream)
            compute_lower_events[i % 2].record(stream=compute_stream)

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i}) L_{i, i}^{-1}
            X_upper_nested_dissection_buffer_d[i % 2, :, :] = (
                -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :]
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_diagonal_top_block_d[:, :] @ buffer_d_i[:, :]
            ) @ L_inv_temp_d[:, :]
            compute_upper_nested_dissection_buffer_h2d_events[(i + 1) % 2].record(
                stream=compute_stream
            )
            compute_upper_nested_dissection_buffer_events[i % 2].record(
                stream=compute_stream
            )

            # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i}) L_{i, i}^{-1}
            X_diagonal_blocks_d[i % 2, :, :] = (
                L_inv_temp_d[:, :].conj().T
                - X_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                @ L_lower_diagonal_blocks_d_i[:, :]
                - X_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T
                @ buffer_d_i[:, :]
            ) @ L_inv_temp_d[:, :]
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(compute_lower_events[i % 2])
        X_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=X_lower_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        d2h_lower_events[i % 2].record(stream=d2h_stream)

        d2h_stream.wait_event(compute_upper_nested_dissection_buffer_events[i % 2])
        X_upper_nested_dissection_buffer_d[i % 2, :, :].get(
            out=buffer[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        X_diagonal_blocks_d[i % 2, :, :].get(
            out=X_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

    # Copy back the first block of the nested dissection buffer to the
    # tridiagonal storage.
    with d2h_stream:
        d2h_stream.synchronize()
        X_lower_diagonal_blocks[0, :, :] = buffer[1, :, :].conj().T

    cp.cuda.Device().synchronize()
