# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
    _get_module_from_str,
    _get_cholesky,
)

from serinv.block_primitive import trsm, gemm, syherk

def pobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    **kwargs,
):
    """Perform the Cholesky factorization of a block tridiagonal with arrowhead
    matrix.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.

    Keyword Arguments
    -----------------
    device_streaming : bool, optional
        If True, the algorithm will perform host-device streaming. (default: False)
    buffer : ArrayLike, optional
        The buffer for the permuted arrowhead factorization. (default: None)
    factorize_last_block : bool, optional
        If True, the algorithm will factorize the last block, its used to perform
        partial operation in the case of a distributed algorithm. (default: True)

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.
    - If a buffer is provided, the algorithm will perform the factorization of a permuted arrowhead matrix.

    """
    expected_kwargs = {"device_streaming", "buffer", "factorize_last_block"}
    unexpected_kwargs = set(kwargs) - expected_kwargs
    if unexpected_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

    device_streaming: bool = kwargs.get("device_streaming", False)
    buffer = kwargs.get("buffer", None)
    factorize_last_block = kwargs.get("factorize_last_block", True)

    if buffer is not None:
        # Permuted arrowhead
        if device_streaming:
            return _pobtaf_permuted_streaming(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_lower_arrow_blocks,
                A_arrow_tip_block,
                buffer,
            )
        else:
            return _pobtaf_permuted(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_lower_arrow_blocks,
                A_arrow_tip_block,
                buffer,
            )
    else:
        # Natural arrowhead
        if device_streaming:
            return _pobtaf_streaming(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_lower_arrow_blocks,
                A_arrow_tip_block,
                factorize_last_block,
            )
        else:
            return _pobtaf(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_lower_arrow_blocks,
                A_arrow_tip_block,
                factorize_last_block,
            )


def _pobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    factorize_last_block: bool,
):
    xp, la = _get_module_from_array(arr=A_diagonal_blocks)
    cholesky = _get_cholesky(module_str=xp.__name__)

    n_diag_blocks: int = A_diagonal_blocks.shape[0]

    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_lower_arrow_blocks = A_lower_arrow_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Forward block-Cholesky
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = cholesky(A_diagonal_blocks[i, :, :], lower=True)

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[i, :, :] = (
            trsm(
                L_diagonal_blocks[i, :, :],
                A_lower_diagonal_blocks[i, :, :],
                trans='C',lower=True, side=1
            )
            
        )
        

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_lower_arrow_blocks[i, :, :] = (
            trsm(
                L_diagonal_blocks[i, :, :],
                A_lower_arrow_blocks[i, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        A_diagonal_blocks[i + 1, :, :] = (
            syherk(
                L_lower_diagonal_blocks[i, :, :],
                A_diagonal_blocks[i + 1, :, :],
                alpha=-1.0, beta=1.0, lower=True, cu_chol=True
            )
        )
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        A_lower_arrow_blocks[i + 1, :, :] = (
            gemm(
                L_lower_arrow_blocks[i, :, :],
                L_lower_diagonal_blocks[i, :, :],
                A_lower_arrow_blocks[i + 1, :, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        A_arrow_tip_block[:, :] = (
            syherk(
                L_lower_arrow_blocks[i, :, :],
                A_arrow_tip_block[:, :],
                alpha=-1.0, beta=1.0, lower=True, cu_chol=True
            )
        )

    if factorize_last_block:
        # L_{ndb, ndb} = chol(A_{ndb, ndb})
        L_diagonal_blocks[-1, :, :] = cholesky(A_diagonal_blocks[-1, :, :], lower=True)

        # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
        L_lower_arrow_blocks[-1, :, :] = (
            trsm(
                L_diagonal_blocks[-1, :, :],
                A_lower_arrow_blocks[-1, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
        A_arrow_tip_block[:, :] = (
            syherk(
                L_lower_arrow_blocks[-1, :, :],
                A_arrow_tip_block[:, :],
                alpha=-1.0, beta=1.0, lower=True, cu_chol=True
            )
        )
        

        # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
        L_arrow_tip_block[:, :] = cholesky(A_arrow_tip_block[:, :], lower=True)


def _pobtaf_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
):
    xp, la = _get_module_from_array(arr=A_diagonal_blocks)
    cholesky = _get_cholesky(module_str=xp.__name__)

    n_diag_blocks: int = A_diagonal_blocks.shape[0]

    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_lower_arrow_blocks = A_lower_arrow_blocks
    L_arrow_tip_block = A_arrow_tip_block

    buffer[1, :, :] = A_lower_diagonal_blocks[0, :, :].conj().T

    # Forward block-Cholesky, performed by a "middle" process
    for i in range(1, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = cholesky(A_diagonal_blocks[i, :, :])

        # Compute lower factors
        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks[i, :, :] = (
            trsm(
                L_diagonal_blocks[i, :, :],
                A_lower_diagonal_blocks[i, :, :],
                trans='C',lower=True, side=1
            )
        )

        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        buffer[i, :, :] = (
            trsm(
                L_diagonal_blocks[i, :, :],
                buffer[i, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_lower_arrow_blocks[i, :, :] = (
            trsm(
                L_diagonal_blocks[i, :, :],
                A_lower_arrow_blocks[i, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        A_diagonal_blocks[i + 1, :, :] = (
            gemm(
                L_lower_diagonal_blocks[i, :, :],
                L_lower_diagonal_blocks[i, :, :],
                A_diagonal_blocks[i + 1, :, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        A_lower_arrow_blocks[i + 1, :, :] = (
            gemm(
                L_lower_arrow_blocks[i, :, :],
                L_lower_diagonal_blocks[i, :, :],
                A_lower_arrow_blocks[i + 1, :, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        L_arrow_tip_block[:, :] = (
            gemm(
                L_lower_arrow_blocks[i, :, :],
                L_lower_arrow_blocks[i, :, :],
                L_arrow_tip_block[:, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )

        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.conj().T
        A_diagonal_blocks[0, :, :] = (
            gemm(
                buffer[i, :, :],
                buffer[i, :, :],
                A_diagonal_blocks[0, :, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )

        # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.conj().T
        buffer[i + 1, :, :] = (
            gemm(
                buffer[i, :, :],
                L_lower_diagonal_blocks[i, :, :],
                trans_b='C', alpha=-1.0
            )     
        )

        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.conj().T
        A_lower_arrow_blocks[0, :, :] = (
            gemm(
                L_lower_arrow_blocks[i, :, :],
                buffer[i, :, :],
                A_lower_arrow_blocks[0, :, :],
                trans_b='C', alpha=-1.0, beta=1.0
            )
        )


def _pobtaf_streaming(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    factorize_last_block: bool,
):
    arr_module, _ = _get_module_from_array(arr=A_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")
    cholesky = _get_cholesky(module_str="cupy")

    # Streams and events
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]

    # L host aliases
    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_lower_arrow_blocks = A_lower_arrow_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Device buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_arrow_blocks_d = cp.empty(
        (2, *A_lower_arrow_blocks.shape[1:]), dtype=A_lower_arrow_blocks.dtype
    )
    A_arrow_tip_block_d = cp.empty_like(A_arrow_tip_block)

    # X Device buffers arrays pointers
    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_lower_arrow_blocks_d = A_lower_arrow_blocks_d
    L_arrow_tip_block_d = A_arrow_tip_block_d

    # Forward pass
    # --- C: events + transfers---
    compute_lower_h2d_events[1].record(stream=compute_stream)
    compute_arrow_h2d_events[1].record(stream=compute_stream)
    A_arrow_tip_block_d.set(arr=A_arrow_tip_block[:, :], stream=compute_stream)

    # --- H2D: transfers ---
    A_diagonal_blocks_d[0, :, :].set(arr=A_diagonal_blocks[0, :, :], stream=h2d_stream)
    h2d_diagonal_events[0].record(stream=h2d_stream)
    A_lower_arrow_blocks_d[0, :, :].set(
        arr=A_lower_arrow_blocks[0, :, :], stream=h2d_stream
    )
    h2d_arrow_events[0].record(stream=h2d_stream)

    n_diag_blocks: int = A_diagonal_blocks.shape[0]
    if n_diag_blocks > 1:
        A_lower_diagonal_blocks_d[0, :, :].set(
            arr=A_lower_diagonal_blocks[0, :, :], stream=h2d_stream
        )
        h2d_lower_events[0].record(stream=h2d_stream)

    # --- D2H: event ---
    d2h_diagonal_events[1].record(stream=d2h_stream)

    for i in range(0, n_diag_blocks - 1):
        # --- Computations ---
        # L_{i, i} = chol(A_{i, i})
        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :] = cholesky(
                A_diagonal_blocks_d[i % 2, :, :]
            )
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        L_diagonal_blocks_d[i % 2, :, :].get(
            out=L_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        d2h_diagonal_events[i % 2].record(stream=d2h_stream)

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        if i + 1 < n_diag_blocks - 1:
            h2d_stream.wait_event(compute_lower_h2d_events[(i + 1) % 2])
            A_lower_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_lower_diagonal_blocks[i + 1, :, :], stream=h2d_stream
            )
            h2d_lower_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_lower_diagonal_blocks_d[i % 2, :, :],
                    trans='C',lower=True, side=1
                )
            )
            compute_lower_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_lower_events[i % 2])
        L_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=L_lower_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        h2d_stream.wait_event(compute_arrow_h2d_events[(i + 1) % 2])
        A_lower_arrow_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_lower_arrow_blocks[i + 1, :, :], stream=h2d_stream
        )
        h2d_arrow_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_arrow_events[i % 2])
            L_lower_arrow_blocks_d[i % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_lower_arrow_blocks_d[i % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )
            compute_arrow_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_arrow_events[i % 2])
        L_lower_arrow_blocks_d[i % 2, :, :].get(
            out=L_lower_arrow_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        # Update next diagonal block
        h2d_stream.wait_event(d2h_diagonal_events[(i + 1) % 2])
        A_diagonal_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_diagonal_blocks[i + 1, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[(i + 1) % 2])
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
            A_diagonal_blocks_d[(i + 1) % 2, :, :] = (
                gemm(
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    A_diagonal_blocks_d[(i + 1) % 2, :, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
            A_lower_arrow_blocks_d[(i + 1) % 2, :, :] = (
                gemm(
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    A_lower_arrow_blocks_d[(i + 1) % 2, :, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )
            compute_lower_h2d_events[i % 2].record(stream=compute_stream)

            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
            A_arrow_tip_block_d[:, :] = (
                gemm(
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    A_arrow_tip_block_d[:, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )
            compute_arrow_h2d_events[i % 2].record(stream=compute_stream)

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
        if factorize_last_block:
            L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = cholesky(
                A_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :]
            )
        compute_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    d2h_stream.wait_event(compute_diagonal_events[(n_diag_blocks - 1) % 2])
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_diagonal_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    with compute_stream:
        compute_stream.wait_event(h2d_arrow_events[(n_diag_blocks - 1) % 2])
        if factorize_last_block:
            L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                    A_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )
        compute_arrow_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    d2h_stream.wait_event(compute_arrow_events[(n_diag_blocks - 1) % 2])
    L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_lower_arrow_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    with compute_stream:
        if factorize_last_block:
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
            A_arrow_tip_block_d[:, :] = (
                gemm(
                    L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                    L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                    A_arrow_tip_block_d[:, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

            # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
            L_arrow_tip_block_d[:, :] = cholesky(A_arrow_tip_block_d[:, :])

        L_arrow_tip_block_d[:, :].get(
            out=L_arrow_tip_block[:, :], stream=compute_stream
        )

    cp.cuda.Device().synchronize()


def _pobtaf_permuted_streaming(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
):
    arr_module, _ = _get_module_from_array(arr=A_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")
    cholesky = _get_cholesky(module_str="cupy")

    # Streams and events
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    cp_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_lower_events_h2d_release = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    cp_arrow_events_h2d_release = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]

    # Host aliases & buffers
    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_lower_arrow_blocks = A_lower_arrow_blocks

    # Device aliases & buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty(
        (2, *A_lower_diagonal_blocks.shape[1:]),
        dtype=A_lower_diagonal_blocks.dtype,
    )
    A_lower_arrow_blocks_d = cp.empty(
        (2, *A_lower_arrow_blocks.shape[1:]),
        dtype=A_lower_arrow_blocks.dtype,
    )
    A_arrow_tip_block_d = cp.zeros_like(A_arrow_tip_block)

    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_lower_arrow_blocks_d = A_lower_arrow_blocks_d

    n_diag_blocks = A_diagonal_blocks.shape[0]

    # Device aliases & buffers specific to the middle process
    A_diagonal_top_block_d = cp.empty_like(A_diagonal_blocks[0])
    A_arrow_bottom_top_block_d = cp.empty_like(A_lower_arrow_blocks[0])
    buffer_d = cp.empty(
        (2, *buffer.shape[1:]),
        dtype=buffer.dtype,
    )

    L_upper_nested_dissection_buffer_d = buffer_d

    cp_upper_nested_dissection_buffer_events = [cp.cuda.Event(), cp.cuda.Event()]

    # Forward block-Cholesky, performed by a "middle" process
    # --- Initial Host 2 Device transfers ---
    # --- H2D: transfers ---
    A_diagonal_blocks_d[1, :, :].set(arr=A_diagonal_blocks[1, :, :], stream=h2d_stream)
    A_diagonal_top_block_d[:, :].set(arr=A_diagonal_blocks[0, :, :], stream=h2d_stream)
    buffer_d[1, :, :].set(
        arr=A_lower_diagonal_blocks[0, :, :].conj().T, stream=h2d_stream
    )
    A_arrow_tip_block_d.set(arr=A_arrow_tip_block[:, :], stream=h2d_stream)
    h2d_diagonal_events[1].record(stream=h2d_stream)

    if n_diag_blocks > 2:
        A_lower_diagonal_blocks_d[1, :, :].set(
            arr=A_lower_diagonal_blocks[1, :, :], stream=h2d_stream
        )
        h2d_lower_events[1].record(stream=h2d_stream)

    A_lower_arrow_blocks_d[1, :, :].set(
        arr=A_lower_arrow_blocks[1, :, :], stream=h2d_stream
    )
    A_arrow_bottom_top_block_d[:, :].set(
        arr=A_lower_arrow_blocks[0, :, :], stream=h2d_stream
    )
    h2d_arrow_events[1].record(stream=h2d_stream)

    # --- CP: events ---
    cp_lower_events_h2d_release[1].record(stream=compute_stream)
    cp_arrow_events_h2d_release[1].record(stream=compute_stream)

    # --- D2H: event ---
    d2h_diagonal_events[1].record(stream=d2h_stream)

    for i in range(1, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :] = cholesky(
                A_diagonal_blocks_d[i % 2, :, :]
            )
            cp_diagonal_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(cp_diagonal_events[i % 2])
        L_diagonal_blocks_d[i % 2, :, :].get(
            out=L_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        d2h_diagonal_events[i % 2].record(stream=d2h_stream)

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        if i + 1 < n_diag_blocks - 1:
            h2d_stream.wait_event(cp_lower_events_h2d_release[(i + 1) % 2])
            A_lower_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_lower_diagonal_blocks[i + 1, :, :], stream=h2d_stream
            )
            h2d_lower_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_lower_diagonal_blocks_d[i % 2, :, :],
                    trans='C',lower=True, side=1
                )
            )
            cp_lower_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(cp_lower_events[i % 2])
        L_lower_diagonal_blocks_d[i % 2, :, :].get(
            out=L_lower_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        h2d_stream.wait_event(cp_arrow_events_h2d_release[(i + 1) % 2])
        A_lower_arrow_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_lower_arrow_blocks[i + 1, :, :], stream=h2d_stream
        )
        h2d_arrow_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_arrow_events[i % 2])
            L_lower_arrow_blocks_d[i % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_lower_arrow_blocks_d[i % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )
            cp_arrow_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(cp_arrow_events[i % 2])
        L_lower_arrow_blocks_d[i % 2, :, :].get(
            out=L_lower_arrow_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        with compute_stream:
            L_upper_nested_dissection_buffer_d[i % 2, :, :] = (
                trsm(
                    L_diagonal_blocks_d[i % 2, :, :],
                    buffer_d[i % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )
            cp_upper_nested_dissection_buffer_events[i % 2].record(
                stream=compute_stream
            )

        d2h_stream.wait_event(cp_upper_nested_dissection_buffer_events[i % 2])
        L_upper_nested_dissection_buffer_d[i % 2, :, :].get(
            out=buffer[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        # Update next diagonal block
        h2d_stream.wait_event(d2h_diagonal_events[(i + 1) % 2])
        A_diagonal_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_diagonal_blocks[i + 1, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            # Update next diagonal block
            compute_stream.wait_event(h2d_diagonal_events[(i + 1) % 2])
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
            A_diagonal_blocks_d[(i + 1) % 2, :, :] = (
                gemm(
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    A_diagonal_blocks_d[(i + 1) % 2, :, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
            A_lower_arrow_blocks_d[(i + 1) % 2, :, :] = (
                gemm(
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    A_lower_arrow_blocks_d[(i + 1) % 2, :, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

            # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.conj().T
            buffer_d[(i + 1) % 2, :, :] = (
                gemm(
                    L_upper_nested_dissection_buffer_d[i % 2, :, :],
                    L_lower_diagonal_blocks_d[i % 2, :, :],
                    trans_b='C', alpha=-1.0
                )
            )
            cp_lower_events_h2d_release[i % 2].record(stream=compute_stream)

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
            A_arrow_tip_block_d[:, :] = (
                gemm(
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    A_arrow_tip_block_d[:, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

            # Update the top (first blocks) of the arrowhead
            # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.conj().T
            A_arrow_bottom_top_block_d[:, :] = (
                gemm(
                    L_lower_arrow_blocks_d[i % 2, :, :],
                    L_upper_nested_dissection_buffer_d[i % 2, :, :],
                    A_arrow_bottom_top_block_d[:, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )
            cp_arrow_events_h2d_release[i % 2].record(stream=compute_stream)

            # Update top and next upper/lower blocks of 2-sided factorization pattern
            # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.conj().T
            A_diagonal_top_block_d[:, :] = (
                gemm(
                    L_upper_nested_dissection_buffer_d[i % 2, :, :],
                    L_upper_nested_dissection_buffer_d[i % 2, :, :],
                    A_diagonal_top_block_d[:, :],
                    trans_b='C', alpha=-1.0, beta=1.0
                )
            )

    # --- Device 2 Host transfers ---
    d2h_stream.wait_event(cp_lower_events_h2d_release[(n_diag_blocks - 2) % 2])
    A_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=A_diagonal_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )
    A_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=A_lower_arrow_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )
    buffer_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=buffer[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    d2h_stream.wait_event(cp_arrow_events_h2d_release[(n_diag_blocks - 2) % 2])
    A_arrow_bottom_top_block_d.get(
        out=A_lower_arrow_blocks[0, :, :],
        stream=d2h_stream,
        blocking=False,
    )
    A_diagonal_top_block_d.get(
        out=A_diagonal_blocks[0, :, :],
        stream=compute_stream,
        blocking=False,
    )

    A_arrow_tip_block_d.get(
        out=A_arrow_tip_block[:, :],
        stream=d2h_stream,
        blocking=False,
    )

    cp.cuda.Device().synchronize()
