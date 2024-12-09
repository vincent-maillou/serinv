# Copyright 2023-2024 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
    _get_module_from_str,
    _get_cholesky,
)


def pobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike = None,
    **kwargs,
):
    """Perform the Cholesky factorization of a block tridiagonal with arrowhead
    matrix using a sequential block algorithm.

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place).
    - If a device array is given, the algorithm will run on the GPU.
    """
    device_streaming: bool = kwargs.get("device_streaming", False)
    permuted_arrow: bool = kwargs.get("permuted_arrow", False)

    # Call, if possible and requested, streaming codes
    if device_streaming:
        if permuted_arrow:
            raise NotImplementedError(
                "H2D streaming not implemented for permuted_arrow scheme."
            )
        else:
            return _pobtaf_streaming(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_arrow_bottom_blocks,
                A_arrow_tip_block,
            )

    # Call "direct-array" (non-streaming) codes
    if permuted_arrow:
        if buffer is None:
            raise ValueError("Buffer for permuted_arrow must be provided.")
        return _pobtaf_permuted(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
            buffer,
        )
    else:
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
):
    xp, la = _get_module_from_array(arr=A_diagonal_blocks)
    cholesky = _get_cholesky(module_str=xp.__name__)

    n_diag_blocks: int = A_diagonal_blocks.shape[0]

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


def _pobtaf_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    buffer: ArrayLike,
):
    raise NotImplementedError("permuted_arrow scheme not implemented.")


def _pobtaf_streaming(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
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
    L_arrow_bottom_blocks = A_arrow_bottom_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Device buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_arrow_bottom_blocks_d = cp.empty(
        (2, *A_arrow_bottom_blocks.shape[1:]), dtype=A_arrow_bottom_blocks.dtype
    )
    A_arrow_tip_block_d = cp.empty_like(A_arrow_tip_block)

    # X Device buffers arrays pointers
    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_arrow_bottom_blocks_d = A_arrow_bottom_blocks_d
    L_arrow_tip_block_d = A_arrow_tip_block_d

    # Forward pass
    # --- C: events + transfers---
    compute_lower_h2d_events[1].record(stream=compute_stream)
    compute_arrow_h2d_events[1].record(stream=compute_stream)
    A_arrow_tip_block_d.set(arr=A_arrow_tip_block[:, :], stream=compute_stream)

    # --- H2D: transfers ---
    A_diagonal_blocks_d[0, :, :].set(arr=A_diagonal_blocks[0, :, :], stream=h2d_stream)
    h2d_diagonal_events[0].record(stream=h2d_stream)
    A_arrow_bottom_blocks_d[0, :, :].set(
        arr=A_arrow_bottom_blocks[0, :, :], stream=h2d_stream
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
                cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_lower_diagonal_blocks_d[i % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
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
        A_arrow_bottom_blocks_d[(i + 1) % 2, :, :].set(
            arr=A_arrow_bottom_blocks[i + 1, :, :], stream=h2d_stream
        )
        h2d_arrow_events[(i + 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            compute_stream.wait_event(h2d_arrow_events[i % 2])
            L_arrow_bottom_blocks_d[i % 2, :, :] = (
                cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2, :, :],
                    A_arrow_bottom_blocks_d[i % 2, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )
            compute_arrow_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_arrow_events[i % 2])
        L_arrow_bottom_blocks_d[i % 2, :, :].get(
            out=L_arrow_bottom_blocks[i, :, :],
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
                A_diagonal_blocks_d[(i + 1) % 2, :, :]
                - L_lower_diagonal_blocks_d[i % 2, :, :]
                @ L_lower_diagonal_blocks_d[i % 2, :, :].conj().T
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
            A_arrow_bottom_blocks_d[(i + 1) % 2, :, :] = (
                A_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                - L_arrow_bottom_blocks_d[i % 2, :, :]
                @ L_lower_diagonal_blocks_d[i % 2, :, :].conj().T
            )
            compute_lower_h2d_events[i % 2].record(stream=compute_stream)

            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
            A_arrow_tip_block_d[:, :] = (
                A_arrow_tip_block_d[:, :]
                - L_arrow_bottom_blocks_d[i % 2, :, :]
                @ L_arrow_bottom_blocks_d[i % 2, :, :].conj().T
            )
            compute_arrow_h2d_events[i % 2].record(stream=compute_stream)

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
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
        L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            cu_la.solve_triangular(
                L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                A_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )
        compute_arrow_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

    d2h_stream.wait_event(compute_arrow_events[(n_diag_blocks - 1) % 2])
    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_arrow_bottom_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    with compute_stream:
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
        A_arrow_tip_block_d[:, :] = (
            A_arrow_tip_block_d[:, :]
            - L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :]
            @ L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
        )

        # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
        L_arrow_tip_block_d[:, :] = cholesky(A_arrow_tip_block_d[:, :])

        L_arrow_tip_block_d[:, :].get(
            out=L_arrow_tip_block[:, :], stream=compute_stream
        )

    cp.cuda.Device().synchronize()
