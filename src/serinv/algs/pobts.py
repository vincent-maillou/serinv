# Copyright 2023-2025 ETH Zurich. All rights reserved.


from serinv import (
    ArrayLike,
    _get_module_from_array,
    _get_module_from_str,
)


def pobts(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
    trans="N",
    **kwargs,
) -> ArrayLike:
    """Solve a block tridiagonal arrowhead linear system given its Cholesky factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    """
    device_streaming: bool = kwargs.get("device_streaming", False)
    buffer = kwargs.get("buffer", None)
    # solve_last_rhs = kwargs.get("solve_last_rhs", True)

    if buffer is not None:
        # Permuted arrowhead
        if device_streaming:
            raise NotImplementedError("Permuted arrowhead is not implemented.")
        else:
            _pobts_permuted(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                B,
                buffer,
                trans,
            )
    else:
        # Natural arrowhead
        if device_streaming:
            _pobts_streaming(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                B,
                trans,
            )
        else:
            _pobts(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                B,
                trans,
            )


def _pobts(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
    trans: str,
):
    _, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    if trans == "N":
        # ----- Forward substitution -----
        B[0:diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[0],
            B[0:diag_blocksize],
            lower=True,
        )

        for i in range(1, n_diag_blocks):
            # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize]
                - L_lower_diagonal_blocks[i - 1]
                @ B[(i - 1) * diag_blocksize : (i) * diag_blocksize],
                lower=True,
            )
    elif trans == "T" or trans == "C":
        # ----- Backward substitution -----
        # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
        B[-diag_blocksize:] = la.solve_triangular(
            L_diagonal_blocks[-1],
            B[-diag_blocksize:],
            lower=True,
            trans="C",
        )

        for i in range(n_diag_blocks - 2, -1, -1):
            # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize]
                - L_lower_diagonal_blocks[i].conj().T
                @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize],
                lower=True,
                trans="C",
            )
    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")


def _pobts_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
    buffer: ArrayLike,
    trans: str,
):
    _, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    if trans == "N":
        # ----- Forward substitution -----
        for i in range(1, n_diag_blocks - 1):
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize],
                lower=True,
            )

            # Update the next RHS block
            B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize] -= (
                L_lower_diagonal_blocks[i]
                @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
            )

            # Update the first RHS block (permutation-linked)
            B[:diag_blocksize] -= (
                buffer[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
            )
    elif trans == "T" or trans == "C":
        # ----- Backward substitution -----
        for i in range(n_diag_blocks - 2, 0, -1):
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize]
                - L_lower_diagonal_blocks[i].conj().T
                @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                - buffer[i].conj().T @ B[:diag_blocksize],
                lower=True,
                trans="C",
            )
    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")


def _pobts_streaming(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
    trans: str,
):
    arr_module, _ = _get_module_from_array(arr=L_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")

    # Vars
    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    # Streams
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    # Device Buffers
    # B Buffers
    B_shape = B[0 : diag_blocksize] 
    B_d = cp.empty(
        (2, *B_shape.shape), dtype=B_shape.dtype
    )
    B_previous_d = cp.empty(
        (2, *B_shape.shape), dtype=B_shape.dtype
    )

    # L Buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )

    # Events
    compute_B_events = [cp.cuda.Event(), cp.cuda.Event()]
    previous_B_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    d2h_events = [cp.cuda.Event(), cp.cuda.Event()]

    if trans == "N":
        B_d[0].set(arr=B[:diag_blocksize], stream=h2d_stream)
        L_diagonal_blocks_d[0].set(arr=L_diagonal_blocks[0], stream=h2d_stream)

        h2d_events[0].record(stream=h2d_stream)

        with compute_stream:
            B_d[0] = (
                cu_la.solve_triangular(
                    L_diagonal_blocks_d[0],
                    B_d[0],
                    lower=True,
                )
            )

            compute_B_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_B_events[0])
        B_d[0].get(out=B[:diag_blocksize], stream=d2h_stream, blocking=False,)
        d2h_events[0].record(stream=d2h_stream)

        if n_diag_blocks > 1:

            B_d[1].set(
                arr=B[diag_blocksize : (2 * diag_blocksize)], 
                stream=h2d_stream
            )
            L_diagonal_blocks_d[1].set(arr=L_diagonal_blocks[1], stream=h2d_stream)
            L_lower_diagonal_blocks_d[1].set(arr=L_lower_diagonal_blocks[0], stream=h2d_stream)
            h2d_stream.wait_event(previous_B_events[0])
            B_previous_d[0].set(arr=B[:diag_blocksize], stream=h2d_stream)
            h2d_events[0].record(stream=h2d_stream)

        for i in range(1, n_diag_blocks):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
            
            if i + 1 < n_diag_blocks:
                h2d_stream.wait_event(compute_B_events[(i + 1) % 2])
                B_d[(i + 1) % 2].set(arr=B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize], stream=h2d_stream)
                L_diagonal_blocks_d[(i + 1) % 2].set(arr=L_diagonal_blocks[i + 1], stream=h2d_stream)
                L_lower_diagonal_blocks_d[(i + 1) % 2].set(arr=L_lower_diagonal_blocks[i], stream=h2d_stream)
                h2d_events[i % 2].record(stream=h2d_stream)
            
            with compute_stream:
                compute_stream.wait_event(h2d_events[(i + 1) % 2])
                compute_stream.wait_event(d2h_events[(i + 1) % 2])
                print(B)
                print(B_d)
                print(i % 2)
                B_previous_d[i % 2] = cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2],
                    B_d[i % 2]
                    - L_lower_diagonal_blocks_d[i % 2]
                    @ B_previous_d[(i + 1) % 2],
                    lower=True,
                )

                compute_B_events[i % 2].record(compute_stream)

            d2h_stream.wait_event(compute_B_events[(i - 1) % 2])
            B_previous_d[(i + 1) % 2].get(out=B[i * diag_blocksize : (i + 1) * diag_blocksize], stream=d2h_stream, blocking=False)
            d2h_events[i % 2].record(stream=d2h_stream)

        if n_diag_blocks > 1:
            d2h_stream.wait_event(compute_B_events[(n_diag_blocks - 1) % 2])
            B_previous_d[(n_diag_blocks - 1) % 2].get(out=B[-diag_blocksize:], stream=d2h_stream, blocking=False)
        
    
    elif trans == "T" or trans == "C":
        B_d[(n_diag_blocks - 1) % 2].set(arr=B[-diag_blocksize:], stream=h2d_stream)
        L_diagonal_blocks_d[(n_diag_blocks - 1) % 2].set(arr=L_diagonal_blocks[-1], stream=h2d_stream)

        h2d_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

        with compute_stream:
            B_d[(n_diag_blocks - 1) % 2] = (
                cu_la.solve_triangular(
                    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2],
                    B_d[(n_diag_blocks - 1) % 2],
                    lower=True,
                    trans="C",
                )
            )

            compute_B_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_B_events[(n_diag_blocks - 1) % 2])
        B_d[(n_diag_blocks - 1) % 2].get(out=B[-diag_blocksize:], stream=d2h_stream, blocking=False,)
        d2h_events[(n_diag_blocks - 1) % 2].record(stream=d2h_stream)

        if n_diag_blocks > 1:

            B_d[n_diag_blocks % 2].set(
                arr=B[-(2 * diag_blocksize) : -diag_blocksize], 
                stream=h2d_stream
            )
            L_diagonal_blocks_d[n_diag_blocks % 2].set(arr=L_diagonal_blocks[-2], stream=h2d_stream)
            L_lower_diagonal_blocks_d[n_diag_blocks % 2].set(arr=L_lower_diagonal_blocks[-1], stream=h2d_stream)
            h2d_stream.wait_event(previous_B_events[(n_diag_blocks - 1) % 2])
            B_previous_d[(n_diag_blocks - 1) % 2].set(arr=B[-diag_blocksize:], stream=h2d_stream)
            h2d_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

        for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
            if i > 0:
                h2d_stream.wait_event(compute_B_events[(i - 1) % 2])
                B_d[(i - 1) % 2].set(arr=B[(i - 1) * diag_blocksize : i * diag_blocksize], stream=h2d_stream)
                L_diagonal_blocks_d[(i - 1) % 2].set(arr=L_diagonal_blocks[i - 1], stream=h2d_stream)
                L_lower_diagonal_blocks_d[(i - 1) % 2].set(arr=L_lower_diagonal_blocks[i - 1], stream=h2d_stream)
                h2d_events[i % 2].record(stream=h2d_stream)
            
            with compute_stream:
                compute_stream.wait_event(h2d_events[(i - 1) % 2])
                compute_stream.wait_event(d2h_events[(i - 1) % 2])
                B_previous_d[i % 2] = cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2],
                    B_d[i % 2]
                    - L_lower_diagonal_blocks_d[i % 2].conj().T
                    @ B_previous_d[(i + 1) % 2],
                    lower=True,
                    trans="C",
                )

                compute_B_events[i % 2].record(compute_stream)

            d2h_stream.wait_event(compute_B_events[(i - 1) % 2])
            B_previous_d[(i + 1) % 2].get(out=B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize], stream=d2h_stream, blocking=False)
            d2h_events[i % 2].record(stream=d2h_stream)

        if n_diag_blocks > 1:
            d2h_stream.wait_event(compute_B_events[0])
            B_previous_d[0].get(out=B[:diag_blocksize], stream=d2h_stream, blocking=False)

    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")
    
    cp.cuda.Device().synchronize()