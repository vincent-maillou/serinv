# Copyright 2023-2025 ETH Zurich. All rights reserved.


from serinv import (
    ArrayLike,
    _get_module_from_array,
    _get_module_from_str,
)


def pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
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
    partial = kwargs.get("partial", False)

    if buffer is not None:
        # Permuted arrowhead
        if device_streaming:
            raise NotImplementedError(
                "Streaming is not implemented for the permuted arrowhead."
            )
        else:
            _pobtas_permuted(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_lower_arrow_blocks,
                L_arrow_tip_block,
                B,
                buffer,
                trans,
            )
    else:
        # Natural arrowhead
        if device_streaming:
            _pobtas_streaming(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_lower_arrow_blocks,
                L_arrow_tip_block,
                B,
                trans,
                partial,
            )
        else:
            _pobtas(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_lower_arrow_blocks,
                L_arrow_tip_block,
                B,
                trans,
                partial,
            )


def _pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    trans: str,
    partial: bool,
):
    _, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    if trans == "N":
        # ----- Forward substitution -----
        for i in range(0, n_diag_blocks - 1):
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize],
                lower=True,
            )

            B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize] -= (
                L_lower_diagonal_blocks[i]
                @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
            )

            B[-arrow_blocksize:] -= (
                L_lower_arrow_blocks[i]
                @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
            )

        if not partial:
            # In the case of the partial solve, we do not solve the last block and
            # arrow tip block of the RHS.
            B[(n_diag_blocks - 1) * diag_blocksize : n_diag_blocks * diag_blocksize] = (
                la.solve_triangular(
                    L_diagonal_blocks[n_diag_blocks - 1],
                    B[
                        (n_diag_blocks - 1)
                        * diag_blocksize : n_diag_blocks
                        * diag_blocksize
                    ],
                    lower=True,
                )
            )

            B[-arrow_blocksize:] -= (
                L_lower_arrow_blocks[-1]
                @ B[
                    (n_diag_blocks - 1)
                    * diag_blocksize : n_diag_blocks
                    * diag_blocksize
                ]
            )

            # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
            B[-arrow_blocksize:] = la.solve_triangular(
                L_arrow_tip_block[:], B[-arrow_blocksize:], lower=True
            )
    elif trans == "T" or trans == "C":
        # ----- Backward substitution -----
        if not partial:
            # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
            B[-arrow_blocksize:] = la.solve_triangular(
                L_arrow_tip_block[:],
                B[-arrow_blocksize:],
                lower=True,
                trans="C",
            )

            # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
            B[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = (
                la.solve_triangular(
                    L_diagonal_blocks[-1],
                    B[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
                    - L_lower_arrow_blocks[-1].conj().T @ B[-arrow_blocksize:],
                    lower=True,
                    trans="C",
                )
            )

        for i in range(n_diag_blocks - 2, -1, -1):
            # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize]
                - L_lower_diagonal_blocks[i].conj().T
                @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                - L_lower_arrow_blocks[i].conj().T @ B[-arrow_blocksize:],
                lower=True,
                trans="C",
            )
    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")


def _pobtas_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    buffer: ArrayLike,
    trans: str,
):
    _, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
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

            # Update the tip RHS block
            B[-arrow_blocksize:] -= (
                L_lower_arrow_blocks[i]
                @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
            )
    elif trans == "T" or trans == "C":
        # ----- Backward substitution -----
        for i in range(n_diag_blocks - 2, 0, -1):
            B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
                L_diagonal_blocks[i],
                B[i * diag_blocksize : (i + 1) * diag_blocksize]
                - L_lower_diagonal_blocks[i].conj().T
                @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                - L_lower_arrow_blocks[i].conj().T @ B[-arrow_blocksize:]
                - buffer[i].conj().T @ B[:diag_blocksize],
                lower=True,
                trans="C",
            )
    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")
    
def _pobtas_streaming(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    trans: str,
    partial: bool,
):
    arr_module, _ = _get_module_from_array(arr=L_diagonal_blocks)
    if arr_module.__name__ != "numpy":
        raise NotImplementedError(
            "Host<->Device streaming only works when host-arrays are given."
        )

    cp, cu_la = _get_module_from_str(module_str="cupy")

    # Vars
    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    # Streams
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    

    # Device Buffers
    # B Buffers
    B_shape = B[-arrow_blocksize:] # block template
    B_arrow_tip_d = cp.empty_like(B_shape)

    B_shape = B[0 : diag_blocksize] 
    B_d = cp.empty(
        (2, *B_shape.shape), dtype=B_shape.dtype
    )
    

    # L Buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_lower_arrow_blocks_d = cp.empty(
        (2, *L_lower_arrow_blocks.shape[1:]), dtype=L_diagonal_blocks.dtype
    )
    L_arrow_tip_block_d = cp.empty_like(L_arrow_tip_block)

    if trans == "N":

        # delete helper variable
        del B_shape

        # Events
        h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
        h2d_lower_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
        h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
        h2d_B_events = [cp.cuda.Event(), cp.cuda.Event()]

        d2h_B_events = [cp.cuda.Event(), cp.cuda.Event()]
        d2h_tip_events = [cp.cuda.Event(), cp.cuda.Event()]

        compute_current_B_events = [cp.cuda.Event(), cp.cuda.Event()]
        compute_next_B_events = [cp.cuda.Event(), cp.cuda.Event()]
        compute_arrow_B_events = [cp.cuda.Event(), cp.cuda.Event()]

        compute_partial_events = [cp.cuda.Event(), cp.cuda.Event()]

        # Forward Pass
        # --- C: events + transfers ---
        compute_current_B_events[1].record(stream=compute_stream)
        compute_next_B_events[1].record(stream=compute_stream)
        compute_arrow_B_events[1].record(stream=compute_stream)

        B_arrow_tip_d.set(arr=B[-arrow_blocksize:], stream=h2d_stream)
        L_arrow_tip_block_d.set(arr=L_arrow_tip_block[:], stream=h2d_stream)

        # --- H2D: transfers ---
        B_d[0].set(arr=B[0 : diag_blocksize], stream = h2d_stream)
        h2d_B_events[0].record(stream=h2d_stream)
        
        L_diagonal_blocks_d[0].set(arr=L_diagonal_blocks[0], stream=h2d_stream)
        h2d_diagonal_events[0].record(stream=h2d_stream)

        L_lower_arrow_blocks_d[0].set(arr=L_lower_arrow_blocks[0], stream=h2d_stream)
        h2d_arrow_events[0].record(stream=h2d_stream)

        # --- D2H: event ---
        d2h_B_events[1].record(stream=d2h_stream)
        
        n_diag_blocks: int = L_diagonal_blocks.shape[0]

        if n_diag_blocks > 1:

            L_lower_diagonal_blocks_d[0].set(arr=L_lower_diagonal_blocks[0], stream=h2d_stream)
            h2d_lower_diagonal_events[0].record(stream=h2d_stream)

        
        # --- Forward substitution ---
        for i in range(0, n_diag_blocks - 1):

            if i + 1 < n_diag_blocks:
                # stream next B block
                h2d_stream.wait_event(compute_arrow_B_events[(i + 1) % 2])

                B_d[(i + 1) % 2].set(
                    arr=B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize],
                    stream = h2d_stream
                )

                h2d_B_events[(i + 1) % 2].record(stream=h2d_stream)

            if i + 1 < n_diag_blocks - 1:
                # stream next diagonal block
                h2d_stream.wait_event(compute_current_B_events[(i + 1) % 2])

                L_diagonal_blocks_d[(i + 1) % 2].set(
                    arr=L_diagonal_blocks[i + 1], 
                    stream=h2d_stream
                )

                h2d_diagonal_events[(i + 1) % 2].record(stream=h2d_stream)


            with compute_stream:
                # Compute step 1 : compute B
                compute_stream.wait_event(h2d_diagonal_events[i % 2])

                B_d[i % 2] = cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2],
                    B_d[i % 2],
                    lower=True,
                )

                compute_current_B_events[i % 2].record(stream=compute_stream)
            
            # stream B back
            d2h_stream.wait_event(compute_current_B_events[i % 2])

            B_d[i % 2].get(
                out=B[i * diag_blocksize : (i + 1) * diag_blocksize],
                stream=d2h_stream,
                blocking=False,
            )

            d2h_B_events[i % 2].record(stream=d2h_stream)

            if i + 1 < n_diag_blocks - 1:
                # stream next lower diagonal block
                h2d_stream.wait_event(compute_next_B_events[(i + 1) % 2])

                L_lower_diagonal_blocks_d[(i + 1) % 2].set(
                    arr=L_lower_diagonal_blocks[i + 1], 
                    stream=h2d_stream
                )

                h2d_lower_diagonal_events[(i + 1) % 2].record(stream=h2d_stream)
            
            with compute_stream:
                # Compute step 2 : update next B
                compute_stream.wait_event(h2d_B_events[(i + 1) % 2])

                B_d[(i + 1) % 2] -= (
                    L_lower_diagonal_blocks_d[i % 2]
                    @ B_d[i % 2]
                )

                compute_next_B_events[i % 2].record(stream=compute_stream)                
                
            if i + 1 < n_diag_blocks - 1:
                # stream next lower arrow block
                h2d_stream.wait_event(compute_arrow_B_events[(i + 1) % 2])

                L_lower_arrow_blocks_d[(i + 1) % 2].set(
                    arr=L_lower_arrow_blocks[i + 1], 
                    stream=h2d_stream
                )

                h2d_arrow_events[(i + 1) % 2].record(stream=h2d_stream)

            with compute_stream:
                # Compute step 3 : update arrow tip
                compute_stream.wait_event(h2d_arrow_events[i % 2])
                
                B_arrow_tip_d -= (
                    L_lower_arrow_blocks_d[i % 2]
                    @ B_d[i % 2]
                )

                compute_arrow_B_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(compute_arrow_B_events[i % 2])
            B_arrow_tip_d.get(out=B[-arrow_blocksize:], stream=d2h_stream, blocking=False,)
            d2h_tip_events[i % 2].record(stream=d2h_stream)


        if not partial:
            # In the case of the partial solve, we do not solve the last block and
            # arrow tip block of the RHS.
            
            h2d_stream.wait_event(d2h_tip_events[n_diag_blocks % 2])
            L_diagonal_blocks_d[(n_diag_blocks - 1) % 2].set(arr=L_diagonal_blocks[n_diag_blocks - 1], stream=h2d_stream)
            h2d_diagonal_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

            L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2].set(arr=L_lower_arrow_blocks[-1], stream=h2d_stream)
            h2d_arrow_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

            
            with compute_stream:

                compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
                B_d[(n_diag_blocks - 1) % 2] = (cu_la.solve_triangular(L_diagonal_blocks_d[(n_diag_blocks - 1) % 2], B_d[(n_diag_blocks - 1) % 2], lower=True,))
                compute_partial_events[0].record(stream=compute_stream)

            d2h_stream.wait_event(compute_partial_events[0])
            B_d[(n_diag_blocks - 1) % 2].get(out=B[(n_diag_blocks - 1) * diag_blocksize : n_diag_blocks * diag_blocksize], stream=d2h_stream, blocking=False,)
            d2h_B_events[0].record(stream=d2h_stream)

            with compute_stream:
                compute_stream.wait_event(h2d_arrow_events[(n_diag_blocks - 1) % 2])

                B_arrow_tip_d -= (L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2] @ B_d[(n_diag_blocks - 1) % 2])
                compute_partial_events[1].record(stream=compute_stream)

                compute_stream.wait_event(compute_partial_events[1])
                B_arrow_tip_d = cu_la.solve_triangular(L_arrow_tip_block_d, B_arrow_tip_d, lower=True)
                compute_partial_events[0].record(stream=compute_stream)

            d2h_stream.wait_event(compute_partial_events[0])
            B_arrow_tip_d.get(out=B[-arrow_blocksize:], stream=d2h_stream, blocking=False,)

    elif trans == "T" or trans == "C":
        # Buffers
        B_previous_d = cp.empty(
            (2, *B_shape.shape), dtype=B_shape.dtype
        )
        del B_shape
        
        # Events
        compute_B_events = [cp.cuda.Event(), cp.cuda.Event()]
        previous_B_events = [cp.cuda.Event(), cp.cuda.Event()]
        h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
        d2h_events = [cp.cuda.Event(), cp.cuda.Event()]
        
        # Forward Pass
        # --- C: events + transfers ---

        B_arrow_tip_d.set(arr=B[-arrow_blocksize:], stream=h2d_stream)
        L_arrow_tip_block_d.set(arr=L_arrow_tip_block[:], stream=h2d_stream)
        B_d[(n_diag_blocks - 1) % 2].set(arr=B[-arrow_blocksize - diag_blocksize : -arrow_blocksize], stream=h2d_stream)
        L_diagonal_blocks_d[(n_diag_blocks - 1) % 2].set(arr=L_diagonal_blocks[-1], stream=h2d_stream)
        L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2].set(arr=L_lower_arrow_blocks[-1], stream=h2d_stream)

        h2d_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)
        



        # ----- Backward substitution -----
        if not partial:
            # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
            with compute_stream:
                compute_stream.wait_event(h2d_events[(n_diag_blocks - 1) % 2])
                B_arrow_tip_d = cu_la.solve_triangular( 
                    L_arrow_tip_block_d,
                    B_arrow_tip_d,
                    lower=True,
                    trans="C",
                )

                B_d[(n_diag_blocks - 1) % 2] = (
                    cu_la.solve_triangular(
                        L_diagonal_blocks_d[(n_diag_blocks - 1) % 2],
                        B_d[(n_diag_blocks - 1) % 2]
                        - L_lower_arrow_blocks_d[(n_diag_blocks - 1) % 2].conj().T @ B_arrow_tip_d,
                        lower=True,
                        trans="C",
                    )
                )

                compute_B_events[(n_diag_blocks - 1) % 2].record(stream=compute_stream)

            d2h_stream.wait_event(compute_B_events[(n_diag_blocks - 1) % 2])
            B_arrow_tip_d.get(out=B[-arrow_blocksize:], stream=d2h_stream, blocking=False,)
            B_d[(n_diag_blocks - 1) % 2].get(out=B[-arrow_blocksize - diag_blocksize : -arrow_blocksize], stream=d2h_stream, blocking=False,)
            d2h_events[(n_diag_blocks - 1) % 2].record(stream=d2h_stream)
            

        if n_diag_blocks > 1:

            B_d[n_diag_blocks % 2].set(
                arr=B[-arrow_blocksize - (2 * diag_blocksize) : -arrow_blocksize - diag_blocksize], 
                stream=h2d_stream
            )
            L_diagonal_blocks_d[n_diag_blocks % 2].set(arr=L_diagonal_blocks[-2], stream=h2d_stream)
            L_lower_arrow_blocks_d[n_diag_blocks % 2].set(arr=L_lower_arrow_blocks[-2], stream=h2d_stream)
            L_lower_diagonal_blocks_d[n_diag_blocks % 2].set(arr=L_lower_diagonal_blocks[-1], stream=h2d_stream)
            h2d_stream.wait_event(previous_B_events[(n_diag_blocks - 1) % 2])
            B_previous_d[(n_diag_blocks - 1) % 2].set(arr=B[-arrow_blocksize - diag_blocksize : -arrow_blocksize], stream=h2d_stream)
            h2d_events[(n_diag_blocks - 1) % 2].record(stream=h2d_stream)

        for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
            print("---")
            if i > 0:
                h2d_stream.wait_event(compute_B_events[(i - 1) % 2])
                B_d[(i - 1) % 2].set(arr=B[(i - 1) * diag_blocksize : i * diag_blocksize], stream=h2d_stream)
                L_diagonal_blocks_d[(i - 1) % 2].set(arr=L_diagonal_blocks[i - 1], stream=h2d_stream)
                L_lower_diagonal_blocks_d[(i - 1) % 2].set(arr=L_lower_diagonal_blocks[i - 1], stream=h2d_stream)
                L_lower_arrow_blocks_d[(i - 1) % 2].set(arr=L_lower_arrow_blocks[i - 1], stream=h2d_stream)
                h2d_events[i % 2].record(stream=h2d_stream)
            
            with compute_stream:
                compute_stream.wait_event(h2d_events[(i - 1) % 2])
                compute_stream.wait_event(d2h_events[(i - 1) % 2])
                B_previous_d[i % 2] = cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2],
                    B_d[i % 2]
                    - L_lower_diagonal_blocks_d[i % 2].conj().T
                    @ B_previous_d[(i + 1) % 2]
                    - L_lower_arrow_blocks_d[i % 2].conj().T @ B_arrow_tip_d,
                    lower=True,
                    trans="C",
                )

                compute_B_events[i % 2].record(compute_stream)

            d2h_stream.wait_event(compute_B_events[(i - 1) % 2])
            B_previous_d[(i + 1) % 2].get(out=B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize], stream=d2h_stream, blocking=False)
            d2h_events[i % 2].record(stream=d2h_stream)

        #B_previous_d[0].get(out=B[:diag_blocksize], stream=d2h_stream, blocking=False)
            
    else:
        raise ValueError(f"Invalid transpose argument: {trans}.")
    

    cp.cuda.Device().synchronize()