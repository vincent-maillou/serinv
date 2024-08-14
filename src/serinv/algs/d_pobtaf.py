# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cu_la
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike
from mpi4py import MPI

# comm_rank = MPI.COMM_WORLD.Get_rank()
# comm_size = MPI.COMM_WORLD.Get_size()


def d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    device_streaming: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the distributed Cholesky factorization of a block tridiagonal
    with arrowhead matrix.

    Note:
    -----
    - The matrix, A, is assumed to be symmetric positive definite.
    - Will overwrite the inputs and store the results in them (in-place). Returns
        are aliases on the inputs.
    - If a device array is given, the algorithm will run on the GPU.

    Complexity analysis:
        Parameters:
            n : number of diagonal blocks
            b : diagonal block size
            a : arrow block size
            p : number of processes

        The FLOPS count of the top process is the same as the one of the sequential
        block Cholesky factorization algorithm (pobtaf). Following the assumption
        of a load balancing between the first processes and the others ("middle"), the FLOPS
        count is derived assuming only "middle" processes.

        The factorization process is embarassingly parallel. Only the tip of the arrow
        is updated through an MPI.Allreduce operation at the end of the factorization.

        FLOPS count:
            POTRF_b^3 : (n/p-2) * (1/3 * b^3 + 1/2 * b^2 + 1/6 * b)
            GEMM_b^3 : (n/p-2) * 4 * b^3
            GEMM_b^2_a : (n/p-2) * 4 * b^2 * a
            GEMM_a^2_b : (n/p-2) * 4 * a^2 * b
            TRSM_b^3 : (n/p-2) * 2 * b^3
            TRSM_a_b^2 : (n/p-2) * a * b^2

        Total FLOPS:
            T_{flops_{d_pobtaf}} = (n/p-2) * (19/3 * b^3 + (1/2 + 5*a) * b^2 + (1/6 + 4*a^2) * b)

        Complexity:
            By making the assumption that b >> a, the complexity of the POBTAF
            algorithm is O(n/p * b^3).

    Parameters
    ----------
    A_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of A.
    A_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of A.
    A_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of A.
    A_arrow_tip_block_global : ArrayLike
        Arrow tip block of A.
    device_streaming : bool
        Whether to use streamed GPU computation.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L, alias on A_diagonal_blocks_local.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L, alias on A_lower_diagonal_blocks_local.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L, alias on A_arrow_bottom_blocks_local.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L, alias on A_arrow_tip_block_global.
    L_upper_nested_dissection_buffer_local : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    """

    if (
        CUPY_AVAIL
        and cp.get_array_module(A_diagonal_blocks_local) == np
        and device_streaming
    ):
        return _streaming_d_pobtaf(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block_global,
            comm=comm
        )

    return _d_pobtaf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_tip_block_global,
        comm=comm
    )


def _d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(A_diagonal_blocks_local)
        if xp == cp:
            la = cu_la
            cholesky = cholesky_lowerfill
        else:
            cholesky = np.linalg.cholesky
    else:
        xp = np
        cholesky = np.linalg.cholesky

    n_diag_blocks_local = A_diagonal_blocks_local.shape[0]

    L_diagonal_blocks_local = A_diagonal_blocks_local
    L_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local
    L_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local

    L_upper_nested_dissection_buffer_local = None

    Update_arrow_tip_block = xp.zeros_like(A_arrow_tip_block_global)

    if comm_rank == 0:
        # Forward block-Cholesky, performed by a "top" process
        for i in range(0, n_diag_blocks_local - 1):
            # L_{i, i} = chol(A_{i, i})
            L_diagonal_blocks_local[i, :, :] = cholesky(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks_local[i, :, :] = (
                la.solve_triangular(
                    L_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            L_arrow_bottom_blocks_local[i, :, :] = (
                la.solve_triangular(
                    L_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - L_lower_diagonal_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].conj().T
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].conj().T
            )

            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_arrow_bottom_blocks_local[i, :, :].conj().T
            )
    else:
        A_upper_nested_dissection_buffer_local = xp.empty_like(A_diagonal_blocks_local)
        L_upper_nested_dissection_buffer_local = A_upper_nested_dissection_buffer_local

        A_upper_nested_dissection_buffer_local[1, :, :] = (
            A_lower_diagonal_blocks_local[0, :, :].conj().T
        )

        # Forward block-Cholesky, performed by a "middle" process
        for i in range(1, n_diag_blocks_local - 1):
            # L_{i, i} = chol(A_{i, i})
            L_diagonal_blocks_local[i, :, :] = cholesky(
                A_diagonal_blocks_local[i, :, :]
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks_local[i, :, :] = (
                la.solve_triangular(
                    L_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )

            # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
            L_upper_nested_dissection_buffer_local[i, :, :] = (
                la.solve_triangular(
                    L_diagonal_blocks_local[i, :, :],
                    A_upper_nested_dissection_buffer_local[i, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            L_arrow_bottom_blocks_local[i, :, :] = (
                la.solve_triangular(
                    L_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    lower=True,
                )
                .conj()
                .T
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - L_lower_diagonal_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].conj().T
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].conj().T
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_arrow_bottom_blocks_local[i, :, :].conj().T
            )

            # Update top and next upper/lower blocks of 2-sided factorization pattern
            # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.conj().T
            A_diagonal_blocks_local[0, :, :] = (
                A_diagonal_blocks_local[0, :, :]
                - L_upper_nested_dissection_buffer_local[i, :, :]
                @ L_upper_nested_dissection_buffer_local[i, :, :].conj().T
            )

            # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.conj().T
            A_upper_nested_dissection_buffer_local[i + 1, :, :] = (
                -L_upper_nested_dissection_buffer_local[i, :, :]
                @ L_lower_diagonal_blocks_local[i, :, :].conj().T
            )

            # Update the top (first blocks) of the arrowhead
            # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.conj().T
            A_arrow_bottom_blocks_local[0, :, :] = (
                A_arrow_bottom_blocks_local[0, :, :]
                - L_arrow_bottom_blocks_local[i, :, :]
                @ L_upper_nested_dissection_buffer_local[i, :, :].conj().T
            )

    # Check if operations are happening on the device, in this case we need to get
    # back the tip blocks on the host to perform the accumulation through MPI.
    if CUPY_AVAIL and xp == cp:
        Update_arrow_tip_block_host = cpx.empty_like_pinned(Update_arrow_tip_block)
        Update_arrow_tip_block.get(out=Update_arrow_tip_block_host)
    else:
        Update_arrow_tip_block_host = Update_arrow_tip_block

    # Accumulate the distributed update of the arrow tip block
    comm.Allreduce(
        MPI.IN_PLACE,
        Update_arrow_tip_block_host,
        op=MPI.SUM,
    )

    if CUPY_AVAIL and xp == cp:
        Update_arrow_tip_block.set(arr=Update_arrow_tip_block_host)

    A_arrow_tip_block_global[:, :] += Update_arrow_tip_block[:, :]
    L_arrow_tip_block_global = A_arrow_tip_block_global

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    )


def _streaming_d_pobtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
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

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Host aliases & buffers
    L_diagonal_blocks_local = A_diagonal_blocks_local
    L_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local
    L_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local
    Update_arrow_tip_block_host = cpx.zeros_like_pinned(A_arrow_tip_block_global)

    # Device aliases & buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks_local.shape[1:]), dtype=A_diagonal_blocks_local.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty(
        (2, *A_lower_diagonal_blocks_local.shape[1:]),
        dtype=A_lower_diagonal_blocks_local.dtype,
    )
    A_arrow_bottom_blocks_d = cp.empty(
        (2, *A_arrow_bottom_blocks_local.shape[1:]),
        dtype=A_arrow_bottom_blocks_local.dtype,
    )
    Update_arrow_tip_block_d = cp.zeros_like(A_arrow_tip_block_global)

    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_arrow_bottom_blocks_d = A_arrow_bottom_blocks_d

    n_diag_blocks_local = A_diagonal_blocks_local.shape[0]
    if comm_rank == 0:
        # Host aliases & buffers specific to the top process
        L_upper_nested_dissection_buffer_local = None

        # --- Initial Host 2 Device transfers ---
        # --- H2D: transfers ---
        A_diagonal_blocks_d[0, :, :].set(
            arr=A_diagonal_blocks_local[0, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[0].record(stream=h2d_stream)

        A_arrow_bottom_blocks_d[0, :, :].set(
            arr=A_arrow_bottom_blocks_local[0, :, :], stream=h2d_stream
        )
        h2d_arrow_events[0].record(stream=h2d_stream)

        if n_diag_blocks_local > 1:
            A_lower_diagonal_blocks_d[0, :, :].set(
                arr=A_lower_diagonal_blocks_local[0, :, :], stream=h2d_stream
            )
            h2d_lower_events[0].record(stream=h2d_stream)

        # --- CP: events ---
        cp_lower_events_h2d_release[1].record(stream=compute_stream)
        cp_arrow_events_h2d_release[1].record(stream=compute_stream)

        # --- D2H: event ---
        d2h_diagonal_events[1].record(stream=d2h_stream)

        # Forward block-Cholesky, performed by a "top" process
        for i in range(0, n_diag_blocks_local - 1):
            # --- Computations ---
            # L_{i, i} = chol(A_{i, i})
            with compute_stream:
                compute_stream.wait_event(h2d_diagonal_events[i % 2])
                L_diagonal_blocks_d[i % 2, :, :] = cholesky_lowerfill(
                    A_diagonal_blocks_d[i % 2, :, :]
                )
                cp_diagonal_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_diagonal_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :].get(
                out=L_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )
            d2h_diagonal_events[i % 2].record(stream=d2h_stream)

            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            if i + 1 < n_diag_blocks_local - 1:
                h2d_stream.wait_event(cp_lower_events_h2d_release[(i + 1) % 2])
                A_lower_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                    arr=A_lower_diagonal_blocks_local[i + 1, :, :], stream=h2d_stream
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
                cp_lower_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :].get(
                out=L_lower_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            h2d_stream.wait_event(cp_arrow_events_h2d_release[(i + 1) % 2])
            A_arrow_bottom_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_arrow_bottom_blocks_local[i + 1, :, :], stream=h2d_stream
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
                cp_arrow_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_arrow_events[i % 2])
            L_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=L_arrow_bottom_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            # Update next diagonal block
            h2d_stream.wait_event(d2h_diagonal_events[(i + 1) % 2])
            A_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_diagonal_blocks_local[i + 1, :, :], stream=h2d_stream
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
                cp_lower_events_h2d_release[i % 2].record(stream=compute_stream)

                # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
                Update_arrow_tip_block_d[:, :] = (
                    Update_arrow_tip_block_d[:, :]
                    - L_arrow_bottom_blocks_d[i % 2, :, :]
                    @ L_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                )
                cp_arrow_events_h2d_release[i % 2].record(stream=compute_stream)

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(
            cp_lower_events_h2d_release[(n_diag_blocks_local - 2) % 2]
        )
        A_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].get(
            out=A_diagonal_blocks_local[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        d2h_stream.wait_event(
            cp_arrow_events_h2d_release[(n_diag_blocks_local - 2) % 2]
        )
        A_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].get(
            out=A_arrow_bottom_blocks_local[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )
    else:
        # Host aliases & buffers specific to the middle process
        A_upper_nested_dissection_buffer_local = cpx.empty_like_pinned(
            A_diagonal_blocks_local
        )
        L_upper_nested_dissection_buffer_local = A_upper_nested_dissection_buffer_local

        # Device aliases & buffers specific to the middle process
        A_diagonal_top_block_d = cp.empty_like(A_diagonal_blocks_local[0])
        A_arrow_bottom_top_block_d = cp.empty_like(A_arrow_bottom_blocks_local[0])
        A_upper_nested_dissection_buffer_d = cp.empty(
            (2, *A_upper_nested_dissection_buffer_local.shape[1:]),
            dtype=A_upper_nested_dissection_buffer_local.dtype,
        )

        L_upper_nested_dissection_buffer_d = A_upper_nested_dissection_buffer_d

        cp_upper_nested_dissection_buffer_events = [cp.cuda.Event(), cp.cuda.Event()]

        # Forward block-Cholesky, performed by a "middle" process
        # --- Initial Host 2 Device transfers ---
        # --- H2D: transfers ---
        A_diagonal_blocks_d[1, :, :].set(
            arr=A_diagonal_blocks_local[1, :, :], stream=h2d_stream
        )
        A_diagonal_top_block_d[:, :].set(
            arr=A_diagonal_blocks_local[0, :, :], stream=h2d_stream
        )
        A_upper_nested_dissection_buffer_d[1, :, :].set(
            arr=A_lower_diagonal_blocks_local[0, :, :].conj().T, stream=h2d_stream
        )
        h2d_diagonal_events[1].record(stream=h2d_stream)

        if n_diag_blocks_local > 2:
            A_lower_diagonal_blocks_d[1, :, :].set(
                arr=A_lower_diagonal_blocks_local[1, :, :], stream=h2d_stream
            )
            h2d_lower_events[1].record(stream=h2d_stream)

        A_arrow_bottom_blocks_d[1, :, :].set(
            arr=A_arrow_bottom_blocks_local[1, :, :], stream=h2d_stream
        )
        A_arrow_bottom_top_block_d[:, :].set(
            arr=A_arrow_bottom_blocks_local[0, :, :], stream=h2d_stream
        )
        h2d_arrow_events[1].record(stream=h2d_stream)

        # --- CP: events ---
        cp_lower_events_h2d_release[1].record(stream=compute_stream)
        cp_arrow_events_h2d_release[1].record(stream=compute_stream)

        # --- D2H: event ---
        d2h_diagonal_events[1].record(stream=d2h_stream)

        for i in range(1, n_diag_blocks_local - 1):

            # L_{i, i} = chol(A_{i, i})
            with compute_stream:
                compute_stream.wait_event(h2d_diagonal_events[i % 2])
                L_diagonal_blocks_d[i % 2, :, :] = cholesky_lowerfill(
                    A_diagonal_blocks_d[i % 2, :, :]
                )
                cp_diagonal_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_diagonal_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :].get(
                out=L_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )
            d2h_diagonal_events[i % 2].record(stream=d2h_stream)

            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            if i + 1 < n_diag_blocks_local - 1:
                h2d_stream.wait_event(cp_lower_events_h2d_release[(i + 1) % 2])
                A_lower_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                    arr=A_lower_diagonal_blocks_local[i + 1, :, :], stream=h2d_stream
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
                cp_lower_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :].get(
                out=L_lower_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            h2d_stream.wait_event(cp_arrow_events_h2d_release[(i + 1) % 2])
            A_arrow_bottom_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_arrow_bottom_blocks_local[i + 1, :, :], stream=h2d_stream
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
                cp_arrow_events[i % 2].record(stream=compute_stream)

            d2h_stream.wait_event(cp_arrow_events[i % 2])
            L_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=L_arrow_bottom_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
            with compute_stream:
                L_upper_nested_dissection_buffer_d[i % 2, :, :] = (
                    cu_la.solve_triangular(
                        L_diagonal_blocks_d[i % 2, :, :],
                        A_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T,
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
                out=L_upper_nested_dissection_buffer_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            # Update next diagonal block
            h2d_stream.wait_event(d2h_diagonal_events[(i + 1) % 2])
            A_diagonal_blocks_d[(i + 1) % 2, :, :].set(
                arr=A_diagonal_blocks_local[i + 1, :, :], stream=h2d_stream
            )
            h2d_diagonal_events[(i + 1) % 2].record(stream=h2d_stream)

            with compute_stream:
                # Update next diagonal block
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

                # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.conj().T
                A_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :] = (
                    -L_upper_nested_dissection_buffer_d[i % 2, :, :]
                    @ L_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                )
                cp_lower_events_h2d_release[i % 2].record(stream=compute_stream)

                # Update the block at the tip of the arrowhead
                # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
                Update_arrow_tip_block_d[:, :] = (
                    Update_arrow_tip_block_d[:, :]
                    - L_arrow_bottom_blocks_d[i % 2, :, :]
                    @ L_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                )

                # Update the top (first blocks) of the arrowhead
                # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.conj().T
                A_arrow_bottom_top_block_d[:, :] = (
                    A_arrow_bottom_top_block_d[:, :]
                    - L_arrow_bottom_blocks_d[i % 2, :, :]
                    @ L_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T
                )
                cp_arrow_events_h2d_release[i % 2].record(stream=compute_stream)

                # Update top and next upper/lower blocks of 2-sided factorization pattern
                # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.conj().T
                A_diagonal_top_block_d[:, :] = (
                    A_diagonal_top_block_d[:, :]
                    - L_upper_nested_dissection_buffer_d[i % 2, :, :]
                    @ L_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T
                )

        # --- Device 2 Host transfers ---
        d2h_stream.wait_event(
            cp_lower_events_h2d_release[(n_diag_blocks_local - 2) % 2]
        )
        A_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].get(
            out=A_diagonal_blocks_local[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        A_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].get(
            out=A_arrow_bottom_blocks_local[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        A_upper_nested_dissection_buffer_d[(n_diag_blocks_local - 1) % 2, :, :].get(
            out=A_upper_nested_dissection_buffer_local[-1, :, :],
            stream=d2h_stream,
            blocking=False,
        )

        d2h_stream.wait_event(
            cp_arrow_events_h2d_release[(n_diag_blocks_local - 2) % 2]
        )
        A_arrow_bottom_top_block_d.get(
            out=A_arrow_bottom_blocks_local[0, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        A_diagonal_top_block_d.get(
            out=A_diagonal_blocks_local[0, :, :],
            stream=compute_stream,
            blocking=False,
        )

    Update_arrow_tip_block_d.get(
        out=Update_arrow_tip_block_host,
        stream=d2h_stream,
        blocking=False,
    )

    cp.cuda.Device().synchronize()

    # Accumulate the distributed update of the arrow tip block
    comm.Allreduce(
        MPI.IN_PLACE,
        Update_arrow_tip_block_host,
        op=MPI.SUM,
    )

    A_arrow_tip_block_global[:, :] += Update_arrow_tip_block_host[:, :]
    L_arrow_tip_block_global = A_arrow_tip_block_global

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    )
