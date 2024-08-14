# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike
from mpi4py import MPI

# comm_rank = MPI.COMM_WORLD.Get_rank()
# comm_size = MPI.COMM_WORLD.Get_size()

from serinv.algs import pobtaf, pobtasi, d_pobtaf

DOUBLE_COMPLEX = MPI.C_DOUBLE_COMPLEX if MPI.Get_library_version().startswith('Open MPI') else MPI.DOUBLE_COMPLEX
mpi_datatype = {np.float64: MPI.DOUBLE, np.complex128: DOUBLE_COMPLEX}
if CUPY_AVAIL:
    mpi_datatype[cp.float64] = MPI.DOUBLE
    mpi_datatype[cp.complex128] = DOUBLE_COMPLEX


def d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike = None,
    device_streaming: bool = False,
    nested_solving: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform a distributed selected-inversion of a block tridiagonal with
    arrowhead matrix.

    Note:
    -----
    - Will overwrite the inputs and store the results in them (in-place). Returns
        are aliases on the inputs.
    - If a device array is given, the algorithm will run on the GPU.

    Complexity analysis:
        Parameters:
            n : number of diagonal blocks
            b : diagonal block size
            a : arrow block size
            p : number of processes
            n_r = (2*p-1): number of diagonal blocks of the reduced system

        The FLOPS count of the top process is the same as the one of the sequential
        block selected inversion algorithm (pobtasi). Following the assumption
        of a load balancing between the first processes and the others ("middle"), the FLOPS
        count is derived assuming only "middle" processes.

        The selected inversion procedure require the inversion of a reduced system
        made of boundary blocks of the distributed factorization of the matrix. The
        reduced system is constructed by gathering the boundary blocks of the factorization
        on each processes and performing a selected inversion of the reduced system.
        This selected inversion is assumed to be performed using the sequential
        block-Cholesky and selected inversion algorithm (pobtaf + pobtasi).

        Inversion of the reduced system:

        FLOPS count:
            Selected inversion of the reduced system:
                T_{flops_{POBTAF}} = n_r * (10/3 * b^3 + (1/2 + 3*a) * b^2 + (1/6 + 2*a^2) * b) - 3 * b^3 - 2*a*b^2 + 1/3 * a^3 + 1/2 * a^2 + 1/6 * a
                T_{flops_{POBTASI}} = (n_r-1) * (9*b^3 + 10*b^2 + 2*a^2*b) + 2*b^3 + 5*a*b^2 + 2*a^2*b + 2*a^3

            Distributed selected inversion:
                GEMM_b^3 : (n/p-1) * 18 * b^3
                GEMM_b^2_a : (n/p-1) * 14 * b^2 * a
                TRSM_b^3 : (n/p-1) * b^3

                Total FLOPS: (n/p-1) * (19*b^3 + 14*a*b^2)

            T_{flops_{d\_pobtasi}} = (n/p-1) * (19*b^3 + 14*a*b^2) + T_{flops_{POBTAF}} + T_{flops_{POBTASI}}

        Complexity:
            By making the assumption that b >> a, the complexity of the d_pobtasi
            algorithm is O(n/p * b^3 + p * b^3) or O((n+p^2)/p * b^3).

    Parameters
    ----------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L.
    L_upper_nested_dissection_buffer_local : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    device_streaming : bool
        Whether to use streamed GPU computation.
    nested_solving : bool
        If true, recursively calls d_pobtasi to solve the reduced system.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    X_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of X, alias on L_diagonal_blocks_local.
    X_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of X, alias on L_lower_diagonal_blocks_local.
    X_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of X, alias on L_arrow_bottom_blocks_local.
    X_arrow_tip_block_global : ArrayLike
        Arrow tip block of X, alias on L_arrow_tip_block_global.
    """

    if (
        CUPY_AVAIL
        and cp.get_array_module(L_diagonal_blocks_local) == np
        and device_streaming
    ):
        return _streaming_d_pobtasi(
            L_diagonal_blocks_local,
            L_lower_diagonal_blocks_local,
            L_arrow_bottom_blocks_local,
            L_arrow_tip_block_global,
            L_upper_nested_dissection_buffer_local,
            nested_solving=nested_solving,
            comm=comm
        )

    return _d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        nested_solving=nested_solving,
        comm=comm
    )


def _d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike,
    nested_solving: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    la = np_la
    if CUPY_AVAIL:
        xp = cp.get_array_module(L_diagonal_blocks_local)
        if xp == cp:
            la = cu_la
    else:
        xp = np

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    arrow_size = L_arrow_tip_block_global.shape[0]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_upper_nested_dissection_buffer_local = L_upper_nested_dissection_buffer_local

    A_reduced_system_diagonal_blocks = xp.zeros(
        (2 * comm_size - 1, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = xp.zeros(
        (2 * comm_size - 2, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = xp.zeros(
        (2 * comm_size - 1, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = L_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        A_reduced_system_diagonal_blocks[0, :, :] = L_diagonal_blocks_local[-1, :, :]
        A_reduced_system_lower_diagonal_blocks[0, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]
        A_reduced_system_arrow_bottom_blocks[0, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        A_reduced_system_diagonal_blocks[2 * comm_rank - 1, :, :] = (
            L_diagonal_blocks_local[0, :, :]
        )
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            -1, :, :
        ]

        A_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :] = (
            L_upper_nested_dissection_buffer_local[-1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
                L_lower_diagonal_blocks_local[-1, :, :]
            )

        A_reduced_system_arrow_bottom_blocks[2 * comm_rank - 1, :, :] = (
            L_arrow_bottom_blocks_local[0, :, :]
        )
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
            L_arrow_bottom_blocks_local[-1, :, :]
        )

    if CUPY_AVAIL and xp == cp:
        A_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_diagonal_blocks
        )
        A_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_arrow_bottom_blocks
        )

        A_reduced_system_diagonal_blocks.get(out=A_reduced_system_diagonal_blocks_host)
        A_reduced_system_lower_diagonal_blocks.get(
            out=A_reduced_system_lower_diagonal_blocks_host
        )
        A_reduced_system_arrow_bottom_blocks.get(
            out=A_reduced_system_arrow_bottom_blocks_host
        )
    else:
        A_reduced_system_diagonal_blocks_host = A_reduced_system_diagonal_blocks
        A_reduced_system_lower_diagonal_blocks_host = (
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = A_reduced_system_arrow_bottom_blocks

    comm.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks_host,
        op=MPI.SUM,
    )
    comm.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks_host,
        op=MPI.SUM,
    )
    comm.Allreduce(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks_host,
        op=MPI.SUM,
    )

    if CUPY_AVAIL and xp == cp:
        A_reduced_system_diagonal_blocks.set(arr=A_reduced_system_diagonal_blocks_host)
        A_reduced_system_lower_diagonal_blocks.set(
            arr=A_reduced_system_lower_diagonal_blocks_host
        )
        A_reduced_system_arrow_bottom_blocks.set(
            arr=A_reduced_system_arrow_bottom_blocks_host
        )

    # Perform the inversion of the reduced system.
    n_diag_blocks = 2 * comm_size - 1
    reduced_rank = comm_rank
    reduced_size = comm_size // 2
    # TODO: Better load balancing
    n_diag_blocks_per_processes = n_diag_blocks // reduced_size
    if nested_solving and reduced_size > 1:
        # Extract the arrays' local slices for each MPI process
        reduced_color = int(comm_rank < reduced_size)
        reduced_key = comm_rank
        reduced_comm = comm.Split(color=reduced_color, key=reduced_key)

        # for rank in range(comm_size):
        #     if rank == comm_rank:
        #         print(f"Rank {comm_rank} reduced_color: {reduced_color}, reduced_key: {reduced_key}", flush=True)
        #     comm.Barrier()

        X_reduced_system_diagonal_blocks = xp.empty((n_diag_blocks, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_lower_diagonal_blocks = xp.empty((n_diag_blocks - 1, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_bottom_blocks = xp.empty((n_diag_blocks, arrow_size, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_tip_block = xp.empty((arrow_size, arrow_size), dtype=L_diagonal_blocks_local.dtype)

        if reduced_color == 1:
            
            # NOTE: Making copies for validation
            if reduced_rank == reduced_size - 1:
                A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[
                    reduced_rank * n_diag_blocks_per_processes :,
                    :,
                    :,
                ]
                A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[
                    reduced_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
                    :,
                    :,
                ]
                A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[
                    reduced_rank * n_diag_blocks_per_processes :,
                    :,
                    :,
                ]
            else:
                A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[
                    reduced_rank
                    * n_diag_blocks_per_processes : (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
                A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[
                    reduced_rank
                    * n_diag_blocks_per_processes : (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
                A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[
                    reduced_rank
                    * n_diag_blocks_per_processes : (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
            A_reduced_system_arrow_tip_block_global = A_reduced_system_arrow_tip_block

            # # NOTE: Making copies for validation
            # if reduced_rank == reduced_size - 1:
            #     A_reduced_system_diagonal_blocks_local = xp.empty((n_diag_blocks - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_diagonal_blocks_local[:] = A_reduced_system_diagonal_blocks[
            #         reduced_rank * n_diag_blocks_per_processes :,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_lower_diagonal_blocks_local = xp.empty((n_diag_blocks - 1 - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_lower_diagonal_blocks_local[:] = A_reduced_system_lower_diagonal_blocks[
            #         reduced_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_arrow_bottom_blocks_local = xp.empty((n_diag_blocks - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, arrow_size), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_arrow_bottom_blocks_local[:] = A_reduced_system_arrow_bottom_blocks[
            #         reduced_rank * n_diag_blocks_per_processes :,
            #         :,
            #         :,
            #     ]
            # else:
            #     A_reduced_system_diagonal_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_diagonal_blocks_local[:] = A_reduced_system_diagonal_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_lower_diagonal_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_lower_diagonal_blocks_local[:] = A_reduced_system_lower_diagonal_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_arrow_bottom_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, arrow_size), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_arrow_bottom_blocks_local[:] = A_reduced_system_arrow_bottom_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            # A_reduced_system_arrow_tip_block_global = xp.copy(A_reduced_system_arrow_tip_block)

            (
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
            ) = d_pobtaf(
                A_reduced_system_diagonal_blocks_local,
                A_reduced_system_lower_diagonal_blocks_local,
                A_reduced_system_arrow_bottom_blocks_local,
                A_reduced_system_arrow_tip_block_global,
                device_streaming=True if CUPY_AVAIL and xp == cp else False,
                comm=reduced_comm
            )
            (
                X_reduced_system_diagonal_blocks_local,
                X_reduced_system_lower_diagonal_blocks_local,
                X_reduced_system_arrow_bottom_blocks_local,
                X_reduced_system_arrow_tip_block_tmp
            ) = d_pobtasi(
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
                device_streaming=True if CUPY_AVAIL and xp == cp else False,
                nested_solving=False,
                comm=reduced_comm
            )

            # # NOTE: Validation
            # (
            #     L_reduced_system_diagonal_blocks,
            #     L_reduced_system_lower_diagonal_blocks,
            #     L_reduced_system_arrow_bottom_blocks,
            #     L_reduced_system_arrow_tip_block,
            # ) = pobtaf(
            #     A_reduced_system_diagonal_blocks,
            #     A_reduced_system_lower_diagonal_blocks,
            #     A_reduced_system_arrow_bottom_blocks,
            #     A_reduced_system_arrow_tip_block,
            #     device_streaming=True if CUPY_AVAIL and xp == cp else False,
            # )

            # (
            #     X_reduced_system_diagonal_blocks_ref,
            #     X_reduced_system_lower_diagonal_blocks_ref,
            #     X_reduced_system_arrow_bottom_blocks_ref,
            #     X_reduced_system_arrow_tip_block_ref,
            # ) = pobtasi(
            #     L_reduced_system_diagonal_blocks,
            #     L_reduced_system_lower_diagonal_blocks,
            #     L_reduced_system_arrow_bottom_blocks,
            #     L_reduced_system_arrow_tip_block,
            #     device_streaming=True if CUPY_AVAIL and xp == cp else False,
            # )

            # assert xp.allclose(X_reduced_system_arrow_tip_block_ref, X_reduced_system_arrow_tip_block_tmp)
            # if reduced_rank == reduced_size - 1:
            #     assert xp.allclose(X_reduced_system_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_lower_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_lower_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_arrow_bottom_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_arrow_bottom_blocks_local)
            # else:
            #     assert xp.allclose(X_reduced_system_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_lower_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_lower_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_arrow_bottom_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_arrow_bottom_blocks_local)

            # NOTE: For some reason, the returned X_reduced_system_arrow_tip_block is not contiguous in memory.
            X_reduced_system_arrow_tip_block[:] = X_reduced_system_arrow_tip_block_tmp

            # # TODO: Gather the results to the X_reduced_system buffers. Is this needed or can we just use the local buffers?
            # # NOTE: We gather naively for now; optimize later.
            # if reduced_rank == 0:
            #     X_reduced_system_diagonal_blocks[:n_diag_blocks_per_processes] = X_reduced_system_diagonal_blocks_local
            #     X_reduced_system_lower_diagonal_blocks[:n_diag_blocks_per_processes] = X_reduced_system_lower_diagonal_blocks_local
            #     X_reduced_system_arrow_bottom_blocks[:n_diag_blocks_per_processes] = X_reduced_system_arrow_bottom_blocks_local

            #     for rank in range(1, reduced_size - 1):
            #         comm.Recv(X_reduced_system_diagonal_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=0)
            #         comm.Recv(X_reduced_system_lower_diagonal_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=1)
            #         comm.Recv(X_reduced_system_arrow_bottom_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=2)
            #     rank = reduced_size - 1
            #     comm.Recv(X_reduced_system_diagonal_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=0)
            #     comm.Recv(X_reduced_system_lower_diagonal_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=1)
            #     comm.Recv(X_reduced_system_arrow_bottom_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=2)
            # else:
            #     comm.Send(X_reduced_system_diagonal_blocks_local, dest=0, tag=0)
            #     comm.Send(X_reduced_system_lower_diagonal_blocks_local, dest=0, tag=1)
            #     comm.Send(X_reduced_system_arrow_bottom_blocks_local, dest=0, tag=2)
        else:
            X_reduced_system_diagonal_blocks_local = None
            X_reduced_system_lower_diagonal_blocks_local = None
            X_reduced_system_arrow_bottom_blocks_local = None

        # comm.Bcast(X_reduced_system_diagonal_blocks, root=0)
        # comm.Bcast(X_reduced_system_lower_diagonal_blocks, root=0)
        # comm.Bcast(X_reduced_system_arrow_bottom_blocks, root=0)
        # comm.Bcast(X_reduced_system_arrow_tip_block, root=0)

        diag_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        lower_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        arrow_count = n_diag_blocks_per_processes * diag_blocksize * arrow_size
        diag_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        lower_count_last = (n_diag_blocks - 1 - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        arrow_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * arrow_size

        if reduced_color == 0:
            send_diag_count = 0
            send_lower_count = 0
            send_arrow_count = 0
        else:
            if reduced_rank != reduced_size - 1:
                send_diag_count = diag_count
                send_lower_count = lower_count
                send_arrow_count = arrow_count
            else:
                send_diag_count = diag_count_last
                send_lower_count = lower_count_last
                send_arrow_count = arrow_count_last
        
        recv_diag_counts = [diag_count] * (reduced_size - 1) + [diag_count_last] + [0] * (comm_size - reduced_size)
        recv_lower_counts = [lower_count] * (reduced_size - 1) + [lower_count_last] + [0] * (comm_size - reduced_size)
        recv_arrow_counts = [arrow_count] * (reduced_size - 1) + [arrow_count_last] + [0] * (comm_size - reduced_size)
        diag_displ = list(range(0, diag_count * reduced_size, diag_count)) + [0] * (comm_size - reduced_size)
        lower_displ = list(range(0, lower_count * reduced_size, lower_count)) + [0] * (comm_size - reduced_size)
        arrow_displ = list(range(0, arrow_count * reduced_size, arrow_count)) + [0] * (comm_size - reduced_size)

        # if comm_rank == 0:
        #     print(f"{CUPY_AVAIL and xp == cp}", flush=True)
        # comm.Barrier()

        if CUPY_AVAIL and xp == cp:

            if reduced_color == 1:
                X_reduced_system_diagonal_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_diagonal_blocks_local)
                X_reduced_system_lower_diagonal_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_lower_diagonal_blocks_local)
                X_reduced_system_arrow_bottom_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_arrow_bottom_blocks_local)
                X_reduced_system_diagonal_blocks_local.get(out=X_reduced_system_diagonal_blocks_local_host)
                X_reduced_system_lower_diagonal_blocks_local.get(out=X_reduced_system_lower_diagonal_blocks_local_host)
                X_reduced_system_arrow_bottom_blocks_local.get(out=X_reduced_system_arrow_bottom_blocks_local_host)
            else:
                X_reduced_system_diagonal_blocks_local_host = None
                X_reduced_system_lower_diagonal_blocks_local_host = None
                X_reduced_system_arrow_bottom_blocks_local_host = None
            X_reduced_system_arrow_tip_block_host = cpx.empty_like_pinned(X_reduced_system_arrow_tip_block)
            if comm_rank == 0:
                X_reduced_system_arrow_tip_block.get(out=X_reduced_system_arrow_tip_block_host)

            X_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(X_reduced_system_diagonal_blocks)
            X_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(X_reduced_system_lower_diagonal_blocks)
            X_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(X_reduced_system_arrow_bottom_blocks)
        else:
            X_reduced_system_diagonal_blocks_local_host = X_reduced_system_diagonal_blocks_local
            X_reduced_system_lower_diagonal_blocks_local_host = X_reduced_system_lower_diagonal_blocks_local
            X_reduced_system_arrow_bottom_blocks_local_host = X_reduced_system_arrow_bottom_blocks_local
            X_reduced_system_diagonal_blocks_host = X_reduced_system_diagonal_blocks
            X_reduced_system_lower_diagonal_blocks_host = X_reduced_system_lower_diagonal_blocks
            X_reduced_system_arrow_bottom_blocks_host = X_reduced_system_arrow_bottom_blocks
            X_reduced_system_arrow_tip_block_host = X_reduced_system_arrow_tip_block

        mpi_dtype = mpi_datatype[L_diagonal_blocks_local.dtype.type]
        comm.Allgatherv([X_reduced_system_diagonal_blocks_local_host, send_diag_count, mpi_dtype],
                        [X_reduced_system_diagonal_blocks_host, recv_diag_counts, diag_displ, mpi_dtype])
        comm.Allgatherv([X_reduced_system_lower_diagonal_blocks_local_host, send_lower_count, mpi_dtype],
                        [X_reduced_system_lower_diagonal_blocks_host, recv_lower_counts, lower_displ, mpi_dtype])
        comm.Allgatherv([X_reduced_system_arrow_bottom_blocks_local_host, send_arrow_count, mpi_dtype],
                        [X_reduced_system_arrow_bottom_blocks_host, recv_arrow_counts, arrow_displ, mpi_dtype])
        comm.Bcast(X_reduced_system_arrow_tip_block_host, root=0)

        if CUPY_AVAIL and xp == cp:
            X_reduced_system_diagonal_blocks.set(arr=X_reduced_system_diagonal_blocks_host)
            X_reduced_system_lower_diagonal_blocks.set(arr=X_reduced_system_lower_diagonal_blocks_host)
            X_reduced_system_arrow_bottom_blocks.set(arr=X_reduced_system_arrow_bottom_blocks_host)
            X_reduced_system_arrow_tip_block.set(arr=X_reduced_system_arrow_tip_block_host)
                
    else:
        (
            L_reduced_system_diagonal_blocks,
            L_reduced_system_lower_diagonal_blocks,
            L_reduced_system_arrow_bottom_blocks,
            L_reduced_system_arrow_tip_block,
        ) = pobtaf(
            A_reduced_system_diagonal_blocks,
            A_reduced_system_lower_diagonal_blocks,
            A_reduced_system_arrow_bottom_blocks,
            A_reduced_system_arrow_tip_block,
            device_streaming=True if CUPY_AVAIL and xp == cp else False,
        )

        (
            X_reduced_system_diagonal_blocks,
            X_reduced_system_lower_diagonal_blocks,
            X_reduced_system_arrow_bottom_blocks,
            X_reduced_system_arrow_tip_block,
        ) = pobtasi(
            L_reduced_system_diagonal_blocks,
            L_reduced_system_lower_diagonal_blocks,
            L_reduced_system_arrow_bottom_blocks,
            L_reduced_system_arrow_tip_block,
            device_streaming=True if CUPY_AVAIL and xp == cp else False,
        )

    X_arrow_tip_block_global = X_reduced_system_arrow_tip_block

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        X_lower_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[0, :, :]
        )
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        X_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        X_upper_nested_dissection_buffer_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            X_lower_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]
            )

        X_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

    # Backward selected-inversion
    L_inv_temp = xp.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_temp = xp.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_temp = xp.empty_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]

            # Compute lower factors
            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            # --- Lower-diagonal blocks ---
            # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_tip_block_global[:, :] @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # --- Diagonal block part ---
            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]
    else:
        L_upper_nested_dissection_buffer_temp = xp.empty_like(
            L_upper_nested_dissection_buffer_local[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]
            L_upper_nested_dissection_buffer_temp[:, :] = (
                L_upper_nested_dissection_buffer_local[i, :, :]
            )

            L_inv_temp[:, :] = la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                xp.eye(diag_blocksize),
                lower=True,
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -X_upper_nested_dissection_buffer_local[i + 1, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_upper_nested_dissection_buffer_local[i, :, :] = (
                -X_upper_nested_dissection_buffer_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_diagonal_blocks_local[0, :, :]
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # Arrowhead
            # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
            X_arrow_bottom_blocks_local[i, :, :] = (
                -X_arrow_bottom_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[0, :, :]
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_tip_block_global[:, :] @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_upper_nested_dissection_buffer_local[i, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = (
            X_upper_nested_dissection_buffer_local[1, :, :].conj().T
        )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )


def _streaming_d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    L_upper_nested_dissection_buffer_local: ArrayLike,
    nested_solving: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    arrow_size = L_arrow_tip_block_global.shape[0]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_upper_nested_dissection_buffer_local = L_upper_nested_dissection_buffer_local

    A_reduced_system_diagonal_blocks = cpx.zeros_pinned(
        (2 * comm_size, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = cpx.zeros_pinned(
        (2 * comm_size, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = cpx.zeros_pinned(
        (2 * comm_size, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = L_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        A_reduced_system_diagonal_blocks[1, :, :] = L_diagonal_blocks_local[-1, :, :]
        A_reduced_system_lower_diagonal_blocks[1, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]
        A_reduced_system_arrow_bottom_blocks[1, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            0, :, :
        ]
        A_reduced_system_diagonal_blocks[2 * comm_rank + 1, :, :] = (
            L_diagonal_blocks_local[-1, :, :]
        )

        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            L_upper_nested_dissection_buffer_local[-1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            A_reduced_system_lower_diagonal_blocks[2 * comm_rank + 1, :, :] = (
                L_lower_diagonal_blocks_local[-1, :, :]
            )

        A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
            L_arrow_bottom_blocks_local[0, :, :]
        )
        A_reduced_system_arrow_bottom_blocks[2 * comm_rank + 1, :, :] = (
            L_arrow_bottom_blocks_local[-1, :, :]
        )

    comm.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_diagonal_blocks,
    )

    comm.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_lower_diagonal_blocks,
    )

    comm.Allgather(
        MPI.IN_PLACE,
        A_reduced_system_arrow_bottom_blocks,
    )

    # Perform the inversion of the reduced system.

    # Perform the inversion of the reduced system.
    n_diag_blocks = 2 * comm_size - 1
    reduced_rank = comm_rank
    reduced_size = comm_size // 2
    # TODO: Better load balancing
    n_diag_blocks_per_processes = n_diag_blocks // reduced_size
    if nested_solving and reduced_size > 1:
        # Extract the arrays' local slices for each MPI process
        reduced_color = int(comm_rank < reduced_size)
        reduced_key = comm_rank
        reduced_comm = comm.Split(color=reduced_color, key=reduced_key)

        # for rank in range(comm_size):
        #     if rank == comm_rank:
        #         print(f"Rank {comm_rank} reduced_color: {reduced_color}, reduced_key: {reduced_key}", flush=True)
        #     comm.Barrier()

        X_reduced_system_diagonal_blocks = cpx.empty_pinned((n_diag_blocks, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_lower_diagonal_blocks = cpx.empty_pinned((n_diag_blocks - 1, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_bottom_blocks = cpx.empty_pinned((n_diag_blocks, arrow_size, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_tip_block = cpx.empty_pinned((arrow_size, arrow_size), dtype=L_diagonal_blocks_local.dtype)

        if reduced_color == 1:
            
            # NOTE: Making copies for validation
            if reduced_rank == reduced_size - 1:
                A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[
                    1 + reduced_rank * n_diag_blocks_per_processes :,
                    :,
                    :,
                ]
                A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[
                    1 + reduced_rank * n_diag_blocks_per_processes : n_diag_blocks,
                    :,
                    :,
                ]
                A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[
                    1 + reduced_rank * n_diag_blocks_per_processes :,
                    :,
                    :,
                ]
            else:
                A_reduced_system_diagonal_blocks_local = A_reduced_system_diagonal_blocks[
                    1 + reduced_rank
                    * n_diag_blocks_per_processes : 1 + (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
                A_reduced_system_lower_diagonal_blocks_local = A_reduced_system_lower_diagonal_blocks[
                    1 + reduced_rank
                    * n_diag_blocks_per_processes : 1 + (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
                A_reduced_system_arrow_bottom_blocks_local = A_reduced_system_arrow_bottom_blocks[
                    1 + reduced_rank
                    * n_diag_blocks_per_processes : 1 + (reduced_rank + 1)
                    * n_diag_blocks_per_processes,
                    :,
                    :,
                ]
            A_reduced_system_arrow_tip_block_global = A_reduced_system_arrow_tip_block

            # # NOTE: Making copies for validation
            # if reduced_rank == reduced_size - 1:
            #     A_reduced_system_diagonal_blocks_local = xp.empty((n_diag_blocks - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_diagonal_blocks_local[:] = A_reduced_system_diagonal_blocks[
            #         reduced_rank * n_diag_blocks_per_processes :,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_lower_diagonal_blocks_local = xp.empty((n_diag_blocks - 1 - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_lower_diagonal_blocks_local[:] = A_reduced_system_lower_diagonal_blocks[
            #         reduced_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_arrow_bottom_blocks_local = xp.empty((n_diag_blocks - reduced_rank * n_diag_blocks_per_processes, diag_blocksize, arrow_size), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_arrow_bottom_blocks_local[:] = A_reduced_system_arrow_bottom_blocks[
            #         reduced_rank * n_diag_blocks_per_processes :,
            #         :,
            #         :,
            #     ]
            # else:
            #     A_reduced_system_diagonal_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_diagonal_blocks_local[:] = A_reduced_system_diagonal_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_lower_diagonal_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_lower_diagonal_blocks_local[:] = A_reduced_system_lower_diagonal_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            #     A_reduced_system_arrow_bottom_blocks_local = xp.empty((n_diag_blocks_per_processes, diag_blocksize, arrow_size), dtype=L_diagonal_blocks_local.dtype)
            #     A_reduced_system_arrow_bottom_blocks_local[:] = A_reduced_system_arrow_bottom_blocks[
            #         reduced_rank
            #         * n_diag_blocks_per_processes : (reduced_rank + 1)
            #         * n_diag_blocks_per_processes,
            #         :,
            #         :,
            #     ]
            # A_reduced_system_arrow_tip_block_global = xp.copy(A_reduced_system_arrow_tip_block)

            (
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
            ) = d_pobtaf(
                A_reduced_system_diagonal_blocks_local,
                A_reduced_system_lower_diagonal_blocks_local,
                A_reduced_system_arrow_bottom_blocks_local,
                A_reduced_system_arrow_tip_block_global,
                # A_reduced_system_diagonal_blocks[1:, :, :],
                # A_reduced_system_lower_diagonal_blocks[1:-1, :, :],
                # A_reduced_system_arrow_bottom_blocks[1:, :, :],
                # A_reduced_system_arrow_tip_block,
                device_streaming=True,
                comm=reduced_comm
            )
            (
                X_reduced_system_diagonal_blocks_local,
                X_reduced_system_lower_diagonal_blocks_local,
                X_reduced_system_arrow_bottom_blocks_local,
                X_reduced_system_arrow_tip_block_tmp
            ) = d_pobtasi(
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
                device_streaming=True,
                nested_solving=False,
                comm=reduced_comm
            )

            # # NOTE: Validation
            # (
            #     L_reduced_system_diagonal_blocks,
            #     L_reduced_system_lower_diagonal_blocks,
            #     L_reduced_system_arrow_bottom_blocks,
            #     L_reduced_system_arrow_tip_block,
            # ) = pobtaf(
            #     A_reduced_system_diagonal_blocks,
            #     A_reduced_system_lower_diagonal_blocks,
            #     A_reduced_system_arrow_bottom_blocks,
            #     A_reduced_system_arrow_tip_block,
            #     device_streaming=True if CUPY_AVAIL and xp == cp else False,
            # )

            # (
            #     X_reduced_system_diagonal_blocks_ref,
            #     X_reduced_system_lower_diagonal_blocks_ref,
            #     X_reduced_system_arrow_bottom_blocks_ref,
            #     X_reduced_system_arrow_tip_block_ref,
            # ) = pobtasi(
            #     L_reduced_system_diagonal_blocks,
            #     L_reduced_system_lower_diagonal_blocks,
            #     L_reduced_system_arrow_bottom_blocks,
            #     L_reduced_system_arrow_tip_block,
            #     device_streaming=True if CUPY_AVAIL and xp == cp else False,
            # )

            # assert xp.allclose(X_reduced_system_arrow_tip_block_ref, X_reduced_system_arrow_tip_block_tmp)
            # if reduced_rank == reduced_size - 1:
            #     assert xp.allclose(X_reduced_system_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_lower_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_lower_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_arrow_bottom_blocks_ref[reduced_rank * n_diag_blocks_per_processes:], X_reduced_system_arrow_bottom_blocks_local)
            # else:
            #     assert xp.allclose(X_reduced_system_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_lower_diagonal_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_lower_diagonal_blocks_local)
            #     assert xp.allclose(X_reduced_system_arrow_bottom_blocks_ref[reduced_rank * n_diag_blocks_per_processes:(reduced_rank + 1) * n_diag_blocks_per_processes], X_reduced_system_arrow_bottom_blocks_local)

            # NOTE: For some reason, the returned X_reduced_system_arrow_tip_block is not contiguous in memory.
            X_reduced_system_arrow_tip_block[:] = X_reduced_system_arrow_tip_block_tmp

            # # TODO: Gather the results to the X_reduced_system buffers. Is this needed or can we just use the local buffers?
            # # NOTE: We gather naively for now; optimize later.
            # if reduced_rank == 0:
            #     X_reduced_system_diagonal_blocks[:n_diag_blocks_per_processes] = X_reduced_system_diagonal_blocks_local
            #     X_reduced_system_lower_diagonal_blocks[:n_diag_blocks_per_processes] = X_reduced_system_lower_diagonal_blocks_local
            #     X_reduced_system_arrow_bottom_blocks[:n_diag_blocks_per_processes] = X_reduced_system_arrow_bottom_blocks_local

            #     for rank in range(1, reduced_size - 1):
            #         comm.Recv(X_reduced_system_diagonal_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=0)
            #         comm.Recv(X_reduced_system_lower_diagonal_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=1)
            #         comm.Recv(X_reduced_system_arrow_bottom_blocks[rank * n_diag_blocks_per_processes:(rank + 1) * n_diag_blocks_per_processes], source=rank, tag=2)
            #     rank = reduced_size - 1
            #     comm.Recv(X_reduced_system_diagonal_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=0)
            #     comm.Recv(X_reduced_system_lower_diagonal_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=1)
            #     comm.Recv(X_reduced_system_arrow_bottom_blocks[rank * n_diag_blocks_per_processes:], source=rank, tag=2)
            # else:
            #     comm.Send(X_reduced_system_diagonal_blocks_local, dest=0, tag=0)
            #     comm.Send(X_reduced_system_lower_diagonal_blocks_local, dest=0, tag=1)
            #     comm.Send(X_reduced_system_arrow_bottom_blocks_local, dest=0, tag=2)
        else:
            X_reduced_system_diagonal_blocks_local = None
            X_reduced_system_lower_diagonal_blocks_local = None
            X_reduced_system_arrow_bottom_blocks_local = None

        # comm.Bcast(X_reduced_system_diagonal_blocks, root=0)
        # comm.Bcast(X_reduced_system_lower_diagonal_blocks, root=0)
        # comm.Bcast(X_reduced_system_arrow_bottom_blocks, root=0)
        # comm.Bcast(X_reduced_system_arrow_tip_block, root=0)

        diag_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        lower_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        arrow_count = n_diag_blocks_per_processes * diag_blocksize * arrow_size
        diag_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        lower_count_last = (n_diag_blocks - 1 - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        arrow_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * arrow_size

        if reduced_color == 0:
            send_diag_count = 0
            send_lower_count = 0
            send_arrow_count = 0
        else:
            if reduced_rank != reduced_size - 1:
                send_diag_count = diag_count
                send_lower_count = lower_count
                send_arrow_count = arrow_count
            else:
                send_diag_count = diag_count_last
                send_lower_count = lower_count_last
                send_arrow_count = arrow_count_last
        
        recv_diag_counts = [diag_count] * (reduced_size - 1) + [diag_count_last] + [0] * (comm_size - reduced_size)
        recv_lower_counts = [lower_count] * (reduced_size - 1) + [lower_count_last] + [0] * (comm_size - reduced_size)
        recv_arrow_counts = [arrow_count] * (reduced_size - 1) + [arrow_count_last] + [0] * (comm_size - reduced_size)
        diag_displ = list(range(0, diag_count * reduced_size, diag_count)) + [0] * (comm_size - reduced_size)
        lower_displ = list(range(0, lower_count * reduced_size, lower_count)) + [0] * (comm_size - reduced_size)
        arrow_displ = list(range(0, arrow_count * reduced_size, arrow_count)) + [0] * (comm_size - reduced_size)

        # if comm_rank == 0:
        #     print(f"{CUPY_AVAIL and xp == cp}", flush=True)
        # comm.Barrier()

        # if CUPY_AVAIL and xp == cp:
        if False:

            if reduced_color == 1:
                X_reduced_system_diagonal_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_diagonal_blocks_local)
                X_reduced_system_lower_diagonal_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_lower_diagonal_blocks_local)
                X_reduced_system_arrow_bottom_blocks_local_host = cpx.empty_like_pinned(X_reduced_system_arrow_bottom_blocks_local)
                X_reduced_system_diagonal_blocks_local.get(out=X_reduced_system_diagonal_blocks_local_host)
                X_reduced_system_lower_diagonal_blocks_local.get(out=X_reduced_system_lower_diagonal_blocks_local_host)
                X_reduced_system_arrow_bottom_blocks_local.get(out=X_reduced_system_arrow_bottom_blocks_local_host)
            else:
                X_reduced_system_diagonal_blocks_local_host = None
                X_reduced_system_lower_diagonal_blocks_local_host = None
                X_reduced_system_arrow_bottom_blocks_local_host = None
            X_reduced_system_arrow_tip_block_host = cpx.empty_like_pinned(X_reduced_system_arrow_tip_block)
            if comm_rank == 0:
                X_reduced_system_arrow_tip_block.get(out=X_reduced_system_arrow_tip_block_host)

            X_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(X_reduced_system_diagonal_blocks)
            X_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(X_reduced_system_lower_diagonal_blocks)
            X_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(X_reduced_system_arrow_bottom_blocks)
        else:
            X_reduced_system_diagonal_blocks_local_host = X_reduced_system_diagonal_blocks_local
            X_reduced_system_lower_diagonal_blocks_local_host = X_reduced_system_lower_diagonal_blocks_local
            X_reduced_system_arrow_bottom_blocks_local_host = X_reduced_system_arrow_bottom_blocks_local
            X_reduced_system_diagonal_blocks_host = X_reduced_system_diagonal_blocks
            X_reduced_system_lower_diagonal_blocks_host = X_reduced_system_lower_diagonal_blocks
            X_reduced_system_arrow_bottom_blocks_host = X_reduced_system_arrow_bottom_blocks
            X_reduced_system_arrow_tip_block_host = X_reduced_system_arrow_tip_block

        mpi_dtype = mpi_datatype[L_diagonal_blocks_local.dtype.type]
        comm.Allgatherv([X_reduced_system_diagonal_blocks_local_host, send_diag_count, mpi_dtype],
                        [X_reduced_system_diagonal_blocks_host, recv_diag_counts, diag_displ, mpi_dtype])
        comm.Allgatherv([X_reduced_system_lower_diagonal_blocks_local_host, send_lower_count, mpi_dtype],
                        [X_reduced_system_lower_diagonal_blocks_host, recv_lower_counts, lower_displ, mpi_dtype])
        comm.Allgatherv([X_reduced_system_arrow_bottom_blocks_local_host, send_arrow_count, mpi_dtype],
                        [X_reduced_system_arrow_bottom_blocks_host, recv_arrow_counts, arrow_displ, mpi_dtype])
        comm.Bcast(X_reduced_system_arrow_tip_block_host, root=0)

        if False:
            X_reduced_system_diagonal_blocks.set(arr=X_reduced_system_diagonal_blocks_host)
            X_reduced_system_lower_diagonal_blocks.set(arr=X_reduced_system_lower_diagonal_blocks_host)
            X_reduced_system_arrow_bottom_blocks.set(arr=X_reduced_system_arrow_bottom_blocks_host)
            X_reduced_system_arrow_tip_block.set(arr=X_reduced_system_arrow_tip_block_host)
    
    else:
        (
            L_reduced_system_diagonal_blocks,
            L_reduced_system_lower_diagonal_blocks,
            L_reduced_system_arrow_bottom_blocks,
            L_reduced_system_arrow_tip_block,
        ) = pobtaf(
            A_reduced_system_diagonal_blocks[1:, :, :],
            A_reduced_system_lower_diagonal_blocks[1:-1, :, :],
            A_reduced_system_arrow_bottom_blocks[1:, :, :],
            A_reduced_system_arrow_tip_block,
            device_streaming=True,
        )

        (
            X_reduced_system_diagonal_blocks,
            X_reduced_system_lower_diagonal_blocks,
            X_reduced_system_arrow_bottom_blocks,
            X_reduced_system_arrow_tip_block,
        ) = pobtasi(
            L_reduced_system_diagonal_blocks,
            L_reduced_system_lower_diagonal_blocks,
            L_reduced_system_arrow_bottom_blocks,
            L_reduced_system_arrow_tip_block,
            device_streaming=True,
        )

    X_arrow_tip_block_global = X_reduced_system_arrow_tip_block

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        X_lower_diagonal_blocks_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[0, :, :]
        )
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        X_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        X_upper_nested_dissection_buffer_local[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            X_lower_diagonal_blocks_local[-1, :, :] = (
                X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]
            )

        X_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        X_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

    # Backward selected-inversion
    # Device buffers
    L_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks_local.shape[1:]), dtype=L_diagonal_blocks_local.dtype
    )
    L_lower_diagonal_blocks_d = cp.empty(
        (2, *L_diagonal_blocks_local.shape[1:]), dtype=L_diagonal_blocks_local.dtype
    )
    L_arrow_bottom_blocks_d = cp.empty(
        (2, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    X_arrow_tip_block_d = cp.empty_like(X_arrow_tip_block_global)

    # X Device buffers arrays pointers
    X_diagonal_blocks_d = L_diagonal_blocks_d
    X_lower_diagonal_blocks_d = L_lower_diagonal_blocks_d
    X_arrow_bottom_blocks_d = L_arrow_bottom_blocks_d

    # Buffers for the intermediate results of the backward block-selected inversion
    L_inv_temp_d = cp.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_d_i = cp.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_d_i = cp.empty_like(L_arrow_bottom_blocks_local[0])

    # Copy/Compute overlap strems
    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_diagonal_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_lower_events = [cp.cuda.Event(), cp.cuda.Event()]

    if comm_rank == 0:
        # --- Host 2 Device transfers ---
        X_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_diagonal_blocks_local[-1, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[(n_diag_blocks_local - 1) % 2].record(h2d_stream)

        X_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_arrow_bottom_blocks_local[-1, :, :], stream=h2d_stream
        )
        h2d_arrow_events[(n_diag_blocks_local - 1) % 2].record(h2d_stream)

        X_arrow_tip_block_d.set(arr=X_arrow_tip_block_global, stream=compute_stream)

        for i in range(n_diag_blocks_local - 2, -1, -1):
            # --- Host 2 Device transfers ---
            h2d_stream.wait_event(compute_diagonal_h2d_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :].set(
                arr=L_diagonal_blocks_local[i, :, :], stream=h2d_stream
            )
            h2d_diagonal_events[i % 2].record(stream=h2d_stream)

            h2d_stream.wait_event(d2h_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :].set(
                arr=L_lower_diagonal_blocks_local[i, :, :], stream=h2d_stream
            )
            h2d_lower_events[i % 2].record(stream=h2d_stream)

            h2d_stream.wait_event(compute_arrow_h2d_events[i % 2])
            L_arrow_bottom_blocks_d[i % 2, :, :].set(
                arr=L_arrow_bottom_blocks_local[i, :, :], stream=h2d_stream
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
                L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[
                    i % 2, :, :
                ]

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
                out=X_lower_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )
            d2h_lower_events[i % 2].record(stream=d2h_stream)

            d2h_stream.wait_event(compute_arrow_events[i % 2])
            X_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=X_arrow_bottom_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            d2h_stream.wait_event(compute_diagonal_events[i % 2])
            X_diagonal_blocks_d[i % 2, :, :].get(
                out=X_diagonal_blocks_local[i, :, :], stream=d2h_stream, blocking=False
            )
    else:
        # Device aliases & buffers specific to the middle process
        X_diagonal_top_block_d = cp.empty_like(X_diagonal_blocks_local[0])
        X_arrow_bottom_top_block_d = cp.empty_like(X_arrow_bottom_blocks_local[0])
        L_upper_nested_dissection_buffer_d = cp.empty(
            (2, *L_upper_nested_dissection_buffer_local.shape[1:]),
            dtype=L_upper_nested_dissection_buffer_local.dtype,
        )

        X_upper_nested_dissection_buffer_d = L_upper_nested_dissection_buffer_d

        L_upper_nested_dissection_buffer_d_i = cp.empty_like(
            L_upper_nested_dissection_buffer_local[0, :, :]
        )

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
        X_diagonal_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_diagonal_blocks_local[-1, :, :], stream=h2d_stream
        )
        X_diagonal_top_block_d.set(
            arr=X_diagonal_blocks_local[0, :, :], stream=h2d_stream
        )
        h2d_diagonal_events[(n_diag_blocks_local - 1) % 2].record(h2d_stream)

        X_arrow_bottom_blocks_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_arrow_bottom_blocks_local[-1, :, :], stream=h2d_stream
        )
        X_arrow_bottom_top_block_d.set(
            arr=X_arrow_bottom_blocks_local[0, :, :], stream=h2d_stream
        )
        h2d_arrow_events[(n_diag_blocks_local - 1) % 2].record(h2d_stream)

        L_upper_nested_dissection_buffer_d[(n_diag_blocks_local - 1) % 2, :, :].set(
            arr=L_upper_nested_dissection_buffer_local[-1, :, :], stream=h2d_stream
        )
        h2d_upper_nested_dissection_buffer_events[(n_diag_blocks_local - 1) % 2].record(
            stream=h2d_stream
        )

        X_arrow_tip_block_d.set(arr=X_arrow_tip_block_global, stream=compute_stream)

        for i in range(n_diag_blocks_local - 2, 0, -1):
            # --- Host 2 Device transfers ---
            h2d_stream.wait_event(compute_diagonal_h2d_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :].set(
                arr=L_diagonal_blocks_local[i, :, :], stream=h2d_stream
            )
            h2d_diagonal_events[i % 2].record(stream=h2d_stream)

            h2d_stream.wait_event(d2h_lower_events[i % 2])
            L_lower_diagonal_blocks_d[i % 2, :, :].set(
                arr=L_lower_diagonal_blocks_local[i, :, :], stream=h2d_stream
            )
            h2d_lower_events[i % 2].record(stream=h2d_stream)

            h2d_stream.wait_event(compute_arrow_h2d_events[i % 2])
            L_arrow_bottom_blocks_d[i % 2, :, :].set(
                arr=L_arrow_bottom_blocks_local[i, :, :], stream=h2d_stream
            )
            h2d_arrow_events[i % 2].record(stream=h2d_stream)

            h2d_stream.wait_event(
                compute_upper_nested_dissection_buffer_h2d_events[i % 2]
            )
            L_upper_nested_dissection_buffer_d[i % 2, :, :].set(
                arr=L_upper_nested_dissection_buffer_local[i, :, :], stream=h2d_stream
            )
            h2d_upper_nested_dissection_buffer_events[i % 2].record(stream=h2d_stream)

            with compute_stream:
                compute_stream.wait_event(h2d_diagonal_events[i % 2])
                L_inv_temp_d[:, :] = cu_la.solve_triangular(
                    L_diagonal_blocks_d[i % 2, :, :],
                    cp.eye(diag_blocksize),
                    lower=True,
                )

                compute_stream.wait_event(h2d_lower_events[i % 2])
                L_lower_diagonal_blocks_d_i[:, :] = L_lower_diagonal_blocks_d[
                    i % 2, :, :
                ]

                compute_stream.wait_event(h2d_arrow_events[i % 2])
                L_arrow_bottom_blocks_d_i[:, :] = L_arrow_bottom_blocks_d[i % 2, :, :]

                compute_stream.wait_event(
                    h2d_upper_nested_dissection_buffer_events[i % 2]
                )
                L_upper_nested_dissection_buffer_d_i[:, :] = (
                    L_upper_nested_dissection_buffer_d[i % 2, :, :]
                )

                # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
                X_lower_diagonal_blocks_d[i % 2, :, :] = (
                    -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :].conj().T
                    @ L_upper_nested_dissection_buffer_d_i[:, :]
                    - X_diagonal_blocks_d[(i + 1) % 2, :, :]
                    @ L_lower_diagonal_blocks_d_i[:, :]
                    - X_arrow_bottom_blocks_d[(i + 1) % 2, :, :].conj().T
                    @ L_arrow_bottom_blocks_d_i[:, :]
                ) @ L_inv_temp_d[:, :]
                compute_diagonal_h2d_events[(i + 1) % 2].record(stream=compute_stream)
                compute_lower_events[i % 2].record(stream=compute_stream)

                # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
                X_upper_nested_dissection_buffer_d[i % 2, :, :] = (
                    -X_upper_nested_dissection_buffer_d[(i + 1) % 2, :, :]
                    @ L_lower_diagonal_blocks_d_i[:, :]
                    - X_diagonal_top_block_d[:, :]
                    @ L_upper_nested_dissection_buffer_d_i[:, :]
                    - X_arrow_bottom_top_block_d[:, :].conj().T
                    @ L_arrow_bottom_blocks_d_i[:, :]
                ) @ L_inv_temp_d[:, :]
                compute_upper_nested_dissection_buffer_h2d_events[(i + 1) % 2].record(
                    stream=compute_stream
                )
                compute_upper_nested_dissection_buffer_events[i % 2].record(
                    stream=compute_stream
                )

                # Arrowhead
                # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
                X_arrow_bottom_blocks_d[i % 2, :, :] = (
                    -X_arrow_bottom_blocks_d[(i + 1) % 2, :, :]
                    @ L_lower_diagonal_blocks_d_i[:, :]
                    - X_arrow_bottom_top_block_d[:, :]
                    @ L_upper_nested_dissection_buffer_d_i[:, :]
                    - X_arrow_tip_block_d[:, :] @ L_arrow_bottom_blocks_d_i[:, :]
                ) @ L_inv_temp_d[:, :]
                compute_arrow_h2d_events[(i + 1) % 2].record(stream=compute_stream)
                compute_arrow_events[i % 2].record(stream=compute_stream)

                # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
                X_diagonal_blocks_d[i % 2, :, :] = (
                    L_inv_temp_d[:, :].conj().T
                    - X_lower_diagonal_blocks_d[i % 2, :, :].conj().T
                    @ L_lower_diagonal_blocks_d_i[:, :]
                    - X_upper_nested_dissection_buffer_d[i % 2, :, :].conj().T
                    @ L_upper_nested_dissection_buffer_d_i[:, :]
                    - X_arrow_bottom_blocks_d[i % 2, :, :].conj().T
                    @ L_arrow_bottom_blocks_d_i[:, :]
                ) @ L_inv_temp_d[:, :]
                compute_diagonal_events[i % 2].record(stream=compute_stream)

            # --- Device 2 Host transfers ---
            d2h_stream.wait_event(compute_lower_events[i % 2])
            X_lower_diagonal_blocks_d[i % 2, :, :].get(
                out=X_lower_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )
            d2h_lower_events[i % 2].record(stream=d2h_stream)

            d2h_stream.wait_event(compute_arrow_events[i % 2])
            X_arrow_bottom_blocks_d[i % 2, :, :].get(
                out=X_arrow_bottom_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            d2h_stream.wait_event(compute_upper_nested_dissection_buffer_events[i % 2])
            X_upper_nested_dissection_buffer_d[i % 2, :, :].get(
                out=X_upper_nested_dissection_buffer_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

            d2h_stream.wait_event(compute_diagonal_events[i % 2])
            X_diagonal_blocks_d[i % 2, :, :].get(
                out=X_diagonal_blocks_local[i, :, :],
                stream=d2h_stream,
                blocking=False,
            )

        # Copy back the first block of the nested dissection buffer to the
        # tridiagonal storage.
        with d2h_stream:
            d2h_stream.synchronize()
            X_lower_diagonal_blocks_local[0, :, :] = (
                X_upper_nested_dissection_buffer_local[1, :, :].conj().T
            )

    cp.cuda.Device().synchronize()

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )
