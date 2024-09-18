# Copyright 2023-2024 ETH Zurich. All rights reserved.
from serinv import SolverConfig
from serinv.algs import pobtaf, pobtasi

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

NCCL_AVAIL = False
if CUPY_AVAIL:
    try:
        from cupy.cuda import nccl
        nccl.get_version()  # Check if NCCL is available

        NCCL_AVAIL = True
    except (AttributeError, ImportError, ModuleNotFoundError):
        pass

import numpy as np
import scipy.linalg as np_la
import time

import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
from mpi4py import MPI
from numpy.typing import ArrayLike

from serinv.algs import pobtaf, pobtasi, d_pobtaf

FLOAT_COMPLEX = MPI.C_FLOAT_COMPLEX
DOUBLE_COMPLEX = MPI.C_DOUBLE_COMPLEX if MPI.Get_library_version().startswith('Open MPI') else MPI.DOUBLE_COMPLEX
mpi_datatype = {np.float32: MPI.FLOAT, np.complex64: FLOAT_COMPLEX,
                np.float64: MPI.DOUBLE, np.complex128: DOUBLE_COMPLEX}
if CUPY_AVAIL:
    mpi_datatype[cp.float32] = MPI.FLOAT
    mpi_datatype[cp.complex64] = FLOAT_COMPLEX
    mpi_datatype[cp.float64] = MPI.DOUBLE
    mpi_datatype[cp.complex128] = DOUBLE_COMPLEX
if NCCL_AVAIL:
    nccl_datatype = {np.float32: nccl.NCCL_FLOAT, cp.float32: nccl.NCCL_FLOAT, cp.complex64: nccl.NCCL_FLOAT,
                     np.float64: nccl.NCCL_DOUBLE, cp.float64: nccl.NCCL_DOUBLE, cp.complex128: nccl.NCCL_DOUBLE}



def d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike = None,
    solver_config: SolverConfig = SolverConfig(),
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
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

            T_{flops_{d_pobtasi}} = (n/p-1) * (19*b^3 + 14*a*b^2) + T_{flops_{POBTAF}} + T_{flops_{POBTASI}}

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
    B_permutation_upper : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    solver_config : SolverConfig, optional
        Configuration of the solver.

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

    if CUPY_AVAIL:
        array_module = cp.get_array_module(L_diagonal_blocks_local)
        if solver_config.device_streaming and array_module == np:
            # Device streaming
            return _streaming_d_pobtasi(
                L_diagonal_blocks_local,
                L_lower_diagonal_blocks_local,
                L_arrow_bottom_blocks_local,
                L_arrow_tip_block_global,
                B_permutation_upper,
                solver_config,
                comm
        )

        if array_module == cp:
            # Device computation
            return _device_d_pobtasi(
                L_diagonal_blocks_local,
                L_lower_diagonal_blocks_local,
                L_arrow_bottom_blocks_local,
                L_arrow_tip_block_global,
                B_permutation_upper,
                solver_config,
                comm
            )

    # Host computation
    return _host_d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        B_permutation_upper,
        comm
    )


def d_pobtasi_rss(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike = None,
    solver_config: SolverConfig = SolverConfig(),
    comm: MPI.Comm = MPI.COMM_WORLD,
    nested_comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    """Perform the selected-inversion of the reduced system constructed from the
    distributed factorization of the block tridiagonal with arrowhead matrix.

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
    B_permutation_upper : ArrayLike, optional
        Local upper buffer used in the permuted factorization. None for
        uppermost process.
    solver_config : SolverConfig, optional
        Configuration of the solver.

    Returns
    -------
    L_diagonal_blocks_local : ArrayLike
        Local slice of the diagonal blocks of L, initialized with the solution
        of the reduced system.
    L_lower_diagonal_blocks_local : ArrayLike
        Local slice of the lower diagonal blocks of L, initialized with the solution
        of the reduced system.
    L_arrow_bottom_blocks_local : ArrayLike
        Local slice of the arrow bottom blocks of L, initialized with the solution
        of the reduced system.
    L_arrow_tip_block_global : ArrayLike
        Arrow tip block of L, initialized with the solution of the reduced system.
    B_permutation_upper : ArrayLike, optional
        Local upper buffer used in the permuted factorization, initialized
        with the solution of the reduced system. None for uppermost process.
    """

    la = np_la

    if CUPY_AVAIL:
        array_module = cp.get_array_module(L_diagonal_blocks_local)
        if solver_config.device_streaming and array_module == np:
            # Device streaming
            return _streaming_d_pobtasi_rss(
                L_diagonal_blocks_local,
                L_lower_diagonal_blocks_local,
                L_arrow_bottom_blocks_local,
                L_arrow_tip_block_global,
                B_permutation_upper,
                solver_config,
                comm
            )

        if array_module == cp:
            # Device computation
            return _device_d_pobtasi_rss(
                L_diagonal_blocks_local,
                L_lower_diagonal_blocks_local,
                L_arrow_bottom_blocks_local,
                L_arrow_tip_block_global,
                B_permutation_upper,
                solver_config,
                comm,
                nested_comm
            )

    # Host computation
    return _host_d_pobtasi_rss(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        B_permutation_upper,
        solver_config,
        comm
    )


def _host_d_pobtasi_rss(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike,
    solver_config: SolverConfig = SolverConfig(),
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    A_reduced_system_diagonal_blocks = np.zeros(
        (2 * comm_size, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = np.zeros(
        (2 * comm_size, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = np.zeros(
        (2 * comm_size, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = L_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        # R_{0,0} = A^{p_0}_{0,0}
        A_reduced_system_diagonal_blocks[1, :, :] = L_diagonal_blocks_local[-1, :, :]

        # R_{1, 0} = A^{p_0}_{1, 0}
        A_reduced_system_lower_diagonal_blocks[1, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]

        # R_{n, 0} = A^{p_0}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[1, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        # R_{2p_i-1, 2p_i-1} = A^{p_i}_{0, 0}
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            0, :, :
        ]

        # R_{2p_i, 2p_i-1} = A^{p_i}_{-1, -1}
        A_reduced_system_diagonal_blocks[
            2 * comm_rank + 1, :, :
        ] = L_diagonal_blocks_local[-1, :, :]

        # R_{2p_i-1, 2p_i-2} = B^{p_i}_{-1}^\dagger
        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            B_permutation_upper[-1, :, :].conj().T
        )

        if comm_rank < comm_size - 1:
            # R_{2p_i, 2p_i-1} = A^{p_i}_{1, 0}
            A_reduced_system_lower_diagonal_blocks[
                2 * comm_rank + 1, :, :
            ] = L_lower_diagonal_blocks_local[-1, :, :]

        # R_{n, 2p_i-1} = A^{p_i}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ] = L_arrow_bottom_blocks_local[0, :, :]

        # R_{n, 2p_i} = A^{p_i}_{n, -1}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank + 1, :, :
        ] = L_arrow_bottom_blocks_local[-1, :, :]

    # Communicate the reduced system
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
    n_diag_blocks = 2 * comm_size - 1
    reduced_rank = comm_rank
    reduced_size = comm_size // 2
    n_diag_blocks_per_processes = n_diag_blocks // reduced_size
    if solver_config.nested_solving and reduced_size > 1:
        diag_blocksize = L_diagonal_blocks_local.shape[1]
        arrow_size = L_arrow_tip_block_global.shape[0]
        reduced_color = int(comm_rank < reduced_size)
        reduced_key = comm_rank
        reduced_comm = comm.Split(color=reduced_color, key=reduced_key)

        X_reduced_system_diagonal_blocks = np.empty((n_diag_blocks, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_lower_diagonal_blocks = np.empty((n_diag_blocks - 1, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_bottom_blocks = np.empty((n_diag_blocks, arrow_size, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_tip_block = np.empty((arrow_size, arrow_size), dtype=L_diagonal_blocks_local.dtype)

        if reduced_color == 1:

            A_reduced_system_diagonal_blocks = A_reduced_system_diagonal_blocks[1:]
            A_reduced_system_lower_diagonal_blocks = A_reduced_system_lower_diagonal_blocks[1:-1]
            A_reduced_system_arrow_bottom_blocks = A_reduced_system_arrow_bottom_blocks[1:]
            
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

            reduced_solver_config = solver_config.copy(update={"nested_solving": False})
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
                solver_config=reduced_solver_config,
                comm=reduced_comm
            )
            (
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
            ) = d_pobtasi_rss(
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
                solver_config=reduced_solver_config,
                comm=reduced_comm,
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
                solver_config=reduced_solver_config,
                comm=reduced_comm
            )
            # NOTE: For some reason, the returned X_reduced_system_arrow_tip_block is not contiguous in memory.
            X_reduced_system_arrow_tip_block[:] = X_reduced_system_arrow_tip_block_tmp
        else:
            X_reduced_system_diagonal_blocks_local = None
            X_reduced_system_lower_diagonal_blocks_local = None
            X_reduced_system_arrow_bottom_blocks_local = None

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
        L_arrow_tip_block_global[:] = X_reduced_system_arrow_tip_block_host
                
    else:

        # Perform the inversion of the reduced system.
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
        )

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        L_lower_diagonal_blocks_local[
            -1, :, :
        ] = X_reduced_system_lower_diagonal_blocks[0, :, :]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        L_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        B_permutation_upper[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            L_lower_diagonal_blocks_local[
                -1, :, :
            ] = X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]

        L_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        B_permutation_upper,
    )


def _device_d_pobtasi_rss(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike,
    solver_config: SolverConfig,
    comm: MPI.Comm = MPI.COMM_WORLD,
    reduced_comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    if NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator):
        comm_rank = comm.rank_id()
        comm_size = comm.size()
    else:
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

    A_reduced_system_diagonal_blocks = cp.zeros(
        (2 * comm_size, *L_diagonal_blocks_local.shape[1:]),
        dtype=L_diagonal_blocks_local.dtype,
    )
    A_reduced_system_lower_diagonal_blocks = cp.zeros(
        (2 * comm_size, *L_lower_diagonal_blocks_local.shape[1:]),
        dtype=L_lower_diagonal_blocks_local.dtype,
    )
    A_reduced_system_arrow_bottom_blocks = cp.zeros(
        (2 * comm_size, *L_arrow_bottom_blocks_local.shape[1:]),
        dtype=L_arrow_bottom_blocks_local.dtype,
    )
    # Alias on the tip block for the reduced system
    A_reduced_system_arrow_tip_block = L_arrow_tip_block_global

    # Construct the reduced system from the factorized blocks distributed over the
    # processes.
    if comm_rank == 0:
        # R_{0,0} = A^{p_0}_{0,0}
        A_reduced_system_diagonal_blocks[1, :, :] = L_diagonal_blocks_local[-1, :, :]

        # R_{1, 0} = A^{p_0}_{1, 0}
        A_reduced_system_lower_diagonal_blocks[1, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]

        # R_{n, 0} = A^{p_0}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[1, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        # R_{2p_i-1, 2p_i-1} = A^{p_i}_{0, 0}
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            0, :, :
        ]

        # R_{2p_i, 2p_i-1} = A^{p_i}_{-1, -1}
        A_reduced_system_diagonal_blocks[
            2 * comm_rank + 1, :, :
        ] = L_diagonal_blocks_local[-1, :, :]

        # R_{2p_i-1, 2p_i-2} = B^{p_i}_{-1}^\dagger
        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            B_permutation_upper[-1, :, :].conj().T
        )

        if comm_rank < comm_size - 1:
            # R_{2p_i, 2p_i-1} = A^{p_i}_{1, 0}
            A_reduced_system_lower_diagonal_blocks[
                2 * comm_rank + 1, :, :
            ] = L_lower_diagonal_blocks_local[-1, :, :]

        # R_{n, 2p_i-1} = A^{p_i}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ] = L_arrow_bottom_blocks_local[0, :, :]

        # R_{n, 2p_i} = A^{p_i}_{n, -1}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank + 1, :, :
        ] = L_arrow_bottom_blocks_local[-1, :, :]

    if (NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator)) or solver_config.cuda_aware_mpi:
        A_reduced_system_diagonal_blocks_host = A_reduced_system_diagonal_blocks
        A_reduced_system_lower_diagonal_blocks_host = A_reduced_system_lower_diagonal_blocks
        A_reduced_system_arrow_bottom_blocks_host = A_reduced_system_arrow_bottom_blocks
    else:
        # Allocate reduced system on host using pinned memory
        A_reduced_system_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_diagonal_blocks
        )
        A_reduced_system_lower_diagonal_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_lower_diagonal_blocks
        )
        A_reduced_system_arrow_bottom_blocks_host = cpx.empty_like_pinned(
            A_reduced_system_arrow_bottom_blocks
        )

        # Copy the reduced system to the host
        A_reduced_system_diagonal_blocks.get(out=A_reduced_system_diagonal_blocks_host)
        A_reduced_system_lower_diagonal_blocks.get(
            out=A_reduced_system_lower_diagonal_blocks_host
        )
        A_reduced_system_arrow_bottom_blocks.get(
            out=A_reduced_system_arrow_bottom_blocks_host
        )

    # Communicate the reduced system
    start = time.perf_counter_ns()
    if NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator):
        print(f"Rank {comm_rank}: POBTASI Allgather with NCCL.", flush=True)
        szd = A_reduced_system_diagonal_blocks_host.size // comm_size
        szl = A_reduced_system_lower_diagonal_blocks_host.size // comm_size
        sza = A_reduced_system_arrow_bottom_blocks_host.size // comm_size
        datatype = nccl_datatype[A_reduced_system_diagonal_blocks_host.dtype.type]
        itemsize = A_reduced_system_diagonal_blocks_host.dtype.itemsize
        dispd = szd * comm_rank * itemsize
        displ = szl * comm_rank * itemsize
        dispa = sza * comm_rank * itemsize
        if np.iscomplexobj(A_reduced_system_diagonal_blocks_host):
            szd *= 2
            szl *= 2
            sza *= 2
        comm.allGather(
            A_reduced_system_diagonal_blocks_host.data.ptr + dispd,
            A_reduced_system_diagonal_blocks_host.data.ptr,
            szd,
            datatype,
            cp.cuda.Stream.null.ptr,
        )
        finish_1 = time.perf_counter_ns()
        comm.allGather(
            A_reduced_system_lower_diagonal_blocks_host.data.ptr + displ,
            A_reduced_system_lower_diagonal_blocks_host.data.ptr,
            szl,
            datatype,
            cp.cuda.Stream.null.ptr,
        )
        finish_2 = time.perf_counter_ns()
        comm.allGather(
            A_reduced_system_arrow_bottom_blocks_host.data.ptr + dispa,
            A_reduced_system_arrow_bottom_blocks_host.data.ptr,
            sza,
            datatype,
            cp.cuda.Stream.null.ptr,
        )
        cp.cuda.Stream.null.synchronize()
    else:
        comm.Allgather(
            MPI.IN_PLACE,
            A_reduced_system_diagonal_blocks_host,
        )
        finish_1 = time.perf_counter_ns()
        comm.Allgather(
            MPI.IN_PLACE,
            A_reduced_system_lower_diagonal_blocks_host,
        )
        finish_2 = time.perf_counter_ns()
        comm.Allgather(
            MPI.IN_PLACE,
            A_reduced_system_arrow_bottom_blocks_host,
        )
    finish_3 = time.perf_counter_ns()
    print(f"Rank {comm_rank}: POBTASI Allgather 1 {(finish_1-start) // 1000000} ms.", flush=True)
    print(f"Rank {comm_rank}: POBTASI Allgather 2 {(finish_2-finish_1) // 1000000} ms.", flush=True)
    print(f"Rank {comm_rank}: POBTASI Allgather 3 {(finish_3-finish_2) // 1000000} ms.", flush=True)


    if not ((NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator)) or solver_config.cuda_aware_mpi):
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
    n_diag_blocks_per_processes = int(np.ceil(n_diag_blocks / reduced_size))
    if solver_config.nested_solving and reduced_size > 1:
        diag_blocksize = L_diagonal_blocks_local.shape[1]
        arrow_size = L_arrow_tip_block_global.shape[0]
        reduced_color = int(comm_rank < reduced_size)
        reduced_key = comm_rank
        start = time.perf_counter_ns()
        if NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator):
            pass
            # comm_id = nccl.get_unique_id()
            # comm_id = MPI.COMM_WORLD.bcast(comm_id, root=0)
            # if reduced_color == 1:
            #     reduced_comm = nccl.NcclCommunicator(reduced_size, comm_id, reduced_rank)
            # cp.cuda.runtime.deviceSynchronize()
        else:
            reduced_comm = comm.Split(color=reduced_color, key=reduced_key)
        finish = time.perf_counter_ns()
        print(f"Rank {comm_rank}: POBTASI Split {(finish-start) // 1000000} ms.", flush=True)

        start = time.perf_counter_ns()
        X_reduced_system_diagonal_blocks = cp.empty((n_diag_blocks, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_lower_diagonal_blocks = cp.empty((n_diag_blocks - 1, diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_bottom_blocks = cp.empty((n_diag_blocks, arrow_size, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
        X_reduced_system_arrow_tip_block = cp.empty((arrow_size, arrow_size), dtype=L_diagonal_blocks_local.dtype)
        finish = time.perf_counter_ns()
        print(f"Rank {comm_rank}: POBTASI Empty {(finish-start) // 1000000} ms.", flush=True)

        if reduced_color == 1:

            start = time.perf_counter_ns()

            A_reduced_system_diagonal_blocks = A_reduced_system_diagonal_blocks[1:]
            A_reduced_system_lower_diagonal_blocks = A_reduced_system_lower_diagonal_blocks[1:-1]
            A_reduced_system_arrow_bottom_blocks = A_reduced_system_arrow_bottom_blocks[1:]
            
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

            reduced_solver_config = solver_config.copy(update={"nested_solving": False})
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
                solver_config=reduced_solver_config,
                comm=reduced_comm
            )
            (
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
            ) = d_pobtasi_rss(
                L_reduced_system_diagonal_blocks_local,
                L_reduced_system_lower_diagonal_blocks_local,
                L_reduced_system_arrow_bottom_blocks_local,
                L_reduced_system_arrow_tip_block_global,
                L_reduced_system_upper_nested_dissection_buffer,
                solver_config=reduced_solver_config,
                comm=reduced_comm,
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
                solver_config=reduced_solver_config,
                comm=reduced_comm
            )
            # NOTE: For some reason, the returned X_reduced_system_arrow_tip_block is not contiguous in memory.
            X_reduced_system_arrow_tip_block[:] = X_reduced_system_arrow_tip_block_tmp
            cp.cuda.Stream.null.synchronize()
            finish = time.perf_counter_ns()
            print(f"Rank {comm_rank}: POBTASI RSS Nested {(finish-start) // 1000000} ms.", flush=True)
        else:
            X_reduced_system_diagonal_blocks_local = None
            X_reduced_system_lower_diagonal_blocks_local = None
            X_reduced_system_arrow_bottom_blocks_local = None

        if not ((NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator)) or solver_config.cuda_aware_mpi):

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

        start = time.perf_counter_ns()

        mpi_dtype = mpi_datatype[L_diagonal_blocks_local.dtype.type]

        diag_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        lower_count = n_diag_blocks_per_processes * diag_blocksize * diag_blocksize
        arrow_count = n_diag_blocks_per_processes * diag_blocksize * arrow_size
        diag_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        lower_count_last = (n_diag_blocks - 1 - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * diag_blocksize
        arrow_count_last = (n_diag_blocks - (reduced_size - 1) * n_diag_blocks_per_processes) * diag_blocksize * arrow_size

        if NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator):
            diag_ptr = X_reduced_system_diagonal_blocks_local_host.data.ptr if comm_rank < reduced_size else 0
            lower_ptr = X_reduced_system_lower_diagonal_blocks_local_host.data.ptr if comm_rank < reduced_size else 0
            arrow_ptr = X_reduced_system_arrow_bottom_blocks_local_host.data.ptr if comm_rank < reduced_size else 0
            datatype = nccl_datatype[X_reduced_system_diagonal_blocks.dtype.type]
            itemsize = X_reduced_system_diagonal_blocks.dtype.itemsize
            dispd = diag_count * itemsize
            displ = lower_count * itemsize
            dispa = arrow_count * itemsize
            sz = X_reduced_system_arrow_tip_block_host.size
            if np.iscomplexobj(X_reduced_system_arrow_tip_block_host):
                diag_count *= 2
                lower_count *= 2
                arrow_count *= 2
                diag_count_last *= 2
                lower_count_last *= 2
                arrow_count_last *= 2
                sz *= 2
            for i in range(reduced_size - 1):
                comm.broadcast(diag_ptr, X_reduced_system_diagonal_blocks_host.data.ptr + i * dispd, diag_count, datatype, i, cp.cuda.Stream.null.ptr)
                comm.broadcast(lower_ptr, X_reduced_system_lower_diagonal_blocks_host.data.ptr + i * displ, lower_count, datatype, i, cp.cuda.Stream.null.ptr)
                comm.broadcast(arrow_ptr , X_reduced_system_arrow_bottom_blocks_host.data.ptr + i * dispa, arrow_count, datatype, i, cp.cuda.Stream.null.ptr)
            i = reduced_size - 1
            comm.broadcast(diag_ptr, X_reduced_system_diagonal_blocks_host.data.ptr + i * dispd, diag_count_last, datatype, i, cp.cuda.Stream.null.ptr)
            comm.broadcast(lower_ptr, X_reduced_system_lower_diagonal_blocks_host.data.ptr + i * displ, lower_count_last, datatype, i, cp.cuda.Stream.null.ptr)
            comm.broadcast(arrow_ptr , X_reduced_system_arrow_bottom_blocks_host.data.ptr + i * dispa, arrow_count_last, datatype, i, cp.cuda.Stream.null.ptr)
            comm.broadcast(X_reduced_system_arrow_tip_block_host.data.ptr, X_reduced_system_arrow_tip_block_host.data.ptr, sz, datatype, 0, cp.cuda.Stream.null.ptr)
            cp.cuda.Stream.null.synchronize()
        else:

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

            comm.Allgatherv([X_reduced_system_diagonal_blocks_local_host, send_diag_count, mpi_dtype],
                            [X_reduced_system_diagonal_blocks_host, recv_diag_counts, diag_displ, mpi_dtype])
            comm.Allgatherv([X_reduced_system_lower_diagonal_blocks_local_host, send_lower_count, mpi_dtype],
                            [X_reduced_system_lower_diagonal_blocks_host, recv_lower_counts, lower_displ, mpi_dtype])
            comm.Allgatherv([X_reduced_system_arrow_bottom_blocks_local_host, send_arrow_count, mpi_dtype],
                            [X_reduced_system_arrow_bottom_blocks_host, recv_arrow_counts, arrow_displ, mpi_dtype])
            comm.Bcast(X_reduced_system_arrow_tip_block_host, root=0)
        finish = time.perf_counter_ns()
        print(f"Rank {comm_rank}: POBTASI Allgather x 3 + Bcast {(finish-start) // 1000000} ms.", flush=True)
        L_arrow_tip_block_global[:] = cp.asarray(X_reduced_system_arrow_tip_block_host)

        if not ((NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator)) or solver_config.cuda_aware_mpi):
            X_reduced_system_diagonal_blocks.set(arr=X_reduced_system_diagonal_blocks_host)
            X_reduced_system_lower_diagonal_blocks.set(arr=X_reduced_system_lower_diagonal_blocks_host)
            X_reduced_system_arrow_bottom_blocks.set(arr=X_reduced_system_arrow_bottom_blocks_host)
            X_reduced_system_arrow_tip_block.set(arr=X_reduced_system_arrow_tip_block_host)
                
    else:

        # Perform the inversion of the reduced system.
        start = time.perf_counter_ns()
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
        )
        cp.cuda.Stream.null.synchronize()
        finish = time.perf_counter_ns()
        print(f"Rank {comm_rank}: POBTASI RSS {(finish-start) // 1000000} ms.", flush=True)

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    start = time.perf_counter_ns()
    if comm_rank == 0:
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        L_lower_diagonal_blocks_local[
            -1, :, :
        ] = X_reduced_system_lower_diagonal_blocks[0, :, :]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        L_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        B_permutation_upper[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            L_lower_diagonal_blocks_local[
                -1, :, :
            ] = X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]

        L_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]
    finish = time.perf_counter_ns()
    print(f"Rank {comm_rank}: POBTASI Update {(finish-start) // 1000000} ms.", flush=True)

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        B_permutation_upper,
    )


def _streaming_d_pobtasi_rss(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike,
    solver_config: SolverConfig,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

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
        # R_{0,0} = A^{p_0}_{0,0}
        A_reduced_system_diagonal_blocks[1, :, :] = L_diagonal_blocks_local[-1, :, :]

        # R_{1, 0} = A^{p_0}_{1, 0}
        A_reduced_system_lower_diagonal_blocks[1, :, :] = L_lower_diagonal_blocks_local[
            -1, :, :
        ]

        # R_{n, 0} = A^{p_0}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[1, :, :] = L_arrow_bottom_blocks_local[
            -1, :, :
        ]
    else:
        # R_{2p_i-1, 2p_i-1} = A^{p_i}_{0, 0}
        A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
            0, :, :
        ]

        # R_{2p_i, 2p_i-1} = A^{p_i}_{-1, -1}
        A_reduced_system_diagonal_blocks[
            2 * comm_rank + 1, :, :
        ] = L_diagonal_blocks_local[-1, :, :]

        # R_{2p_i-1, 2p_i-2} = B^{p_i}_{-1}^\dagger
        A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
            B_permutation_upper[-1, :, :].conj().T
        )

        if comm_rank < comm_size - 1:
            # R_{2p_i, 2p_i-1} = A^{p_i}_{1, 0}
            A_reduced_system_lower_diagonal_blocks[
                2 * comm_rank + 1, :, :
            ] = L_lower_diagonal_blocks_local[-1, :, :]

        # R_{n, 2p_i-1} = A^{p_i}_{n, 0}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ] = L_arrow_bottom_blocks_local[0, :, :]

        # R_{n, 2p_i} = A^{p_i}_{n, -1}
        A_reduced_system_arrow_bottom_blocks[
            2 * comm_rank + 1, :, :
        ] = L_arrow_bottom_blocks_local[-1, :, :]

    # Communicate the reduced system
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

    # Update of the local slices by there corresponding blocks in the inverted
    # reduced system.
    if comm_rank == 0:
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[0, :, :]
        L_lower_diagonal_blocks_local[
            -1, :, :
        ] = X_reduced_system_lower_diagonal_blocks[0, :, :]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            0, :, :
        ]
    else:
        L_diagonal_blocks_local[0, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_diagonal_blocks_local[-1, :, :] = X_reduced_system_diagonal_blocks[
            2 * comm_rank, :, :
        ]

        B_permutation_upper[-1, :, :] = (
            X_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :].conj().T
        )
        if comm_rank < comm_size - 1:
            L_lower_diagonal_blocks_local[
                -1, :, :
            ] = X_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :]

        L_arrow_bottom_blocks_local[0, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank - 1, :, :
        ]
        L_arrow_bottom_blocks_local[-1, :, :] = X_reduced_system_arrow_bottom_blocks[
            2 * comm_rank, :, :
        ]

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        B_permutation_upper,
    )


def _host_d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_arrow_tip_block_global = L_arrow_tip_block_global

    # Backward selected-inversion
    L_inv_temp = np.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_temp = np.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_temp = np.empty_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]

            # Compute lower factors
            L_inv_temp[:, :] = np_la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                np.eye(diag_blocksize),
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
        L_upper_nested_dissection_buffer_temp = np.empty_like(
            B_permutation_upper[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]
            L_upper_nested_dissection_buffer_temp[:, :] = B_permutation_upper[i, :, :]

            L_inv_temp[:, :] = np_la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                np.eye(diag_blocksize),
                lower=True,
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -B_permutation_upper[i + 1, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            B_permutation_upper[i, :, :] = (
                -B_permutation_upper[i + 1, :, :] @ L_lower_diagonal_blocks_temp[:, :]
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

            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - B_permutation_upper[i, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = B_permutation_upper[1, :, :].conj().T

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )


def _device_d_pobtasi(
    L_diagonal_blocks_local: ArrayLike,
    L_lower_diagonal_blocks_local: ArrayLike,
    L_arrow_bottom_blocks_local: ArrayLike,
    L_arrow_tip_block_global: ArrayLike,
    B_permutation_upper: ArrayLike,
    solver_config: SolverConfig,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    if NCCL_AVAIL and isinstance(comm, nccl.NcclCommunicator):
        comm_rank = comm.rank_id()
        comm_size = comm.size()
    else:
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_arrow_tip_block_global = L_arrow_tip_block_global

    # Backward selected-inversion
    L_inv_temp = cp.empty_like(L_diagonal_blocks_local[0])
    L_lower_diagonal_blocks_temp = cp.empty_like(L_lower_diagonal_blocks_local[0])
    L_arrow_bottom_blocks_temp = cp.empty_like(L_arrow_bottom_blocks_local[0])

    if comm_rank == 0:
        for i in range(n_diag_blocks_local - 2, -1, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]

            # Compute lower factors
            L_inv_temp[:, :] = cu_la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                cp.eye(diag_blocksize),
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
        L_upper_nested_dissection_buffer_temp = cp.empty_like(
            B_permutation_upper[0, :, :]
        )

        for i in range(n_diag_blocks_local - 2, 0, -1):
            L_lower_diagonal_blocks_temp[:, :] = L_lower_diagonal_blocks_local[i, :, :]
            L_arrow_bottom_blocks_temp[:, :] = L_arrow_bottom_blocks_local[i, :, :]
            L_upper_nested_dissection_buffer_temp[:, :] = B_permutation_upper[i, :, :]

            L_inv_temp[:, :] = cu_la.solve_triangular(
                L_diagonal_blocks_local[i, :, :],
                cp.eye(diag_blocksize),
                lower=True,
            )

            # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_lower_diagonal_blocks_local[i, :, :] = (
                -B_permutation_upper[i + 1, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_diagonal_blocks_local[i + 1, :, :]
                @ L_lower_diagonal_blocks_temp[:, :]
                - X_arrow_bottom_blocks_local[i + 1, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

            # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
            B_permutation_upper[i, :, :] = (
                -B_permutation_upper[i + 1, :, :] @ L_lower_diagonal_blocks_temp[:, :]
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

            # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
            X_diagonal_blocks_local[i, :, :] = (
                L_inv_temp[:, :].conj().T
                - X_lower_diagonal_blocks_local[i, :, :].conj().T
                @ L_lower_diagonal_blocks_temp[:, :]
                - B_permutation_upper[i, :, :].conj().T
                @ L_upper_nested_dissection_buffer_temp[:, :]
                - X_arrow_bottom_blocks_local[i, :, :].conj().T
                @ L_arrow_bottom_blocks_temp[:, :]
            ) @ L_inv_temp[:, :]

        # Copy back the 2 first blocks that have been produced in the 2-sided pattern
        # to the tridiagonal storage.
        X_lower_diagonal_blocks_local[0, :, :] = B_permutation_upper[1, :, :].conj().T

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
    B_permutation_upper: ArrayLike,
    solver_config: SolverConfig,
    comm: MPI.Comm = MPI.COMM_WORLD
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    diag_blocksize = L_diagonal_blocks_local.shape[1]
    n_diag_blocks_local = L_diagonal_blocks_local.shape[0]

    X_diagonal_blocks_local = L_diagonal_blocks_local
    X_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local
    X_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local
    X_arrow_tip_block_global = L_arrow_tip_block_global

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
            (2, *B_permutation_upper.shape[1:]),
            dtype=B_permutation_upper.dtype,
        )

        X_upper_nested_dissection_buffer_d = L_upper_nested_dissection_buffer_d

        L_upper_nested_dissection_buffer_d_i = cp.empty_like(
            B_permutation_upper[0, :, :]
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
            arr=B_permutation_upper[-1, :, :], stream=h2d_stream
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
                arr=B_permutation_upper[i, :, :], stream=h2d_stream
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
                L_upper_nested_dissection_buffer_d_i[
                    :, :
                ] = L_upper_nested_dissection_buffer_d[i % 2, :, :]

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
                out=B_permutation_upper[i, :, :],
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
                B_permutation_upper[1, :, :].conj().T
            )

    cp.cuda.Device().synchronize()

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    )
