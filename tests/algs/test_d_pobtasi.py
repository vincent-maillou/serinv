# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import pytest
from mpi4py import MPI

from serinv import SolverConfig
from serinv.algs import d_pobtaf, d_pobtasi, d_pobtasi_rss

try:
    import cupy as cp
    import cupyx as cpx

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

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

# cuda_aware = CUPY_AVAIL and device_array and not device_streaming
cuda_aware = False
use_nccl = True
nested_solving = False
if NCCL_AVAIL:
    from cupy.cuda import nccl
    comm_id = nccl.get_unique_id()
    comm_id = MPI.COMM_WORLD.bcast(comm_id, root=0)
    nccl_comm = nccl.NcclCommunicator(comm_size, comm_id, comm_rank)
    cp.cuda.runtime.deviceSynchronize()
    if nested_solving:
        reduced_size = comm_size // 2
        reduced_rank = comm_rank
        reduced_color = int(comm_rank < reduced_size)
        reduced_key = comm_rank
        reduced_comm_id = nccl.get_unique_id()
        reduced_comm_id = MPI.COMM_WORLD.bcast(reduced_comm_id, root=0)
        if reduced_color == 1:
            nccl_reduced_comm = nccl.NcclCommunicator(reduced_size, reduced_comm_id, reduced_rank)
        else:
            nccl_reduced_comm = None
        cp.cuda.runtime.deviceSynchronize()
    else:
        nccl_reduced_comm = None
else:
    use_nccl = False


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("device_streaming", [False, True])
# @pytest.mark.parametrize("diagonal_blocksize", [2])
# @pytest.mark.parametrize("arrowhead_blocksize", [2])
# @pytest.mark.parametrize("n_diag_blocks", [comm_size * 3])
# @pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
# @pytest.mark.parametrize("dtype", [np.float64, np.complex128])
# @pytest.mark.parametrize("device_streaming", [False])
def test_d_pobtasi(
    dd_bta,
    bta_dense_to_arrays,
    bta_symmetrize,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    device_streaming,
):
    if CUPY_AVAIL:
        xp = cp.get_array_module(dd_bta)
    else:
        xp = np

    # Input matrix
    A = bta_symmetrize(dd_bta)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        _,
        A_arrow_bottom_blocks,
        _,
        A_arrow_tip_block_global,
    ) = bta_dense_to_arrays(A, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks)

    # Save the local slice of the array for each MPI process
    n_diag_blocks_per_processes = n_diag_blocks // comm_size
    A_diagonal_blocks_local = A_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if comm_rank == comm_size - 1:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            :,
            :,
        ]
    else:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
            :,
            :,
        ]

    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if CUPY_AVAIL and device_streaming and not device_array:
        A_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(A_diagonal_blocks_local)
        A_diagonal_blocks_local_pinned[:, :, :] = A_diagonal_blocks_local[:, :, :]
        A_lower_diagonal_blocks_local_pinned = cpx.zeros_like_pinned(
            A_lower_diagonal_blocks_local
        )
        A_lower_diagonal_blocks_local_pinned[:, :, :] = A_lower_diagonal_blocks_local[
            :, :, :
        ]
        A_arrow_bottom_blocks_local_pinned = cpx.zeros_like_pinned(
            A_arrow_bottom_blocks_local
        )
        A_arrow_bottom_blocks_local_pinned[:, :, :] = A_arrow_bottom_blocks_local[
            :, :, :
        ]
        A_arrow_tip_block_global_pinned = cpx.zeros_like_pinned(
            A_arrow_tip_block_global
        )
        A_arrow_tip_block_global_pinned[:, :] = A_arrow_tip_block_global[:, :]

        A_diagonal_blocks_local = A_diagonal_blocks_local_pinned
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local_pinned
        A_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local_pinned
        A_arrow_tip_block_global = A_arrow_tip_block_global_pinned

    # Reference solution
    X_ref = xp.linalg.inv(A)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        _,
        X_ref_arrow_bottom_blocks,
        _,
        X_ref_arrow_tip_block_global,
    ) = bta_dense_to_arrays(
        X_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    X_ref_diagonal_blocks_local = X_ref_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    if comm_rank == comm_size - 1:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            :,
            :,
        ]
    else:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
            :,
            :,
        ]

    X_ref_arrow_bottom_blocks_local = X_ref_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
        :,
        :,
    ]

    # SerinV solver configuration
    if device_array and use_nccl:
        comm = nccl_comm
        reduced_comm = nccl_reduced_comm
    else:
        comm = MPI.COMM_WORLD
        reduced_comm = None
    solver_config = SolverConfig(device_streaming=device_streaming, cuda_aware_mpi=cuda_aware, nccl=use_nccl, nested_solving=nested_solving)

    # Distributed factorization
    (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    ) = d_pobtaf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_tip_block_global,
        solver_config,
        comm,
    )

    # Distributed selected-inversion
    # Inversion of the reduced system
    (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    ) = d_pobtasi_rss(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        solver_config,
        comm,
        reduced_comm,
    )

    # Inversion of the full system
    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_tip_block_global,
    ) = d_pobtasi(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
        solver_config,
        comm,
    )

    assert xp.allclose(X_diagonal_blocks_local, X_ref_diagonal_blocks_local)
    assert xp.allclose(
        X_lower_diagonal_blocks_local,
        X_ref_lower_diagonal_blocks_local,
    )
    assert xp.allclose(
        X_arrow_bottom_blocks_local,
        X_ref_arrow_bottom_blocks_local,
    )
    assert xp.allclose(X_arrow_tip_block_global, X_ref_arrow_tip_block_global)

    if device_array:
        assert X_diagonal_blocks_local.data == A_diagonal_blocks_local.data
        assert X_lower_diagonal_blocks_local.data == A_lower_diagonal_blocks_local.data
        assert X_arrow_bottom_blocks_local.data == A_arrow_bottom_blocks_local.data
        assert X_arrow_tip_block_global.data == A_arrow_tip_block_global.data
    else:
        assert (
            X_diagonal_blocks_local.ctypes.data == A_diagonal_blocks_local.ctypes.data
        )
        assert (
            X_lower_diagonal_blocks_local.ctypes.data
            == A_lower_diagonal_blocks_local.ctypes.data
        )
        assert (
            X_arrow_bottom_blocks_local.ctypes.data
            == A_arrow_bottom_blocks_local.ctypes.data
        )
        assert (
            X_arrow_tip_block_global.ctypes.data == A_arrow_tip_block_global.ctypes.data
        )
