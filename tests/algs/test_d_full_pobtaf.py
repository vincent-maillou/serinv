# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import pytest
from mpi4py import MPI

from serinv import SolverConfig
from serinv.algs import d_pobtaf, d_full_pobtaf, pobtaf

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
# @pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("diagonal_blocksize", [3])
# @pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2])
# @pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
@pytest.mark.parametrize("n_diag_blocks", [16])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64])  # , np.complex128])
@pytest.mark.parametrize("device_streaming", [False])  # , True])
def test_d_pobtaf(
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
    log_det_sign_ref, log_det_val_ref = xp.linalg.slogdet(A)

    if comm_rank == 0:
        print(f"Reference log det: {log_det_sign_ref * log_det_val_ref}")

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

    if n_diag_blocks % comm_size != 0:
        raise ValueError("n_diag_blocks must be divisible by comm_size")

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

    # SerinV solver configuration
    solver_config = SolverConfig(device_streaming=device_streaming)

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
    )

    # # Create a reduced system out of the factorized blocks
    # A_reduced_system_diagonal_blocks = np.zeros(
    #     (2 * comm_size - 1, *A_diagonal_blocks_local.shape[1:]), dtype=A.dtype
    # )
    # A_reduced_system_lower_diagonal_blocks = np.zeros(
    #     (2 * comm_size - 2, *A_lower_diagonal_blocks_local.shape[1:]), dtype=A.dtype
    # )
    # A_reduced_system_arrow_bottom_blocks = np.zeros(
    #     (2 * comm_size - 1, *A_arrow_bottom_blocks_local.shape[1:]), dtype=A.dtype
    # )
    # A_reduced_system_arrow_tip_block = np.zeros_like(A_arrow_tip_block_global)

    # if comm_rank == 0:
    #     # Top process storing reduced blocks in reduced system
    #     A_reduced_system_diagonal_blocks[0, :, :] = L_diagonal_blocks_local[-1, :, :]
    #     A_reduced_system_lower_diagonal_blocks[0, :, :] = L_lower_diagonal_blocks_local[
    #         -1, :, :
    #     ]
    #     A_reduced_system_arrow_bottom_blocks[0, :, :] = L_arrow_bottom_blocks_local[
    #         -1, :, :
    #     ]
    # else:
    #     # Middle processes storing reduced blocks in reduced system
    #     A_reduced_system_diagonal_blocks[2 * comm_rank - 1, :, :] = (
    #         L_diagonal_blocks_local[0, :, :]
    #     )
    #     A_reduced_system_diagonal_blocks[2 * comm_rank, :, :] = L_diagonal_blocks_local[
    #         -1, :, :
    #     ]

    #     A_reduced_system_lower_diagonal_blocks[2 * comm_rank - 1, :, :] = (
    #         L_upper_nested_dissection_buffer_local[-1, :, :].conj().T
    #     )
    #     if comm_rank < comm_size - 1:
    #         A_reduced_system_lower_diagonal_blocks[2 * comm_rank, :, :] = (
    #             L_lower_diagonal_blocks_local[-1, :, :]
    #         )

    #     A_reduced_system_arrow_bottom_blocks[2 * comm_rank - 1, :, :] = (
    #         L_arrow_bottom_blocks_local[0, :, :]
    #     )
    #     A_reduced_system_arrow_bottom_blocks[2 * comm_rank, :, :] = (
    #         L_arrow_bottom_blocks_local[-1, :, :]
    #     )

    # A_reduced_system_arrow_tip_block[:, :] = L_arrow_tip_block_global[:, :]

    # MPI.COMM_WORLD.Barrier()

    # MPI.COMM_WORLD.Allreduce(
    #     MPI.IN_PLACE,
    #     A_reduced_system_diagonal_blocks,
    #     op=MPI.SUM,
    # )
    # MPI.COMM_WORLD.Allreduce(
    #     MPI.IN_PLACE,
    #     A_reduced_system_lower_diagonal_blocks,
    #     op=MPI.SUM,
    # )
    # MPI.COMM_WORLD.Allreduce(
    #     MPI.IN_PLACE,
    #     A_reduced_system_arrow_bottom_blocks,
    #     op=MPI.SUM,
    # )

    # ## default pobtaf overwrites the input arrays, not very clear
    # # me no like at all
    # (
    #     L_rs_diagonal_blocks_serinv,
    #     L_rs_lower_diagonal_blocks_serinv,
    #     L_rs_arrow_bottom_blocks_serinv,
    #     L_rs_arrow_tip_block_serinv,
    # ) = pobtaf(
    #     A_reduced_system_diagonal_blocks,
    #     A_reduced_system_lower_diagonal_blocks,
    #     A_reduced_system_arrow_bottom_blocks,
    #     A_reduced_system_arrow_tip_block,
    # )

    (
        L_rs_diagonal_blocks_serinv,
        L_rs_lower_diagonal_blocks_serinv,
        L_rs_arrow_bottom_blocks_serinv,
        L_rs_arrow_tip_block_serinv,
    ) = d_full_pobtaf(
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_arrow_tip_block_global,
        L_upper_nested_dissection_buffer_local,
    )

    ################################################################################
    # compute local log dets
    log_det_local = 0.0
    # distributed part
    if comm_rank == 0:
        for i in range(n_diag_blocks_per_processes - 1):
            for j in range(diagonal_blocksize):
                log_det_local += 2 * np.log(L_diagonal_blocks_local[i, j, j])

    else:
        for i in range(1, n_diag_blocks_per_processes - 1):
            for j in range(diagonal_blocksize):
                log_det_local += 2 * np.log(L_diagonal_blocks_local[i, j, j])

    if comm_rank == 0:
        for i in range(2 * comm_size - 1):
            for j in range(diagonal_blocksize):
                log_det_local += 2 * np.log(L_rs_diagonal_blocks_serinv[i, j, j])

        # Tip of the arrow
        for j in range(arrowhead_blocksize):
            log_det_local += 2 * np.log(L_rs_arrow_tip_block_serinv[j, j])

    MPI.COMM_WORLD.Barrier()
    # MPI sum together log determinant
    log_det = MPI.COMM_WORLD.allreduce(log_det_local, op=MPI.SUM)
    ################################################################################

    # reference log det
    # log_det_sign_ref, log_det_val_ref = xp.linalg.slogdet(A)

    if comm_rank == 0:
        print(f"distributed log det: {log_det}")
        print(
            f"norm(distributed_log_det - reference_log_det): {np.abs(log_det - log_det_sign_ref * log_det_val_ref):.6f}"
        )

    # put results back into arrays

    # generate correct permutation matrix

    # Check the result
    # print("L_rs_diagonal_blocks_serinv", L_rs_diagonal_blocks_serinv)
    assert xp.allclose(log_det, log_det_sign_ref * log_det_val_ref)
