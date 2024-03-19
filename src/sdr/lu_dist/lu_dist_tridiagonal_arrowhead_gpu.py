"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cpla
    from mpi4py import MPI
except ImportError:
    pass

import numpy as np

import time

from sdr.lu.lu_factorize_gpu import lu_factorize_tridiag_arrowhead_gpu
from sdr.lu.lu_selected_inversion_gpu import lu_sinv_tridiag_arrowhead_gpu


def lu_dist_tridiagonal_arrowhead_gpu(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_bridges_upper: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Perform the distributed LU factorization and selected inversion of a
    block tridiagonal arrowhead matrix.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local par of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.
    A_bridges_lower : np.ndarray
        Lower bridges.
    A_bridges_upper : np.ndarray
        Upper bridges.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global part of the arrow tip array of the inverse.
    X_bridges_lower : np.ndarray
        Lower bridges of the inverse. Only the one that belong to the local
        process are correct
    X_bridges_upper : np.ndarray
        Upper bridges of the inverse. Only the one that belong to the local
        process are correct

    Notes
    -----
    The algorithm use a non-pivoting LU factorization, hence the input matrix
    is considered diagonally dominant or block diagonally dominant.
    """
    timings: dict[str, float] = {}

    timings["t_mem"] = 0.0
    timings["t_lu"] = 0.0
    timings["t_trsm"] = 0.0
    timings["t_gemm"] = 0.0
    timings["t_comm"] = 0.0

    sections: dict[str, float] = {}

    t_factorize = 0.0

    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_partition = A_diagonal_blocks_local.shape[1] // diag_blocksize

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        t_factorize_start = time.perf_counter_ns()
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            Update_arrow_tip,
            A_diagonal_blocks_local_updated,
            A_arrow_bottom_blocks_local_updated,
            A_arrow_right_blocks_local_updated,
            timings_factorize,
        ) = top_factorize_gpu(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_arrow_tip_block,
        )
        t_factorize_stop = time.perf_counter_ns()
        t_factorize = t_factorize_stop - t_factorize_start

        t_reduced_system_start = time.perf_counter_ns()
        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_right_blocks,
            A_rs_arrow_tip_block,
            timings_reduced_system,
        ) = create_reduced_system(
            A_diagonal_blocks_local_updated,
            A_arrow_bottom_blocks_local_updated,
            A_arrow_right_blocks_local_updated,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_bridges_upper,
        )
        t_reduced_system_stop = time.perf_counter_ns()
        t_reduced_system = t_reduced_system_stop - t_reduced_system_start
    else:
        t_mem_start = time.perf_counter_ns()
        # Arrays that store the update of the 2sided pattern for the middle processes
        A_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )
        A_left_2sided_arrow_blocks_local = np.empty(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )

        A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_top_2sided_arrow_blocks_local[:, diag_blocksize : 2 * diag_blocksize] = (
            A_upper_diagonal_blocks_local[:, :diag_blocksize]
        )

        A_left_2sided_arrow_blocks_local[:diag_blocksize, :] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_left_2sided_arrow_blocks_local[diag_blocksize : 2 * diag_blocksize, :] = (
            A_lower_diagonal_blocks_local[:, :diag_blocksize]
        )
        t_mem_stop = time.perf_counter_ns()
        t_mem = t_mem_stop - t_mem_start

        timings["t_mem"] += t_mem

        t_factorize_start = time.perf_counter_ns()
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            U_left_2sided_arrow_blocks,
            Update_arrow_tip,
            A_diagonal_blocks_local_updated,
            A_arrow_bottom_blocks_local_updated,
            A_arrow_right_blocks_local_updated,
            A_top_2sided_arrow_blocks_local_updated,
            A_left_2sided_arrow_blocks_local_updated,
            timings_factorize,
        ) = middle_factorize_gpu(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_top_2sided_arrow_blocks_local,
            A_left_2sided_arrow_blocks_local,
            A_arrow_tip_block,
        )
        t_factorize_stop = time.perf_counter_ns()
        t_factorize = t_factorize_stop - t_factorize_start

        t_reduced_system_start = time.perf_counter_ns()
        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_upper_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_right_blocks,
            A_rs_arrow_tip_block,
            timings_reduced_system,
        ) = create_reduced_system(
            A_diagonal_blocks_local_updated,
            A_arrow_bottom_blocks_local_updated,
            A_arrow_right_blocks_local_updated,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_bridges_upper,
            A_top_2sided_arrow_blocks_local_updated,
            A_left_2sided_arrow_blocks_local_updated,
        )
        t_reduced_system_stop = time.perf_counter_ns()
        t_reduced_system = t_reduced_system_stop - t_reduced_system_start

    timings["t_mem"] += timings_factorize["t_mem"]
    timings["t_lu"] += timings_factorize["t_lu"]
    timings["t_trsm"] += timings_factorize["t_trsm"]
    timings["t_gemm"] += timings_factorize["t_gemm"]

    timings["t_mem"] += timings_reduced_system["t_mem"]
    timings["t_comm"] += timings_reduced_system["t_comm"]

    sections["t_factorize"] = t_factorize
    sections["t_reduced_system"] = t_reduced_system

    t_sinv_start = time.perf_counter_ns()
    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
        timings_sinv,
    ) = inverse_reduced_system_gpu(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_upper_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_right_blocks,
        A_rs_arrow_tip_block,
    )
    t_sinv_stop = time.perf_counter_ns()
    t_sinv = t_sinv_stop - t_sinv_start

    timings["t_mem"] += timings_sinv["t_mem"]
    timings["t_lu"] += timings_sinv["t_lu"]
    timings["t_trsm"] += timings_sinv["t_trsm"]
    timings["t_gemm"] += timings_sinv["t_gemm"]

    sections["t_sinv"] = t_sinv

    t_update_reduced_system_start = time.perf_counter_ns()
    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_left_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
        timings_update_reduced_system,
    ) = update_sinv_reduced_system(
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
        n_diag_blocks_partition,
        diag_blocksize,
        arrow_blocksize,
    )
    t_update_reduced_system_stop = time.perf_counter_ns()
    t_update_reduced_system = (
        t_update_reduced_system_stop - t_update_reduced_system_start
    )

    timings["t_mem"] += timings_update_reduced_system["t_mem"]

    sections["t_update_reduced_system"] = t_update_reduced_system

    t_sinv_start = time.perf_counter_ns()
    if comm_rank == 0:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            _,
            timings_sinv,
        ) = top_sinv_gpu(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
        )
    else:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            _,
            timings_sinv,
        ) = middle_sinv_gpu(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_upper_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_arrow_right_blocks_local,
            X_top_2sided_arrow_blocks_local,
            X_left_2sided_arrow_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            U_left_2sided_arrow_blocks,
        )
    t_sinv_stop = time.perf_counter_ns()
    t_sinv = t_sinv_stop - t_sinv_start

    timings["t_mem"] += timings_sinv["t_mem"]
    timings["t_trsm"] += timings_sinv["t_trsm"]
    timings["t_gemm"] += timings_sinv["t_gemm"]

    sections["t_sinv"] = t_sinv

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
        timings,
        sections,
    )


def top_factorize_gpu(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Proceed to the top-down LU factorization, called by the first process.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.

    Returns
    -------
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local L factor.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the local L factor.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the local L factor.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local U factor.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the local U factor.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the local U factor.
    Update_arrow_tip_local : np.ndarray
        Local update of the arrow tip block.
    """
    timings_factorize: dict[str, float] = {}

    t_mem = 0.0
    t_lu = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    nblocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    A_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(A_diagonal_blocks_local)
    A_lower_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        A_lower_diagonal_blocks_local
    )
    A_upper_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        A_upper_diagonal_blocks_local
    )
    A_arrow_bottom_blocks_local_gpu: np.ndarray = cp.asarray(
        A_arrow_bottom_blocks_local
    )
    A_arrow_right_blocks_local_gpu: np.ndarray = cp.asarray(A_arrow_right_blocks_local)

    # Host side arrays
    A_diagonal_blocks_updated: np.ndarray = cpx.empty_pinned(
        (diag_blocksize, diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_arrow_bottom_blocks_updated: np.ndarray = cpx.empty_pinned(
        (diag_blocksize, diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_arrow_right_blocks_updated: np.ndarray = cpx.empty_pinned(
        (diag_blocksize, diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )

    L_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_lower_diagonal_blocks_local
    )
    L_arrow_bottom_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_arrow_bottom_blocks_local
    )

    U_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_local)
    U_upper_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_upper_diagonal_blocks_local
    )
    U_arrow_right_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_arrow_right_blocks_local
    )

    Update_arrow_tip_local: np.ndarray = cpx.empty_like_pinned(A_arrow_tip_block)

    # Device side arrays
    L_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(L_diagonal_blocks_local)
    L_lower_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        L_lower_diagonal_blocks_local
    )
    L_arrow_bottom_blocks_local_gpu: np.ndarray = cp.empty_like(
        L_arrow_bottom_blocks_local
    )

    U_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(U_diagonal_blocks_local)
    U_upper_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        U_upper_diagonal_blocks_local
    )
    U_arrow_right_blocks_local_gpu: np.ndarray = cp.empty_like(
        U_arrow_right_blocks_local
    )

    Update_arrow_tip_local_gpu: np.ndarray = cp.zeros_like(
        Update_arrow_tip_local
    )  # Have to be zero-initialized
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(nblocks - 1):
        t_lu_start = time.perf_counter_ns()
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
        ) = cpla.lu(
            A_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            permute_l=True,
        )
        stream.synchronize()
        t_lu_stop = time.perf_counter_ns()
        t_lu += t_lu_stop - t_lu_start

        t_trsm_start = time.perf_counter_ns()
        # Compute lower factors
        U_inv_temp_gpu = cpla.solve_triangular(
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # L_{i+1, i} = A_{i+1, i} @ U_local{i, i}^{-1}
        L_lower_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_trsm_start = time.perf_counter_ns()
        # Compute upper factors
        L_inv_temp_gpu = cpla.solve_triangular(
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=True,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # U_{i, i+1} = L_local{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp_gpu
            @ A_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = (
            L_inv_temp_gpu
            @ A_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_gemm_start = time.perf_counter_ns()
        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local_gpu[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local_gpu[:, :] = (
            Update_arrow_tip_local_gpu[:, :]
            - L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_lu_start = time.perf_counter_ns()
    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_local_gpu[:, -diag_blocksize:],
        U_diagonal_blocks_local_gpu[:, -diag_blocksize:],
    ) = cpla.lu(A_diagonal_blocks_local_gpu[:, -diag_blocksize:], permute_l=True)
    stream.synchronize()
    t_lu_stop = time.perf_counter_ns()
    t_lu += t_lu_stop - t_lu_start

    t_mem_start = time.perf_counter_ns()
    A_diagonal_blocks_updated = A_diagonal_blocks_local_gpu[:, -diag_blocksize:].get()
    A_arrow_bottom_blocks_updated = A_arrow_bottom_blocks_local_gpu[
        :, -diag_blocksize:
    ].get()
    A_arrow_right_blocks_updated = A_arrow_right_blocks_local_gpu[
        -diag_blocksize:, :
    ].get()

    L_diagonal_blocks_local = L_diagonal_blocks_local_gpu.get()
    L_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local_gpu.get()
    L_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local_gpu.get()

    U_diagonal_blocks_local = U_diagonal_blocks_local_gpu.get()
    U_upper_diagonal_blocks_local = U_upper_diagonal_blocks_local_gpu.get()
    U_arrow_right_blocks_local = U_arrow_right_blocks_local_gpu.get()

    Update_arrow_tip_local = Update_arrow_tip_local_gpu.get()
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings_factorize["t_mem"] = t_mem
    timings_factorize["t_lu"] = t_lu
    timings_factorize["t_trsm"] = t_trsm
    timings_factorize["t_gemm"] = t_gemm

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        U_diagonal_blocks_local,
        U_upper_diagonal_blocks_local,
        U_arrow_right_blocks_local,
        Update_arrow_tip_local,
        A_diagonal_blocks_updated,
        A_arrow_bottom_blocks_updated,
        A_arrow_right_blocks_updated,
        timings_factorize,
    )


def middle_factorize_gpu(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray,
    A_left_2sided_arrow_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Proceed to the 2-sided LU factorization, called by the middle processes.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array.
    A_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_top_2sided_arrow_blocks_local : np.ndarray
        Array that stores the top update of the 2sided pattern for the middle processes.
    A_left_2sided_arrow_blocks_local : np.ndarray
        Array that stores the left update of the 2sided pattern for the middle processes.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.

    Returns
    -------
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local L factor.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the local L factor.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the local L factor.
    L_upper_2sided_arrow_blocks_local : np.ndarray
        Upper 2sided arrow blocks of the local L factor.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the local U factor.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the local U factor.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the local U factor.
    U_left_2sided_arrow_blocks_local : np.ndarray
        Left 2sided arrow blocks of the local U factor.
    Update_arrow_tip_local : np.ndarray
        Local update of the arrow tip block.
    """
    timings_factorize: dict[str, float] = {}

    t_mem = 0.0
    t_lu = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_blocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    A_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(A_diagonal_blocks_local)
    A_lower_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        A_lower_diagonal_blocks_local
    )
    A_upper_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        A_upper_diagonal_blocks_local
    )
    A_arrow_bottom_blocks_local_gpu: np.ndarray = cp.asarray(
        A_arrow_bottom_blocks_local
    )
    A_arrow_right_blocks_local_gpu: np.ndarray = cp.asarray(A_arrow_right_blocks_local)
    A_top_2sided_arrow_blocks_local_gpu: np.ndarray = cp.asarray(
        A_top_2sided_arrow_blocks_local
    )
    A_left_2sided_arrow_blocks_local_gpu: np.ndarray = cp.asarray(
        A_left_2sided_arrow_blocks_local
    )

    # Host side arrays
    A_diagonal_blocks_local_updated = cpx.empty_pinned(
        (diag_blocksize, 2 * diag_blocksize), dtype=A_diagonal_blocks_local.dtype
    )

    A_arrow_bottom_blocks_local_updated = cpx.empty_pinned(
        (arrow_blocksize, 2 * diag_blocksize), dtype=A_arrow_bottom_blocks_local.dtype
    )

    A_arrow_right_blocks_local_updated = cpx.empty_pinned(
        (2 * diag_blocksize, arrow_blocksize), dtype=A_arrow_right_blocks_local.dtype
    )

    A_top_2sided_arrow_blocks_local_updated = cpx.empty_pinned(
        (diag_blocksize, 2 * diag_blocksize),
        dtype=A_top_2sided_arrow_blocks_local.dtype,
    )

    A_left_2sided_arrow_blocks_local_updated = cpx.empty_pinned(
        (2 * diag_blocksize, diag_blocksize),
        dtype=A_left_2sided_arrow_blocks_local.dtype,
    )

    L_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_diagonal_blocks_local_gpu
    )
    L_lower_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_lower_diagonal_blocks_local_gpu
    )
    L_arrow_bottom_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_arrow_bottom_blocks_local_gpu
    )
    L_upper_2sided_arrow_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_top_2sided_arrow_blocks_local_gpu
    )

    U_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_diagonal_blocks_local_gpu
    )
    U_upper_diagonal_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_upper_diagonal_blocks_local_gpu
    )
    U_arrow_right_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_arrow_right_blocks_local_gpu
    )
    U_left_2sided_arrow_blocks_local: np.ndarray = cpx.empty_like_pinned(
        A_left_2sided_arrow_blocks_local_gpu
    )

    Update_arrow_tip_local: np.ndarray = cpx.empty_like_pinned(A_arrow_tip_block)

    # Device side arrays
    L_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(L_diagonal_blocks_local)
    L_lower_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        L_lower_diagonal_blocks_local
    )
    L_arrow_bottom_blocks_local_gpu: np.ndarray = cp.empty_like(
        L_arrow_bottom_blocks_local
    )
    L_upper_2sided_arrow_blocks_local_gpu: np.ndarray = cp.empty_like(
        L_upper_2sided_arrow_blocks_local
    )

    U_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(U_diagonal_blocks_local)
    U_upper_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        U_upper_diagonal_blocks_local
    )
    U_arrow_right_blocks_local_gpu: np.ndarray = cp.empty_like(
        U_arrow_right_blocks_local
    )
    U_left_2sided_arrow_blocks_local_gpu: np.ndarray = cp.empty_like(
        U_left_2sided_arrow_blocks_local
    )

    Update_arrow_tip_local_gpu: np.ndarray = cp.zeros_like(
        Update_arrow_tip_local
    )  # Have to be zero-initialized
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(1, n_blocks - 1):
        t_lu_start = time.perf_counter_ns()
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
        ) = cpla.lu(
            A_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            permute_l=True,
        )
        stream.synchronize()
        t_lu_stop = time.perf_counter_ns()
        t_lu += t_lu_stop - t_lu_start

        t_trsm_start = time.perf_counter_ns()
        # Compute lower factors
        U_inv_temp_gpu = cpla.solve_triangular(
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )

        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        L_upper_2sided_arrow_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_top_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_trsm_start = time.perf_counter_ns()
        # Compute upper factors
        L_inv_temp_gpu = cpla.solve_triangular(
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=True,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp_gpu
            @ A_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
        U_left_2sided_arrow_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = (
            L_inv_temp_gpu
            @ A_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = (
            L_inv_temp_gpu
            @ A_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_gemm_start = time.perf_counter_ns()
        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local_gpu[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local_gpu[:, :] = (
            Update_arrow_tip_local_gpu[:, :]
            - L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
        A_diagonal_blocks_local_gpu[:, :diag_blocksize] = (
            A_diagonal_blocks_local_gpu[:, :diag_blocksize]
            - L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_{i+1, top} = - L_{i+1, i} @ U_{i, top}
        A_left_2sided_arrow_blocks_local_gpu[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            -L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_local[top, i+1] = - L[top, i] @ U_[i, i+1]
        A_top_2sided_arrow_blocks_local_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            -L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
        A_arrow_bottom_blocks_local_gpu[:, :diag_blocksize] = (
            A_arrow_bottom_blocks_local_gpu[:, :diag_blocksize]
            - L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )

        # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local_gpu[:diag_blocksize, :] = (
            A_arrow_right_blocks_local_gpu[:diag_blocksize, :]
            - L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_lu_start = time.perf_counter_ns()
    # Compute the last LU blocks of the 2-sided factorization pattern
    (
        L_diagonal_blocks_local_gpu[:, (n_blocks - 1) * diag_blocksize :],
        U_diagonal_blocks_local_gpu[:, (n_blocks - 1) * diag_blocksize :],
    ) = cpla.lu(
        A_diagonal_blocks_local[:, (n_blocks - 1) * diag_blocksize :], permute_l=True
    )
    stream.synchronize()
    t_lu_stop = time.perf_counter_ns()
    t_lu += t_lu_stop - t_lu_start

    t_trsm_start = time.perf_counter_ns()
    # Compute last lower factors
    U_inv_temp_gpu = cpla.solve_triangular(
        U_diagonal_blocks_local_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=False,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    # L_{top, nblocks} = A_{top, nblocks} @ U{nblocks, nblocks}^{-1}
    L_upper_2sided_arrow_blocks_local_gpu[:, -diag_blocksize:] = (
        A_top_2sided_arrow_blocks_local_gpu[:, -diag_blocksize:] @ U_inv_temp_gpu
    )

    # L_{ndb+1, nblocks} = A_{ndb+1, nblocks} @ U{nblocks, nblocks}^{-1}
    L_arrow_bottom_blocks_local_gpu[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks_local_gpu[:, -diag_blocksize:] @ U_inv_temp_gpu
    )
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_trsm_start = time.perf_counter_ns()
    # Compute last upper factors
    L_inv_temp_gpu = cpla.solve_triangular(
        L_diagonal_blocks_local_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=True,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    # U_{nblocks, top} = L{nblocks, nblocks}^{-1} @ A_{nblocks, top}
    U_left_2sided_arrow_blocks_local_gpu[-diag_blocksize:, :] = (
        L_inv_temp_gpu @ A_left_2sided_arrow_blocks_local_gpu[-diag_blocksize:, :]
    )

    # U_{nblocks, ndb+1} = L{nblocks, nblocks}^{-1} @ A_{nblocks, ndb+1}
    U_arrow_right_blocks_local_gpu[-diag_blocksize:, :] = (
        L_inv_temp_gpu @ A_arrow_right_blocks_local_gpu[-diag_blocksize:, :]
    )
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    # NOTE: On purpose, we don't update the tip of the arrowhead since the
    # propagation will appear during the inversion of the reduced system

    # Compute the top (first) LU blocks of the 2-sided factorization pattern
    # and its respective parts of the arrowhead
    t_lu_start = time.perf_counter_ns()
    # L_{top, top}, U_{top, top} = lu_dcmp(A_{top, top})
    (
        L_diagonal_blocks_local_gpu[:, :diag_blocksize],
        U_diagonal_blocks_local_gpu[:, :diag_blocksize],
    ) = cpla.lu(A_diagonal_blocks_local_gpu[:, :diag_blocksize], permute_l=True)
    stream.synchronize()
    t_lu_stop = time.perf_counter_ns()
    t_lu += t_lu_stop - t_lu_start

    t_trsm_start = time.perf_counter_ns()
    # Compute top lower factors
    U_inv_temp_gpu = cpla.solve_triangular(
        U_diagonal_blocks_local_gpu[:, :diag_blocksize],
        cp.eye(diag_blocksize),
        lower=False,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    # L_{top+1, top} = A_{top+1, top} @ U{top, top}^{-1}
    L_lower_diagonal_blocks_local_gpu[:, :diag_blocksize] = (
        A_lower_diagonal_blocks_local_gpu[:, :diag_blocksize] @ U_inv_temp_gpu
    )

    # L_{ndb+1, top} = A_{ndb+1, top} @ U{top, top}^{-1}
    L_arrow_bottom_blocks_local_gpu[:, :diag_blocksize] = (
        A_arrow_bottom_blocks_local_gpu[:, :diag_blocksize] @ U_inv_temp_gpu
    )
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_trsm_start = time.perf_counter_ns()
    # Compute top upper factors
    L_inv_temp_gpu = cpla.solve_triangular(
        L_diagonal_blocks_local_gpu[:, :diag_blocksize],
        cp.eye(diag_blocksize),
        lower=True,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    # U_{top, top+1} = L{top, top}^{-1} @ A_{top, top+1}
    U_upper_diagonal_blocks_local_gpu[:, :diag_blocksize] = (
        L_inv_temp_gpu @ A_upper_diagonal_blocks_local_gpu[:, :diag_blocksize]
    )

    # U_{top, ndb+1} = L{top, top}^{-1} @ A_{top, ndb+1}
    U_arrow_right_blocks_local_gpu[:diag_blocksize, :] = (
        L_inv_temp_gpu @ A_arrow_right_blocks_local_gpu[:diag_blocksize, :]
    )
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_mem_start = time.perf_counter_ns()
    A_diagonal_blocks_local_updated[:, :diag_blocksize] = A_diagonal_blocks_local_gpu[
        :, :diag_blocksize
    ].get()
    A_diagonal_blocks_local_updated[:, -diag_blocksize:] = A_diagonal_blocks_local_gpu[
        :, -diag_blocksize:
    ].get()

    A_arrow_bottom_blocks_local_updated[:, :diag_blocksize] = (
        A_arrow_bottom_blocks_local_gpu[:, :diag_blocksize].get()
    )
    A_arrow_bottom_blocks_local_updated[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks_local_gpu[:, -diag_blocksize:].get()
    )

    A_arrow_right_blocks_local_updated[:diag_blocksize, :] = (
        A_arrow_right_blocks_local_gpu[:diag_blocksize, :].get()
    )
    A_arrow_right_blocks_local_updated[-diag_blocksize:, :] = (
        A_arrow_right_blocks_local_gpu[-diag_blocksize:, :].get()
    )

    A_top_2sided_arrow_blocks_local_updated[:, :diag_blocksize] = (
        A_top_2sided_arrow_blocks_local_gpu[:, :diag_blocksize].get()
    )
    A_top_2sided_arrow_blocks_local_updated[:, -diag_blocksize:] = (
        A_top_2sided_arrow_blocks_local_gpu[:, -diag_blocksize:].get()
    )

    A_left_2sided_arrow_blocks_local_updated[:diag_blocksize, :] = (
        A_left_2sided_arrow_blocks_local_gpu[:diag_blocksize, :].get()
    )
    A_left_2sided_arrow_blocks_local_updated[-diag_blocksize:, :] = (
        A_left_2sided_arrow_blocks_local_gpu[-diag_blocksize:, :].get()
    )

    L_diagonal_blocks_local = L_diagonal_blocks_local_gpu.get()
    L_lower_diagonal_blocks_local = L_lower_diagonal_blocks_local_gpu.get()
    L_arrow_bottom_blocks_local = L_arrow_bottom_blocks_local_gpu.get()
    L_upper_2sided_arrow_blocks_local = L_upper_2sided_arrow_blocks_local_gpu.get()

    U_diagonal_blocks_local = U_diagonal_blocks_local_gpu.get()
    U_upper_diagonal_blocks_local = U_upper_diagonal_blocks_local_gpu.get()
    U_arrow_right_blocks_local = U_arrow_right_blocks_local_gpu.get()
    U_left_2sided_arrow_blocks_local = U_left_2sided_arrow_blocks_local_gpu.get()

    Update_arrow_tip_local = Update_arrow_tip_local_gpu.get()
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings_factorize["t_mem"] = t_mem
    timings_factorize["t_lu"] = t_lu
    timings_factorize["t_trsm"] = t_trsm
    timings_factorize["t_gemm"] = t_gemm

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_upper_2sided_arrow_blocks_local,
        U_diagonal_blocks_local,
        U_upper_diagonal_blocks_local,
        U_arrow_right_blocks_local,
        U_left_2sided_arrow_blocks_local,
        Update_arrow_tip_local,
        A_diagonal_blocks_local_updated,
        A_arrow_bottom_blocks_local_updated,
        A_arrow_right_blocks_local_updated,
        A_top_2sided_arrow_blocks_local_updated,
        A_left_2sided_arrow_blocks_local_updated,
        timings_factorize,
    )


def create_reduced_system(
    A_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    Update_arrow_tip: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_bridges_upper: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray = None,
    A_left_2sided_arrow_blocks_local: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the reduced system and broadcast it to all the processes.

    Parameters
    ----------
    A_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array.
    A_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array.
    A_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array.
    A_arrow_tip_block : np.ndarray
        Tip block of the arrow.
    Update_arrow_tip : np.ndarray
        Update of the arrow tip block.
    A_bridges_lower : np.ndarray
        Lower part of the bridges array.
    A_bridges_upper : np.ndarray
        Upper part of the bridges array.
    A_top_2sided_arrow_blocks_local : np.ndarray, optional
        Array that stores the top update of the 2sided pattern for the middle processes.
    A_left_2sided_arrow_blocks_local : np.ndarray, optional
        Array that stores the left update of the 2sided pattern for the middle processes.

    Returns
    -------
    A_rs_diagonal_blocks_sum : np.ndarray
        Diagonal blocks of the reduced system.
    A_rs_lower_diagonal_blocks_sum : np.ndarray
        Lower diagonal blocks of the reduced system.
    A_rs_upper_diagonal_blocks_sum : np.ndarray
        Upper diagonal blocks of the reduced system.
    A_rs_arrow_bottom_blocks_sum : np.ndarray
        Arrow bottom blocks of the reduced system.
    A_rs_arrow_right_blocks_sum : np.ndarray
        Arrow right blocks of the reduced system.
    A_rs_arrow_tip_block_sum : np.ndarray
        Tip block of the reduced system.

    Notes
    -----
    This function represent the parallel bottleneck of the algorithm. It uses
    the MPI_Allreduce operation to sum the reduced system of each process and
    broadcast it to all the processes. After this communication step no more
    communication is needed.
    """
    timings_reduced_system: dict[str, float] = {}

    t_mem = 0.0
    t_comm = 0.0

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    t_mem_start = time.perf_counter_ns()
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create empty matrix for reduced system -> (2 * #process - 1) * diag_blocksize + arrowhead_size
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_reduced_system = 2 * comm_size - 1

    A_rs_diagonal_blocks = np.zeros(
        (diag_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )  # Have to be zero-initialized
    A_rs_lower_diagonal_blocks = np.zeros(
        (diag_blocksize, (n_diag_blocks_reduced_system - 1) * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )  # Have to be zero-initialized
    A_rs_upper_diagonal_blocks = np.zeros(
        (diag_blocksize, (n_diag_blocks_reduced_system - 1) * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )  # Have to be zero-initialized
    A_rs_arrow_bottom_blocks = np.zeros(
        (arrow_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )  # Have to be zero-initialized
    A_rs_arrow_right_blocks = np.zeros(
        (n_diag_blocks_reduced_system * diag_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )  # Have to be zero-initialized
    A_rs_arrow_tip_block = np.empty(
        (arrow_blocksize, arrow_blocksize), dtype=A_diagonal_blocks_local.dtype
    )

    A_rs_arrow_tip_block = Update_arrow_tip

    if comm_rank == 0:
        A_rs_diagonal_blocks[:, :diag_blocksize] = A_diagonal_blocks_local[:, :]
        A_rs_upper_diagonal_blocks[:, :diag_blocksize] = A_bridges_upper[
            :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
        ]

        A_rs_arrow_bottom_blocks[:, :diag_blocksize] = A_arrow_bottom_blocks_local[:, :]
        A_rs_arrow_right_blocks[:diag_blocksize, :] = A_arrow_right_blocks_local[:, :]
    else:
        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        A_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index] = (
            A_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ]
        )

        A_rs_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
        )

        A_rs_upper_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_top_2sided_arrow_blocks_local[:, -diag_blocksize:]
        )

        A_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_left_2sided_arrow_blocks_local[-diag_blocksize:, :]
        )

        A_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_diagonal_blocks_local[:, -diag_blocksize:]

        if comm_rank != comm_size - 1:
            A_rs_upper_diagonal_blocks[
                :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
            ] = A_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ]

        A_rs_arrow_bottom_blocks[:, start_index : start_index + diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
        )

        A_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_arrow_bottom_blocks_local[:, -diag_blocksize:]

        A_rs_arrow_right_blocks[start_index : start_index + diag_blocksize, :] = (
            A_arrow_right_blocks_local[:diag_blocksize, :]
        )

        A_rs_arrow_right_blocks[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, :
        ] = A_arrow_right_blocks_local[-diag_blocksize:, :]

    # Send the reduced_system with MPIallReduce SUM operation
    A_rs_diagonal_blocks_sum = np.zeros_like(A_rs_diagonal_blocks)
    A_rs_lower_diagonal_blocks_sum = np.zeros_like(A_rs_lower_diagonal_blocks)
    A_rs_upper_diagonal_blocks_sum = np.zeros_like(A_rs_upper_diagonal_blocks)
    A_rs_arrow_bottom_blocks_sum = np.zeros_like(A_rs_arrow_bottom_blocks)
    A_rs_arrow_right_blocks_sum = np.zeros_like(A_rs_arrow_right_blocks)
    A_rs_arrow_tip_block_sum = np.zeros_like(A_rs_arrow_tip_block)
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    comm.Barrier()
    t_comm_start = time.perf_counter_ns()
    comm.Allreduce(
        [A_rs_diagonal_blocks, MPI.DOUBLE],
        [A_rs_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_lower_diagonal_blocks, MPI.DOUBLE],
        [A_rs_lower_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_upper_diagonal_blocks, MPI.DOUBLE],
        [A_rs_upper_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_bottom_blocks, MPI.DOUBLE],
        [A_rs_arrow_bottom_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_right_blocks, MPI.DOUBLE],
        [A_rs_arrow_right_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_tip_block, MPI.DOUBLE],
        [A_rs_arrow_tip_block_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    t_comm_stop = time.perf_counter_ns()
    t_comm = t_comm_stop - t_comm_start

    t_mem_start = time.perf_counter_ns()
    # Add the global arrow tip to the reduced system arrow-tip summation
    A_rs_arrow_tip_block_sum += A_arrow_tip_block
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings_reduced_system["t_mem"] = t_mem
    timings_reduced_system["t_comm"] = t_comm

    return (
        A_rs_diagonal_blocks_sum,
        A_rs_lower_diagonal_blocks_sum,
        A_rs_upper_diagonal_blocks_sum,
        A_rs_arrow_bottom_blocks_sum,
        A_rs_arrow_right_blocks_sum,
        A_rs_arrow_tip_block_sum,
        timings_reduced_system,
    )


def inverse_reduced_system_gpu(
    A_rs_diagonal_blocks: np.ndarray,
    A_rs_lower_diagonal_blocks: np.ndarray,
    A_rs_upper_diagonal_blocks: np.ndarray,
    A_rs_arrow_bottom_blocks: np.ndarray,
    A_rs_arrow_right_blocks: np.ndarray,
    A_rs_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert the reduced system using a sequential selected inversion algorithm.

    Parameters
    ----------
    A_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the reduced system.
    A_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the reduced system.
    A_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the reduced system.
    A_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the reduced system.
    A_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the reduced system.
    A_rs_arrow_tip_block : np.ndarray
        Tip block of the reduced system.

    Returns
    -------
    X_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the inverse of the reduced system.
    X_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the inverse of the reduced system.
    X_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the inverse of the reduced system.
    X_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the inverse of the reduced system.
    X_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the inverse of the reduced system.
    X_rs_arrow_tip_block : np.ndarray
        Tip block of the inverse of the reduced system.

    Notes
    -----
    The inversion of the reduced system is performed using a sequential
    selected inversion algorithm.
    """
    timings_sinv: dict[str, float] = {}

    timings_sinv["t_mem"] = 0.0
    timings_sinv["t_lu"] = 0.0
    timings_sinv["t_trsm"] = 0.0
    timings_sinv["t_gemm"] = 0.0

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        timings_factorize_tridiag_arrowhead,
    ) = lu_factorize_tridiag_arrowhead_gpu(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_upper_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_right_blocks,
        A_rs_arrow_tip_block,
    )

    timings_sinv["t_mem"] += timings_factorize_tridiag_arrowhead["mem"]
    timings_sinv["t_lu"] += timings_factorize_tridiag_arrowhead["lu"]
    timings_sinv["t_trsm"] += timings_factorize_tridiag_arrowhead["trsm"]
    timings_sinv["t_gemm"] += timings_factorize_tridiag_arrowhead["gemm"]

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
        timings_sinv_tridiag_arrowhead,
    ) = lu_sinv_tridiag_arrowhead_gpu(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
    )

    timings_sinv["t_mem"] += timings_sinv_tridiag_arrowhead["mem"]
    timings_sinv["t_trsm"] += timings_sinv_tridiag_arrowhead["trsm"]
    timings_sinv["t_gemm"] += timings_sinv_tridiag_arrowhead["gemm"]

    return (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_upper_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_right_blocks,
        X_rs_arrow_tip_block,
        timings_sinv,
    )


def update_sinv_reduced_system(
    X_rs_diagonal_blocks: np.ndarray,
    X_rs_lower_diagonal_blocks: np.ndarray,
    X_rs_upper_diagonal_blocks: np.ndarray,
    X_rs_arrow_bottom_blocks: np.ndarray,
    X_rs_arrow_right_blocks: np.ndarray,
    X_rs_arrow_tip_block: np.ndarray,
    n_diag_blocks_partition: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fill the local parts of the inverse with the suited blocks of the
    inverted reduced system.

    Parameters
    ----------
    X_rs_diagonal_blocks : np.ndarray
        Diagonal blocks of the inverse of the reduced system.
    X_rs_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the inverse of the reduced system.
    X_rs_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the inverse of the reduced system.
    X_rs_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the inverse of the reduced system.
    X_rs_arrow_right_blocks : np.ndarray
        Arrow right blocks of the inverse of the reduced system.
    X_rs_arrow_tip_block : np.ndarray
        Tip block of the inverse of the reduced system.
    n_diag_blocks_partition : int
        Number of diagonal blocks in the partition.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_top_2sided_arrow_blocks_local : np.ndarray
        2sided pattern array storing corner blocks of the inverse, for the middle processes.
    X_left_2sided_arrow_blocks_local : np.ndarray
        2sided pattern array storing corner blocks of the inverse, for the middle processes.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    X_bridges_lower : np.ndarray
        Lower part of the bridges array of the inverse.
    X_bridges_upper : np.ndarray
        Upper part of the bridges array of the inverse.
    """
    timings_update_reduced_system: dict[str, float] = {}

    t_mem = 0.0

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    t_mem_start = time.perf_counter_ns()
    X_diagonal_blocks_local = cpx.empty_pinned(
        (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_lower_diagonal_blocks_local = cpx.empty_pinned(
        (diag_blocksize, (n_diag_blocks_partition - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_upper_diagonal_blocks_local = cpx.empty_pinned(
        (diag_blocksize, (n_diag_blocks_partition - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_arrow_bottom_blocks_local = cpx.empty_pinned(
        (arrow_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_arrow_right_blocks_local = cpx.empty_pinned(
        (n_diag_blocks_partition * diag_blocksize, arrow_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    X_bridges_upper = np.empty(
        (diag_blocksize, (comm_size - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_bridges_lower = np.empty(
        (diag_blocksize, (comm_size - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    if comm_rank == 0:
        X_top_2sided_arrow_blocks_local = None
        X_left_2sided_arrow_blocks_local = None

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, :diag_blocksize
        ]

        X_bridges_upper[
            :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
        ] = X_rs_upper_diagonal_blocks[:, :diag_blocksize]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, :diag_blocksize
        ]
        X_arrow_right_blocks_local[-diag_blocksize:, :] = X_rs_arrow_right_blocks[
            :diag_blocksize, :
        ]
    else:
        X_top_2sided_arrow_blocks_local = cpx.empty_pinned(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=X_rs_diagonal_blocks.dtype,
        )
        X_left_2sided_arrow_blocks_local = cpx.empty_pinned(
            (n_diag_blocks_partition * diag_blocksize, diag_blocksize),
            dtype=X_rs_diagonal_blocks.dtype,
        )

        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        X_bridges_lower[
            :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
        ] = X_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index]

        X_diagonal_blocks_local[:, :diag_blocksize] = X_rs_diagonal_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_top_2sided_arrow_blocks_local[:, -diag_blocksize:] = (
            X_rs_upper_diagonal_blocks[:, start_index : start_index + diag_blocksize]
        )

        X_left_2sided_arrow_blocks_local[-diag_blocksize:, :diag_blocksize] = (
            X_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize]
        )

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        if comm_rank != comm_size - 1:
            X_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ] = X_rs_upper_diagonal_blocks[
                :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
            ]

        X_arrow_bottom_blocks_local[:, :diag_blocksize] = X_rs_arrow_bottom_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        X_arrow_right_blocks_local[:diag_blocksize, :] = X_rs_arrow_right_blocks[
            start_index : start_index + diag_blocksize, :
        ]

        X_arrow_right_blocks_local[-diag_blocksize:, :] = X_rs_arrow_right_blocks[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, :
        ]

    X_global_arrow_tip = X_rs_arrow_tip_block
    t_mem_stop = time.perf_counter_ns()
    t_mem = t_mem_stop - t_mem_start

    timings_update_reduced_system["t_mem"] = t_mem

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_left_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
        X_bridges_upper,
        timings_update_reduced_system,
    )


def top_sinv_gpu(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_upper_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_arrow_right_blocks_local: np.ndarray,
    X_global_arrow_tip: np.ndarray,
    L_diagonal_blocks_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
    U_diagonal_blocks_local: np.ndarray,
    U_upper_diagonal_blocks_local: np.ndarray,
    U_arrow_right_blocks_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the lower factor of the local partition.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the lower factor of the local partition.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the lower factor of the local partition.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the upper factor of the local partition.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the upper factor of the local partition.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the upper factor of the local partition.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip : np.ndarray
        Global arrow tip block of the inverse.
    """
    timings_sinv: dict[str, float] = {}

    t_mem = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    # Device side arrays
    X_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(X_diagonal_blocks_local)
    X_diagonal_blocks_local_gpu[:, -diag_blocksize:] = cp.asarray(
        X_diagonal_blocks_local[:, -diag_blocksize:]
    )

    X_lower_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_lower_diagonal_blocks_local
    )

    X_upper_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_upper_diagonal_blocks_local
    )

    X_arrow_bottom_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_arrow_bottom_blocks_local
    )
    X_arrow_bottom_blocks_local_gpu[:, -diag_blocksize:] = cp.asarray(
        X_arrow_bottom_blocks_local[:, -diag_blocksize:]
    )

    X_arrow_right_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_arrow_right_blocks_local
    )
    X_arrow_right_blocks_local_gpu[-diag_blocksize:, :] = cp.asarray(
        X_arrow_right_blocks_local[-diag_blocksize:, :]
    )

    X_global_arrow_tip_gpu: np.ndarray = cp.asarray(X_global_arrow_tip)

    L_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(L_diagonal_blocks_local)
    L_lower_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        L_lower_diagonal_blocks_local
    )
    L_arrow_bottom_blocks_local_gpu: np.ndarray = cp.asarray(
        L_arrow_bottom_blocks_local
    )

    U_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(U_diagonal_blocks_local)
    U_upper_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        U_upper_diagonal_blocks_local
    )
    U_arrow_right_blocks_local_gpu: np.ndarray = cp.asarray(U_arrow_right_blocks_local)

    L_blk_inv_gpu = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype
    )
    U_blk_inv_gpu = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype
    )
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(n_blocks - 2, -1, -1):
        t_trsm_start = time.perf_counter_ns()
        # ----- Block-tridiagonal solver -----
        L_blk_inv_gpu = cpla.solve_triangular(
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv_gpu = cpla.solve_triangular(
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # --- Lower-diagonal blocks ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip_gpu[:, :]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # --- Upper-diagonal blocks ---
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = U_blk_inv_gpu @ (
            -U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
        )

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = U_blk_inv_gpu @ (
            -U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_global_arrow_tip_gpu[:, :]
        )

        # # --- Diagonal blocks ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            U_blk_inv_gpu
            - X_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_mem_start = time.perf_counter_ns()
    X_diagonal_blocks_local = X_diagonal_blocks_local_gpu.get()
    X_lower_diagonal_blocks_local = X_lower_diagonal_blocks_local_gpu.get()
    X_upper_diagonal_blocks_local = X_upper_diagonal_blocks_local_gpu.get()
    X_arrow_bottom_blocks_local = X_arrow_bottom_blocks_local_gpu.get()
    X_arrow_right_blocks_local = X_arrow_right_blocks_local_gpu.get()
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings_sinv["t_mem"] = t_mem
    timings_sinv["t_trsm"] = t_trsm
    timings_sinv["t_gemm"] = t_gemm

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip,
        timings_sinv,
    )


def middle_sinv_gpu(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_upper_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_arrow_right_blocks_local: np.ndarray,
    X_top_2sided_arrow_blocks_local: np.ndarray,
    X_left_2sided_arrow_blocks_local: np.ndarray,
    X_global_arrow_tip_block_local: np.ndarray,
    L_diagonal_blocks_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
    L_upper_2sided_arrow_blocks_local: np.ndarray,
    U_diagonal_blocks_local: np.ndarray,
    U_upper_diagonal_blocks_local: np.ndarray,
    U_arrow_right_blocks_local: np.ndarray,
    U_left_2sided_arrow_blocks_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_top_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing top blocks of the inverse.
    X_left_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing left blocks of the inverse.
    X_global_arrow_tip_block_local : np.ndarray
        Global arrow tip block of the inverse.
    L_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the lower factor of the local partition.
    L_lower_diagonal_blocks_local : np.ndarray
        Lower diagonal blocks of the lower factor of the local partition.
    L_arrow_bottom_blocks_local : np.ndarray
        Arrow bottom blocks of the lower factor of the local partition.
    L_upper_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing top blocks of the lower factor of the local partition.
    U_diagonal_blocks_local : np.ndarray
        Diagonal blocks of the upper factor of the local partition.
    U_upper_diagonal_blocks_local : np.ndarray
        Upper diagonal blocks of the upper factor of the local partition.
    U_arrow_right_blocks_local : np.ndarray
        Arrow right blocks of the upper factor of the local partition.
    U_left_2sided_arrow_blocks_local : np.ndarray
        2-sided pattern array storing left blocks of the upper factor of the local partition.

    Returns
    -------
    X_diagonal_blocks_local : np.ndarray
        Local part of the diagonal array of the inverse.
    X_lower_diagonal_blocks_local : np.ndarray
        Local part of the lower diagonal array of the inverse.
    X_upper_diagonal_blocks_local : np.ndarray
        Local part of the upper diagonal array of the inverse.
    X_arrow_bottom_blocks_local : np.ndarray
        Local part of the arrow bottom array of the inverse.
    X_arrow_right_blocks_local : np.ndarray
        Local part of the arrow right array of the inverse.
    X_global_arrow_tip_block_local : np.ndarray
        Global arrow tip block of the inverse.
    """
    timings_sinv: dict[str, float] = {}

    t_mem = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    # Device side arrays
    X_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(X_diagonal_blocks_local)
    X_diagonal_blocks_local_gpu[:, :diag_blocksize] = cp.asarray(
        X_diagonal_blocks_local[:, :diag_blocksize]
    )
    X_diagonal_blocks_local_gpu[:, -diag_blocksize:] = cp.asarray(
        X_diagonal_blocks_local[:, -diag_blocksize:]
    )
    X_lower_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_lower_diagonal_blocks_local
    )
    X_upper_diagonal_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_upper_diagonal_blocks_local
    )
    X_arrow_bottom_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_arrow_bottom_blocks_local
    )
    X_arrow_bottom_blocks_local_gpu[:, :diag_blocksize] = cp.asarray(
        X_arrow_bottom_blocks_local[:, :diag_blocksize]
    )
    X_arrow_bottom_blocks_local_gpu[:, -diag_blocksize:] = cp.asarray(
        X_arrow_bottom_blocks_local[:, -diag_blocksize:]
    )

    X_arrow_right_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_arrow_right_blocks_local
    )
    X_arrow_right_blocks_local_gpu[:diag_blocksize, :] = cp.asarray(
        X_arrow_right_blocks_local[:diag_blocksize, :]
    )
    X_arrow_right_blocks_local_gpu[-diag_blocksize:, :] = cp.asarray(
        X_arrow_right_blocks_local[-diag_blocksize:, :]
    )

    X_top_2sided_arrow_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_top_2sided_arrow_blocks_local
    )
    X_top_2sided_arrow_blocks_local_gpu[:, :diag_blocksize] = cp.asarray(
        X_top_2sided_arrow_blocks_local[:, :diag_blocksize]
    )
    X_top_2sided_arrow_blocks_local_gpu[:, -diag_blocksize:] = cp.asarray(
        X_top_2sided_arrow_blocks_local[:, -diag_blocksize:]
    )

    X_left_2sided_arrow_blocks_local_gpu: np.ndarray = cp.empty_like(
        X_left_2sided_arrow_blocks_local
    )
    X_left_2sided_arrow_blocks_local_gpu[:diag_blocksize, :] = cp.asarray(
        X_left_2sided_arrow_blocks_local[:diag_blocksize, :]
    )
    X_left_2sided_arrow_blocks_local_gpu[-diag_blocksize:, :] = cp.asarray(
        X_left_2sided_arrow_blocks_local[-diag_blocksize:, :]
    )

    X_global_arrow_tip_block_local_gpu: np.ndarray = cp.asarray(
        X_global_arrow_tip_block_local
    )

    L_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(L_diagonal_blocks_local)
    L_lower_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        L_lower_diagonal_blocks_local
    )
    L_arrow_bottom_blocks_local_gpu: np.ndarray = cp.asarray(
        L_arrow_bottom_blocks_local
    )
    L_upper_2sided_arrow_blocks_local_gpu: np.ndarray = cp.asarray(
        L_upper_2sided_arrow_blocks_local
    )

    U_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(U_diagonal_blocks_local)
    U_upper_diagonal_blocks_local_gpu: np.ndarray = cp.asarray(
        U_upper_diagonal_blocks_local
    )
    U_arrow_right_blocks_local_gpu: np.ndarray = cp.asarray(U_arrow_right_blocks_local)
    U_left_2sided_arrow_blocks_local_gpu: np.ndarray = cp.asarray(
        U_left_2sided_arrow_blocks_local
    )

    L_blk_inv_gpu = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype
    )
    U_blk_inv_gpu = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype
    )
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(n_blocks - 2, 0, -1):
        t_trsm_start = time.perf_counter_ns()
        # ----- Block-tridiagonal solver -----
        L_blk_inv_gpu = cpla.solve_triangular(
            L_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv_gpu = cpla.solve_triangular(
            U_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            cp.eye(diag_blocksize),
            lower=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # X_{i+1, i} = (- X_{i+1, top} L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_left_2sided_arrow_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, top} X_{top, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = U_blk_inv_gpu @ (
            -U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_diagonal_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_top_2sided_arrow_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
        )

        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{top, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_top_2sided_arrow_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_top_2sided_arrow_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local_gpu[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local_gpu[:diag_blocksize, :]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # X_{i, top} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, top} - U_{i, top} X_{top, top} - U_{i, ndb+1} X_{ndb+1, top})
        X_left_2sided_arrow_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = U_blk_inv_gpu @ (
            -U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_left_2sided_arrow_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_diagonal_blocks_local_gpu[:, :diag_blocksize]
            - U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_bottom_blocks_local_gpu[:, :diag_blocksize]
        )

        # Arrowhead
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local_gpu[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip_block_local_gpu[:, :]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, top} X_{top, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local_gpu[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = U_blk_inv_gpu @ (
            -U_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ X_arrow_right_blocks_local_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - U_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_arrow_right_blocks_local_gpu[:diag_blocksize, :]
            - U_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ X_global_arrow_tip_block_local_gpu[:, :]
        )

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, top} L_{top, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            U_blk_inv_gpu
            - X_upper_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_left_2sided_arrow_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_upper_2sided_arrow_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_right_blocks_local_gpu[
                i * diag_blocksize : (i + 1) * diag_blocksize, :
            ]
            @ L_arrow_bottom_blocks_local_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv_gpu
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_mem_start = time.perf_counter_ns()
    # Copy back the 2 first blocks that have been produced in the 2-sided pattern
    # to the tridiagonal storage.
    X_upper_diagonal_blocks_local_gpu[:, :diag_blocksize] = (
        X_top_2sided_arrow_blocks_local_gpu[:, diag_blocksize : 2 * diag_blocksize]
    )
    X_lower_diagonal_blocks_local_gpu[:, :diag_blocksize] = (
        X_left_2sided_arrow_blocks_local_gpu[diag_blocksize : 2 * diag_blocksize, :]
    )

    X_diagonal_blocks_local = X_diagonal_blocks_local_gpu.get()
    X_lower_diagonal_blocks_local = X_lower_diagonal_blocks_local_gpu.get()
    X_upper_diagonal_blocks_local = X_upper_diagonal_blocks_local_gpu.get()
    X_arrow_bottom_blocks_local = X_arrow_bottom_blocks_local_gpu.get()
    X_arrow_right_blocks_local = X_arrow_right_blocks_local_gpu.get()
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings_sinv["t_mem"] = t_mem
    timings_sinv["t_trsm"] = t_trsm
    timings_sinv["t_gemm"] = t_gemm

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip_block_local,
        timings_sinv,
    )
