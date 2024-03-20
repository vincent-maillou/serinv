"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Contains the lu selected factorization routines on GPU.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cpla
except ImportError:
    pass

import numpy as np

import time


def lu_factorize_tridiag_gpu(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the non-pivoted LU factorization of a block tridiagonal matrix.
    The matrix is assumed to be non-singular and blocks are assumed to be of the
    same size given in a sequential array.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray
        The blocks on the upper diagonal of the matrix.

    Returns
    -------
    L_diagonal_blocks : cp.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : cp.ndarray
        Lower diagonal blocks of the lower factor.
    U_diagonal_blocks : cp.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : cp.ndarray
        Upper diagonal blocks of the upper factor
    """
    timings: dict[str, float] = {}

    t_mem = 0.0
    t_lu = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    blocksize = A_diagonal_blocks.shape[0]
    nblocks = A_diagonal_blocks.shape[1] // blocksize

    A_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_diagonal_blocks)
    A_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_lower_diagonal_blocks)
    A_upper_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_upper_diagonal_blocks)

    # Host side arrays
    L_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_gpu)
    L_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        A_lower_diagonal_blocks_gpu
    )
    U_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks_gpu)
    U_upper_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(
        A_upper_diagonal_blocks_gpu
    )

    # Device side arrays
    L_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_lower_diagonal_blocks)

    U_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_upper_diagonal_blocks)
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(0, nblocks - 1, 1):
        t_lu_start = time.perf_counter_ns()
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            U_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
        ) = cpla.lu(
            A_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            permute_l=True,
        )
        stream.synchronize()
        t_lu_stop = time.perf_counter_ns()
        t_lu += t_lu_stop - t_lu_start

        t_trsm_start = time.perf_counter_ns()
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_gpu[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = cpla.solve_triangular(
            U_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
            cp.eye(blocksize),
            trans=0,
            lower=False,
            unit_diagonal=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        L_lower_diagonal_blocks_gpu[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            A_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ L_lower_diagonal_blocks_gpu[
                :,
                i * blocksize : (i + 1) * blocksize,
            ]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_trsm_start = time.perf_counter_ns()
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_gpu[
            :,
            i * blocksize : (i + 1) * blocksize,
        ] = (
            cpla.solve_triangular(
                L_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize],
                cp.eye(blocksize),
                trans=0,
                lower=True,
                unit_diagonal=True,
            )
            @ A_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_gemm_start = time.perf_counter_ns()
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize] = (
            A_diagonal_blocks_gpu[:, (i + 1) * blocksize : (i + 2) * blocksize]
            - L_lower_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
            @ U_upper_diagonal_blocks_gpu[:, i * blocksize : (i + 1) * blocksize]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_lu_start = time.perf_counter_ns()
    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_gpu[:, -blocksize:],
        U_diagonal_blocks_gpu[:, -blocksize:],
    ) = cpla.lu(A_diagonal_blocks_gpu[:, -blocksize:], permute_l=True)

    t_mem_start = time.perf_counter_ns()
    L_diagonal_blocks_gpu.get(out=L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu.get(out=L_lower_diagonal_blocks)
    U_diagonal_blocks_gpu.get(out=U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu.get(out=U_upper_diagonal_blocks)
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings["mem"] = t_mem
    timings["lu"] = t_lu
    timings["trsm"] = t_trsm
    timings["gemm"] = t_gemm

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        timings,
    )


def lu_factorize_tridiag_arrowhead_gpu(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_right_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the lu factorization of a block tridiagonal arrowhead
    matrix. The matrix is assumed to be non singular.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : np.ndarray
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : np.ndarray
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : np.ndarray
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : np.ndarray
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : np.ndarray
        The block at the tip of the arrowhead.

    Returns
    -------
    L_diagonal_blocks : cp.ndarray
        Diagonal blocks of the lower factor.
    L_lower_diagonal_blocks : cp.ndarray
        Lower diagonal blocks of the lower factor.
    L_arrow_bottom_blocks : np.ndarray
        Bottom arrow blocks of the lower factor.
    U_diagonal_blocks : cp.ndarray
        Diagonal blocks of the upper factor.
    U_upper_diagonal_blocks : cp.ndarray
        Upper diagonal blocks of the upper factor
    U_arrow_right_blocks : np.ndarray
        Right arrow blocks of the upper factor
    """
    timings: dict[str, float] = {}

    t_mem = 0.0
    t_lu = 0.0
    t_trsm = 0.0
    t_gemm = 0.0

    stream = cp.cuda.Stream()
    stream.use()

    t_mem_start = time.perf_counter_ns()
    diag_blocksize = A_diagonal_blocks.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks.shape[0]

    n_diag_blocks = A_diagonal_blocks.shape[1] // diag_blocksize

    A_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_diagonal_blocks)
    A_lower_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_lower_diagonal_blocks)
    A_upper_diagonal_blocks_gpu: cp.ndarray = cp.asarray(A_upper_diagonal_blocks)
    A_arrow_bottom_blocks_gpu: cp.ndarray = cp.asarray(A_arrow_bottom_blocks)
    A_arrow_right_blocks_gpu: cp.ndarray = cp.asarray(A_arrow_right_blocks)
    A_arrow_tip_block_gpu: cp.ndarray = cp.asarray(A_arrow_tip_block)

    # Host side arrays
    L_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks)
    L_lower_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_lower_diagonal_blocks)
    L_arrow_bottom_blocks: cpx.ndarray = cpx.empty_pinned(
        (arrow_blocksize, n_diag_blocks * diag_blocksize + arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    U_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_diagonal_blocks)
    U_upper_diagonal_blocks: cp.ndarray = cpx.empty_like_pinned(A_upper_diagonal_blocks)
    U_arrow_right_blocks: cp.ndarray = cpx.empty_pinned(
        (n_diag_blocks * diag_blocksize + arrow_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )

    # Device side arrays
    L_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu: cp.ndarray = cp.empty_like(L_arrow_bottom_blocks)

    U_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu: cp.ndarray = cp.empty_like(U_upper_diagonal_blocks)
    U_arrow_right_blocks_gpu: cp.ndarray = cp.empty_like(U_arrow_right_blocks)

    L_inv_temp_gpu: cp.ndarray = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    U_inv_temp_gpu: cp.ndarray = cp.empty(
        (diag_blocksize, diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    for i in range(0, n_diag_blocks - 1):
        t_lu_start = time.perf_counter_ns()
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = cpla.lu(
            A_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )
        stream.synchronize()
        t_lu_stop = time.perf_counter_ns()
        t_lu += t_lu_stop - t_lu_start

        t_trsm_start = time.perf_counter_ns()
        # Compute lower factors
        U_inv_temp_gpu = cpla.solve_triangular(
            U_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=False,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_inv_temp_gpu
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp_gpu
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_trsm_start = time.perf_counter_ns()
        # Compute upper factors
        L_inv_temp_gpu = cpla.solve_triangular(
            L_diagonal_blocks_gpu[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            cp.eye(diag_blocksize),
            lower=True,
        )
        stream.synchronize()
        t_trsm_stop = time.perf_counter_ns()
        t_trsm += t_trsm_stop - t_trsm_start

        t_gemm_start = time.perf_counter_ns()
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_gpu[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            L_inv_temp_gpu
            @ A_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp_gpu
            @ A_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

        t_gemm_start = time.perf_counter_ns()
        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_diagonal_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_gpu[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_gpu[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            - L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_upper_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_gpu[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
        ] = (
            A_arrow_right_blocks_gpu[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :
            ]
            - L_lower_diagonal_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        A_arrow_tip_block_gpu[:, :] = (
            A_arrow_tip_block_gpu[:, :]
            - L_arrow_bottom_blocks_gpu[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ U_arrow_right_blocks_gpu[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        stream.synchronize()
        t_gemm_stop = time.perf_counter_ns()
        t_gemm += t_gemm_stop - t_gemm_start

    t_lu_start = time.perf_counter_ns()
    # L_{ndb, ndb}, U_{ndb, ndb} = lu_dcmp(A_{ndb, ndb})
    (
        L_diagonal_blocks_gpu[:, -diag_blocksize:],
        U_diagonal_blocks_gpu[:, -diag_blocksize:],
    ) = cpla.lu(
        A_diagonal_blocks_gpu[:, -diag_blocksize:],
        permute_l=True,
    )
    stream.synchronize()
    t_lu_stop = time.perf_counter_ns()
    t_lu += t_lu_stop - t_lu_start

    t_trsm_start = time.perf_counter_ns()
    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ U_{ndb, ndb}^{-1}
    L_arrow_temp = cpla.solve_triangular(
        U_diagonal_blocks_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=False,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    L_arrow_bottom_blocks_gpu[
        :, -diag_blocksize - arrow_blocksize : -arrow_blocksize
    ] = (A_arrow_bottom_blocks_gpu[:, -diag_blocksize:] @ L_arrow_temp)
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_trsm_start = time.perf_counter_ns()
    # U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} @ A_{ndb, ndb+1}
    U_arrow_temp = cpla.solve_triangular(
        L_diagonal_blocks_gpu[:, -diag_blocksize:],
        cp.eye(diag_blocksize),
        lower=True,
    )
    stream.synchronize()
    t_trsm_stop = time.perf_counter_ns()
    t_trsm += t_trsm_stop - t_trsm_start

    t_gemm_start = time.perf_counter_ns()
    U_arrow_right_blocks_gpu[
        -diag_blocksize - arrow_blocksize : -arrow_blocksize, :
    ] = (U_arrow_temp @ A_arrow_right_blocks_gpu[-diag_blocksize:, :])
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_gemm_start = time.perf_counter_ns()
    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ U_{ndb, ndb+1}
    A_arrow_tip_block_gpu[:, :] = (
        A_arrow_tip_block_gpu[:, :]
        - L_arrow_bottom_blocks_gpu[
            :, -diag_blocksize - arrow_blocksize : -arrow_blocksize
        ]
        @ U_arrow_right_blocks_gpu[
            -diag_blocksize - arrow_blocksize : -arrow_blocksize, :
        ]
    )
    stream.synchronize()
    t_gemm_stop = time.perf_counter_ns()
    t_gemm += t_gemm_stop - t_gemm_start

    t_lu_start = time.perf_counter_ns()
    # L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = lu_dcmp(A_{ndb+1, ndb+1})
    (
        L_arrow_bottom_blocks_gpu[:, -arrow_blocksize:],
        U_arrow_right_blocks_gpu[-arrow_blocksize:, :],
    ) = cpla.lu(A_arrow_tip_block_gpu[:, :], permute_l=True)

    t_mem_start = time.perf_counter_ns()
    L_diagonal_blocks_gpu.get(out=L_diagonal_blocks)
    L_lower_diagonal_blocks_gpu.get(out=L_lower_diagonal_blocks)
    L_arrow_bottom_blocks_gpu.get(out=L_arrow_bottom_blocks)
    U_diagonal_blocks_gpu.get(out=U_diagonal_blocks)
    U_upper_diagonal_blocks_gpu.get(out=U_upper_diagonal_blocks)
    U_arrow_right_blocks_gpu.get(out=U_arrow_right_blocks)
    stream.synchronize()
    t_mem_stop = time.perf_counter_ns()
    t_mem += t_mem_stop - t_mem_start

    timings["mem"] = t_mem
    timings["lu"] = t_lu
    timings["trsm"] = t_trsm
    timings["gemm"] = t_gemm

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        U_diagonal_blocks,
        U_upper_diagonal_blocks,
        U_arrow_right_blocks,
        timings,
    )
