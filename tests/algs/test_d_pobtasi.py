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

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("device_streaming", [False, True])
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
        device_streaming,
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
