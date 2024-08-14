# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

from os import environ

environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
from mpi4py import MPI

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()

from serinv.algs import d_pobtaf, d_pobtasi


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
@pytest.mark.parametrize("device_array", [False, True], ids=["host", "device"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("device_streaming", [False, True])
@pytest.mark.parametrize("nested_solving", [True, False])
def test_d_pobtasi(
    dd_bta,
    bta_dense_to_arrays,
    bta_symmetrize,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_diag_blocks,
    device_array,
    device_streaming,
    nested_solving,
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
        device_streaming,
        nested_solving=nested_solving,
    )

    def _relerror(a, b):
        return xp.linalg.norm(a - b) / xp.linalg.norm(a)
    
    for rank in range(comm_size):
        if rank == comm_rank:
            print(f"rank {comm_rank} relerror diagonal blocks: {_relerror(X_ref_diagonal_blocks_local, X_diagonal_blocks_local)}")
            print(f"rank {comm_rank} relerror lower diagonal blocks: {_relerror(X_ref_lower_diagonal_blocks_local, X_lower_diagonal_blocks_local)}")
            print(f"rank {comm_rank} relerror arrow bottom blocks: {_relerror(X_ref_arrow_bottom_blocks_local, X_arrow_bottom_blocks_local)}")
            print(f"rank {comm_rank} relerror arrow tip block: {_relerror(X_ref_arrow_tip_block_global, X_arrow_tip_block_global)}", flush=True)
        MPI.COMM_WORLD.Barrier()

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
        if not nested_solving:
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
        # NOTE: This assertion fails with nested solving because we have to copy the arrow tip block to a contiguous buffer
        if not nested_solving:
            assert (
                X_arrow_tip_block_global.ctypes.data == A_arrow_tip_block_global.ctypes.data
            )
