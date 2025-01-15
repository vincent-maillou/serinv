# Copyright 2023-2025 ETH Zurich. All rights reserved.

from mpi4py import MPI

import numpy as np
import pytest

from serinv import CUPY_AVAIL, _get_module_from_array

from ..testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.algs import pobtasi
from serinv.wrappers import (
    ppobtaf,
    allocate_permutation_buffer,
    allocate_ppobtars,
)

if CUPY_AVAIL:
    import cupyx as cpx

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_diag_blocks", [comm_size * 3, comm_size * 4, comm_size * 5])
@pytest.mark.parametrize("array_type", ["host", "device", "streaming"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("preallocate_permutation_buffer", [True, False])
@pytest.mark.parametrize("preallocate_reduced_system", [True, False])
@pytest.mark.parametrize("comm_strategy", ["allreduce", "allgather", "gather-scatter"])
def test_ppobtaf(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    array_type: str,
    dtype: np.dtype,
    preallocate_permutation_buffer: bool,
    preallocate_reduced_system: bool,
    comm_strategy: str,
):
    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    symmetrize(A)

    xp, _ = _get_module_from_array(A)

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
    ]

    if comm_rank == comm_size - 1:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
        ]
    else:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]

    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
    ]

    if CUPY_AVAIL and array_type == "streaming":
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
    ]

    if comm_rank == comm_size - 1:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
        ]
    else:
        X_ref_lower_diagonal_blocks_local = X_ref_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]

    X_ref_arrow_bottom_blocks_local = X_ref_arrow_bottom_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
    ]

    # Allocate permutation buffer
    if preallocate_permutation_buffer:
        permutation_buffer = allocate_permutation_buffer(
            A_diagonal_blocks_local,
            device_streaming=True if array_type == "streaming" else False,
        )
    else:
        permutation_buffer = None

    # Allocate reduced system
    if preallocate_reduced_system:
        (
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            _L_tip_update,
        ) = allocate_ppobtars(
            A_diagonal_blocks=A_diagonal_blocks_local,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks=A_arrow_bottom_blocks_local,
            A_arrow_tip_block=A_arrow_tip_block_global,
            comm_size=comm_size,
            array_module=xp.__name__,
            device_streaming=True if array_type == "streaming" else False,
            strategy=comm_strategy,
        )
    else:
        _L_diagonal_blocks = None
        _L_lower_diagonal_blocks = None
        _L_lower_arrow_blocks = None
        _L_tip_update = None

    # Distributed factorization
    (
        _L_diagonal_blocks,
        _L_lower_diagonal_blocks,
        _L_lower_arrow_blocks,
        permutation_buffer,
    ) = ppobtaf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_tip_block_global,
        device_streaming=True if array_type == "streaming" else False,
        A_permutation_buffer=permutation_buffer,
        _L_diagonal_blocks=_L_diagonal_blocks,
        _L_lower_diagonal_blocks=_L_lower_diagonal_blocks,
        _L_lower_arrow_blocks=_L_lower_arrow_blocks,
        _L_tip_update=_L_tip_update,
        strategy=comm_strategy,
    )

    if comm_strategy == "gather-scatter":
        if comm_rank == 0:
            pobtasi(
                _L_diagonal_blocks[1:],
                _L_lower_diagonal_blocks[1:-1],
                _L_lower_arrow_blocks[1:],
                A_arrow_tip_block_global,
                device_streaming=True if array_type == "streaming" else False,
            )

            assert xp.allclose(
                _L_diagonal_blocks[1],
                X_ref_diagonal_blocks_local[-1],
            )
            assert xp.allclose(
                _L_lower_diagonal_blocks[1],
                X_ref_lower_diagonal_blocks_local[-1],
            )
            assert xp.allclose(
                _L_lower_arrow_blocks[1],
                X_ref_arrow_bottom_blocks_local[-1],
            )
            assert xp.allclose(A_arrow_tip_block_global, X_ref_arrow_tip_block_global)
    else:
        pobtasi(
            _L_diagonal_blocks,
            _L_lower_diagonal_blocks,
            _L_lower_arrow_blocks,
            A_arrow_tip_block_global,
            device_streaming=True if array_type == "streaming" else False,
        )

        if comm_rank == 0:
            assert xp.allclose(
                _L_diagonal_blocks[0],
                X_ref_diagonal_blocks_local[-1],
            )
            assert xp.allclose(
                _L_lower_diagonal_blocks[0],
                X_ref_lower_diagonal_blocks_local[-1],
            )
            assert xp.allclose(
                _L_lower_arrow_blocks[0],
                X_ref_arrow_bottom_blocks_local[-1],
            )
            assert xp.allclose(A_arrow_tip_block_global, X_ref_arrow_tip_block_global)
        else:
            assert xp.allclose(
                _L_diagonal_blocks[2 * comm_rank - 1],
                X_ref_diagonal_blocks_local[0],
            )
            assert xp.allclose(
                _L_diagonal_blocks[2 * comm_rank],
                X_ref_diagonal_blocks_local[-1],
            )

            if comm_rank < comm_size - 1:
                assert xp.allclose(
                    _L_lower_diagonal_blocks[2 * comm_rank],
                    X_ref_lower_diagonal_blocks_local[-1],
                )

            assert xp.allclose(
                _L_lower_arrow_blocks[2 * comm_rank - 1],
                X_ref_arrow_bottom_blocks_local[0],
            )
            assert xp.allclose(
                _L_lower_arrow_blocks[2 * comm_rank],
                X_ref_arrow_bottom_blocks_local[-1],
            )

            assert xp.allclose(A_arrow_tip_block_global, X_ref_arrow_tip_block_global)
