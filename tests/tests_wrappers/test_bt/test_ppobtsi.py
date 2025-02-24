# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import backend_flags, _get_module_from_array
import pytest

if backend_flags["mpi_avail"]:
    from mpi4py import MPI
else:
    pytest.skip("mpi4py is not available", allow_module_level=True)

import numpy as np

from ...testing_utils import bt_dense_to_arrays, dd_bt, symmetrize

from serinv.utils import allocate_pobtx_permutation_buffers
from serinv.wrappers import (
    ppobtf,
    ppobtsi,
    allocate_pobtrs,
)

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("comm_strategy", ["allgather", "gather-scatter"])
def test_ppobtsi(
    diagonal_blocksize: int,
    partition_size: int,
    array_type: str,
    dtype: np.dtype,
    comm_strategy: str,
):
    n_diag_blocks = partition_size * comm_size

    A = dd_bt(
        diagonal_blocksize,
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
    ) = bt_dense_to_arrays(A, diagonal_blocksize, n_diag_blocks)

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

    # Reference solution
    X_ref = xp.linalg.inv(A)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        _,
    ) = bt_dense_to_arrays(X_ref, diagonal_blocksize, n_diag_blocks)

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

    # Allocate permutation buffer
    buffer = allocate_pobtx_permutation_buffers(
        A_diagonal_blocks_local,
    )

    # Allocate reduced system
    pobtrs: dict = allocate_pobtrs(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        comm_size=comm_size,
        array_module=xp.__name__,
        strategy=comm_strategy,
    )

    # Distributed factorization
    ppobtf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        buffer=buffer,
        pobtrs=pobtrs,
        strategy=comm_strategy,
    )

    # Distributed Selected-Inversion of the full system
    ppobtsi(
        L_diagonal_blocks=A_diagonal_blocks_local,
        L_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        buffer=buffer,
        pobtrs=pobtrs,
        strategy=comm_strategy,
    )

    assert xp.allclose(A_diagonal_blocks_local, X_ref_diagonal_blocks_local)
    assert xp.allclose(
        A_lower_diagonal_blocks_local,
        X_ref_lower_diagonal_blocks_local,
    )