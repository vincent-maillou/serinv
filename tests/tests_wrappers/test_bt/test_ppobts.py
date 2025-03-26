# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import backend_flags, _get_module_from_array
import pytest

if backend_flags["mpi_avail"]:
    from mpi4py import MPI
else:
    pytest.skip("mpi4py is not available", allow_module_level=True)

import numpy as np

from ...testing_utils import bt_dense_to_arrays, dd_bt, symmetrize, rhs

from serinv.utils import allocate_pobtx_permutation_buffers
from serinv.wrappers import (
    ppobtf,
    ppobts,
    allocate_pobtrs,
)

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("comm_strategy", ["allgather"])
@pytest.mark.parametrize("n_rhs", [1, 2, 3])
def test_ppobts(
    n_rhs: int,
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
    ) = bt_dense_to_arrays(A.copy(), diagonal_blocksize, n_diag_blocks)

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

    B = rhs(
        n_rhs=n_rhs,
        diagonal_blocksize=diagonal_blocksize,
        arrowhead_blocksize=0,
        n_diag_blocks=n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    B_local = xp.zeros(
        (n_diag_blocks_per_processes * diagonal_blocksize, n_rhs),
        dtype=dtype,
    )

    B_local[:] = B[
        comm_rank
        * n_diag_blocks_per_processes
        * diagonal_blocksize : (comm_rank + 1)
        * n_diag_blocks_per_processes
        * diagonal_blocksize
    ]

    # Reference solution
    X_ref = xp.linalg.solve(A.copy(), B.copy())

    X_ref_local = xp.zeros_like(B_local)
    X_ref_local = X_ref[
        comm_rank
        * n_diag_blocks_per_processes
        * diagonal_blocksize : (comm_rank + 1)
        * n_diag_blocks_per_processes
        * diagonal_blocksize
    ]

    # Allocate permutation buffer
    buffer = allocate_pobtx_permutation_buffers(
        A_diagonal_blocks_local,
    )

    # Allocate reduced system
    pobtrs: dict = allocate_pobtrs(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        B=B_local,
        comm=MPI.COMM_WORLD,
        array_module=xp.__name__,
        strategy=comm_strategy,
    )

    # Distributed factorization
    ppobtf(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        buffer=buffer,
        pobtrs=pobtrs,
        comm=MPI.COMM_WORLD,
        strategy=comm_strategy,
    )

    # Distributed Solve
    ppobts(
        L_diagonal_blocks=A_diagonal_blocks_local,
        L_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        B=B_local,
        buffer=buffer,
        pobtrs=pobtrs,
        comm=MPI.COMM_WORLD,
        strategy=comm_strategy,
    )

    assert xp.allclose(X_ref_local, B_local)
