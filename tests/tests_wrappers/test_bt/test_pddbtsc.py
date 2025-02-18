# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import backend_flags, _get_module_from_array
import pytest

if backend_flags["mpi_avail"]:
    from mpi4py import MPI
else:
    pytest.skip("mpi4py is not available", allow_module_level=True)

import numpy as np

from ...testing_utils import bt_dense_to_arrays, dd_bt, symmetrize

from serinv.algs import ddbtsci
from serinv.utils import allocate_ddbtx_permutation_buffers
from serinv.wrappers import (
    pddbtsc,
    allocate_ddbtrs,
)

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_pddbtsc(
    diagonal_blocksize: int,
    partition_size: int,
    array_type: str,
    dtype: np.dtype,
    comm_strategy: str,
    type_of_equation: str,
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
        A_upper_diagonal_blocks,
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
        A_upper_diagonal_blocks_local = A_upper_diagonal_blocks[
            comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
        ]
    else:
        A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]
        A_upper_diagonal_blocks_local = A_upper_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]

    # Reference solution
    X_ref = xp.linalg.inv(A)

    (
        X_ref_diagonal_blocks,
        _,
        _,
    ) = bt_dense_to_arrays(X_ref, diagonal_blocksize, n_diag_blocks)

    X_ref_diagonal_blocks_local = X_ref_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
    ]

    if type_of_equation == "AX=I":
        rhs = None
        quadratic = None
    elif type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        B = dd_bt(
            diagonal_blocksize,
            n_diag_blocks,
            device_array=True if array_type == "device" else False,
            dtype=dtype,
        )

        symmetrize(B)

        (
            B_diagonal_blocks,
            B_lower_diagonal_blocks,
            B_upper_diagonal_blocks,
        ) = bt_dense_to_arrays(B, diagonal_blocksize, n_diag_blocks)

        B_diagonal_blocks_local = B_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]

        if comm_rank == comm_size - 1:
            B_lower_diagonal_blocks_local = B_lower_diagonal_blocks[
                comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            ]
            B_upper_diagonal_blocks_local = B_upper_diagonal_blocks[
                comm_rank * n_diag_blocks_per_processes : n_diag_blocks - 1,
            ]
        else:
            B_lower_diagonal_blocks_local = B_lower_diagonal_blocks[
                comm_rank
                * n_diag_blocks_per_processes : (comm_rank + 1)
                * n_diag_blocks_per_processes,
            ]
            B_upper_diagonal_blocks_local = B_upper_diagonal_blocks[
                comm_rank
                * n_diag_blocks_per_processes : (comm_rank + 1)
                * n_diag_blocks_per_processes,
            ]

        rhs: dict = {
            "B_diagonal_blocks": B_diagonal_blocks_local,
            "B_lower_diagonal_blocks": B_lower_diagonal_blocks_local,
            "B_upper_diagonal_blocks": B_upper_diagonal_blocks_local,
        }

        quadratic = True

    buffers: dict = allocate_ddbtx_permutation_buffers(
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        quadratic=True if type_of_equation == "AXA.T=B" else False,
    )

    ddbtrs: dict = allocate_ddbtrs(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks_local,
        comm_size=comm_size,
        array_module="numpy" if array_type == "host" else "cupy",
        strategy=comm_strategy,
        quadratic=quadratic,
    )

    pddbtsc(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks_local,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
        ddbtrs=ddbtrs,
    )

    ddbtsci(
        A_diagonal_blocks=ddbtrs["A_diagonal_blocks"],
        A_lower_diagonal_blocks=ddbtrs["A_lower_diagonal_blocks"],
        A_upper_diagonal_blocks=ddbtrs["A_upper_diagonal_blocks"],
        rhs=ddbtrs.get("_rhs", None),
        quadratic=quadratic,
    )

    if type_of_equation == "AXA.T=B":
        Xl_ref = X_ref @ B @ X_ref.conj().T

        (
            Xl_diagonal_blocks_ref,
            _,
            _,
        ) = bt_dense_to_arrays(Xl_ref, diagonal_blocksize, n_diag_blocks)

        Xl_ref_diagonal_blocks_local = Xl_diagonal_blocks_ref[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]

        _rhs = ddbtrs["_rhs"]

    if comm_rank == 0:
        assert xp.allclose(
            X_ref_diagonal_blocks_local[-1],
            ddbtrs["A_diagonal_blocks"][0],
        )

        if type_of_equation == "AXA.T=B":
            assert xp.allclose(
                Xl_ref_diagonal_blocks_local[-1],
                _rhs["B_diagonal_blocks"][0],
            )
    elif comm_rank == comm_size - 1:
        assert xp.allclose(
            X_ref_diagonal_blocks_local[0],
            ddbtrs["A_diagonal_blocks"][-1],
        )

        if type_of_equation == "AXA.T=B":
            assert xp.allclose(
                Xl_ref_diagonal_blocks_local[0],
                _rhs["B_diagonal_blocks"][-1],
            )
    else:
        assert xp.allclose(
            X_ref_diagonal_blocks_local[0],
            ddbtrs["A_diagonal_blocks"][2 * comm_rank - 1],
        )
        assert xp.allclose(
            X_ref_diagonal_blocks_local[-1],
            ddbtrs["A_diagonal_blocks"][2 * comm_rank],
        )

        if type_of_equation == "AXA.T=B":
            assert xp.allclose(
                Xl_ref_diagonal_blocks_local[0],
                _rhs["B_diagonal_blocks"][2 * comm_rank - 1],
            )
            assert xp.allclose(
                Xl_ref_diagonal_blocks_local[-1],
                _rhs["B_diagonal_blocks"][2 * comm_rank],
            )
