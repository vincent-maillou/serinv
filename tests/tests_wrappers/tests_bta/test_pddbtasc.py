# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import backend_flags, _get_module_from_array
import pytest

if backend_flags["mpi_avail"]:
    from mpi4py import MPI
else:
    pytest.skip("mpi4py is not available", allow_module_level=True)

import numpy as np

from ...testing_utils import bta_dense_to_arrays, dd_bta, symmetrize

from serinv.algs import ddbtasci, ddbtasc
from serinv.utils import allocate_ddbtax_permutation_buffers
from serinv.wrappers import (
    pddbtasc,
    allocate_ddbtars,
)

if backend_flags["cupy_avail"]:
    import cupyx as cpx

from os import environ

environ["OMP_NUM_THREADS"] = "1"


comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("comm_strategy", ["allgather"])
@pytest.mark.parametrize("type_of_equation", ["AX=I", "AXA.T=B"])
def test_pddbtasc(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    partition_size: int,
    array_type: str,
    dtype: np.dtype,
    comm_strategy: str,
    type_of_equation: str,
):
    n_diag_blocks = partition_size * comm_size

    A = dd_bta(
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        device_array=True if array_type == "device" else False,
        dtype=dtype,
    )

    xp, _ = _get_module_from_array(A)

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_lower_arrow_blocks,
        A_upper_arrow_blocks,
        A_arrow_tip_block,
    ) = bta_dense_to_arrays(
        A.copy(), diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    # Save the local slice of the array for each MPI process
    n_diag_blocks_per_processes = n_diag_blocks // comm_size

    A_diagonal_blocks_local = A_diagonal_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
    ]
    A_lower_arrow_blocks_local = A_lower_arrow_blocks[
        comm_rank
        * n_diag_blocks_per_processes : (comm_rank + 1)
        * n_diag_blocks_per_processes,
    ]
    A_upper_arrow_blocks_local = A_upper_arrow_blocks[
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
    X_ref = xp.linalg.inv(A.copy())

    (
        X_ref_diagonal_blocks,
        _,
        _,
        _,
        _,
        X_ref_arrow_tip_block,
    ) = bta_dense_to_arrays(
        X_ref.copy(), diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
    )

    if type_of_equation == "AX=I":
        rhs = None
        quadratic = None
    elif type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        B = dd_bta(
            diagonal_blocksize,
            arrowhead_blocksize,
            n_diag_blocks,
            device_array=True if array_type == "device" else False,
            dtype=dtype,
        )

        symmetrize(B)

        (
            B_diagonal_blocks,
            B_lower_diagonal_blocks,
            B_upper_diagonal_blocks,
            B_lower_arrow_blocks,
            B_upper_arrow_blocks,
            B_arrow_tip_block,
        ) = bta_dense_to_arrays(
            B, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
        )

        B_diagonal_blocks_local = B_diagonal_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]
        B_lower_arrow_blocks_local = B_lower_arrow_blocks[
            comm_rank
            * n_diag_blocks_per_processes : (comm_rank + 1)
            * n_diag_blocks_per_processes,
        ]
        B_upper_arrow_blocks_local = B_upper_arrow_blocks[
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
            "B_lower_arrow_blocks": B_lower_arrow_blocks_local,
            "B_upper_arrow_blocks": B_upper_arrow_blocks_local,
            "B_arrow_tip_block": B_arrow_tip_block,
        }

        quadratic = True

    # print(f"{comm_rank}, quadratic: {quadratic}, rhs: {rhs}")

    buffers: dict = allocate_ddbtax_permutation_buffers(
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        quadratic=quadratic,
    )

    ddbtars: dict = allocate_ddbtars(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks_local,
        A_lower_arrow_blocks=A_lower_arrow_blocks_local,
        A_upper_arrow_blocks=A_upper_arrow_blocks_local,
        A_arrow_tip_block=A_arrow_tip_block,
        comm_size=comm_size,
        array_module="numpy" if array_type == "host" else "cupy",
        strategy=comm_strategy,
        quadratic=quadratic,
    )

    pddbtasc(
        A_diagonal_blocks=A_diagonal_blocks_local,
        A_lower_diagonal_blocks=A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks=A_upper_diagonal_blocks_local,
        A_lower_arrow_blocks=A_lower_arrow_blocks_local,
        A_upper_arrow_blocks=A_upper_arrow_blocks_local,
        A_arrow_tip_block=A_arrow_tip_block,
        rhs=rhs,
        quadratic=quadratic,
        buffers=buffers,
        ddbtars=ddbtars,
    )

    ddbtasci(
        A_diagonal_blocks=ddbtars["A_diagonal_blocks"],
        A_lower_diagonal_blocks=ddbtars["A_lower_diagonal_blocks"],
        A_upper_diagonal_blocks=ddbtars["A_upper_diagonal_blocks"],
        A_lower_arrow_blocks=ddbtars["A_lower_arrow_blocks"],
        A_upper_arrow_blocks=ddbtars["A_upper_arrow_blocks"],
        A_arrow_tip_block=ddbtars["A_arrow_tip_block"],
        rhs=ddbtars.get("_rhs", None),
        quadratic=quadratic,
    )

    assert xp.allclose(
        X_ref_arrow_tip_block,
        ddbtars["A_arrow_tip_block"],
    )

    if type_of_equation == "AX=B":
        ...
    elif type_of_equation == "AXA.T=B":
        Xl_ref = X_ref @ B @ X_ref.conj().T

        (
            _,
            _,
            _,
            _,
            _,
            Xl_arrow_tip_block_ref,
        ) = bta_dense_to_arrays(
            Xl_ref, diagonal_blocksize, arrowhead_blocksize, n_diag_blocks
        )

        assert xp.allclose(
            Xl_arrow_tip_block_ref,
            ddbtars["_rhs"]["B_arrow_tip_block"],
        )
