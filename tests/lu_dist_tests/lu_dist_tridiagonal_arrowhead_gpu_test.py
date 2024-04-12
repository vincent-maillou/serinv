"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Integration testing of the lu_dist algorithm for tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import sys

from copy import deepcopy
from os import environ

import numpy as np
import pytest
from mpi4py import MPI

try:
    from sdr.lu_dist.lu_dist_tridiagonal_arrowhead_gpu import (
        lu_dist_tridiagonal_arrowhead_gpu,
    )
except ImportError:
    pass

from sdr.utils.dist_utils import (
    get_partitions_indices,
    extract_partition_tridiagonal_arrowhead_array,
    extract_bridges_tridiagonal_array,
)
from sdr.utils import matrix_generation_dense
from sdr.utils.matrix_transformation_arrays import (
    convert_block_tridiagonal_arrowhead_dense_to_arrays,
)

environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.skipif(
    "cupy" not in sys.modules, reason="requires a working cupy installation"
)
@pytest.mark.gpu
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "nblocks, diag_blocksize, arrow_blocksize",
    [
        (6, 2, 2),
        (6, 3, 2),
        (6, 2, 3),
        (13, 2, 2),
        (13, 3, 2),
        (13, 2, 3),
        (13, 10, 2),
        (13, 2, 10),
    ],
)
def test_lu_dist(
    nblocks: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    diagonal_dominant = True
    symmetric = False
    seed = 63

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if nblocks // comm_size < 3:
        pytest.skip("Not enough blocks for the number of processes. Skipping test.")

    A = matrix_generation_dense.generate_block_tridiagonal_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    (
        start_blockrows,
        partition_sizes,
        end_blockrows,
    ) = get_partitions_indices(n_partitions=comm_size, total_size=nblocks - 1)

    # ----- Reference/Checking data -----
    A_ref = deepcopy(A)

    X_ref = np.linalg.inv(A_ref)

    (
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        X_ref_upper_diagonal_blocks,
        X_ref_arrow_bottom_blocks,
        X_ref_arrow_right_blocks,
        X_ref_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        X_ref, diag_blocksize, arrow_blocksize
    )

    (
        X_ref_diagonal_blocks_local,
        X_ref_lower_diagonal_blocks_local,
        X_ref_upper_diagonal_blocks_local,
        X_ref_arrow_bottom_blocks_local,
        X_ref_arrow_right_blocks_local,
        X_ref_arrow_tip_block_local,
    ) = extract_partition_tridiagonal_arrowhead_array(
        X_ref_diagonal_blocks,
        X_ref_lower_diagonal_blocks,
        X_ref_upper_diagonal_blocks,
        X_ref_arrow_bottom_blocks,
        X_ref_arrow_right_blocks,
        X_ref_arrow_tip_block,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
    )

    (
        X_ref_bridges_lower,
        X_ref_bridges_upper,
    ) = extract_bridges_tridiagonal_array(
        X_ref_lower_diagonal_blocks, X_ref_upper_diagonal_blocks, start_blockrows
    )
    # -----------------------------------

    (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    ) = convert_block_tridiagonal_arrowhead_dense_to_arrays(
        A, diag_blocksize, arrow_blocksize
    )

    (
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_right_blocks_local,
        A_arrow_tip_block_local,
    ) = extract_partition_tridiagonal_arrowhead_array(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
    )

    (A_bridges_lower, A_bridges_upper) = extract_bridges_tridiagonal_array(
        A_lower_diagonal_blocks, A_upper_diagonal_blocks, start_blockrows
    )

    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_arrow_tip_block_local,
        X_bridges_lower,
        X_bridges_upper,
    ) = lu_dist_tridiagonal_arrowhead_gpu(
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_right_blocks_local,
        A_arrow_tip_block_local,
        A_bridges_lower,
        A_bridges_upper,
    )

    assert np.allclose(X_ref_diagonal_blocks_local, X_diagonal_blocks_local)
    assert np.allclose(X_ref_lower_diagonal_blocks_local, X_lower_diagonal_blocks_local)
    assert np.allclose(X_ref_upper_diagonal_blocks_local, X_upper_diagonal_blocks_local)
    assert np.allclose(X_ref_arrow_bottom_blocks_local, X_arrow_bottom_blocks_local)
    assert np.allclose(X_ref_arrow_right_blocks_local, X_arrow_right_blocks_local)
    assert np.allclose(X_ref_arrow_tip_block_local, X_arrow_tip_block_local)

    # Check for bridges correctness
    if comm_rank == 0:
        assert np.allclose(
            X_ref_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ],
            X_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ],
        )
    elif comm_rank == comm_size - 1:
        assert np.allclose(
            X_ref_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ],
            X_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ],
        )
    else:
        assert np.allclose(
            X_ref_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ],
            X_bridges_upper[
                :, comm_rank * diag_blocksize : (comm_rank + 1) * diag_blocksize
            ],
        )
        assert np.allclose(
            X_ref_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ],
            X_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ],
        )


if __name__ == "__main__":
    test_lu_dist(10, 3, 2)
