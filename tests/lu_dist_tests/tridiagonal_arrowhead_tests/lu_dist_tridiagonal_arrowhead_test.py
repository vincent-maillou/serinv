"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Integration testing of the lu_dist algorithm for tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils import matrix_transform
from sdr.utils import dist_utils
from sdr.lu_dist.lu_dist_tridiagonal_arrowhead import lu_dist_tridiagonal_arrowhead


import numpy as np
import copy as cp
import pytest
from mpi4py import MPI


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
    ]
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
    
    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )
    
    (
        start_blockrows, 
        partition_sizes, 
        end_blockrows
    ) = dist_utils.get_partitions_indices(
        n_partitions = comm_size, 
        total_size = nblocks - 1
    )
    
    # ----- Reference/Checking data -----
    A_ref = cp.deepcopy(A)

    X_ref = np.linalg.inv(A_ref)
    
    (
        X_ref_local, 
        X_ref_arrow_bottom, 
        X_ref_arrow_right,
        X_ref_arrow_tip
    ) = dist_utils.extract_partition_tridiagonal_arrowhead_dense(
        X_ref,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )

    (
        X_ref_bridges_upper, 
        X_ref_bridges_lower
    ) = dist_utils.extract_bridges_tridiagonal_dense(
        X_ref, 
        diag_blocksize, 
        start_blockrows
    )

    X_ref_local = matrix_transform.cut_to_blocktridiag(X_ref_local, diag_blocksize)
    # -----------------------------------

    (
        A_local, 
        A_arrow_bottom, 
        A_arrow_right,
        A_arrow_tip
    ) = dist_utils.extract_partition_tridiagonal_arrowhead_dense(
        A,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )
    
    Bridges_upper, Bridges_lower = dist_utils.extract_bridges_tridiagonal_dense(
        A, diag_blocksize, start_blockrows
    )

    (
        X_local, 
        X_bridges_upper, 
        X_bridges_lower, 
        X_arrow_bottom, 
        X_arrow_right, 
        X_global_arrow_tip
    ) = lu_dist_tridiagonal_arrowhead(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        A_arrow_tip,
        Bridges_upper,
        Bridges_lower,
        diag_blocksize,
        arrow_blocksize,
    )

    X_local = matrix_transform.cut_to_blocktridiag(X_local, diag_blocksize)

    assert np.allclose(X_ref_local, X_local)
    assert np.allclose(X_ref_arrow_bottom, X_arrow_bottom)
    assert np.allclose(X_ref_arrow_right, X_arrow_right)
    assert np.allclose(X_ref_arrow_tip, X_global_arrow_tip)

    # Check for bridges correctness
    if comm_rank == 0:
        assert np.allclose(X_ref_bridges_upper[comm_rank], X_bridges_upper[comm_rank])
    elif comm_rank == comm_size-1:
        assert np.allclose(X_ref_bridges_lower[comm_rank-1], X_bridges_lower[comm_rank-1])
    else:
        assert np.allclose(X_ref_bridges_upper[comm_rank], X_bridges_upper[comm_rank])
        assert np.allclose(X_ref_bridges_lower[comm_rank-1], X_bridges_lower[comm_rank-1])
        