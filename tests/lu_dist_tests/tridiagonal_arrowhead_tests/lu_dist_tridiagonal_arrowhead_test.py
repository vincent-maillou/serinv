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
from sdr.utils.matrix_transform import from_dense_to_arrowhead_arrays

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
        X_ref_diagonal_blocks, 
        X_ref_lower_diagonal_blocks, 
        X_ref_upper_diagonal_blocks, 
        X_ref_arrow_bottom_blocks, 
        X_ref_arrow_right_blocks, 
        X_ref_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(
        X_ref, 
        diag_blocksize, 
        arrow_blocksize
    )

    (
        X_ref_diagonal_blocks_local, 
        X_ref_lower_diagonal_blocks_local, 
        X_ref_upper_diagonal_blocks_local, 
        X_ref_arrow_bottom_blocks_local, 
        X_ref_arrow_right_blocks_local, 
        X_ref_arrow_tip_block_local
    ) = dist_utils.extract_partition_tridiagonal_arrowhead_array(
        X_ref_diagonal_blocks, 
        X_ref_lower_diagonal_blocks, 
        X_ref_upper_diagonal_blocks, 
        X_ref_arrow_bottom_blocks, 
        X_ref_arrow_right_blocks, 
        X_ref_arrow_tip_block,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank]
    )

    (
        X_ref_bridges_upper, 
        X_ref_bridges_lower
    ) = dist_utils.extract_bridges_tridiagonal_array(
        X_ref_lower_diagonal_blocks,
        X_ref_upper_diagonal_blocks, 
        start_blockrows
    )
    # -----------------------------------


    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 6)
    # fig.suptitle("Process " + str(comm_rank))
    # ax[0, 0].matshow(X_ref_diagonal_blocks)
    # ax[0, 0].set_title("X_ref_diagonal_blocks")
    # ax[1, 0].matshow(X_ref_diagonal_blocks_local)
    # ax[1, 0].set_title("X_ref_diagonal_blocks_local")

    # ax[0, 1].matshow(X_ref_lower_diagonal_blocks)
    # ax[0, 1].set_title("X_ref_lower_diagonal_blocks")
    # ax[1, 1].matshow(X_ref_lower_diagonal_blocks_local)
    # ax[1, 1].set_title("X_ref_lower_diagonal_blocks_local")

    # ax[0, 2].matshow(X_ref_upper_diagonal_blocks)
    # ax[0, 2].set_title("X_ref_upper_diagonal_blocks")
    # ax[1, 2].matshow(X_ref_upper_diagonal_blocks_local)
    # ax[1, 2].set_title("X_ref_upper_diagonal_blocks_local")

    # ax[0, 3].matshow(X_ref_arrow_bottom_blocks)
    # ax[0, 3].set_title("X_ref_arrow_bottom_blocks")
    # ax[1, 3].matshow(X_ref_arrow_bottom_blocks_local)
    # ax[1, 3].set_title("X_ref_arrow_bottom_blocks_local")

    # ax[0, 4].matshow(X_ref_arrow_right_blocks)
    # ax[0, 4].set_title("X_ref_arrow_right_blocks")
    # ax[1, 4].matshow(X_ref_arrow_right_blocks_local)
    # ax[1, 4].set_title("X_ref_arrow_right_blocks_local")

    # ax[0, 5].matshow(X_ref_arrow_tip_block)
    # ax[0, 5].set_title("X_ref_arrow_tip_block")
    # ax[1, 5].matshow(X_ref_arrow_tip_block_local)
    # ax[1, 5].set_title("X_ref_arrow_tip_block_local")
    # plt.show()


    (
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_right_blocks, 
        A_arrow_tip_block,
    ) = from_dense_to_arrowhead_arrays(
        A, 
        diag_blocksize, 
        arrow_blocksize
    )

    (
        A_diagonal_blocks_local, 
        A_lower_diagonal_blocks_local, 
        A_upper_diagonal_blocks_local, 
        A_arrow_bottom_blocks_local, 
        A_arrow_right_blocks_local, 
        A_arrow_tip_block_local
    ) = dist_utils.extract_partition_tridiagonal_arrowhead_array(
        A_diagonal_blocks, 
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks, 
        A_arrow_bottom_blocks, 
        A_arrow_right_blocks, 
        A_arrow_tip_block,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank]
    )

    (
        A_bridges_lower,
        A_bridges_upper
    ) = dist_utils.extract_bridges_tridiagonal_array(
        A_lower_diagonal_blocks, 
        A_upper_diagonal_blocks,  
        start_blockrows
    )

    # (
    #     X_diagonal_blocks_local, 
    #     X_lower_diagonal_blocks_local, 
    #     X_upper_diagonal_blocks_local, 
    #     X_arrow_bottom_blocks_local, 
    #     X_arrow_right_blocks_local, 
    #     X_arrow_tip_block_local,
    #     X_bridges_upper, 
    #     X_bridges_lower
    # ) = lu_dist_tridiagonal_arrowhead(
    #     A_diagonal_blocks_local, 
    #     A_lower_diagonal_blocks_local, 
    #     A_upper_diagonal_blocks_local, 
    #     A_arrow_bottom_blocks_local, 
    #     A_arrow_right_blocks_local, 
    #     A_arrow_tip_block_local,
    #     A_bridges_upper, 
    #     A_bridges_lower
    # )

    lu_dist_tridiagonal_arrowhead(
        A_diagonal_blocks_local, 
        A_lower_diagonal_blocks_local, 
        A_upper_diagonal_blocks_local, 
        A_arrow_bottom_blocks_local, 
        A_arrow_right_blocks_local, 
        A_arrow_tip_block_local,
        A_bridges_upper, 
        A_bridges_lower
    )

    # X_local = matrix_transform.cut_to_blocktridiag(X_local, diag_blocksize)

    # assert np.allclose(X_ref_local, X_local)
    # assert np.allclose(X_ref_arrow_bottom, X_arrow_bottom)
    # assert np.allclose(X_ref_arrow_right, X_arrow_right)
    # assert np.allclose(X_ref_arrow_tip, X_global_arrow_tip)

    # # Check for bridges correctness
    # if comm_rank == 0:
    #     assert np.allclose(X_ref_bridges_upper[comm_rank], X_bridges_upper[comm_rank])
    # elif comm_rank == comm_size-1:
    #     assert np.allclose(X_ref_bridges_lower[comm_rank-1], X_bridges_lower[comm_rank-1])
    # else:
    #     assert np.allclose(X_ref_bridges_upper[comm_rank], X_bridges_upper[comm_rank])
    #     assert np.allclose(X_ref_bridges_lower[comm_rank-1], X_bridges_lower[comm_rank-1])
        
if __name__ == "__main__":
    test_lu_dist(10, 3, 2)