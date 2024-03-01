"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Example of the lu_dist algorithm for tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils import matrix_transform as mt
from sdr.utils import dist_utils as du
from sdr.lu_dist.lu_dist_tridiagonal_arrowhead import lu_dist_tridiagonal_arrowhead

import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


# ----- Integration test -----
if __name__ == "__main__":
    nblocks = 13
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    (
        start_blockrows, 
        partition_sizes, 
        end_blockrows
    ) = du.get_partitions_indices(
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
    ) = du.extract_partition_tridiagonal_arrowhead_dense(
        X_ref,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )

    Bridges_upper_inv_ref, Bridges_lower_inv_ref = du.extract_bridges_tridiagonal_dense(
        X_ref, diag_blocksize, start_blockrows
    )

    X_ref_local = mt.cut_to_blocktridiag(X_ref_local, diag_blocksize)
    # ----- Reference/Checking data -----


    (
        A_local, 
        A_arrow_bottom, 
        A_arrow_right,
        A_arrow_tip
    ) = du.extract_partition_tridiagonal_arrowhead_dense(
        A,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )
    
    Bridges_upper, Bridges_lower = du.extract_bridges_tridiagonal_dense(
        A, diag_blocksize, start_blockrows
    )
    
    
    fig, axs = plt.subplots(2, 5)
    fig.suptitle("Initial datas, process: " + str(comm_rank))
    axs[0, 0].matshow(A)
    axs[0, 0].set_title("A")
    axs[0, 1].matshow(A_local)
    axs[0, 1].set_title("A_local")
    axs[0, 2].matshow(A_arrow_bottom)
    axs[0, 2].set_title("A_arrow_bottom")
    axs[0, 3].matshow(A_arrow_right)
    axs[0, 3].set_title("A_arrow_right")
    axs[0, 4].matshow(A_arrow_tip)
    axs[0, 4].set_title("A_arrow_tip")
    
    axs[1, 0].matshow(X_ref)
    axs[1, 0].set_title("X_ref")
    axs[1, 1].matshow(X_ref_local)
    axs[1, 1].set_title("X_ref_local")
    axs[1, 2].matshow(X_ref_arrow_bottom)
    axs[1, 2].set_title("X_ref_arrow_bottom")
    axs[1, 3].matshow(X_ref_arrow_right)
    axs[1, 3].set_title("X_ref_arrow_right")
    axs[1, 4].matshow(X_ref_arrow_tip)
    axs[1, 4].set_title("X_ref_arrow_tip")
    plt.show()


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

    X_local_cut_tridiag = mt.cut_to_blocktridiag(X_local, diag_blocksize)

    fig, axs = plt.subplots(2, 4)
    fig.suptitle("Results, process: " + str(comm_rank))
    axs[0, 0].matshow(X_ref_local)
    axs[0, 0].set_title("X_ref_local")
    axs[0, 1].matshow(X_ref_arrow_bottom)
    axs[0, 1].set_title("X_ref_arrow_bottom")
    axs[0, 2].matshow(X_ref_arrow_right)
    axs[0, 2].set_title("X_ref_arrow_right")
    axs[0, 3].matshow(X_ref_arrow_tip)
    axs[0, 3].set_title("X_ref_arrow_tip")
    
    axs[1, 0].matshow(X_local_cut_tridiag)
    axs[1, 0].set_title("X_local_cut_tridiag")
    axs[1, 1].matshow(X_arrow_bottom)
    axs[1, 1].set_title("X_arrow_bottom")
    axs[1, 2].matshow(X_arrow_right)
    axs[1, 2].set_title("X_arrow_right")
    axs[1, 3].matshow(X_global_arrow_tip)
    axs[1, 3].set_title("X_global_arrow_tip")
    plt.show()

    # ----- VERFIFYING THE RESULTS -----

    assert np.allclose(X_ref_local, X_local_cut_tridiag)
    norme_diff_partition_i = np.linalg.norm(X_ref_local - X_local_cut_tridiag)
    print("Partition n", comm_rank, "  norm diff = ", norme_diff_partition_i)
    
    assert np.allclose(X_ref_arrow_bottom, X_arrow_bottom)
    norme_diff_arrow_bottom_i = np.linalg.norm(X_ref_arrow_bottom - X_arrow_bottom)
    print("Arrow bottom n", comm_rank, "  norm diff = ", norme_diff_arrow_bottom_i)
    
    assert np.allclose(X_ref_arrow_right, X_arrow_right)
    norme_diff_arrow_right_i = np.linalg.norm(X_ref_arrow_right - X_arrow_right)
    print("Arrow right n", comm_rank, "  norm diff = ", norme_diff_arrow_right_i)
    
    assert np.allclose(X_ref_arrow_tip, X_global_arrow_tip)
    norme_diff_arrow_tip_i = np.linalg.norm(X_ref_arrow_tip - X_global_arrow_tip)
    print("Arrow tip n", comm_rank, "  norm diff = ", norme_diff_arrow_tip_i)

    # Check for bridges correctness
    print("Bridges correctness:")
    if comm_rank == 0:
        norme_diff_upper_bridge_i = np.linalg.norm(Bridges_upper_inv_ref[comm_rank] - X_bridges_upper[comm_rank])
        assert np.allclose(Bridges_upper_inv_ref[comm_rank], X_bridges_upper[comm_rank])
        print("    Upper bridge n", comm_rank, "  norm diff = ", norme_diff_upper_bridge_i)            
                        
    elif comm_rank == comm_size-1:
        norme_diff_lower_bridge_i = np.linalg.norm(Bridges_lower_inv_ref[comm_rank-1] - X_bridges_lower[comm_rank-1])
        assert np.allclose(Bridges_lower_inv_ref[comm_rank-1], X_bridges_lower[comm_rank-1])
        print("    Lower bridge n", comm_rank, "  norm diff = ", norme_diff_lower_bridge_i)
        
    else:
        norme_diff_upper_bridge_i = np.linalg.norm(Bridges_upper_inv_ref[comm_rank] - X_bridges_upper[comm_rank])
        assert np.allclose(Bridges_upper_inv_ref[comm_rank], X_bridges_upper[comm_rank])
        print("    Upper bridge n", comm_rank, "  norm diff = ", norme_diff_upper_bridge_i)            
        
        norme_diff_lower_bridge_i = np.linalg.norm(Bridges_lower_inv_ref[comm_rank-1] - X_bridges_lower[comm_rank-1])
        assert np.allclose(Bridges_lower_inv_ref[comm_rank-1], X_bridges_lower[comm_rank-1])
        print("    Lower bridge n", comm_rank, "  norm diff = ", norme_diff_lower_bridge_i)
        
        