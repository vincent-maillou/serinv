"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils import matrix_transform as mt
from sdr.utils import dist_utils as du
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead

import numpy as np
import copy as cp
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpi4py import MPI


def lu_dist_tridiagonal_arrowhead(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    A_global_arrow_tip: np.ndarray,
    Bridges_upper: list,
    Bridges_lower: list,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Perform the distributed LU factorization and selected inversion of a 
    block tridiagonal arrowhead matrix.
     
    Parameters
    ----------
    # TODO:docstring
    
    Returns
    -------
    
    
    """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        (
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            Update_arrow_tip,
            L_local,
            U_local,
            L_arrow_bottom,
            U_arrow_right,
        ) = top_factorize(
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            diag_blocksize,
            arrow_blocksize,
        )
    else:
        (
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            Update_arrow_tip,
            L_local,
            U_local,
            L_arrow_bottom,
            U_arrow_right,
        ) = middle_factorize(
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            diag_blocksize,
            arrow_blocksize,
        )

    reduced_system = create_reduced_system(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        A_global_arrow_tip,
        Bridges_upper,
        Bridges_lower,
        Update_arrow_tip,
        diag_blocksize,
        arrow_blocksize,
    )

    reduced_system_inv = inverse_reduced_system(
        reduced_system, diag_blocksize, arrow_blocksize
    )

    S_local = np.zeros_like(A_local)
    S_arrow_bottom = np.zeros_like(A_arrow_bottom)
    S_arrow_right = np.zeros_like(A_arrow_right)
    S_global_arrow_tip = np.zeros_like(A_global_arrow_tip)

    (
        S_local,
        S_bridges_upper, 
        S_bridges_lower,
        S_arrow_bottom,
        S_arrow_right,
        S_global_arrow_tip,
    ) = update_sinv_reduced_system(
        S_local,
        S_arrow_bottom,
        S_arrow_right,
        S_global_arrow_tip,
        reduced_system_inv,
        Bridges_upper,
        Bridges_lower,
        diag_blocksize,
        arrow_blocksize,
    )
    
    if comm_rank == 0:
        (
            S_local, 
            S_arrow_bottom, 
            S_arrow_right
        ) = top_sinv(
            S_local,
            S_arrow_bottom,
            S_arrow_right,
            S_global_arrow_tip,
            L_local,
            U_local,
            L_arrow_bottom,
            U_arrow_right,
            diag_blocksize,
        )
    else:
        (
            S_local, 
            S_arrow_bottom, 
            S_arrow_right
        ) = middle_sinv(
            S_local,
            S_arrow_bottom,
            S_arrow_right,
            S_global_arrow_tip,
            L_local,
            U_local,
            L_arrow_bottom,
            U_arrow_right,
            diag_blocksize,
        )

    return S_local, S_bridges_upper, S_bridges_lower, S_arrow_bottom, S_arrow_right, S_global_arrow_tip


def top_factorize(
    A_diagonal_blocks_local: np.ndarray, 
    A_lower_diagonal_blocks_local: np.ndarray, 
    A_upper_diagonal_blocks_local: np.ndarray, 
    A_arrow_bottom_blocks_local: np.ndarray, 
    A_arrow_right_blocks_local: np.ndarray, 
    A_arrow_tip_block_local: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    nblocks = A_diagonal_blocks_local.shape[1] // diag_blocksize
    
    L_diagonal_blocks_local = np.zeros_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.zeros_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.zeros_like(A_arrow_bottom_blocks_local)
    U_diagonal_blocks_local = np.zeros_like(A_diagonal_blocks_local)
    U_upper_diagonal_blocks_local = np.zeros_like(A_upper_diagonal_blocks_local)
    U_arrow_right_blocks_local = np.zeros_like(A_arrow_right_blocks_local)
    Update_arrow_tip_local = np.zeros_like(A_arrow_tip_block_local)


    for i in range(nblocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )


        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U_local{i, i}^{-1}
        L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        
        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        

        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        
        # U_{i, i+1} = L_local{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_inv_temp
            @ A_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )
        
        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )
        
        
        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            A_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )


        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_local[:, -diag_blocksize:], 
        U_diagonal_blocks_local[:, -diag_blocksize:]
    ) = la.lu(
        A_diagonal_blocks_local[:, -diag_blocksize:], 
        permute_l=True
    )

    return (
        L_diagonal_blocks_local, 
        L_lower_diagonal_blocks_local, 
        L_arrow_bottom_blocks_local, 
        U_diagonal_blocks_local, 
        U_upper_diagonal_blocks_local, 
        U_arrow_right_blocks_local, 
        Update_arrow_tip_local
    )


def middle_factorize(
    A_diagonal_blocks_local: np.ndarray, 
    A_lower_diagonal_blocks_local: np.ndarray,
    A_upper_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_right_blocks_local: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray,
    A_left_2sided_arrow_blocks_local: np.ndarray,
    A_arrow_tip_block_local: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    n_blocks = A_diagonal_blocks_local.shape[1] // diag_blocksize
    
    L_diagonal_blocks_local = np.zeros_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.zeros_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.zeros_like(A_arrow_bottom_blocks_local)
    L_upper_2sided_arrow_blocks_local = np.zeros_like(A_top_2sided_arrow_blocks_local)
    U_diagonal_blocks_local = np.zeros_like(A_diagonal_blocks_local)
    U_upper_diagonal_blocks_local = np.zeros_like(A_upper_diagonal_blocks_local)
    U_arrow_right_blocks_local = np.zeros_like(A_arrow_right_blocks_local)
    U_left_2sided_arrow_blocks_local = np.zeros_like(A_left_2sided_arrow_blocks_local)
    Update_arrow_tip_local = np.zeros_like(A_arrow_tip_block_local)


    for i in range(1, n_blocks-1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )
        
        
        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )
        
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        
        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_top_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
            @ U_inv_temp
        )
        
        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        
        
        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_inv_temp
            @ A_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )
        
        # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
        U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        
        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )
        
        
        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            A_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )


        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        
        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
        A_diagonal_blocks_local[:, :diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
            - L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        # A_{i+1, top} = - L_{i+1, i} @ U_{i, top}
        A_left_2sided_arrow_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            - L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        # A_local[top, i+1] = - L[top, i] @ U_[i, i+1]
        A_top_2sided_arrow_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            - L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        )
        
        
        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
        A_arrow_bottom_blocks_local[:, :diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
            - L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
        A_arrow_right_blocks_local[:diag_blocksize, :] = (
            A_arrow_right_blocks_local[:diag_blocksize, :]
            - L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
    
    # Compute the last LU blocks of the 2-sided factorization pattern
    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_diagonal_blocks_local[:, -diag_blocksize:], 
        U_diagonal_blocks_local[:, -diag_blocksize:]
    ) = la.lu(
        A_diagonal_blocks_local[:, -diag_blocksize:], 
        permute_l=True
    )
    
    
    # Compute last lower factors
    U_inv_temp = la.solve_triangular(
        U_diagonal_blocks_local[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=False,
    )
    
    # L_{top, nblocks} = A_{top, nblocks} @ U{nblocks, nblocks}^{-1}
    L_upper_2sided_arrow_blocks_local[:, -diag_blocksize:] = (
        A_top_2sided_arrow_blocks_local[:, -diag_blocksize:]
        @ U_inv_temp
    )
    
    # L_{ndb+1, nblocks} = A_{ndb+1, nblocks} @ U{nblocks, nblocks}^{-1}
    L_arrow_bottom_blocks_local[:, -diag_blocksize:] = (
        A_arrow_bottom_blocks_local[:, -diag_blocksize:]
        @ U_inv_temp
    )
    
    
    # Compute last upper factors
    L_inv_temp = la.solve_triangular(
        L_diagonal_blocks_local[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=True,
    )
    
    # U_{nblocks, top} = L{nblocks, nblocks}^{-1} @ A_{nblocks, top}
    U_left_2sided_arrow_blocks_local[-diag_blocksize:, :] = (
        L_inv_temp
        @ A_left_2sided_arrow_blocks_local[-diag_blocksize:, :]
    )
    
    # U_{nblocks, ndb+1} = L{nblocks, nblocks}^{-1} @ A_{nblocks, ndb+1}
    U_arrow_right_blocks_local[-diag_blocksize:, :] = (
        L_inv_temp
        @ A_arrow_right_blocks_local[-diag_blocksize:, :]
    )
    
    
    # NOTE: On purpose, we don't update the tip of the arrowhead since the
    # propagation will appear during the inversion of the reduced system
    
    
    # Compute the top (first) LU blocks of the 2-sided factorization pattern
    # and its respective parts of the arrowhead
    # L_{top, top}, U_{top, top} = lu_dcmp(A_{top, top})
    (
        L_diagonal_blocks_local[:, :diag_blocksize], 
        U_diagonal_blocks_local[:, :diag_blocksize]
    ) = la.lu(
        A_diagonal_blocks_local[:, :diag_blocksize], 
        permute_l=True
    )
    
    
    # Compute top lower factors
    U_inv_temp = la.solve_triangular(
        U_diagonal_blocks_local[:, :diag_blocksize],
        np.eye(diag_blocksize),
        lower=False,
    )
    
    # L_{top+1, top} = A_{top+1, top} @ U{top, top}^{-1}
    L_lower_diagonal_blocks_local[:, :diag_blocksize] = (
        A_lower_diagonal_blocks_local[:, :diag_blocksize]
        @ U_inv_temp
    )
    
    # L_{ndb+1, top} = A_{ndb+1, top} @ U{top, top}^{-1}
    L_arrow_bottom_blocks_local[:, :diag_blocksize] = (
        A_arrow_bottom_blocks_local[:, :diag_blocksize]
        @ U_inv_temp
    )
    
    
    # Compute top upper factors
    L_inv_temp = la.solve_triangular(
        L_diagonal_blocks_local[:, :diag_blocksize], 
        np.eye(diag_blocksize),
        lower=True,
    )
    
    # U_{top, top+1} = L{top, top}^{-1} @ A_{top, top+1}
    U_upper_diagonal_blocks_local[:, :diag_blocksize] = (
        L_inv_temp
        @ A_upper_diagonal_blocks_local[:, :diag_blocksize]
    )
    
    # U_{top, ndb+1} = L{top, top}^{-1} @ A_{top, ndb+1}
    U_arrow_right_blocks_local[:diag_blocksize, :] = (
        L_inv_temp
        @ A_arrow_right_blocks_local[:diag_blocksize, :]
    )
    

    return (
        L_diagonal_blocks_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_upper_2sided_arrow_blocks_local,
        U_diagonal_blocks_local,
        U_upper_diagonal_blocks_local, 
        U_arrow_right_blocks_local,
        U_left_2sided_arrow_blocks_local,
        Update_arrow_tip_local
    )    


def create_reduced_system(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    A_global_arrow_tip: np.ndarray,
    Bridges_upper: list,
    Bridges_lower: list,
    Update_arrow_tip: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create empty matrix for reduced system -> (2*#process - 1)*diag_blocksize + arrowhead_size
    size_reduced_system = (2 * comm_size - 1) * diag_blocksize + arrow_blocksize
    reduced_system = np.zeros((size_reduced_system, size_reduced_system))
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = Update_arrow_tip[:,:]

    if comm_rank == 0:
        reduced_system[:diag_blocksize, :diag_blocksize] = A_local[-diag_blocksize:, -diag_blocksize:]
        reduced_system[:diag_blocksize, diag_blocksize : 2 * diag_blocksize] = Bridges_upper[comm_rank]

        reduced_system[-arrow_blocksize:, :diag_blocksize] = A_arrow_bottom[:, -diag_blocksize:]
        reduced_system[:diag_blocksize, -arrow_blocksize:] = A_arrow_right[-diag_blocksize:, :]
    else:
        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        reduced_system[
            start_index : start_index + diag_blocksize, start_index - diag_blocksize : start_index
        ] = Bridges_lower[comm_rank - 1]

        reduced_system[
            start_index : start_index + diag_blocksize, start_index : start_index + diag_blocksize
        ] = A_local[:diag_blocksize, :diag_blocksize]

        reduced_system[
            start_index : start_index + diag_blocksize,
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
        ] = A_local[:diag_blocksize, -diag_blocksize:]

        reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
            start_index : start_index + diag_blocksize,
        ] = A_local[-diag_blocksize:, :diag_blocksize]

        reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
        ] = A_local[-diag_blocksize:, -diag_blocksize:]

        if comm_rank != comm_size - 1:
            reduced_system[
                start_index + diag_blocksize : start_index + 2 * diag_blocksize,
                start_index + 2 * diag_blocksize : start_index + 3 * diag_blocksize,
            ] = Bridges_upper[comm_rank]

        reduced_system[
            -arrow_blocksize:, start_index : start_index + diag_blocksize
        ] = A_arrow_bottom[:, :diag_blocksize]

        reduced_system[
            -arrow_blocksize:, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_arrow_bottom[:, -diag_blocksize:]

        reduced_system[
            start_index : start_index + diag_blocksize, -arrow_blocksize:
        ] = A_arrow_right[:diag_blocksize, :]

        reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, -arrow_blocksize:
        ] = A_arrow_right[-diag_blocksize:, :]

    # Send the reduced_system with MPIallReduce SUM operation
    reduced_system_sum = np.zeros_like(reduced_system)
    comm.Allreduce(
        [reduced_system, MPI.DOUBLE], [reduced_system_sum, MPI.DOUBLE], op=MPI.SUM
    )

    # Add the global arrow tip to the reduced system arrow-tip summation
    reduced_system_sum[-arrow_blocksize:, -arrow_blocksize:] += A_global_arrow_tip

    return reduced_system_sum


def inverse_reduced_system(
    reduced_system, diag_blocksize, arrow_blocksize
) -> np.ndarray:
    n_diag_blocks = (reduced_system.shape[0] - arrow_blocksize) // diag_blocksize

    # ----- For checking dense inverse -----
    
    S_reduced = la.inv(reduced_system)
    
    # # ------------------------------------
    # # ----- Arrowhead solver -----
    # L_reduced, U_reduced = lu_factorize_tridiag_arrowhead(
    #     reduced_system, diag_blocksize, arrow_blocksize
    # )
    # S_reduced = lu_sinv_tridiag_arrowhead(
    #     L_reduced, U_reduced, diag_blocksize, arrow_blocksize
    # )
    # # # ----------------------------

    return S_reduced


def update_sinv_reduced_system(
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    S_global_arrow_tip: np.ndarray,
    reduced_system: np.ndarray,
    Bridges_upper: list,
    Bridges_lower: list,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        S_local[-diag_blocksize:, -diag_blocksize:] = reduced_system[:diag_blocksize, :diag_blocksize]

        Bridges_upper[comm_rank] = reduced_system[:diag_blocksize, diag_blocksize : 2 * diag_blocksize]

        S_arrow_bottom[:, -diag_blocksize:] = reduced_system[-arrow_blocksize:, :diag_blocksize]
        S_arrow_right[-diag_blocksize:, :] = reduced_system[:diag_blocksize, -arrow_blocksize:]
    else:
        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        Bridges_lower[comm_rank - 1] = reduced_system[
            start_index : start_index + diag_blocksize, start_index - diag_blocksize : start_index
        ]

        S_local[:diag_blocksize, :diag_blocksize] = reduced_system[
            start_index : start_index + diag_blocksize, start_index : start_index + diag_blocksize
        ]

        S_local[:diag_blocksize, -diag_blocksize:] = reduced_system[
            start_index : start_index + diag_blocksize,
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
        ]

        S_local[-diag_blocksize:, :diag_blocksize] = reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
            start_index : start_index + diag_blocksize,
        ]

        S_local[-diag_blocksize:, -diag_blocksize:] = reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
            start_index + diag_blocksize : start_index + 2 * diag_blocksize,
        ]

        if comm_rank != comm_size - 1:
            Bridges_upper[comm_rank] = reduced_system[
                start_index + diag_blocksize : start_index + 2 * diag_blocksize,
                start_index + 2 * diag_blocksize : start_index + 3 * diag_blocksize,
            ]

        S_arrow_bottom[:, :diag_blocksize] = reduced_system[
            -arrow_blocksize:, start_index : start_index + diag_blocksize
        ]

        S_arrow_bottom[:, -diag_blocksize:] = reduced_system[
            -arrow_blocksize:, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        S_arrow_right[:diag_blocksize, :] = reduced_system[
            start_index : start_index + diag_blocksize, -arrow_blocksize:
        ]

        S_arrow_right[-diag_blocksize:, :] = reduced_system[
            start_index + diag_blocksize : start_index + 2 * diag_blocksize, -arrow_blocksize:
        ]

    S_global_arrow_tip = reduced_system[-arrow_blocksize:, -arrow_blocksize:]

    return S_local, Bridges_upper, Bridges_lower, S_arrow_bottom, S_arrow_right, S_global_arrow_tip


def top_sinv(
    X_diagonal_blocks_local, 
    X_lower_diagonal_blocks_local, 
    X_upper_diagonal_blocks_local,
    X_arrow_bottom_blocks_local,
    X_arrow_right_blocks_local,
    X_global_arrow_tip_local,
    L_diagonal_blocks_local, 
    L_lower_diagonal_blocks_local, 
    L_arrow_bottom_blocks_local, 
    U_diagonal_blocks_local, 
    U_upper_diagonal_blocks_local, 
    U_arrow_right_blocks_local
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    L_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
    U_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype)

    for i in range(n_blocks - 2, -1, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )


        # --- Lower-diagonal blocks ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            - X_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv
        
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            - X_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_global_arrow_tip_local[:, :]
            @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv


        # --- Upper-diagonal blocks ---
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = U_blk_inv @ (
            - U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ X_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ X_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
        )
        
        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = U_blk_inv @ (
            - U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ X_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ X_global_arrow_tip_local[:, :]
        )


        # # --- Diagonal blocks ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            U_blk_inv
            - X_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv
        
        
    return (
        X_diagonal_blocks_local, 
        X_lower_diagonal_blocks_local, 
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip_local
    )


def middle_sinv(
    X_diagonal_blocks_local,
    X_lower_diagonal_blocks_local,
    X_upper_diagonal_blocks_local,
    X_arrow_bottom_blocks_local,
    X_arrow_right_blocks_local,
    X_top_2sided_arrow_blocks_local, 
    X_left_2sided_arrow_blocks_local,
    X_global_arrow_tip_block_local,
    L_diagonal_blocks_local, 
    L_lower_diagonal_blocks_local, 
    L_arrow_bottom_blocks_local, 
    L_upper_2sided_arrow_blocks_local,
    U_diagonal_blocks_local, 
    U_upper_diagonal_blocks_local, 
    U_arrow_right_blocks_local, 
    U_left_2sided_arrow_blocks_local
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize
    
    L_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=L_diagonal_blocks_local.dtype)
    U_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=U_diagonal_blocks_local.dtype)
    
    for i in range(n_blocks - 2, 0, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )
        
        # X_{i+1, i} = (- X_{i+1, top} L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            (
                - X_left_2sided_arrow_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] 
                @ L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - X_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] 
                @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
                - X_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] 
                @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )
        
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, top} X_{top, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        X_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            U_blk_inv @ (
                - U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
                @ X_diagonal_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                - U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_top_2sided_arrow_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] 
                - U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            )
        )
        
        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{top, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_top_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            (
                - X_top_2sided_arrow_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
                - X_diagonal_blocks_local[:, :diag_blocksize] 
                @ L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - X_arrow_right_blocks_local[:diag_blocksize, :] 
                @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )
        
        # X_{i, top} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, top} - U_{i, top} X_{top, top} - U_{i, ndb+1} X_{ndb+1, top})
        X_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            U_blk_inv @ (
                - U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
                @ X_left_2sided_arrow_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
                - U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_diagonal_blocks_local[:, :diag_blocksize] 
                - U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_arrow_bottom_blocks_local[:, :diag_blocksize]
            )
        )


        # Arrowhead
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            (
                - X_arrow_bottom_blocks_local[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
                - X_arrow_bottom_blocks_local[:, :diag_blocksize]
                @ L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - X_global_arrow_tip_block_local[:, :] 
                @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, top} X_{top, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            U_blk_inv @ (
                - U_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
                @ X_arrow_right_blocks_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] 
                - U_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_arrow_right_blocks_local[:diag_blocksize, :] 
                - U_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :] 
                @ X_global_arrow_tip_block_local[:, :] 
            )
        )
        
        
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, top} L_{top, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            U_blk_inv
            - X_upper_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ L_lower_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            - X_left_2sided_arrow_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ L_upper_2sided_arrow_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] 
            - X_arrow_right_blocks_local[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ L_arrow_bottom_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv
        
    
    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_upper_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_arrow_right_blocks_local,
        X_global_arrow_tip_block_local
    )
