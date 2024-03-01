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
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as plt
from mpi4py import MPI


def get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> [[], [], []]:
    """Create the partitions start/end indices and sizes for the entire problem.
    If the problem size doesn't match a perfect partitioning w.r.t the distribution,
    partitions will be resized starting from the first one.

    Parameters
    ----------
    n_partitions : int
        Total number of partitions.
    total_size : int
        Total number of blocks in the global matrix. Equal to the sum of the sizes of
        all partitions.
    partitions_distribution : list, optional
        Distribution of the partitions sizes, in percentage. The default is None
        and a uniform distribution is assumed.

    Returns
    -------
    start_blockrows : []
        List of the indices of the first blockrow of each partition in the
        global matrix.
    partition_sizes : []
        List of the sizes of each partition.
    end_blockrows : []
        List of the indices of the last blockrow of each partition in the
        global matrix.

    """

    if n_partitions > total_size:
        raise ValueError(
            "Number of partitions cannot be greater than the total size of the matrix."
        )

    if partitions_distribution is not None:
        if n_partitions != len(partitions_distribution):
            raise ValueError(
                "Number of partitions and number of entries in the distribution list do not match."
            )
        if sum(partitions_distribution) != 100:
            raise ValueError(
                "Sum of the entries in the distribution list is not equal to 100."
            )
    else:
        partitions_distribution = [100 / n_partitions] * n_partitions

    partitions_distribution = np.array(partitions_distribution) / 100

    start_blockrows = []
    partition_sizes = []
    end_blockrows = []

    for i in range(n_partitions):
        partition_sizes.append(math.floor(partitions_distribution[i] * total_size))

    if sum(partition_sizes) != total_size:
        diff = total_size - sum(partition_sizes)
        for i in range(diff):
            partition_sizes[i] += 1

    for i in range(n_partitions):
        start_blockrows.append(sum(partition_sizes[:i]))
        end_blockrows.append(start_blockrows[i] + partition_sizes[i])

    return start_blockrows, partition_sizes, end_blockrows


def extract_partition_tridiagonal_arrowhead_dense(
    A_global: np.ndarray,
    start_blockrow: int,
    partition_size: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    A_local = np.zeros(
        (partition_size * diag_blocksize, partition_size * diag_blocksize), dtype=A_global.dtype
    )
    A_arrow_bottom = np.zeros(
        (arrow_blocksize, partition_size * arrow_blocksize), dtype=A_global.dtype
    )
    A_arrow_right = np.zeros(
        (partition_size * arrow_blocksize, arrow_blocksize), dtype=A_global.dtype
    )

    stop_blockrow = start_blockrow + partition_size

    A_local = A_global[
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize,
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize,
    ]
    A_arrow_bottom = A_global[
        -arrow_blocksize:, start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize
    ]
    A_arrow_right = A_global[
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize, -arrow_blocksize:
    ]
    
    A_arrow_tip = A_global[-arrow_blocksize:, -arrow_blocksize:]

    return A_local, A_arrow_bottom, A_arrow_right, A_arrow_tip


def extract_partition_tridiagonal_arrowhead_array(
    A_diagonal_blocks: np.ndarray, 
    A_lower_diagonal_blocks: np.ndarray, 
    A_upper_diagonal_blocks: np.ndarray, 
    A_arrow_bottom_blocks: np.ndarray, 
    A_arrow_right_blocks: np.ndarray, 
    A_arrow_tip_block: np.ndarray,
    start_blockrow: int,
    partition_size: int,
):
    diag_blocksize = A_diagonal_blocks[0].shape[0]
    arrow_blocksize = A_arrow_bottom_blocks[0].shape[0]
    
    A_diagonal_blocks_local = np.empty((diag_blocksize, partition_size * diag_blocksize), dtype=A_diagonal_blocks.dtype)
    A_lower_diagonal_blocks_local = np.empty((diag_blocksize, (partition_size - 1) * diag_blocksize), dtype=A_diagonal_blocks.dtype)
    A_upper_diagonal_blocks_local = np.empty((diag_blocksize, (partition_size - 1) * diag_blocksize), dtype=A_diagonal_blocks.dtype)
    A_arrow_bottom_blocks_local = np.empty((arrow_blocksize, (partition_size - 1) * diag_blocksize), dtype=A_diagonal_blocks.dtype)
    A_arrow_right_blocks_local = np.empty(((partition_size - 1) * diag_blocksize, arrow_blocksize), dtype=A_diagonal_blocks.dtype)
    A_arrow_tip_block_local = np.empty((arrow_blocksize, arrow_blocksize), dtype=A_diagonal_blocks.dtype)

    stop_blockrow = start_blockrow + partition_size

    A_diagonal_blocks_local = A_diagonal_blocks[:, start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize]
    A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[:, start_blockrow * diag_blocksize : (stop_blockrow - 1) * diag_blocksize]
    A_upper_diagonal_blocks_local = A_upper_diagonal_blocks[:, start_blockrow * diag_blocksize : (stop_blockrow - 1) * diag_blocksize]
    
    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks[:, start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize]
    A_arrow_right_blocks_local = A_arrow_right_blocks[start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize, :]
    
    A_arrow_tip_block_local = A_arrow_tip_block

    return (
        A_diagonal_blocks_local, 
        A_lower_diagonal_blocks_local, 
        A_upper_diagonal_blocks_local, 
        A_arrow_bottom_blocks_local, 
        A_arrow_right_blocks_local, 
        A_arrow_tip_block_local
    )


def extract_bridges_tridiagonal_dense(
    A: np.ndarray,
    diag_blocksize: int,
    start_blockrows: list,
) -> [list, list]:
    
    Bridges_lower: list = []
    Bridges_upper: list = []
    
    for i in range(1, len(start_blockrows)):
        upper_bridge = np.zeros((diag_blocksize, diag_blocksize))
        lower_bridge = np.zeros((diag_blocksize, diag_blocksize))
        
        start_ixd = start_blockrows[i]*diag_blocksize
        
        upper_bridge = A[start_ixd-diag_blocksize:start_ixd, start_ixd:start_ixd+diag_blocksize]
        lower_bridge = A[start_ixd:start_ixd+diag_blocksize, start_ixd-diag_blocksize:start_ixd]
        
        Bridges_upper.append(upper_bridge)
        Bridges_lower.append(lower_bridge)
        
    return Bridges_upper, Bridges_lower


def extract_bridges_tridiagonal_array(
    A_diagonal_blocks: np.ndarray, 
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray, 
    start_blockrows: list,
) -> [list, list]:
    diag_blocksize = A_diagonal_blocks[0].shape[0]
    
    Bridges_lower: list = []
    Bridges_upper: list = []
    
    for i in range(1, len(start_blockrows)):
        upper_bridge = np.empty((diag_blocksize, diag_blocksize))
        lower_bridge = np.empty((diag_blocksize, diag_blocksize))
        
        start_ixd = start_blockrows[i]*diag_blocksize
        
        lower_bridge = A_lower_diagonal_blocks[:, start_ixd-diag_blocksize:start_ixd]
        upper_bridge = A_upper_diagonal_blocks[:, start_ixd:start_ixd+diag_blocksize]
        
        Bridges_upper.append(upper_bridge)
        Bridges_lower.append(lower_bridge)
        
    return Bridges_upper, Bridges_lower

