"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

import numpy as np
import math


def get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> tuple[list, list, list]:
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
    start_blockrows : list
        List of the indices of the first blockrow of each partition in the
        global matrix.
    partition_sizes : list
        List of the sizes of each partition.
    end_blockrows : list
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

    return (start_blockrows, partition_sizes, end_blockrows)


def extract_partition_tridiagonal_arrowhead_dense(
    A_global: np.ndarray,
    start_blockrow: int,
    partition_size: int,
    diag_blocksize: int,
    arrow_blocksize: int,
):
    """Extract the local partition of a block tridiagonal arrowhead matrix,
    passed as dense.

    Parameters
    ----------
    A_global : np.ndarray
        Global block tridiagonal matrix.
    start_blockrow : int
        Index of the first blockrow of the partition in the global matrix.
    partition_size : int
        Size of the partition.
    diag_blocksize : int
        Size of the diagonal blocks.
    arrow_blocksize : int
        Size of the arrow blocks.

    Returns
    -------
    A_local : np.ndarray
        Local diagonal blocks of the partition.
    A_arrow_bottom : np.ndarray
        Local arrow bottom blocks of the partition.
    A_arrow_right : np.ndarray
        Local arrow right blocks of the partition.
    A_arrow_tip : np.ndarray
        Local arrow tip block of the partition.
    """
    A_local = np.zeros(
        (partition_size * diag_blocksize, partition_size * diag_blocksize),
        dtype=A_global.dtype,
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
        -arrow_blocksize:,
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize,
    ]
    A_arrow_right = A_global[
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize,
        -arrow_blocksize:,
    ]

    A_arrow_tip = A_global[-arrow_blocksize:, -arrow_blocksize:]

    return (A_local, A_arrow_bottom, A_arrow_right, A_arrow_tip)


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
    """Extract the local partition of a block tridiagonal arrowhead matrix,
    passed as arrays.

    Parameters
    ----------
    A_diagonal_blocks : np.ndarray
        Diagonal blocks of the block tridiagonal matrix.
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the block tridiagonal matrix.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the block tridiagonal matrix.
    A_arrow_bottom_blocks : np.ndarray
        Arrow bottom blocks of the block tridiagonal matrix.
    A_arrow_right_blocks : np.ndarray
        Arrow right blocks of the block tridiagonal matrix.
    A_arrow_tip_block : np.ndarray
        Arrow tip block of the block tridiagonal matrix.
    start_blockrow : int
        Index of the first blockrow of the partition in the global matrix.
    partition_size : int
        Size of the partition.

    Returns
    -------
    A_diagonal_blocks_local : np.ndarray
        Local diagonal blocks of the partition.
    A_lower_diagonal_blocks_local : np.ndarray
        Local lower diagonal blocks of the partition.
    A_upper_diagonal_blocks_local : np.ndarray
        Local upper diagonal blocks of the partition.
    A_arrow_bottom_blocks_local : np.ndarray
        Local arrow bottom blocks of the partition.
    A_arrow_right_blocks_local : np.ndarray
        Local arrow right blocks of the partition.
    A_arrow_tip_block_local : np.ndarray
        Local arrow tip block of the partition.
    """
    diag_blocksize = A_diagonal_blocks.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks.shape[0]

    A_diagonal_blocks_local = np.empty(
        (diag_blocksize, partition_size * diag_blocksize), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_local = np.empty(
        (diag_blocksize, (partition_size - 1) * diag_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    A_upper_diagonal_blocks_local = np.empty(
        (diag_blocksize, (partition_size - 1) * diag_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    A_arrow_bottom_blocks_local = np.empty(
        (arrow_blocksize, (partition_size - 1) * diag_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    A_arrow_right_blocks_local = np.empty(
        ((partition_size - 1) * diag_blocksize, arrow_blocksize),
        dtype=A_diagonal_blocks.dtype,
    )
    A_arrow_tip_block_local = np.empty(
        (arrow_blocksize, arrow_blocksize), dtype=A_diagonal_blocks.dtype
    )

    stop_blockrow = start_blockrow + partition_size

    A_diagonal_blocks_local = A_diagonal_blocks[
        :, start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize
    ]
    A_lower_diagonal_blocks_local = A_lower_diagonal_blocks[
        :, start_blockrow * diag_blocksize : (stop_blockrow - 1) * diag_blocksize
    ]
    A_upper_diagonal_blocks_local = A_upper_diagonal_blocks[
        :, start_blockrow * diag_blocksize : (stop_blockrow - 1) * diag_blocksize
    ]

    A_arrow_bottom_blocks_local = A_arrow_bottom_blocks[
        :, start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize
    ]
    A_arrow_right_blocks_local = A_arrow_right_blocks[
        start_blockrow * diag_blocksize : stop_blockrow * diag_blocksize, :
    ]

    A_arrow_tip_block_local = A_arrow_tip_block

    return (
        A_diagonal_blocks_local,
        A_lower_diagonal_blocks_local,
        A_upper_diagonal_blocks_local,
        A_arrow_bottom_blocks_local,
        A_arrow_right_blocks_local,
        A_arrow_tip_block_local,
    )


def extract_bridges_tridiagonal_dense(
    A: np.ndarray,
    diag_blocksize: int,
    start_blockrows: list,
) -> tuple[list, list]:
    """Extract the bridge blocks from the lower and upper diagonal blocks of a
    block tridiagonal matrix, passed as dense.

    Parameters
    ----------
    A : np.ndarray
        Block tridiagonal matrix.
    diag_blocksize : int
        Size of the diagonal blocks.
    start_blockrows : list
        List of the indices of the first blockrow of each partition in the
        global matrix.

    Returns
    -------
    Bridges_lower : list
        Lower bridge blocks.
    Bridges_upper : list
        Upper bridge blocks.
    """

    Bridges_lower: list = []
    Bridges_upper: list = []

    for i in range(1, len(start_blockrows)):
        upper_bridge = np.zeros((diag_blocksize, diag_blocksize))
        lower_bridge = np.zeros((diag_blocksize, diag_blocksize))

        start_ixd = start_blockrows[i] * diag_blocksize

        upper_bridge = A[
            start_ixd - diag_blocksize : start_ixd,
            start_ixd : start_ixd + diag_blocksize,
        ]
        lower_bridge = A[
            start_ixd : start_ixd + diag_blocksize,
            start_ixd - diag_blocksize : start_ixd,
        ]

        Bridges_upper.append(upper_bridge)
        Bridges_lower.append(lower_bridge)

    return (Bridges_upper, Bridges_lower)


def extract_bridges_tridiagonal_array(
    A_lower_diagonal_blocks: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray,
    start_blockrows: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the bridge blocks from the lower and upper diagonal blocks of a
    block tridiagonal matrix, passed as arrays.

    Parameters
    ----------
    A_lower_diagonal_blocks : np.ndarray
        Lower diagonal blocks of the block tridiagonal matrix.
    A_upper_diagonal_blocks : np.ndarray
        Upper diagonal blocks of the block tridiagonal matrix.
    start_blockrows : list
        List of the indices of the first blockrow of each partition in the
        global matrix.

    Returns
    -------
    Bridges_lower : np.ndarray
        Lower bridge blocks.
    Bridges_upper : np.ndarray
        Upper bridge blocks.
    """
    diag_blocksize = A_lower_diagonal_blocks.shape[0]
    n_bridges = len(start_blockrows) - 1

    Bridges_lower = np.zeros(
        (diag_blocksize, n_bridges * diag_blocksize),
        dtype=A_lower_diagonal_blocks.dtype,
    )
    Bridges_upper = np.zeros(
        (diag_blocksize, n_bridges * diag_blocksize),
        dtype=A_upper_diagonal_blocks.dtype,
    )

    for i in range(0, n_bridges):
        start_ixd = start_blockrows[i + 1] * diag_blocksize

        Bridges_lower[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = A_lower_diagonal_blocks[:, start_ixd - diag_blocksize : start_ixd]
        Bridges_upper[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = A_upper_diagonal_blocks[:, start_ixd - diag_blocksize : start_ixd]

    return (Bridges_lower, Bridges_upper)
