from sdr.utils import matrix_generation
from sdr.utils import matrix_transform as mt
from sdr.lu.lu_decompose import lu_dcmp_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead

from sdr.lu.lu_decompose import lu_dcmp_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag


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


def extract_partition(
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

    return A_local, A_arrow_bottom, A_arrow_right


def extract_bridges(
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



### ------------------------ PSR ARROWHEAD ------------------------ ###
def top_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> [
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    L_local = np.zeros_like(A_local)
    U_local = np.zeros_like(A_local)
    
    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)
    
    Update_arrow_tip = np.zeros((arrow_blocksize, arrow_blocksize))

    nblocks = A_local.shape[0] // diag_blocksize

    for i in range(nblocks - 1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )


        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=False,
        )

        # L_{i+1, i} = A_{i+1, i} @ U_local{i, i}^{-1}
        L_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = A_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] @ U_inv_temp
        
        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        

        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )
        
        # L_{i, i+1} = L_local{i, i}^{-1} @ A_{i, i+1}
        U_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            L_inv_temp
            @ A_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )
        
        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )
        
        
        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )


        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            A_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )


        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip[:, :] = (
            Update_arrow_tip[:, :]
            - L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )

    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    L_local[-diag_blocksize:, -diag_blocksize:], U_local[-diag_blocksize:, -diag_blocksize:] = la.lu(
        A_local[-diag_blocksize:, -diag_blocksize:], permute_l=True
    )

    return (
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        Update_arrow_tip,
        L_local,
        U_local,
        L_arrow_bottom,
        U_arrow_right,
    )
    
    
def middle_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> [
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    L_local = np.zeros_like(A_local)
    U_local = np.zeros_like(A_local)

    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)
    
    Update_arrow_tip = np.zeros((arrow_blocksize, arrow_blocksize))

    n_blocks = A_local.shape[0] // diag_blocksize

    for i in range(1, n_blocks-1):
        # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
        (
            L_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            U_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
        ) = la.lu(
            A_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            permute_l=True,
        )
        
        
        # Compute lower factors
        U_inv_temp = la.solve_triangular(
            U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=False,
        )
        
        # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
        L_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = A_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] @ U_inv_temp
        
        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        L_local[
            0 : diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = A_local[
            0 : diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] @ U_inv_temp
        
        # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
        L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            A_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_inv_temp
        )
        
        
        # Compute upper factors
        L_inv_temp = la.solve_triangular(
            L_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )
        
        # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
        U_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            L_inv_temp
            @ A_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )
        
        # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
        U_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            0 : diag_blocksize,
        ] = (
            L_inv_temp
            @ A_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                0 : diag_blocksize,
            ]
        )
        
        # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
        U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :] = (
            L_inv_temp
            @ A_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        
        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
        A_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )
        
        
        # Update next upper/lower blocks of the arrowhead
        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
        A_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] = (
            A_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
        A_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] = (
            A_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, -arrow_blocksize:]
        )


        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
        Update_arrow_tip[:, :] = (
            Update_arrow_tip[:, :]
            - L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            @ U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
        )
        
        
        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
        A_local[
            0 : diag_blocksize,
            0 : diag_blocksize,
        ] = (
            A_local[
                0 : diag_blocksize,
                0 : diag_blocksize,
            ]
            - L_local[
                0 : diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                0 : diag_blocksize,
            ]
        )
        
        # A_{i+1, top} = - L_{i+1, i} @ U_{i, top}
        A_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            0 : diag_blocksize,
        ] = (
            - L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                0 : diag_blocksize,
            ]
        )
        
        # A_local[top, i+1] = - L[top, i] @ U_[i, i+1]
        A_local[
            0 : diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            - L_local[
                0 : diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
        )
        
        
        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
        A_arrow_bottom[
            :, 0 : diag_blocksize,
        ] = (
            A_arrow_bottom[
                :, 0 : diag_blocksize,
            ]
            - L_arrow_bottom[
                :, i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                0 : diag_blocksize,
            ]
        )
        
        # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
        A_arrow_right[
            0 : diag_blocksize, :
        ] = (
            A_arrow_right[
                0 : diag_blocksize, :
            ]
            - L_local[
                0 : diag_blocksize, 
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ U_arrow_right[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                :,
            ]
        )
        
    
    # Compute the last LU blocks of the 2-sided factorization pattern
    # L_{nblocks, nblocks}, U_{nblocks, nblocks} = lu_dcmp(A_{nblocks, nblocks})
    (
        L_local[-diag_blocksize:, -diag_blocksize:], 
        U_local[-diag_blocksize:, -diag_blocksize:]
    ) = la.lu(
        A_local[-diag_blocksize:, -diag_blocksize:], permute_l=True
    )
    
    
    # Compute last lower factors
    U_inv_temp = la.solve_triangular(
        U_local[
            -diag_blocksize:, -diag_blocksize:
        ],
        np.eye(diag_blocksize),
        lower=False,
    )
    
    # L_{top, nblocks} = A_{top, nblocks} @ U{nblocks, nblocks}^{-1}
    L_local[
        0 : diag_blocksize,
        -diag_blocksize:,
    ] = A_local[
        0 : diag_blocksize,
        -diag_blocksize:,
    ] @ U_inv_temp
    
    # L_{ndb+1, nblocks} = A_{ndb+1, nblocks} @ U{nblocks, nblocks}^{-1}
    L_arrow_bottom[:, -diag_blocksize:] = (
        A_arrow_bottom[:, -diag_blocksize:]
        @ U_inv_temp
    )
    
    
    # Compute last upper factors
    L_inv_temp = la.solve_triangular(
        L_local[
            -diag_blocksize:, -diag_blocksize:
        ],
        np.eye(diag_blocksize),
        lower=True,
    )
    
    # U_{nblocks, top} = L{nblocks, nblocks}^{-1} @ A_{nblocks, top}
    U_local[
        -diag_blocksize:,
        0 : diag_blocksize,
    ] = (
        L_inv_temp
        @ A_local[
            -diag_blocksize:,
            0 : diag_blocksize,
        ]
    )
    
    # U_{nblocks, ndb+1} = L{nblocks, nblocks}^{-1} @ A_{nblocks, ndb+1}
    U_arrow_right[-diag_blocksize:, :] = (
        L_inv_temp
        @ A_arrow_right[-diag_blocksize:, :]
    )
    
    # NOTE: On purpose, I don't update the tip of the arrowhead since the
    # propagation will appear during the inversion of the reduced system
    
    
    # Compute the top (first) LU blocks of the 2-sided factorization pattern
    # and its respective parts of the arrowhead
    # L_{top, top}, U_{top, top} = lu_dcmp(A_{top, top})
    (
        L_local[0:diag_blocksize, 0:diag_blocksize], 
        U_local[0:diag_blocksize, 0:diag_blocksize]
    ) = la.lu(
        A_local[0:diag_blocksize, 0:diag_blocksize], permute_l=True
    )
    
    
    # Compute top lower factors
    U_inv_temp = la.solve_triangular(
        U_local[
            0:diag_blocksize, 0:diag_blocksize
        ],
        np.eye(diag_blocksize),
        lower=False,
    )
    
    # L_{top+1, top} = A_{top+1, top} @ U{top, top}^{-1}
    L_local[
        diag_blocksize:2*diag_blocksize,
        0:diag_blocksize,
    ] = A_local[
        diag_blocksize:2*diag_blocksize,
        0:diag_blocksize,
    ] @ U_inv_temp
    
    # L_{ndb+1, top} = A_{ndb+1, top} @ U{top, top}^{-1}
    L_arrow_bottom[
        :, 0:diag_blocksize,
    ] = A_arrow_bottom[
        :, 0:diag_blocksize,
    ] @ U_inv_temp
    
    
    # Compute top upper factors
    L_inv_temp = la.solve_triangular(
        L_local[
            0:diag_blocksize, 0:diag_blocksize
        ],
        np.eye(diag_blocksize),
        lower=True,
    )
    
    # U_{top, top+1} = L{top, top}^{-1} @ A_{top, top+1}
    U_local[
        0:diag_blocksize,
        diag_blocksize:2*diag_blocksize,
    ] = (
        L_inv_temp
        @ A_local[
            0:diag_blocksize,
            diag_blocksize:2*diag_blocksize,
        ]
    )
    
    # U_{top, ndb+1} = L{top, top}^{-1} @ A_{top, ndb+1}
    U_arrow_right[
        0:diag_blocksize,
        :,
    ] = (
        L_inv_temp
        @ A_arrow_right[
            0:diag_blocksize,
            :,
        ]
    )
    

    return (
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        Update_arrow_tip,
        L_local,
        U_local,
        L_arrow_bottom,
        U_arrow_right,
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
    
    # ------------------------------------
    # # ----- Arrowhead solver -----
    # L_reduced, U_reduced = lu_dcmp_tridiag_arrowhead(
    #     reduced_system, diag_blocksize, arrow_blocksize
    # )
    # S_reduced = lu_sinv_tridiag_arrowhead(
    #     L_reduced, U_reduced, diag_blocksize, arrow_blocksize
    # )
    # # ----------------------------

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
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    S_global_arrow_tip: np.ndarray,
    L_local: np.ndarray,
    U_local: np.ndarray,
    L_arrow_bottom: np.ndarray,
    U_arrow_right: np.ndarray,
    diag_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:

    L_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=L_local.dtype)
    U_blk_inv = np.empty((diag_blocksize, diag_blocksize), dtype=U_local.dtype)

    n_blocks = S_local.shape[0] // diag_blocksize
    for i in range(n_blocks - 2, -1, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )


        # --- Lower-diagonal blocks ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            - S_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - S_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv
        
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            - S_arrow_bottom[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - S_global_arrow_tip[:, :]
            @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv


        # --- Upper-diagonal blocks ---
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        S_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = U_blk_inv @ (
            - U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ S_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ S_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
        )
        
        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        S_arrow_right[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = U_blk_inv @ (
            -U_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ S_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :]
            - U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ S_global_arrow_tip[:, :]
        )


        # # --- Diagonal blocks ---
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            U_blk_inv
            - S_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - S_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :]
            @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
        ) @ L_blk_inv
        
        
    return S_local, S_arrow_bottom, S_arrow_right


def middle_sinv(
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    S_global_arrow_tip: np.ndarray,
    L_local: np.ndarray,
    U_local: np.ndarray,
    L_arrow_bottom: np.ndarray,
    U_arrow_right: np.ndarray,
    diag_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:

    n_blocks = S_local.shape[0] // diag_blocksize
    for i in range(n_blocks - 2, 0, -1):
        # ----- Block-tridiagonal solver -----
        L_blk_inv = la.solve_triangular(
            L_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=True,
        )
        U_blk_inv = la.solve_triangular(
            U_local[i * diag_blocksize : (i + 1) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize],
            np.eye(diag_blocksize),
            lower=False,
        )
        
        # X_{i+1, i} = (- X_{i+1, top} L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_local[
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            (
                - S_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, 0:diag_blocksize] @ L_local[0:diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - S_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] @ L_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize]
                - S_arrow_right[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, :] @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )
        
        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, top} X_{top, i+1} - U_{i, ndb+1} X_{ndb+1, i+1})
        S_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            U_blk_inv @ (
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] @ S_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, 0:diag_blocksize] @ S_local[0:diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] 
                - U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :] @ S_arrow_bottom[:, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            )
        )
        
        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{top, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_local[
            0:diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            (
                - S_local[0:diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] @ L_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - S_local[0:diag_blocksize, 0:diag_blocksize] @ L_local[0:diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize]
                - S_arrow_right[0:diag_blocksize, :] @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )
        
        # X_{i, top} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, top} - U_{i, top} X_{top, top} - U_{i, ndb+1} X_{ndb+1, top})
        S_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            0:diag_blocksize,
        ] = (
            U_blk_inv @ (
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize] @ S_local[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize, 0:diag_blocksize]
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, 0:diag_blocksize] @ S_local[0:diag_blocksize, 0:diag_blocksize] 
                - U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :] @ S_arrow_bottom[:, 0:diag_blocksize]
            )
        )


        # Arrowhead
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_arrow_bottom[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            (
                - S_arrow_bottom[:, (i+1) * diag_blocksize : (i+2) * diag_blocksize] @ L_local[(i+1) * diag_blocksize : (i+2) * diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize] 
                - S_arrow_bottom[:, 0 : diag_blocksize] @ L_local[0 : diag_blocksize, i * diag_blocksize : (i + 1) * diag_blocksize]
                - S_global_arrow_tip[:, :] @ L_arrow_bottom[:, i * diag_blocksize : (i + 1) * diag_blocksize]
            ) @ L_blk_inv
        )

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, top} X_{top, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        S_arrow_right[
            i * diag_blocksize : (i + 1) * diag_blocksize, :
        ] = (
            U_blk_inv @ (
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, (i+1) * diag_blocksize : (i+2) * diag_blocksize] @ S_arrow_right[(i+1) * diag_blocksize : (i+2) * diag_blocksize, :]
                - U_local[i * diag_blocksize : (i + 1) * diag_blocksize, 0 : diag_blocksize] @ S_arrow_right[0 : diag_blocksize, :] 
                - U_arrow_right[i * diag_blocksize : (i + 1) * diag_blocksize, :] @ S_global_arrow_tip[:, :]
            )
        )
        
        
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, top} L_{top, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        S_local[
            i * diag_blocksize : (i + 1) * diag_blocksize,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            U_blk_inv
            - S_local[
                i * diag_blocksize : (i + 1) * diag_blocksize,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            @ L_local[
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            - S_local[
                i * diag_blocksize : (i + 1) * diag_blocksize, 
                0:diag_blocksize:
            ]
            @ L_local[
                0:diag_blocksize, 
                i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - S_arrow_right[
                i * diag_blocksize : (i + 1) * diag_blocksize, 
                :
            ]
            @ L_arrow_bottom[
                :, 
                i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_blk_inv
        
        
    
    return S_local, S_arrow_bottom, S_arrow_right



def psr_arrowhead(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    A_global_arrow_tip: np.ndarray,
    Bridges_upper: list,
    Bridges_lower: list,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
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


import copy as cp


""" # ----- Local checking -----
# ...of top process
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    diagonal_dominant = True
    symmetric = False
    seed = 63
    
    A_init = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # ----- Reference -----
    A_ref_init = cp.deepcopy(A_init)

    A_ref_inv = np.linalg.inv(A_ref_init)
    
    A_local_refinv, A_arrow_bottom_refinv, A_arrow_right_refinv = extract_partition(
        A_ref_inv,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )
    
    A_arrow_tip_refinv = A_ref_inv[-arrow_blocksize:, -arrow_blocksize:]
    # ---------------------

    A_local, A_arrow_bottom, A_arrow_right = extract_partition(
        A_init,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )

    A_arrow_tip = A_init[-arrow_blocksize:, -arrow_blocksize:]

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

    fig, axs = plt.subplots(2, 5)
    fig.suptitle("After factorization")
    axs[0 ,0].matshow(A_ref_init)
    axs[0 ,0].set_title("A_ref_init")
    axs[0 ,1].matshow(A_local)
    axs[0 ,1].set_title("A_local")
    axs[0 ,2].matshow(A_arrow_right)
    axs[0 ,2].set_title("A_arrow_right")
    axs[0, 3].matshow(A_arrow_bottom)
    axs[0, 3].set_title("A_arrow_bottom")
    axs[0, 4].matshow(A_arrow_tip)
    axs[0, 4].set_title("A_arrow_tip")
    
    axs[1, 0].matshow(L_local)
    axs[1, 0].set_title("L_local")
    axs[1, 1].matshow(U_local)
    axs[1, 1].set_title("U_local")
    axs[1, 2].matshow(U_arrow_right)
    axs[1, 2].set_title("U_arrow_right")
    axs[1, 3].matshow(L_arrow_bottom)
    axs[1, 3].set_title("L_arrow_bottom")
    axs[1, 4].matshow(Update_arrow_tip)
    axs[1, 4].set_title("Update_arrow_tip")
    plt.show()

    S_local = np.zeros_like(A_local)
    S_arrow_bottom = np.zeros_like(A_arrow_bottom)
    S_arrow_right = np.zeros_like(A_arrow_right)
    S_global_arrow_tip = np.zeros_like(Update_arrow_tip)

    reduced_system = np.zeros((diag_blocksize+arrow_blocksize, diag_blocksize+arrow_blocksize))
    reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_local[-diag_blocksize:, -diag_blocksize:]
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = A_arrow_tip + Update_arrow_tip    
    reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_right[-diag_blocksize:, :]
    reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom[:, -diag_blocksize:]
    
    reduced_system_inv = np.linalg.inv(reduced_system)
    
    S_local[-diag_blocksize:, -diag_blocksize:] = reduced_system_inv[0:diag_blocksize, 0:diag_blocksize]
    S_arrow_bottom[:, -diag_blocksize:] = reduced_system_inv[-arrow_blocksize:, 0:diag_blocksize]
    S_arrow_right[-diag_blocksize:, :] = reduced_system_inv[0:diag_blocksize, -arrow_blocksize:]
    S_global_arrow_tip = reduced_system_inv[-arrow_blocksize:, -arrow_blocksize:]
    
    # Check for correctness of the first blocks of the true inverse
    assert np.allclose(A_local_refinv[-diag_blocksize:, -diag_blocksize:], S_local[-diag_blocksize:, -diag_blocksize:])
    assert np.allclose(A_arrow_bottom_refinv[:, -diag_blocksize:], S_arrow_bottom[:, -diag_blocksize:])
    assert np.allclose(A_arrow_right_refinv[-diag_blocksize:, :], S_arrow_right[-diag_blocksize:, :])
    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip)
    

    
    # ----- Selected inversion part -----
    S_local, S_arrow_bottom, S_arrow_right = top_sinv(
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

    fig, axs = plt.subplots(2, 4)
    fig.suptitle("After selected inversion")
    axs[0 ,0].matshow(A_local_refinv)
    axs[0 ,0].set_title("A_local_refinv")
    axs[0 ,1].matshow(A_arrow_bottom_refinv)
    axs[0 ,1].set_title("A_arrow_bottom_refinv")
    axs[0 ,2].matshow(A_arrow_right_refinv)
    axs[0 ,2].set_title("A_arrow_right_refinv")
    axs[0, 3].matshow(A_arrow_tip_refinv)
    axs[0, 3].set_title("A_arrow_tip_refinv")
    
    axs[1, 0].matshow(S_local)
    axs[1, 0].set_title("S_local")
    axs[1, 1].matshow(S_arrow_bottom)
    axs[1, 1].set_title("S_arrow_bottom")
    axs[1, 2].matshow(S_arrow_right)
    axs[1, 2].set_title("S_arrow_right")
    axs[1, 3].matshow(S_global_arrow_tip)
    axs[1, 3].set_title("S_global_arrow_tip")
    plt.show()
  
    A_local_refinv = mt.cut_to_blocktridiag(A_local_refinv, diag_blocksize)

    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip)
    assert np.allclose(A_arrow_bottom_refinv, S_arrow_bottom)
    assert np.allclose(A_arrow_right_refinv, S_arrow_right)
    assert np.allclose(A_local_refinv, S_local) """
    





""" # ----- Local checking -----
# ...of middle
if __name__ == "__main__":
    nblocks = 5
    diag_blocksize = 3
    arrow_blocksize = 2
    diagonal_dominant = True
    symmetric = False
    seed = 63
    
    A_init = matrix_generation.generate_tridiag_arrowhead_dense(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    # ----- Reference -----
    A_ref_init = cp.deepcopy(A_init)

    A_ref_inv = np.linalg.inv(A_ref_init)
    
    A_local_refinv, A_arrow_bottom_refinv, A_arrow_right_refinv = extract_partition(
        A_ref_inv,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )
    
    A_arrow_tip_refinv = A_ref_inv[-arrow_blocksize:, -arrow_blocksize:]
    # ---------------------

    A_local, A_arrow_bottom, A_arrow_right = extract_partition(
        A_init,
        0,
        nblocks-1,
        diag_blocksize,
        arrow_blocksize,
    )

    A_arrow_tip = A_init[-arrow_blocksize:, -arrow_blocksize:]


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
    

    # fig, axs = plt.subplots(2, 5)
    # fig.suptitle("After factorization")
    # axs[0 ,0].matshow(A_ref_init)
    # axs[0 ,0].set_title("A_ref_init")
    # axs[0 ,1].matshow(A_local)
    # axs[0 ,1].set_title("A_local")
    # axs[0 ,2].matshow(A_arrow_right)
    # axs[0 ,2].set_title("A_arrow_right")
    # axs[0, 3].matshow(A_arrow_bottom)
    # axs[0, 3].set_title("A_arrow_bottom")
    # axs[0, 4].matshow(A_arrow_tip)
    # axs[0, 4].set_title("A_arrow_tip")
    
    # axs[1, 0].matshow(L_local)
    # axs[1, 0].set_title("L_local")
    # axs[1, 1].matshow(U_local)
    # axs[1, 1].set_title("U_local")
    # axs[1, 2].matshow(U_arrow_right)
    # axs[1, 2].set_title("U_arrow_right")
    # axs[1, 3].matshow(L_arrow_bottom)
    # axs[1, 3].set_title("L_arrow_bottom")
    # axs[1, 4].matshow(Update_arrow_tip)
    # axs[1, 4].set_title("Update_arrow_tip")
    # plt.show()


    global_arrow_tip = np.zeros((arrow_blocksize, arrow_blocksize))
    global_arrow_tip = A_arrow_tip + Update_arrow_tip

    reduced_system = np.zeros((2 * diag_blocksize + arrow_blocksize, 2 * diag_blocksize + arrow_blocksize))
    
    # (top, top)
    reduced_system[0:diag_blocksize, 0:diag_blocksize] = A_local[0:diag_blocksize, 0:diag_blocksize]
    
    # (top, nblocks)
    reduced_system[0:diag_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_local[0:diag_blocksize, -diag_blocksize:]
    
    # (top, ndb+1)
    reduced_system[0:diag_blocksize, -arrow_blocksize:] = A_arrow_right[0:diag_blocksize, :]
    
    # (nblocks, top)
    reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, 0:diag_blocksize] = A_local[-diag_blocksize:, 0:diag_blocksize]
    
    # (ndb+1, top)
    reduced_system[-arrow_blocksize:, 0:diag_blocksize] = A_arrow_bottom[:, 0:diag_blocksize]
    
    # (nblocks, nblocks)
    reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_local[-diag_blocksize:, -diag_blocksize:]    
    
    # (nblocks, ndb+1)
    reduced_system[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -arrow_blocksize:] = A_arrow_right[-diag_blocksize:, :]
    
    # (ndb+1, nblocks)
    reduced_system[-arrow_blocksize:, -diag_blocksize-arrow_blocksize:-arrow_blocksize] = A_arrow_bottom[:, -diag_blocksize:]
    
    # (ndb+1, ndb+1)
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = global_arrow_tip
    
    
    # fig, axs = plt.subplots(1, 5)
    # fig.suptitle("Reduced system construction")
    # axs[0].matshow(A_local)
    # axs[0].set_title("A_local")    
    # axs[1].matshow(A_arrow_right)
    # axs[1].set_title("A_arrow_right")    
    # axs[2].matshow(A_arrow_bottom)
    # axs[2].set_title("A_arrow_bottom")    
    # axs[3].matshow(global_arrow_tip)
    # axs[3].set_title("global_arrow_tip")    
    # axs[4].matshow(reduced_system)
    # axs[4].set_title("reduced_system")  
    # plt.show() 
    
    
    reduced_system_inv = np.linalg.inv(reduced_system)


    S_local = np.zeros_like(A_local)
    S_arrow_bottom = np.zeros_like(A_arrow_bottom)
    S_arrow_right = np.zeros_like(A_arrow_right)
    S_global_arrow_tip = np.zeros_like(global_arrow_tip)
    
    # (top, top)
    S_local[0:diag_blocksize, 0:diag_blocksize] = reduced_system_inv[0:diag_blocksize, 0:diag_blocksize]
    
    # (top, nblocks)
    S_local[0:diag_blocksize, -diag_blocksize:] = reduced_system_inv[0:diag_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize]
    
    # (top, ndb+1)
    S_arrow_right[0:diag_blocksize, :] = reduced_system_inv[0:diag_blocksize, -arrow_blocksize:]
    
    # (nblocks, top)
    S_local[-diag_blocksize:, 0:diag_blocksize] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, 0:diag_blocksize]
    
    # (ndb+1, top)
    S_arrow_bottom[:, 0:diag_blocksize] = reduced_system_inv[-arrow_blocksize:, 0:diag_blocksize]
    
    # (nblocks, nblocks)
    S_local[-diag_blocksize:, -diag_blocksize:] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -diag_blocksize-arrow_blocksize:-arrow_blocksize] 
    
    # (nblocks, ndb+1)
    S_arrow_right[-diag_blocksize:, :] = reduced_system_inv[-diag_blocksize-arrow_blocksize:-arrow_blocksize, -arrow_blocksize:]
    
    # (ndb+1, nblocks)
    S_arrow_bottom[:, -diag_blocksize:] = reduced_system_inv[-arrow_blocksize:, -diag_blocksize-arrow_blocksize:-arrow_blocksize]
    
    # (ndb+1, ndb+1)
    S_global_arrow_tip = reduced_system_inv[-arrow_blocksize:, -arrow_blocksize:]

    
    fig, axs = plt.subplots(2, 4)
    fig.suptitle("After inversion of reduced system")
    axs[0 ,0].matshow(A_local_refinv)
    axs[0 ,0].set_title("A_local_refinv")
    axs[0 ,1].matshow(A_arrow_bottom_refinv)
    axs[0 ,1].set_title("A_arrow_bottom_refinv")
    axs[0 ,2].matshow(A_arrow_right_refinv)
    axs[0 ,2].set_title("A_arrow_right_refinv")
    axs[0, 3].matshow(A_arrow_tip_refinv)
    axs[0, 3].set_title("A_arrow_tip_refinv")
    
    axs[1, 0].matshow(S_local)
    axs[1, 0].set_title("S_local")
    axs[1, 1].matshow(S_arrow_bottom)
    axs[1, 1].set_title("S_arrow_bottom")
    axs[1, 2].matshow(S_arrow_right)
    axs[1, 2].set_title("S_arrow_right")
    axs[1, 3].matshow(S_global_arrow_tip)
    axs[1, 3].set_title("S_global_arrow_tip")
    plt.show()
    
    
    # (top, top)
    # S_local[0:diag_blocksize, 0:diag_blocksize] == A_local_refinv[0:diag_blocksize, 0:diag_blocksize]
    assert np.allclose(A_local_refinv[0:diag_blocksize, 0:diag_blocksize], S_local[0:diag_blocksize, 0:diag_blocksize])
    
    # (top, nblocks)
    # S_local[0:diag_blocksize, -diag_blocksize:] == A_local_refinv[0:diag_blocksize, -diag_blocksize:]
    assert np.allclose(A_local_refinv[0:diag_blocksize, -diag_blocksize:], S_local[0:diag_blocksize, -diag_blocksize:])
    
    # (top, ndb+1)
    # S_arrow_right[0:diag_blocksize, :] == A_arrow_right_refinv[0:diag_blocksize, :]
    assert np.allclose(A_arrow_right_refinv[0:diag_blocksize, :], S_arrow_right[0:diag_blocksize, :])
    
    # (nblocks, top)
    # S_local[-diag_blocksize:, 0:diag_blocksize] == A_local_refinv[-diag_blocksize:, 0:diag_blocksize]
    assert np.allclose(A_local_refinv[-diag_blocksize:, 0:diag_blocksize], S_local[-diag_blocksize:, 0:diag_blocksize])
    
    # (ndb+1, top)
    # S_arrow_bottom[:, 0:diag_blocksize] == A_arrow_bottom_refinv[:, 0:diag_blocksize]
    assert np.allclose(A_arrow_bottom_refinv[:, 0:diag_blocksize], S_arrow_bottom[:, 0:diag_blocksize])
    
    # (nblocks, nblocks)
    # S_local[-diag_blocksize:, -diag_blocksize:] == A_local_refinv[-diag_blocksize:, -diag_blocksize:]
    assert np.allclose(A_local_refinv[-diag_blocksize:, -diag_blocksize:], S_local[-diag_blocksize:, -diag_blocksize:])
    
    # (nblocks, ndb+1)
    # S_arrow_right[-diag_blocksize:, :] == A_arrow_right_refinv[-diag_blocksize:, :]
    assert np.allclose(A_arrow_right_refinv[-diag_blocksize:, :], S_arrow_right[-diag_blocksize:, :])
    
    # (ndb+1, nblocks)
    # S_arrow_bottom[:, -diag_blocksize:] == A_arrow_bottom_refinv[:, -diag_blocksize:]
    assert np.allclose(A_arrow_bottom_refinv[:, -diag_blocksize:], S_arrow_bottom[:, -diag_blocksize:])
        
    # (ndb+1, ndb+1)
    # S_global_arrow_tip == A_arrow_tip_refinv
    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip)
    print("...reduced system inversion is correct")
    

    S_local, S_arrow_bottom, S_arrow_right = middle_sinv(
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
    
    
    # These 9 blocks should still be correct after the middle_sinv call (un-touched)
    # (top, top)
    # S_local[0:diag_blocksize, 0:diag_blocksize] == A_local_refinv[0:diag_blocksize, 0:diag_blocksize]
    assert np.allclose(A_local_refinv[0:diag_blocksize, 0:diag_blocksize], S_local[0:diag_blocksize, 0:diag_blocksize])
    
    # (top, nblocks)
    # S_local[0:diag_blocksize, -diag_blocksize:] == A_local_refinv[0:diag_blocksize, -diag_blocksize:]
    assert np.allclose(A_local_refinv[0:diag_blocksize, -diag_blocksize:], S_local[0:diag_blocksize, -diag_blocksize:])
    
    # (top, ndb+1)
    # S_arrow_right[0:diag_blocksize, :] == A_arrow_right_refinv[0:diag_blocksize, :]
    assert np.allclose(A_arrow_right_refinv[0:diag_blocksize, :], S_arrow_right[0:diag_blocksize, :])
    
    # (nblocks, top)
    # S_local[-diag_blocksize:, 0:diag_blocksize] == A_local_refinv[-diag_blocksize:, 0:diag_blocksize]
    assert np.allclose(A_local_refinv[-diag_blocksize:, 0:diag_blocksize], S_local[-diag_blocksize:, 0:diag_blocksize])
    
    # (ndb+1, top)
    # S_arrow_bottom[:, 0:diag_blocksize] == A_arrow_bottom_refinv[:, 0:diag_blocksize]
    assert np.allclose(A_arrow_bottom_refinv[:, 0:diag_blocksize], S_arrow_bottom[:, 0:diag_blocksize])
    
    # (nblocks, nblocks)
    # S_local[-diag_blocksize:, -diag_blocksize:] == A_local_refinv[-diag_blocksize:, -diag_blocksize:]
    assert np.allclose(A_local_refinv[-diag_blocksize:, -diag_blocksize:], S_local[-diag_blocksize:, -diag_blocksize:])
    
    # (nblocks, ndb+1)
    # S_arrow_right[-diag_blocksize:, :] == A_arrow_right_refinv[-diag_blocksize:, :]
    assert np.allclose(A_arrow_right_refinv[-diag_blocksize:, :], S_arrow_right[-diag_blocksize:, :])
    
    # (ndb+1, nblocks)
    # S_arrow_bottom[:, -diag_blocksize:] == A_arrow_bottom_refinv[:, -diag_blocksize:]
    assert np.allclose(A_arrow_bottom_refinv[:, -diag_blocksize:], S_arrow_bottom[:, -diag_blocksize:])
        
    # (ndb+1, ndb+1)
    # S_global_arrow_tip == A_arrow_tip_refinv
    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip)
    print("...reduced system blocks stayed untouched after middle_sinv call")
    

    fig, axs = plt.subplots(2, 4)
    fig.suptitle("After sinv")
    axs[0 ,0].matshow(A_local_refinv)
    axs[0 ,0].set_title("A_local_refinv")
    axs[0 ,1].matshow(A_arrow_bottom_refinv)
    axs[0 ,1].set_title("A_arrow_bottom_refinv")
    axs[0 ,2].matshow(A_arrow_right_refinv)
    axs[0 ,2].set_title("A_arrow_right_refinv")
    axs[0, 3].matshow(A_arrow_tip_refinv)
    axs[0, 3].set_title("A_arrow_tip_refinv")
    
    axs[1, 0].matshow(S_local)
    axs[1, 0].set_title("S_local")
    axs[1, 1].matshow(S_arrow_bottom)
    axs[1, 1].set_title("S_arrow_bottom")
    axs[1, 2].matshow(S_arrow_right)
    axs[1, 2].set_title("S_arrow_right")
    axs[1, 3].matshow(S_global_arrow_tip)
    axs[1, 3].set_title("S_global_arrow_tip")
    plt.show()
    
    
    A_local_refinv = mt.cut_to_blocktridiag(A_local_refinv, diag_blocksize)
    S_local = mt.cut_to_blocktridiag(S_local, diag_blocksize)
    
    assert np.allclose(A_local_refinv, S_local)
    assert np.allclose(A_arrow_bottom_refinv, S_arrow_bottom)
    assert np.allclose(A_arrow_right_refinv, S_arrow_right)
    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip) """
    




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
    ) = get_partitions_indices(
        n_partitions = comm_size, 
        total_size = nblocks - 1
    )


    # ----- Reference/Checking data -----
    A_ref_init = cp.deepcopy(A)

    A_ref_inv = np.linalg.inv(A_ref_init)
    
    A_local_refinv, A_arrow_bottom_refinv, A_arrow_right_refinv = extract_partition(
        A_ref_inv,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )

    A_arrow_tip_refinv = A_ref_inv[-arrow_blocksize:, -arrow_blocksize:]

    Bridges_upper_inv_ref, Bridges_lower_inv_ref = extract_bridges(
        A_ref_inv, diag_blocksize, start_blockrows
    )

    A_local_refinv = mt.cut_to_blocktridiag(A_local_refinv, diag_blocksize)
    # ----- Reference/Checking data -----


    A_local, A_arrow_bottom, A_arrow_right = extract_partition(
        A,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )
    
    A_arrow_tip = A[-arrow_blocksize:, -arrow_blocksize:]
    
    Bridges_upper, Bridges_lower = extract_bridges(
        A, diag_blocksize, start_blockrows
    )
    
    
    # fig, axs = plt.subplots(2, 5)
    # fig.suptitle("Initial datas, process: " + str(comm_rank))
    # axs[0, 0].matshow(A)
    # axs[0, 0].set_title("A")
    # axs[0, 1].matshow(A_local)
    # axs[0, 1].set_title("A_local")
    # axs[0, 2].matshow(A_arrow_bottom)
    # axs[0, 2].set_title("A_arrow_bottom")
    # axs[0, 3].matshow(A_arrow_right)
    # axs[0, 3].set_title("A_arrow_right")
    # axs[0, 4].matshow(A_arrow_tip)
    # axs[0, 4].set_title("A_arrow_tip")
    
    # axs[1, 0].matshow(A_ref_inv)
    # axs[1, 0].set_title("A_ref_inv")
    # axs[1, 1].matshow(A_local_refinv)
    # axs[1, 1].set_title("A_local_refinv")
    # axs[1, 2].matshow(A_arrow_bottom_refinv)
    # axs[1, 2].set_title("A_arrow_bottom_refinv")
    # axs[1, 3].matshow(A_arrow_right_refinv)
    # axs[1, 3].set_title("A_arrow_right_refinv")
    # axs[1, 4].matshow(A_arrow_tip_refinv)
    # axs[1, 4].set_title("A_arrow_tip_refinv")
    # plt.show()


    S_local, S_bridges_upper, S_bridges_lower, S_arrow_bottom, S_arrow_right, S_global_arrow_tip = psr_arrowhead(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        A_arrow_tip,
        Bridges_upper,
        Bridges_lower,
        diag_blocksize,
        arrow_blocksize,
    )

    S_local_cut_tridiag = mt.cut_to_blocktridiag(S_local, diag_blocksize)

    # fig, axs = plt.subplots(2, 4)
    # fig.suptitle("Results, process: " + str(comm_rank))
    # axs[0, 0].matshow(A_local_refinv)
    # axs[0, 0].set_title("A_local_refinv")
    # axs[0, 1].matshow(A_arrow_bottom_refinv)
    # axs[0, 1].set_title("A_arrow_bottom_refinv")
    # axs[0, 2].matshow(A_arrow_right_refinv)
    # axs[0, 2].set_title("A_arrow_right_refinv")
    # axs[0, 3].matshow(A_arrow_tip_refinv)
    # axs[0, 3].set_title("A_arrow_tip_refinv")
    
    # axs[1, 0].matshow(S_local_cut_tridiag)
    # axs[1, 0].set_title("S_local_cut_tridiag")
    # axs[1, 1].matshow(S_arrow_bottom)
    # axs[1, 1].set_title("S_arrow_bottom")
    # axs[1, 2].matshow(S_arrow_right)
    # axs[1, 2].set_title("S_arrow_right")
    # axs[1, 3].matshow(S_global_arrow_tip)
    # axs[1, 3].set_title("S_global_arrow_tip")
    # plt.show()


    # ----- VERFIFYING THE RESULTS -----

    assert np.allclose(A_local_refinv, S_local_cut_tridiag)
    norme_diff_partition_i = np.linalg.norm(A_local_refinv - S_local_cut_tridiag)
    print("Partition n", comm_rank, "  norm diff = ", norme_diff_partition_i)
    
    assert np.allclose(A_arrow_bottom_refinv, S_arrow_bottom)
    norme_diff_arrow_bottom_i = np.linalg.norm(A_arrow_bottom_refinv - S_arrow_bottom)
    print("Arrow bottom n", comm_rank, "  norm diff = ", norme_diff_arrow_bottom_i)
    
    assert np.allclose(A_arrow_right_refinv, S_arrow_right)
    norme_diff_arrow_right_i = np.linalg.norm(A_arrow_right_refinv - S_arrow_right)
    print("Arrow right n", comm_rank, "  norm diff = ", norme_diff_arrow_right_i)
    
    assert np.allclose(A_arrow_tip_refinv, S_global_arrow_tip)
    norme_diff_arrow_tip_i = np.linalg.norm(A_arrow_tip_refinv - S_global_arrow_tip)
    print("Arrow tip n", comm_rank, "  norm diff = ", norme_diff_arrow_tip_i)

    # Check for bridges correctness
    print("Bridges correctness:")
    if comm_rank == 0:
        norme_diff_upper_bridge_i = np.linalg.norm(Bridges_upper_inv_ref[comm_rank] - S_bridges_upper[comm_rank])
        assert np.allclose(Bridges_upper_inv_ref[comm_rank], S_bridges_upper[comm_rank])
        print("    Upper bridge n", comm_rank, "  norm diff = ", norme_diff_upper_bridge_i)            
                        
    elif comm_rank == comm_size-1:
        norme_diff_lower_bridge_i = np.linalg.norm(Bridges_lower_inv_ref[comm_rank-1] - S_bridges_lower[comm_rank-1])
        assert np.allclose(Bridges_lower_inv_ref[comm_rank-1], S_bridges_lower[comm_rank-1])
        print("    Lower bridge n", comm_rank, "  norm diff = ", norme_diff_lower_bridge_i)
        
    else:
        norme_diff_upper_bridge_i = np.linalg.norm(Bridges_upper_inv_ref[comm_rank] - S_bridges_upper[comm_rank])
        assert np.allclose(Bridges_upper_inv_ref[comm_rank], S_bridges_upper[comm_rank])
        print("    Upper bridge n", comm_rank, "  norm diff = ", norme_diff_upper_bridge_i)            
        
        norme_diff_lower_bridge_i = np.linalg.norm(Bridges_lower_inv_ref[comm_rank-1] - S_bridges_lower[comm_rank-1])
        assert np.allclose(Bridges_lower_inv_ref[comm_rank-1], S_bridges_lower[comm_rank-1])
        print("    Lower bridge n", comm_rank, "  norm diff = ", norme_diff_lower_bridge_i)