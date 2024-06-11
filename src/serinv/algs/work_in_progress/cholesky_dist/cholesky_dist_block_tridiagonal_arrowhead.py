# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import numpy.linalg as npla
import scipy.linalg as scla
from mpi4py import MPI

from serinv.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from serinv.cholesky.cholesky_selected_inversion import (
    cholesky_sinv_block_tridiagonal_arrowhead,
)


def cholesky_dist_block_tridiagonal_arrowhead(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    A_bridges_lower: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_partition = A_diagonal_blocks_local.shape[1] // diag_blocksize

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            Update_arrow_tip,
        ) = top_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_tip_block,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
        )
    else:
        # Arrays that store the update of the 2sided pattern for the middle processes
        A_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=A_diagonal_blocks_local.dtype,
        )

        A_top_2sided_arrow_blocks_local[:, :diag_blocksize] = A_diagonal_blocks_local[
            :, :diag_blocksize
        ]
        A_top_2sided_arrow_blocks_local[:, diag_blocksize : 2 * diag_blocksize] = (
            A_lower_diagonal_blocks_local[:, :diag_blocksize].T
        )

        (
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
            Update_arrow_tip,
        ) = middle_factorize(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_top_2sided_arrow_blocks_local,
            A_arrow_tip_block,
        )

        (
            A_rs_diagonal_blocks,
            A_rs_lower_diagonal_blocks,
            A_rs_arrow_bottom_blocks,
            A_rs_arrow_tip_block,
        ) = create_reduced_system(
            A_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_tip_block,
            Update_arrow_tip,
            A_bridges_lower,
            A_top_2sided_arrow_blocks_local,
        )

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_tip_block,
    ) = inverse_reduced_system(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_tip_block,
    )

    (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
    ) = update_sinv_reduced_system(
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_tip_block,
        n_diag_blocks_partition,
        diag_blocksize,
        arrow_blocksize,
    )

    if comm_rank == 0:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            _,
        ) = top_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
        )
    else:
        (
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            _,
        ) = middle_sinv(
            X_diagonal_blocks_local,
            X_lower_diagonal_blocks_local,
            X_arrow_bottom_blocks_local,
            X_top_2sided_arrow_blocks_local,
            X_global_arrow_tip,
            L_diagonal_blocks_inv,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_upper_2sided_arrow_blocks,
        )

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
    )


def top_factorize(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
):
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    nblocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    L_diagonal_blocks_inv_local = np.empty_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.empty_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.empty_like(A_arrow_bottom_blocks_local)
    Update_arrow_tip_local = np.zeros_like(
        A_arrow_tip_block
    )  # Have to be zero-initialized

    for i in range(0, nblocks - 1, 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = npla.cholesky(
            A_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize],
        )

        # Compute lower factors
        L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = scla.solve_triangular(
            L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            np.eye(diag_blocksize),
            lower=True,
        )

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks_local[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            A_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks_local[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A_diagonal_blocks_local[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_diagonal_blocks_local[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A_arrow_bottom_blocks_local[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks_local[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_arrow_bottom_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks_inv_local[:, -diag_blocksize:] = npla.cholesky(
        A_diagonal_blocks_local[:, -diag_blocksize:]
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks_local[:, -diag_blocksize:] = A_arrow_bottom_blocks_local[
        :, -diag_blocksize:
    ] @ scla.solve_triangular(
        L_diagonal_blocks_inv_local[:, -diag_blocksize:],
        np.eye(diag_blocksize),
        lower=True,
    )

    return (
        L_diagonal_blocks_inv_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        Update_arrow_tip_local,
    )


def middle_factorize(
    A_diagonal_blocks_local: np.ndarray,
    A_lower_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray,
    A_arrow_tip_block_local: np.ndarray,
):
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    n_blocks = A_diagonal_blocks_local.shape[1] // diag_blocksize

    L_diagonal_blocks_inv_local = np.empty_like(A_diagonal_blocks_local)
    L_lower_diagonal_blocks_local = np.empty_like(A_lower_diagonal_blocks_local)
    L_arrow_bottom_blocks_local = np.empty_like(A_arrow_bottom_blocks_local)
    L_upper_2sided_arrow_blocks_local = np.empty_like(A_top_2sided_arrow_blocks_local)
    Update_arrow_tip_local = np.zeros_like(
        A_arrow_tip_block_local
    )  # Have to be zero-initialized

    for i in range(1, n_blocks - 1, 1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks_inv_local[
            :,
            i * diag_blocksize : (i + 1) * diag_blocksize,
        ] = npla.cholesky(
            A_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
        )

        # Compute lower factors
        L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = scla.solve_triangular(
            L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ],
            np.eye(diag_blocksize),
            lower=True,
        )

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
        L_upper_2sided_arrow_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_top_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            A_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # Update next diagonal block
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.T
        A_diagonal_blocks_local[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_diagonal_blocks_local[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.T
        A_arrow_bottom_blocks_local[
            :,
            (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
        ] = (
            A_arrow_bottom_blocks_local[
                :,
                (i + 1) * diag_blocksize : (i + 2) * diag_blocksize,
            ]
            - L_arrow_bottom_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ]
            @ L_lower_diagonal_blocks_local[
                :,
                i * diag_blocksize : (i + 1) * diag_blocksize,
            ].T
        )

        # Update the block at the tip of the arrowhead
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.T
        Update_arrow_tip_local[:, :] = (
            Update_arrow_tip_local[:, :]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # Update top and next upper/lower blocks of 2-sided factorization pattern
        # A_{top, top} = A_{top, top} - L_{top, i} @ L_{top, i}.T
        A_diagonal_blocks_local[:, :diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
            - L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # A_{top, i+1} = - L{top, i} @ L_{i+1, i}.T
        A_top_2sided_arrow_blocks_local[
            :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
        ] = (
            -L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

        # Update the top (first blocks) of the arrowhead
        # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ L_{top, i}.T
        A_arrow_bottom_blocks_local[:, :diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
            - L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
        )

    return (
        L_diagonal_blocks_inv_local,
        L_lower_diagonal_blocks_local,
        L_arrow_bottom_blocks_local,
        L_upper_2sided_arrow_blocks_local,
        Update_arrow_tip_local,
    )


def create_reduced_system(
    A_diagonal_blocks_local: np.ndarray,
    A_arrow_bottom_blocks_local: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    Update_arrow_tip: np.ndarray,
    A_bridges_lower: np.ndarray,
    A_top_2sided_arrow_blocks_local: np.ndarray = None,
):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create empty matrix for reduced system -> (2 * #process - 1) * diag_blocksize + arrowhead_size
    diag_blocksize = A_diagonal_blocks_local.shape[0]
    arrow_blocksize = A_arrow_bottom_blocks_local.shape[0]
    n_diag_blocks_reduced_system = 2 * comm_size - 1

    A_rs_diagonal_blocks = np.zeros(
        (diag_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_lower_diagonal_blocks = np.zeros(
        (diag_blocksize, (n_diag_blocks_reduced_system - 1) * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_arrow_bottom_blocks = np.zeros(
        (arrow_blocksize, n_diag_blocks_reduced_system * diag_blocksize),
        dtype=A_diagonal_blocks_local.dtype,
    )
    A_rs_arrow_tip_block = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=A_diagonal_blocks_local.dtype
    )

    A_rs_arrow_tip_block = Update_arrow_tip

    if comm_rank == 0:
        A_rs_diagonal_blocks[:, :diag_blocksize] = A_diagonal_blocks_local[
            :, -diag_blocksize:
        ]
        A_rs_arrow_bottom_blocks[:, :diag_blocksize] = A_arrow_bottom_blocks_local[
            :, -diag_blocksize:
        ]
    else:
        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        A_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index] = (
            A_bridges_lower[
                :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
            ]
        )

        A_rs_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_diagonal_blocks_local[:, :diag_blocksize]
        )

        A_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize] = (
            A_top_2sided_arrow_blocks_local[:, -diag_blocksize:].T
        )

        A_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_diagonal_blocks_local[:, -diag_blocksize:]

        A_rs_arrow_bottom_blocks[:, start_index : start_index + diag_blocksize] = (
            A_arrow_bottom_blocks_local[:, :diag_blocksize]
        )

        A_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ] = A_arrow_bottom_blocks_local[:, -diag_blocksize:]

    # Send the reduced_system with MPIallReduce SUM operation
    A_rs_diagonal_blocks_sum = np.zeros_like(A_rs_diagonal_blocks)
    A_rs_lower_diagonal_blocks_sum = np.zeros_like(A_rs_lower_diagonal_blocks)
    A_rs_arrow_bottom_blocks_sum = np.zeros_like(A_rs_arrow_bottom_blocks)
    A_rs_arrow_tip_block_sum = np.zeros_like(A_rs_arrow_tip_block)

    comm.Allreduce(
        [A_rs_diagonal_blocks, MPI.DOUBLE],
        [A_rs_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_lower_diagonal_blocks, MPI.DOUBLE],
        [A_rs_lower_diagonal_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_bottom_blocks, MPI.DOUBLE],
        [A_rs_arrow_bottom_blocks_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [A_rs_arrow_tip_block, MPI.DOUBLE],
        [A_rs_arrow_tip_block_sum, MPI.DOUBLE],
        op=MPI.SUM,
    )

    # Add the global arrow tip to the reduced system arrow-tip summation
    A_rs_arrow_tip_block_sum += A_arrow_tip_block

    return (
        A_rs_diagonal_blocks_sum,
        A_rs_lower_diagonal_blocks_sum,
        A_rs_arrow_bottom_blocks_sum,
        A_rs_arrow_tip_block_sum,
    )


def inverse_reduced_system(
    A_rs_diagonal_blocks: np.ndarray,
    A_rs_lower_diagonal_blocks: np.ndarray,
    A_rs_arrow_bottom_blocks: np.ndarray,
    A_rs_arrow_tip_block: np.ndarray,
):

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    ) = cholesky_factorize_block_tridiagonal_arrowhead(
        A_rs_diagonal_blocks,
        A_rs_lower_diagonal_blocks,
        A_rs_arrow_bottom_blocks,
        A_rs_arrow_tip_block,
    )

    (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_tip_block,
    ) = cholesky_sinv_block_tridiagonal_arrowhead(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )

    return (
        X_rs_diagonal_blocks,
        X_rs_lower_diagonal_blocks,
        X_rs_arrow_bottom_blocks,
        X_rs_arrow_tip_block,
    )


def update_sinv_reduced_system(
    X_rs_diagonal_blocks,
    X_rs_lower_diagonal_blocks,
    X_rs_arrow_bottom_blocks,
    X_rs_arrow_tip_block,
    n_diag_blocks_partition,
    diag_blocksize,
    arrow_blocksize,
):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    X_diagonal_blocks_local = np.empty(
        (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_lower_diagonal_blocks_local = np.empty(
        (diag_blocksize, (n_diag_blocks_partition - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )
    X_arrow_bottom_blocks_local = np.empty(
        (arrow_blocksize, n_diag_blocks_partition * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    X_bridges_lower = np.empty(
        (diag_blocksize, (comm_size - 1) * diag_blocksize),
        dtype=X_rs_diagonal_blocks.dtype,
    )

    if comm_rank == 0:
        X_top_2sided_arrow_blocks_local = None

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, :diag_blocksize
        ]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, :diag_blocksize
        ]
    else:
        X_top_2sided_arrow_blocks_local = np.empty(
            (diag_blocksize, n_diag_blocks_partition * diag_blocksize),
            dtype=X_rs_diagonal_blocks.dtype,
        )

        start_index = diag_blocksize + (comm_rank - 1) * 2 * diag_blocksize

        X_bridges_lower[
            :, (comm_rank - 1) * diag_blocksize : comm_rank * diag_blocksize
        ] = X_rs_lower_diagonal_blocks[:, start_index - diag_blocksize : start_index]

        X_diagonal_blocks_local[:, :diag_blocksize] = X_rs_diagonal_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_top_2sided_arrow_blocks_local[:, -diag_blocksize:] = (
            X_rs_lower_diagonal_blocks[:, start_index : start_index + diag_blocksize].T
        )

        X_diagonal_blocks_local[:, -diag_blocksize:] = X_rs_diagonal_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

        X_arrow_bottom_blocks_local[:, :diag_blocksize] = X_rs_arrow_bottom_blocks[
            :, start_index : start_index + diag_blocksize
        ]

        X_arrow_bottom_blocks_local[:, -diag_blocksize:] = X_rs_arrow_bottom_blocks[
            :, start_index + diag_blocksize : start_index + 2 * diag_blocksize
        ]

    X_global_arrow_tip = X_rs_arrow_tip_block

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_top_2sided_arrow_blocks_local,
        X_global_arrow_tip,
        X_bridges_lower,
    )


def top_sinv(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_global_arrow_tip_local: np.ndarray,
    L_diagonal_blocks_inv_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
):
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    for i in range(n_blocks - 2, -1, -1):
        # --- Lower-diagonal blocks ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip_local[:, :]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

        # --- Diagonal block part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            - X_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_global_arrow_tip_local,
    )


def middle_sinv(
    X_diagonal_blocks_local: np.ndarray,
    X_lower_diagonal_blocks_local: np.ndarray,
    X_arrow_bottom_blocks_local: np.ndarray,
    X_top_2sided_arrow_blocks_local: np.ndarray,
    X_global_arrow_tip_block_local: np.ndarray,
    L_diagonal_blocks_inv_local: np.ndarray,
    L_lower_diagonal_blocks_local: np.ndarray,
    L_arrow_bottom_blocks_local: np.ndarray,
    L_upper_2sided_arrow_blocks_local: np.ndarray,
):
    diag_blocksize = X_diagonal_blocks_local.shape[0]
    n_blocks = X_diagonal_blocks_local.shape[1] // diag_blocksize

    for i in range(n_blocks - 2, 0, -1):
        # X_{i+1, i} = (- X_{top, i+1}.T L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_lower_diagonal_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_top_2sided_arrow_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

        # X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{ndb+1, top}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_top_2sided_arrow_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_top_2sided_arrow_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_diagonal_blocks_local[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[:, :diag_blocksize].T
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

        # Arrowhead
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom_blocks_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ] = (
            -X_arrow_bottom_blocks_local[
                :, (i + 1) * diag_blocksize : (i + 2) * diag_blocksize
            ]
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[:, :diag_blocksize]
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_global_arrow_tip_block_local[:, :]
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

        # X_{i, i} = (U_{i, i}^{-1} - X_{i+1, i}.T L_{i+1, i} - X_{top, i}.T L_{top, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal_blocks_local[:, i * diag_blocksize : (i + 1) * diag_blocksize] = (
            L_diagonal_blocks_inv_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            - X_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_lower_diagonal_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_top_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_upper_2sided_arrow_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
            - X_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ].T
            @ L_arrow_bottom_blocks_local[
                :, i * diag_blocksize : (i + 1) * diag_blocksize
            ]
        ) @ L_diagonal_blocks_inv_local[
            :, i * diag_blocksize : (i + 1) * diag_blocksize
        ]

    # Copy back the 2 first blocks that have been produced in the 2-sided pattern
    # to the tridiagonal storage.
    X_lower_diagonal_blocks_local[:, :diag_blocksize] = X_top_2sided_arrow_blocks_local[
        :, diag_blocksize : 2 * diag_blocksize
    ].T

    return (
        X_diagonal_blocks_local,
        X_lower_diagonal_blocks_local,
        X_arrow_bottom_blocks_local,
        X_global_arrow_tip_block_local,
    )
