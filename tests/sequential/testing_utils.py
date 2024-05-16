# Copyright 2023-2024 ETH Zurich & USI. All rights reserved.

import numpy as np


def bta_dense_to_arrays(
    bta: np.ndarray,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
):
    A_diagonal_blocks = np.zeros(
        (diagonal_blocksize, n_diag_blocks * diagonal_blocksize), dtype=np.complex128
    )
    A_lower_diagonal_blocks = np.zeros(
        (diagonal_blocksize, (n_diag_blocks - 1) * diagonal_blocksize),
        dtype=np.complex128,
    )
    A_upper_diagonal_blocks = np.zeros(
        (diagonal_blocksize, (n_diag_blocks - 1) * diagonal_blocksize),
        dtype=np.complex128,
    )

    for i in range(n_diag_blocks):
        A_diagonal_blocks[:, i * diagonal_blocksize : (i + 1) * diagonal_blocksize] = (
            bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
            ]
        )
        if i > 0:
            A_lower_diagonal_blocks[
                :, (i - 1) * diagonal_blocksize : i * diagonal_blocksize
            ] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize : i * diagonal_blocksize,
            ]
        if i < n_diag_blocks - 1:
            A_upper_diagonal_blocks[
                :, i * diagonal_blocksize : (i + 1) * diagonal_blocksize
            ] = bta[
                i * diagonal_blocksize : (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize : (i + 2) * diagonal_blocksize,
            ]

    A_arrow_bottom_blocks = bta[-arrowhead_blocksize:, :-arrowhead_blocksize]
    A_arrow_right_blocks = bta[:-arrowhead_blocksize, -arrowhead_blocksize:]
    A_arrow_tip_block = bta[-arrowhead_blocksize:, -arrowhead_blocksize:]

    return (
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_upper_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_right_blocks,
        A_arrow_tip_block,
    )
