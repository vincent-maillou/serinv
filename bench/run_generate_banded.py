""" 
Run the sequential Serinv codes on CPU using the inlamat matrices.
"""

import time


import numpy as np
import scipy.stats
import argparse
from scipy.io import mmwrite


from matutils import (
    bta_to_coo,
)

import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--diagonal_blocksize",
        type=int,
        default=92,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--arrowhead_blocksize",
        type=int,
        default=4,
        help="an integer for the arrowhead block size",
    )
    parser.add_argument(
        "--n_diag_blocks",
        type=int,
        default=50,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/x_gaedkelb/serinv/dev/matrices/",
        help="a string for the file path",
    )

    args = parser.parse_args()

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    # Generate BTA arrays
    A_diagonal_blocks = np.random.rand(
        n_diag_blocks, diagonal_blocksize, diagonal_blocksize
    )
    A_lower_diagonal_blocks = np.random.rand(
        n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize
    )
    A_arrow_bottom_blocks = np.random.rand(
        n_diag_blocks, arrowhead_blocksize, diagonal_blocksize
    )
    A_arrow_tip_block = np.random.rand(arrowhead_blocksize, arrowhead_blocksize)

    # Make diagonally dominante
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=A_diagonal_blocks.dtype)
    for i in range(A_diagonal_blocks.shape[0]):
        colsum = np.sum(A_diagonal_blocks[i, :, :], axis=1) - np.diag(
            A_diagonal_blocks[i, :, :]
        )
        if i > 0:
            colsum += np.sum(A_lower_diagonal_blocks[i - 1, :, :], axis=1)

        A_diagonal_blocks[i, :, :] += np.diag(colsum)

        arrow_colsum[:] += np.sum(A_arrow_bottom_blocks[i, :, :], axis=1)

    A_arrow_tip_block[:, :] += np.diag(
        arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)
    )

    # Make symmetric
    for i in range(n_diag_blocks):
        A_diagonal_blocks[i] += A_diagonal_blocks[i].T
    A_arrow_tip_block += A_arrow_tip_block.T

    A_coo = bta_to_coo(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )

    mmwrite("matrix.mtx", A_coo, symmetry="symmetric")
