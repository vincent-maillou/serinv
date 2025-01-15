""" 
Run the sequential Serinv codes on CPU using the inlamat matrices.
"""

import time


import numpy as np
import scipy.stats
import argparse
from scipy.io import mmwrite
import scipy.sparse as sps


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
        default=2865,
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
        default=60,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/hpc/ihpc/ihpc060h/share/serinv/",
        #default="/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples",
        help="a string for the file path",
    )

    args = parser.parse_args()

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    density = args.density
    file_path = args.file_path

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    # Generate BTA arrays
    tic = time.time()

    if density == 1:
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
    else:
        A_diagonal_blocks = np.zeros(
            (n_diag_blocks, diagonal_blocksize, diagonal_blocksize), dtype=np.float64
        )
        A_lower_diagonal_blocks = np.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize), dtype=np.float64
        )
        A_arrow_bottom_blocks = np.zeros(
            (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize), dtype=np.float64
        )
        A_arrow_tip_block = np.random.rand(arrowhead_blocksize, arrowhead_blocksize)

        for i in range(n_diag_blocks):
            A_diagonal_blocks[i] = sps.random(
                diagonal_blocksize, diagonal_blocksize, density=density, format="csr"
            ).toarray()

            if i > 0:
                A_lower_diagonal_blocks[i - 1] = sps.random(
                    diagonal_blocksize, diagonal_blocksize, density=density, format="csr"
                ).toarray()

            A_arrow_bottom_blocks[i] = sps.random(
                arrowhead_blocksize, diagonal_blocksize, density=density, format="csr"
            ).toarray()
    toc = time.time()
    print("Time to random arrays: ", toc - tic)
    
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

    tic = time.time()
    A_coo = bta_to_coo(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
    )
    toc = time.time()
    print("Time to convert to COO: ", toc - tic)

    # plt.spy(A_coo.toarray())
    # plt.show()

    tic = time.time()
    file = (
        "Qxy_ns"
        + str(diagonal_blocksize)
        + "_nt"
        + str(n_diag_blocks)
        + "_nss0_nb"
        + str(arrowhead_blocksize)
        + "_n"
        + str(n)
        + "_density"
        + str(density)
        + ".mtx"
    )

    storing_path = file_path + file

    mmwrite(storing_path, A_coo, symmetry="symmetric") 
    toc = time.time()
    print("Time to write matrix: ", toc - tic)
    
    print("Matrix written to " + storing_path)
