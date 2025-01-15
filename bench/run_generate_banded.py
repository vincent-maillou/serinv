""" 
Run the sequential Serinv codes on CPU using the inlamat matrices.
"""

import time

tic = time.perf_counter()

import numpy as np
import scipy.stats
import argparse
from scipy.io import mmwrite


from matutils import (
    dd_bta,
    bta_symmetrize,
    bta_to_csc,
)


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
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize


    # Generate the matrix
    A = dd_bta(
        diagonal_blocksize=diagonal_blocksize,
        arrowhead_blocksize=arrowhead_blocksize,
        n_diag_blocks=n_diag_blocks,
        device_array=False,
        dtype=np.float64,
    )




    mmwrite("matrix.mtx", A)