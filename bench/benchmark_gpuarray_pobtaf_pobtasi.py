import time

tic = time.perf_counter()
try:
    import cupy as cp
except:
    pass

from serinv.algs import pobtaf, pobtasi
from load_datmat import csc_to_dense_bta, read_sym_CSC
from utility_functions import bta_arrays_to_dense, bta_dense_to_arrays
import numpy as np

import scipy.stats
import argparse


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
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1,
        help="number of iterations for the benchmarking",
    )
    parser.add_argument(
        "--n_warmups",
        type=int,
        default=1,
        help="number of warm-ups iterations",
    )

    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    n_iterations = args.n_iterations
    n_warmups = args.n_warmups
    timing_sections = True

    print("timing_sections", timing_sections)

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    device_streaming = True

    # if True: compare to reference solution
    DEBUG = False

    # read in file
    file = (
        f"pobta_nb{n_diag_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}.npz"
    )
    
    filename = file_path + file

    data = np.load(filename)

    A_diagonal_blocks = data["A_diagonal_blocks"]
    A_lower_diagonal_blocks = data["A_lower_diagonal_blocks"]
    A_arrow_bottom_blocks = data["A_arrow_bottom_blocks"]
    A_arrow_tip_block = data["A_arrow_tip_block"]
    toc = time.perf_counter()

    print(f"Reading matrix took: {toc - tic:.5f} sec", flush=True)
    print(f"    A_diagonal_blocks.shape", A_diagonal_blocks.shape, flush=True)
    print(f"    A_lower_diagonal_blocks.shape", A_lower_diagonal_blocks.shape, flush=True)
    print(f"    A_arrow_bottom_blocks.shape", A_arrow_bottom_blocks.shape, flush=True)
    print(f"    A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    tic = time.perf_counter()
    A_diagonal_blocks_device = cp.zeros_like(A_diagonal_blocks)
    A_lower_diagonal_blocks_device = cp.zeros_like(A_lower_diagonal_blocks)
    A_arrow_bottom_blocks_device = cp.zeros_like(A_arrow_bottom_blocks)
    A_arrow_tip_block_device = cp.zeros_like(A_arrow_tip_block)
    cp.cuda.Stream.null.synchronize()
    toc = time.perf_counter()

    print(f"Allocating GPU memory took: {toc - tic:.5f} sec", flush=True)
    print(f"Initialization done..", flush=True)

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_device[:, :, :].set(A_diagonal_blocks[:, :, :])
        A_lower_diagonal_blocks_device[:, :, :].set(A_lower_diagonal_blocks[:, :, :])
        A_arrow_bottom_blocks_device[:, :, :].set(A_arrow_bottom_blocks[:, :, :])
        A_arrow_tip_block_device[:, :].set(A_arrow_tip_block[:, :])
        toc = time.perf_counter()

        print(f"  Copying data to GPU memory took: {toc - tic:.5f} sec", flush=True)

        start_time = time.perf_counter()
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
        ) = pobtaf(
            A_diagonal_blocks_device,
            A_lower_diagonal_blocks_device,
            A_arrow_bottom_blocks_device,
            A_arrow_tip_block_device,
            device_streaming,
            timing_sections,
        )
        end_time = time.perf_counter()
        elapsed_time_pobtaf = end_time - start_time

        print(f"    pobtaf took: {elapsed_time_pobtaf:.5f} sec", flush=True)

        start_time = time.perf_counter()
        (
            X_diagonal_blocks,
            X_lower_diagonal_blocks,
            X_arrow_bottom_blocks,
            X_arrow_tip_block,
        ) = pobtasi(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
            device_streaming,
            timing_sections,
        )
        end_time = time.perf_counter()
        elapsed_time_pobtasi = end_time - start_time

        print(f"    pobtasi took: {elapsed_time_pobtasi:.5f} sec", flush=True)

    
