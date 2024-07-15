try:
    import cupyx as cpx

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

print("CUPY_AVAIL: ", CUPY_AVAIL)

from serinv.algs import pobtaf, pobtasi
from load_datmat import csc_to_dense_bta, read_sym_CSC
from utility_functions import bta_arrays_to_dense, bta_dense_to_arrays
import numpy as np

import scipy.stats
import time
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
        "--device_streaming",
        type=bool,
        default=True,
        help="a boolean indicating if device streaming is enabled",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=1,
        help="number of iterations for the benchmarking",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="number of warm-ups iterations",
    )

    args = parser.parse_args()

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    device_streaming = args.device_streaming
    n_iterations = args.n_iterations
    warmups = args.warmups

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    device_streaming = True
    device_array = False

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

    # timings
    t_pobtaf = np.zeros(n_iterations)
    t_pobtasi = np.zeros(n_iterations)

    for i in range(warmups + n_iterations):

        if CUPY_AVAIL and device_streaming and not device_array:
            if i == 0:
                print("Using pinned memory", flush=True)
            A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
            A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :]
            A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(
                A_lower_diagonal_blocks
            )
            A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :]
            A_arrow_bottom_blocks_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks)
            A_arrow_bottom_blocks_pinned[:, :, :] = A_arrow_bottom_blocks[:, :, :]
            A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
            A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :]

        start_time = time.perf_counter()

        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
        ) = pobtaf(
            A_diagonal_blocks_pinned,
            A_lower_diagonal_blocks_pinned,
            A_arrow_bottom_blocks_pinned,
            A_arrow_tip_block_pinned,
            device_streaming,
        )
        end_time = time.perf_counter()
        elapsed_time_pobtaf = end_time - start_time

        if i >= warmups:
            t_pobtaf[i - warmups] = elapsed_time_pobtaf

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
        )

        end_time = time.perf_counter()
        elapsed_time_selinv = end_time - start_time

        if i >= warmups:
            t_pobtasi[i - warmups] = elapsed_time_selinv

        if i < warmups:
            print(
                f"Warmup iteration: {i+1}/{warmups}, Time pobtaf: {elapsed_time_pobtaf:.5f} sec, Time pobtasi: {elapsed_time_selinv:.5f} sec"
            )
        else:
            print(
                f"Bench iteration: {i+1-warmups}/{n_iterations} Time Chol: {elapsed_time_pobtaf:.5f} sec. Time selInv: {elapsed_time_selinv:.5f} sec"
            )

    # Save the raw data
    np.save(
        f"timings_pobtaf_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}.npy",
        t_pobtaf,
    )

    np.save(
        f"timings_pobtasi_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}.npy",
        t_pobtasi,
    )

    mean_pobtaf, lb_mean_pobtaf, ub_mean_pobtaf = mean_confidence_interval(t_pobtaf)
    mean_pobtasi, lb_mean_pobtasi, ub_mean_pobtasi = mean_confidence_interval(t_pobtasi)

    print(
        f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]"
    )

    print(
        f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]"
    )
