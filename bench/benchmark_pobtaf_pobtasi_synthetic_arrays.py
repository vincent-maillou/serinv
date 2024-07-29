try:
    import cupyx as cpx

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

print("CUPY_AVAIL: ", CUPY_AVAIL)

from serinv.algs import pobtaf, pobtasi
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
        default="/home/vault/j101df/j101df10/inla_matrices/synthetic_dataset/sequential/",
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

    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_diag_blocks = args.n_diag_blocks
    file_path = args.file_path
    n_iterations = args.n_iterations
    n_warmups = args.n_warmups

    n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

    device_streaming = True
    device_array = False

    # if True: compare to reference solution
    DEBUG = False

    tic = time.perf_counter()
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

    print(f"Reading file took: {toc - tic:.5f} sec", flush=True)
    print(f"    A_diagonal_blocks.shape", A_diagonal_blocks.shape, flush=True)
    print(f"    A_lower_diagonal_blocks.shape", A_lower_diagonal_blocks.shape, flush=True)
    print(f"    A_arrow_bottom_blocks.shape", A_arrow_bottom_blocks.shape, flush=True)
    print(f"    A_arrow_tip_block.shape", A_arrow_tip_block.shape, flush=True)

    tic = time.perf_counter()
    # Allocate pinned memory buffers
    A_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_diagonal_blocks)
    A_lower_diagonal_blocks_pinned = cpx.zeros_like_pinned(A_lower_diagonal_blocks)
    A_arrow_bottom_blocks_pinned = cpx.zeros_like_pinned(A_arrow_bottom_blocks)
    A_arrow_tip_block_pinned = cpx.zeros_like_pinned(A_arrow_tip_block)
    toc = time.perf_counter()

    print(f"Allocating pinned memory took: {toc - tic:.5f} sec", flush=True)
    print(f"Initialization done..", flush=True)

    t_pobtaf = np.zeros(n_iterations)
    t_pobtasi = np.zeros(n_iterations)

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_pinned[:, :, :] = A_diagonal_blocks[:, :, :].copy()
        A_lower_diagonal_blocks_pinned[:, :, :] = A_lower_diagonal_blocks[:, :, :].copy()
        A_arrow_bottom_blocks_pinned[:, :, :] = A_arrow_bottom_blocks[:, :, :].copy()
        A_arrow_tip_block_pinned[:, :] = A_arrow_tip_block[:, :].copy()
        toc = time.perf_counter()

        print(f"  Copying data to pinned memory took: {toc - tic:.5f} sec", flush=True)

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

        print("I went out of pobtaf!!", flush=True)

        if i >= n_warmups:
            t_pobtaf[i - n_warmups] = elapsed_time_pobtaf

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

        print("I went out of pobtasi!!", flush=True)

        end_time = time.perf_counter()
        elapsed_time_selinv = end_time - start_time

        if i >= n_warmups:
            t_pobtasi[i - n_warmups] = elapsed_time_selinv

        if i < n_warmups:
            print(
                f"  Warmup iteration: {i+1}/{n_warmups}, Time pobtaf: {elapsed_time_pobtaf:.5f} sec, Time pobtasi: {elapsed_time_selinv:.5f} sec", flush=True
            )
        else:
            print(
                f"  Bench iteration: {i+1-n_warmups}/{n_iterations} Time pobtaf: {elapsed_time_pobtaf:.5f} sec. Time pobtasi: {elapsed_time_selinv:.5f} sec", flush=True
            )

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
        f"Mean time pobtaf: {mean_pobtaf:.5f} sec, 95% CI: [{lb_mean_pobtaf:.5f}, {ub_mean_pobtaf:.5f}]", flush=True
    )

    print(
        f"Mean time pobtasi: {mean_pobtasi:.5f} sec, 95% CI: [{lb_mean_pobtasi:.5f}, {ub_mean_pobtasi:.5f}]", flush=True
    )
