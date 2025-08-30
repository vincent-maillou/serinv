import time

tic = time.perf_counter()
import numpy as np
import cupy as cp
import argparse

from serinv.algs import ddbtsc, ddbtsci

def generate_dataset(
    n_blocks: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    bsym: bool,
    quadratic: bool,
    dtype=np.float64,
):
    print(f"Generating sequential quadratic dataset...", flush=True)
    print(f"    - Generating A", flush=True)
    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0
    A_diagonal_blocks = rc * np.random.rand(
        n_blocks, diagonal_blocksize, diagonal_blocksize
    )
    A_lower_diagonal_blocks = rc * np.random.rand(
        n_blocks - 1, diagonal_blocksize, diagonal_blocksize
    )
    A_upper_diagonal_blocks = rc * np.random.rand(
        n_blocks - 1, diagonal_blocksize, diagonal_blocksize
    )
    # A arrowhead part
    A_lower_arrow_blocks = rc * np.random.rand(
        n_blocks, arrowhead_blocksize, diagonal_blocksize
    )
    A_upper_arrow_blocks = rc * np.random.rand(
        n_blocks,
        diagonal_blocksize,
        arrowhead_blocksize,
    )
    A_arrow_tip_block = rc * np.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )
    arrow_colsum = np.zeros((arrowhead_blocksize), dtype=A_diagonal_blocks.dtype)
    for i in range(A_diagonal_blocks.shape[0]):
        colsum = np.sum(A_diagonal_blocks[i], axis=1) - np.diag(
            A_diagonal_blocks[i]
        )
        if i > 0:
            colsum += np.sum(A_lower_diagonal_blocks[i - 1], axis=1)
        if i < n_blocks - 1:
            colsum += np.sum(A_upper_diagonal_blocks[i], axis=1)
        colsum += np.sum(A_upper_arrow_blocks[i], axis=1)
        A_diagonal_blocks[i] += np.diag(colsum)
        arrow_colsum[:] += np.sum(A_lower_arrow_blocks[i], axis=1)
    A_arrow_tip_block[:, :] += np.diag(
        arrow_colsum + np.sum(A_arrow_tip_block[:, :], axis=1)
    )

    A = {
        "A_diagonal_blocks": A_diagonal_blocks,
        "A_lower_diagonal_blocks": A_lower_diagonal_blocks,
        "A_upper_diagonal_blocks": A_upper_diagonal_blocks,
        "A_lower_arrow_blocks": A_lower_arrow_blocks,
        "A_upper_arrow_blocks": A_upper_arrow_blocks,
        "A_arrow_tip_block": A_arrow_tip_block,
    }

    if quadratic:
        print(f"    - Generating B (Quadratic Equation)", flush=True)
        B_diagonal_blocks = rc * np.random.rand(
            n_blocks, diagonal_blocksize, diagonal_blocksize
        )
        B_lower_diagonal_blocks = rc * np.random.rand(
            n_blocks - 1, diagonal_blocksize, diagonal_blocksize
        )
        B_upper_diagonal_blocks = rc * np.random.rand(
            n_blocks - 1, diagonal_blocksize, diagonal_blocksize
        )
        # B arrowhead part
        B_lower_arrow_blocks = rc * np.random.rand(
            n_blocks, arrowhead_blocksize, diagonal_blocksize
        )
        B_upper_arrow_blocks = rc * np.random.rand(
            n_blocks,
            diagonal_blocksize,
            arrowhead_blocksize,
        )
        B_arrow_tip_block = rc * np.random.rand(
            arrowhead_blocksize, arrowhead_blocksize
        )
        arrow_colsum = np.zeros((arrowhead_blocksize), dtype=B_diagonal_blocks.dtype)
        for i in range(B_diagonal_blocks.shape[0]):
            colsum = np.sum(B_diagonal_blocks[i], axis=1) - np.diag(
                B_diagonal_blocks[i]
            )
            if i > 0:
                colsum += np.sum(B_lower_diagonal_blocks[i - 1], axis=1)
            if i < n_blocks - 1:
                colsum += np.sum(B_upper_diagonal_blocks[i], axis=1)
            colsum += np.sum(B_upper_arrow_blocks[i], axis=1)
            B_diagonal_blocks[i] += np.diag(colsum)
            arrow_colsum[:] += np.sum(B_lower_arrow_blocks[i], axis=1)
        B_arrow_tip_block[:, :] += np.diag(
            arrow_colsum + np.sum(B_arrow_tip_block[:, :], axis=1)
        )

        if bsym:
            for i in range(n_blocks):
                B_diagonal_blocks[i] = (
                    B_diagonal_blocks[i] + B_diagonal_blocks[i].conj().T
                ) / 2
                if i < n_blocks - 1:
                    B_upper_diagonal_blocks[i] = B_lower_diagonal_blocks[i].conj().T
                    B_upper_arrow_blocks[i] = B_lower_arrow_blocks[i].conj().T
            B_arrow_tip_block = (B_arrow_tip_block + B_arrow_tip_block.conj().T) / 2

        B = {
            "B_diagonal_blocks": B_diagonal_blocks,
            "B_lower_diagonal_blocks": B_lower_diagonal_blocks,
            "B_upper_diagonal_blocks": B_upper_diagonal_blocks,
            "B_lower_arrow_blocks": B_lower_arrow_blocks,
            "B_upper_arrow_blocks": B_upper_arrow_blocks,
            "B_arrow_tip_block": B_arrow_tip_block,
        }
    else:
        B = None

    return A, B

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--b",
        type=int,
        default=128,
        help="an integer for the diagonal block size",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="an integer for the number of diagonal blocks",
    )
    parser.add_argument(
        "--bsym",
        type=bool,
        default=True,
        help="whether to make B block-symmetric or not",
    )
    parser.add_argument(
        "--q",
        type=int,
        help="wether to run the quadratic or not",
    )
    args = parser.parse_args()
    toc = time.perf_counter()
    print(f"Import and parsing took: {toc - tic:.5f} sec", flush=True)

    quadratic = False if args.q == 0 else True
    n_iterations = 10
    n_warmups = 2

    tic = time.perf_counter()
    A, B = generate_dataset(
        n_blocks = args.n,
        diagonal_blocksize = args.b,
        bsym = args.bsym,
        quadratic = quadratic,
    )
    toc = time.perf_counter()
    print(f"Dataset generation took: {toc - tic:.5f} sec", flush=True)

    A_diagonal_blocks_cpu = A["A_diagonal_blocks"]
    A_lower_diagonal_blocks_cpu = A["A_lower_diagonal_blocks"]
    A_upper_diagonal_blocks_cpu = A["A_upper_diagonal_blocks"]

    A_diagonal_blocks_gpu = cp.empty_like(A_diagonal_blocks_cpu)
    A_lower_diagonal_blocks_gpu = cp.empty_like(A_lower_diagonal_blocks_cpu)
    A_upper_diagonal_blocks_gpu = cp.empty_like(A_upper_diagonal_blocks_cpu)

    if quadratic:
        B_diagonal_blocks_cpu = B["B_diagonal_blocks"]
        B_lower_diagonal_blocks_cpu = B["B_lower_diagonal_blocks"]
        B_upper_diagonal_blocks_cpu = B["B_upper_diagonal_blocks"]

        B_diagonal_blocks_gpu = cp.empty_like(B_diagonal_blocks_cpu)
        B_lower_diagonal_blocks_gpu = cp.empty_like(B_lower_diagonal_blocks_cpu)
        B_upper_diagonal_blocks_gpu = cp.empty_like(B_upper_diagonal_blocks_cpu)

    direction = "downward"

    t_ddbtsc = []
    t_ddbtsci = []

    for i in range(n_warmups + n_iterations):
        print(f"Iteration: {i+1}/{n_warmups+n_iterations}", flush=True)

        tic = time.perf_counter()
        A_diagonal_blocks_gpu.set(arr=A_diagonal_blocks_cpu)
        A_lower_diagonal_blocks_gpu.set(arr=A_lower_diagonal_blocks_cpu)
        A_upper_diagonal_blocks_gpu.set(arr=A_upper_diagonal_blocks_cpu)

        if quadratic:
            B_diagonal_blocks_gpu.set(arr=B_diagonal_blocks_cpu)
            B_lower_diagonal_blocks_gpu.set(arr=B_lower_diagonal_blocks_cpu)
            B_upper_diagonal_blocks_gpu.set(arr=B_upper_diagonal_blocks_cpu)
            rhs = {
                "B_diagonal_blocks": B_diagonal_blocks_gpu,
                "B_lower_diagonal_blocks": B_lower_diagonal_blocks_gpu,
                "B_upper_diagonal_blocks": B_upper_diagonal_blocks_gpu,
            }
        toc = time.perf_counter()
        print(f"Copying data to GPU took: {toc - tic:.5f} sec", flush=True)

        cp.cuda.runtime.deviceSynchronize()
        tic = time.perf_counter()
        ddbtsc(
            A_diagonal_blocks_gpu,
            A_lower_diagonal_blocks_gpu,
            A_upper_diagonal_blocks_gpu,
            rhs=rhs if quadratic else None,
            quadratic=quadratic,
        )
        cp.cuda.runtime.deviceSynchronize()
        toc = time.perf_counter()
        elapsed = toc - tic
        print(f"t_ddbtsc took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_ddbtsc.append(elapsed)

        cp.cuda.runtime.deviceSynchronize()
        tic = time.perf_counter()
        ddbtsci(
            A_diagonal_blocks_gpu,
            A_lower_diagonal_blocks_gpu,
            A_upper_diagonal_blocks_gpu,
            rhs=rhs if quadratic else None,
            quadratic=quadratic,
        )
        cp.cuda.runtime.deviceSynchronize()
        toc = time.perf_counter()
        elapsed = toc - tic
        print(f"t_ddbtsci took: {elapsed:.5f} sec", flush=True)
        if i >= n_warmups:
            t_ddbtsci.append(elapsed)

    print(f"t_ddbtsc: {t_ddbtsc}", flush=True)
    print(f"t_ddbtsci: {t_ddbtsci}", flush=True)

    print(f"avg t_ddbtsc: {np.mean(np.array(t_ddbtsc)):.5f} sec", flush=True)
    print(f"avg t_ddbtsci: {np.mean(np.array(t_ddbtsci)):.5f} sec", flush=True)
    print(f"avg total time: {np.mean(np.array(t_ddbtsc)) + np.mean(np.array(t_ddbtsci)):.5f} sec", flush=True)
