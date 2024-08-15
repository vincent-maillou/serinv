# Copyright 2023-2024 ETH Zurich. All rights reserved.

SEED = 63

import numpy as np

np.random.seed(SEED)

from load_datmat import read_sym_CSC
from complexity_analysis import (
    compute_flops_pobtaf,
    compute_flops_d_pobtaf,
    get_partition_size,
)
import argparse


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
        "--n_blocks",
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
        "--n_processes",
        type=int,
        default=1,
        help="number of processes for the distributed case",
    )

    args = parser.parse_args()
    diagonal_blocksize = args.diagonal_blocksize
    arrowhead_blocksize = args.arrowhead_blocksize
    n_blocks = args.n_blocks
    file_path = args.file_path
    n_processes = args.n_processes

    n = diagonal_blocksize * n_blocks + arrowhead_blocksize

    file = (
        "Qxy_ns"
        + str(diagonal_blocksize)
        + "_nt"
        + str(n_blocks)
        + "_nss0_nb"
        + str(arrowhead_blocksize)
        + "_n"
        + str(n)
        + ".dat"
    )
    
    filename = file_path + file

    A = read_sym_CSC(filename)

    PATH = (
        file_path + "distributed_datasets/"
    )

    load_balancing_ratio = compute_flops_d_pobtaf(
        n=n_blocks, b=diagonal_blocksize, a=arrowhead_blocksize, p=1
    ) / compute_flops_pobtaf(n=n_blocks, b=diagonal_blocksize, a=arrowhead_blocksize)

    partition_sizes = get_partition_size(
        n=n_blocks, p=n_processes, balancing_ratio=load_balancing_ratio
    )

    for p in range(n_processes):

        n_blocks_pi = partition_sizes[p]

        A_diagonal_blocks_pi = np.zeros(
            (n_blocks_pi, diagonal_blocksize, diagonal_blocksize),dtype=A.dtype
        )

        if p == n_processes - 1:
            A_lower_diagonal_blocks_pi = np.zeros(
                (n_blocks_pi - 1, diagonal_blocksize, diagonal_blocksize),dtype=A.dtype
            )
        else:
            A_lower_diagonal_blocks_pi = np.zeros(
                (n_blocks_pi, diagonal_blocksize, diagonal_blocksize),dtype=A.dtype
            )

        A_arrow_bottom_blocks_pi = np.zeros(
            (partition_sizes[p], arrowhead_blocksize, diagonal_blocksize),dtype=A.dtype
        )

        A_offset = int(np.sum(partition_sizes[:p]))

        print(f"P:{p} A_offset:{A_offset} n_blocks_pi:{n_blocks_pi}")

        for i in range(n_blocks_pi):
            A_diagonal_blocks_pi[i, :, :] = A[
                (A_offset + i) * diagonal_blocksize : (A_offset + i + 1) * diagonal_blocksize,
                (A_offset + i) * diagonal_blocksize : (A_offset + i + 1) * diagonal_blocksize,
            ].toarray()

            if i < n_blocks_pi - 1:
                A_lower_diagonal_blocks_pi[i, :, :] = A[
                    (A_offset + i + 1) * diagonal_blocksize : (A_offset + i + 2) * diagonal_blocksize,
                    (A_offset + i) * diagonal_blocksize : (A_offset + i + 1) * diagonal_blocksize,
                ].toarray()

            A_arrow_bottom_blocks_pi[i, :, :] = A[
                -arrowhead_blocksize:,
                (A_offset + i) * diagonal_blocksize : (A_offset + i + 1) * diagonal_blocksize,
            ].toarray()

        file_name = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_blocks_p{p}_np{n_processes-1}.npz"

        np.savez(
            PATH + file_name,
            A_diagonal_blocks=A_diagonal_blocks_pi,
            A_lower_diagonal_blocks=A_lower_diagonal_blocks_pi,
            A_arrow_bottom_blocks=A_arrow_bottom_blocks_pi,
        )

    A_arrow_tip_block = np.zeros((arrowhead_blocksize, arrowhead_blocksize), dtype=A.dtype)
    A_arrow_tip_block[:, :] = A[-arrowhead_blocksize:, -arrowhead_blocksize:].toarray()

    file_name_arrowtip = f"pobta_nb{n_blocks}_ds{diagonal_blocksize}_as{arrowhead_blocksize}_arrowtip_np{n_processes-1}.npz"

    np.savez(
        PATH + file_name_arrowtip,
        A_arrow_tip_block=A_arrow_tip_block,
    )