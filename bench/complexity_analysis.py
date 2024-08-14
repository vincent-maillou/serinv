import numpy as np
import matplotlib.pyplot as plt


def compute_flops_pobtaf(n: int, b: int, a: int):
    return int(
        (
            n
            * (
                10 / 3 * pow(b, 3)
                + (1 / 2 + 3 * a) * pow(b, 2)
                + (1 / 6 + 2 * pow(a, 2)) * b
            )
            - 3 * pow(b, 3)
            - 2 * a * pow(b, 2)
            + 1 / 3 * pow(a, 3)
            + 1 / 2 * pow(a, 2)
            + 1 / 6 * a
        )
    )


def compute_flops_pobtasi(n: int, b: int, a: int):
    return int(
        (
            (n - 1) * (9 * pow(b, 3) + 10 * pow(b, 2) + 2 * pow(a, 2) * b)
            + 2 * pow(b, 3)
            + 5 * a * pow(b, 2)
            + 2 * pow(a, 2) * b
            + 2 * pow(a, 3)
        )
    )


def get_partition_size(n: int, p: int, balancing_ratio: float):
    # Also need to ensure that the partition size is greater than 3
    middle_process_partition_size = int(n / (p - 1 + balancing_ratio))
    if middle_process_partition_size < 3:
        return 3
    return middle_process_partition_size


def compute_flops_d_pobtaf(
    n: int,
    b: int,
    a: int,
    p: int,
):
    return int(
        (n / p - 2)
        * (
            19 / 3 * pow(b, 3)
            + (1 / 2 + 5 * a) * pow(b, 2)
            + (1 / 6 + 4 * pow(a, 2)) * b
        )
    )


if __name__ == "__main__":
    n = [32, 64, 128, 256, 512]
    b = 1024
    a = b // 4

    # Sequential algorithm
    # Complexity analysis
    # POBTAF
    total_flops_pobtaf = [compute_flops_pobtaf(n[i], b, a) for i in range(len(n))]
    print("total_flops_pobtaf\n    ", end="")
    [print(f"{total_flops_pobtaf[i]:.2e}, ", end="") for i in range(len(n))]

    # POBTASI
    total_flops_pobtasi = [compute_flops_pobtasi(n[i], b, a) for i in range(len(n))]
    print("\ntotal_flops_pobtasi\n    ", end="")
    [print(f"{total_flops_pobtasi[i]:.2e}, ", end="") for i in range(len(n))]

    # POBTA A2X
    total_flops_pobtaA2X = [
        total_flops_pobtaf[i] + total_flops_pobtasi[i] for i in range(len(n))
    ]
    print("\ntotal_flops_pobtaA2X\n    ", end="")
    [print(f"{total_flops_pobtaA2X[i]:.2e}, ", end="") for i in range(len(n))]

    print()

    # Distributed algorithms
    # load balancing analysis
    ratio_first_over_middle_partitions = [
        compute_flops_d_pobtaf(n[i], b, a, p=1) / compute_flops_pobtaf(n[i], b, a)
        for i in range(len(n))
    ]
    ratio_avg = np.mean(ratio_first_over_middle_partitions)
    print("ratio_first_over_middle_partitions:", ratio_avg)

    # Complexity analysis
    p = [2, 4, 8, 16, 32]

    # D_POBTAF
    total_flops_d_pobtaf = np.zeros((len(p), len(n)))
    for n_i in range(len(n)):
        for p_i in range(len(p)):
            if n[n_i] >= 3 * p[p_i]:
                n_blocks_balanced_partition = get_partition_size(
                    n=n[n_i], p=p[p_i], balancing_ratio=ratio_avg
                )
                total_flops_d_pobtaf[p_i, n_i] = compute_flops_d_pobtaf(
                    n_blocks_balanced_partition, b, a, p[p_i]
                )
            else:
                total_flops_d_pobtaf[p_i, n_i] = -1

    print("\ntotal_flops_d_pobtaf", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_d_pobtaf[p_i, n_i]:.2e}, ", end="")
