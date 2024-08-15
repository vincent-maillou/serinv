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
    middle_process_partition_size = int(np.ceil(n / (p - 1 + balancing_ratio)))

    if middle_process_partition_size < 3:
        middle_process_partition_size = 3

    first_partition_size = n - (p - 1) * middle_process_partition_size

    extras = 0
    if first_partition_size < 3:
        Neach_section, extras = divmod(n, p)
        first_partition_size = middle_process_partition_size = Neach_section

    partition_sizes = []
    for i in range(p):
        if i == 0:
            partition_sizes.append(first_partition_size)
        else:
            partition_sizes.append(middle_process_partition_size)
        if extras > 0:
            partition_sizes[-1] += 1
            extras -= 1

    assert np.sum(partition_sizes) == n

    return partition_sizes


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
                partition_sizes = get_partition_size(
                    n=n[n_i], p=p[p_i], balancing_ratio=ratio_avg
                )
                total_flops_d_pobtaf[p_i, n_i] = compute_flops_d_pobtaf(
                    partition_sizes[1], b, a, p[p_i]
                )
            else:
                total_flops_d_pobtaf[p_i, n_i] = -1

    print("\ntotal_flops_d_pobtaf", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_d_pobtaf[p_i, n_i]:.2e}, ", end="")

    print()

# Strong scaling partition ratio
if __name__ == "__main__":
    n = 365
    b = 2865
    a = 4

    ratio_first_over_middle_partitions = compute_flops_d_pobtaf(
        n, b, a, p=1
    ) / compute_flops_pobtaf(n, b, a)

    print(
        f"INLA case ratio (n={n}, b={b}, a={a}): {ratio_first_over_middle_partitions}"
    )

    p = [2, 4, 8, 16, 32]

    for i in range(len(p)):
        partition_sizes = get_partition_size(
            n=n, p=p[i], balancing_ratio=ratio_first_over_middle_partitions
        )

        print(
            f"  {p[i]} processes | first_partition: {partition_sizes[0]}, middle_partition: {partition_sizes[1]}, total: {np.sum(partition_sizes)}"
        )
