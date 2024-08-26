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


def compute_flops_reduced_system(
    b: int,
    a: int,
    p: int,
):
    reduced_system_size = 2 * p - 1
    flops_reduced_system = compute_flops_pobtaf(
        reduced_system_size, b, a
    ) + compute_flops_pobtasi(reduced_system_size, b, a)

    return flops_reduced_system


def compute_flops_d_pobtasi(
    n: int,
    b: int,
    a: int,
    p: int,
):
    return (n / p - 1) * (19 * pow(b, 3) + 14 * a * pow(b, 2))


def show_sequential_flops_bargraph(
    n, b, a, flops_pobtaf, flops_pobtasi, flops_pobtaA2X
):

    base_width = 0.4

    pct_pobtaf = [(flops_pobtaf[i] / flops_pobtaA2X[i]) * 100 for i in range(len(n))]
    pct_pobtasi = [(flops_pobtasi[i] / flops_pobtaA2X[i]) * 100 for i in range(len(n))]

    print("pct_pobtaf: ", pct_pobtaf)
    print("pct_pobtasi: ", pct_pobtasi)

    fig, ax = plt.subplots()
    for i in range(len(n)):
        width = base_width * n[i]
        ax.bar(
            n[i],
            flops_pobtaf[i],
            width,
            color="b",
            label="flops pobtaf" if i == 0 else "",
        )
        ax.bar(
            n[i],
            flops_pobtasi[i],
            width,
            bottom=flops_pobtaf[i],
            color="g",
            label="flops pobtasi" if i == 0 else "",
        )

        ax.text(
            n[i],
            flops_pobtaf[i] / 2,
            f"{pct_pobtaf[i]:.2f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            n[i],
            flops_pobtaf[i] + flops_pobtasi[i] / 2,
            f"{pct_pobtasi[i]:.2f}%",
            ha="center",
            va="center",
            color="white",
        )

    print(n)

    ax.legend(loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(val)) for val in n])

    # ax.set_yscale("log", base=2)

    plt.show()


def show_sequential_flops(ax, n, b, a, flops_pobtaf, flops_pobtasi, flops_pobtaA2X):

    # make a scatter plot
    ax.scatter(n, flops_pobtaf, color="b", label="flops POBTAF", marker="x")
    ax.scatter(n, flops_pobtasi, color="g", label="flops POBTASI", marker="x")

    # Annotate the points
    for i in range(len(n)):
        ax.annotate(
            f"{flops_pobtaf[i]:.1e}",
            (n[i], flops_pobtaf[i]),
            textcoords="offset points",
            xytext=(-10, 7),
        )
        ax.annotate(
            f"{flops_pobtasi[i]:.1e}",
            (n[i], flops_pobtasi[i]),
            textcoords="offset points",
            xytext=(-10, 7),
        )

    # Fit linear trend lines
    coeffs_pobtaf = np.polyfit(np.log2(n), np.log2(flops_pobtaf), 1)
    coeffs_pobtasi = np.polyfit(np.log2(n), np.log2(flops_pobtasi), 1)

    # Generate x values for the trend lines
    x_vals = np.linspace(16, 1024, 100)
    y_vals_pobtaf = 2 ** (coeffs_pobtaf[0] * np.log2(x_vals) + coeffs_pobtaf[1])
    y_vals_pobtasi = 2 ** (coeffs_pobtasi[0] * np.log2(x_vals) + coeffs_pobtasi[1])

    # Plot the trend lines
    stroke_width = 0.6
    ax.plot(
        x_vals,
        y_vals_pobtaf,
        color="b",
        linestyle="--",
        linewidth=stroke_width,
        label="Trend POBTAF",
    )
    ax.plot(
        x_vals,
        y_vals_pobtasi,
        color="g",
        linestyle="--",
        linewidth=stroke_width,
        label="Trend POBTASI",
    )

    # Increase the font size
    ax.set_xlabel("Number of diagonal blocks: n", fontsize=14)
    ax.set_ylabel("FLOPS", fontsize=14)
    ax.legend()

    # set x and y range
    ax.set_xlim([n[0] / 1.1, n[-1] * 1.7])
    ax.set_ylim([flops_pobtaf[0] / 1.1, flops_pobtasi[-1] * 1.3])

    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(val)) for val in n], fontsize=14)

    ax.set_yscale("log", base=2)
    ax.set_yticks([])

    # Add custom annotations
    ax.text(
        n[-1] * 1.1,
        flops_pobtaf[0],
        f"b={b}\na={a}",
        fontsize=14,
        color="black",
    )


def show_sequential_flops_pct(ax, n, b, a, flops_pobtaf, flops_pobtasi, flops_pobtaA2X):
    pct_pobtaf = [(flops_pobtaf[i] / flops_pobtaA2X[i]) * 100 for i in range(len(n))]
    pct_pobtasi = [100 - pct_pobtaf[i] for i in range(len(n))]

    # Plot the stacked lines
    ax.fill_between(n, pct_pobtaf, label="POBTAF", color="blue", alpha=0.5)
    ax.fill_between(
        n,
        y1=100,
        y2=pct_pobtaf,
        label="POBTASI",
        color="green",
        alpha=0.5,
        where=[True] * len(n),
        interpolate=True,
    )

    i = 2
    ax.annotate(
        f"≈{pct_pobtaf[i]:.2f}%",
        (n[i], pct_pobtaf[i] / 2),
        textcoords="offset points",
        xytext=(-50, -5),
    )
    ax.annotate(
        f"≈{pct_pobtasi[i]:.2f}%",
        (n[i], pct_pobtaf[i] + pct_pobtasi[i] / 2),
        textcoords="offset points",
        xytext=(-50, -5),
    )

    # Add labels and title
    ax.set_xlabel("Number of diagonal blocks: n", fontsize=14)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.legend()

    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(val)) for val in n], fontsize=14)

    # ax.set_yticks([])


def show_load_balancing(
    n,
    b,
    a,
    load_balancing_d_pobtaf,
    load_balancing_d_pobtasi,
    ratio_d_pobtaf_d_pobtasi,
    ideal_load_balancing,
):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(
        n,
        load_balancing_d_pobtaf,
        label="load balancing d_pobtaf",
        marker="x",
        color="b",
    )
    ax[0].plot(
        n,
        load_balancing_d_pobtasi,
        label="load balancing d_pobtasi",
        marker="x",
        color="g",
    )
    ax[0].plot(
        n,
        ideal_load_balancing,
        label="ideal load balancing",
        marker="x",
        linestyle="--",
        color="r",
    )

    # annotate the ideal load balancing and put the values
    offset = 0.05
    for i in range(len(n)):
        ax[0].text(
            n[i],
            ideal_load_balancing[i] + offset,
            f"{ideal_load_balancing[i]:.2f}",
            ha="center",
            va="center",
            color="black",
        )

    # set x and y range
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlim([n[0] / 1.2, n[-1] * 1.2])
    ax[0].set_xticks([])
    ax[0].set_ylim([0, max(load_balancing_d_pobtasi) * 1.1])
    ax[0].set_ylabel("Load balancing ratio")
    ax[0].legend()

    # plot a stacked bar graph
    width = 0.2
    for i in range(len(n)):
        adjusted_width = width * n[i]
        ax[1].bar(n[i], ratio_d_pobtaf_d_pobtasi[i], color="b", width=adjusted_width)
        ax[1].bar(
            n[i],
            1 - ratio_d_pobtaf_d_pobtasi[i],
            color="g",
            bottom=ratio_d_pobtaf_d_pobtasi[i],
            width=adjusted_width,
        )

    # annotate the values in the bar
    for i in range(len(n)):
        ax[1].text(
            n[i],
            ratio_d_pobtaf_d_pobtasi[i] / 2,
            f"{ratio_d_pobtaf_d_pobtasi[i]*100:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax[1].text(
            n[i],
            ratio_d_pobtaf_d_pobtasi[i] + (1 - ratio_d_pobtaf_d_pobtasi[i]) / 2,
            f"{(1 - ratio_d_pobtaf_d_pobtasi[i])*100:.1f}%",
            ha="center",
            va="center",
            color="white",
        )

    ax[1].set_xlabel("Number of diagonal blocks: n")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlim([n[0] / 1.2, n[-1] * 1.2])
    ax[1].set_xticks(n)
    ax[1].set_xticklabels([str(int(val)) for val in n])

    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel("Operations ratio\nd_pobtaf / d_pobtasi")
    ax[1].set_yticks([])

    fig.tight_layout()

    plt.savefig("load_balancing_analysis.png")

    plt.show()


def plot_theoretical_parallel_efficiency(
    ax,
    n,
    b,
    a,
    p,
    reference,
    flops_matrix,
):
    ideal_d_flops = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ideal_d_flops[p_i, n_i] = reference[n_i] / p[p_i]

    ax.imshow(ideal_d_flops, cmap="viridis", aspect="auto")


if __name__ == "__main__":
    debug_values = True
    plot = False

    n = [32, 64, 128, 256, 512]
    b = 1024
    a = b // 4

    # Sequential algorithm
    # Complexity analysis
    # POBTAF
    total_flops_pobtaf = [compute_flops_pobtaf(n[i], b, a) for i in range(len(n))]

    # POBTASI
    total_flops_pobtasi = [compute_flops_pobtasi(n[i], b, a) for i in range(len(n))]

    # POBTA A2X
    total_flops_pobtaA2X = [
        total_flops_pobtaf[i] + total_flops_pobtasi[i] for i in range(len(n))
    ]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        # Set font size
        plt.rcParams.update({"font.size": 20})

        show_sequential_flops(
            ax[0],
            n,
            b,
            a,
            total_flops_pobtaf,
            total_flops_pobtasi,
            total_flops_pobtaA2X,
        )

        show_sequential_flops_pct(
            ax[1],
            n,
            b,
            a,
            total_flops_pobtaf,
            total_flops_pobtasi,
            total_flops_pobtaA2X,
        )

        # add a) and b) labels
        ax[0].text(
            0.5,
            -0.15,
            "a)",
            fontsize=20,
            # fontweight="bold",
            transform=ax[0].transAxes,
        )

        ax[1].text(
            0.5,
            -0.15,
            "b)",
            fontsize=20,
            # fontweight="bold",
            transform=ax[1].transAxes,
        )

        fig.tight_layout()
        plt.savefig("sequential_complexity_analysis.png")

        plt.show()

    if debug_values:
        print("total_flops_pobtaf:   ", end="")
        [print(f"{total_flops_pobtaf[i]:.2e}, ", end="") for i in range(len(n))]

        print("\ntotal_flops_pobtasi   ", end="")
        [print(f"{total_flops_pobtasi[i]:.2e}, ", end="") for i in range(len(n))]

        print("\ntotal_flops_pobtaA2X  ", end="")
        [print(f"{total_flops_pobtaA2X[i]:.2e}, ", end="") for i in range(len(n))]

        print()

    # Distributed algorithms
    # load balancing analysis
    load_balancing_d_pobtaf = [
        compute_flops_d_pobtaf(n[i], b, a, p=1) / compute_flops_pobtaf(n[i], b, a)
        for i in range(len(n))
    ]

    load_balancing_d_pobtasi = [
        compute_flops_d_pobtasi(n[i], b, a, p=1) / compute_flops_pobtasi(n[i], b, a)
        for i in range(len(n))
    ]

    ratio_d_pobtaf_d_pobtasi = [
        compute_flops_d_pobtaf(n[i], b, a, p=1)
        / compute_flops_d_pobtasi(n[i], b, a, p=1)
        for i in range(len(n))
    ]

    ideal_load_balancing = [
        ratio_d_pobtaf_d_pobtasi[i] * load_balancing_d_pobtaf[i]
        + (1 - ratio_d_pobtaf_d_pobtasi[i]) * load_balancing_d_pobtasi[i]
        for i in range(len(n))
    ]

    if plot:
        show_load_balancing(
            n,
            b,
            a,
            load_balancing_d_pobtaf,
            load_balancing_d_pobtasi,
            ratio_d_pobtaf_d_pobtasi,
            ideal_load_balancing,
        )

    if debug_values:
        print()
        print("load_balancing_d_pobtaf:  ", end="")
        for i in range(len(n)):
            print(f"{load_balancing_d_pobtaf[i]:.2f}, ", end="")
        print()

        print("load_balancing_d_pobtasi: ", end="")
        for i in range(len(n)):
            print(f"{load_balancing_d_pobtasi[i]:.2f}, ", end="")
        print()

        print("ratio_d_pobtaf_d_pobtasi: ", end="")
        for i in range(len(n)):
            print(f"{ratio_d_pobtaf_d_pobtasi[i]:.2f}, ", end="")
        print()

        print("ideal_load_balancing: ", end="")
        for i in range(len(n)):
            print(f"{ideal_load_balancing[i]:.2f}, ", end="")
        print()

    # Complexity analysis
    p = [2, 4, 8, 16, 32]

    # D_POBTAF
    total_flops_d_pobtaf = np.zeros((len(p), len(n)))
    for n_i in range(len(n)):
        for p_i in range(len(p)):
            if n[n_i] >= 3 * p[p_i]:
                partition_sizes = get_partition_size(
                    n=n[n_i], p=p[p_i], balancing_ratio=ideal_load_balancing[n_i]
                )

                # The more full central partition is the '1'
                central_partition_size = partition_sizes[1]

                total_flops_d_pobtaf[p_i, n_i] = compute_flops_d_pobtaf(
                    central_partition_size, b, a, p=1
                )
            else:
                total_flops_d_pobtaf[p_i, n_i] = 0

    print("\ntotal_flops_d_pobtaf", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_d_pobtaf[p_i, n_i]:.2e}, ", end="")
    print()

    # Reduced system
    total_flops_reduced_system = np.zeros((len(p), len(n)))
    for n_i in range(len(n)):
        for p_i in range(len(p)):
            if n[n_i] >= 3 * p[p_i]:
                total_flops_reduced_system[p_i, n_i] = compute_flops_reduced_system(
                    b, a, p[p_i]
                )
            else:
                total_flops_reduced_system[p_i, n_i] = 0

    print("\ntotal_flops_reduced_system (pobtaf + pobtasi)", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_reduced_system[p_i, n_i]:.2e}, ", end="")
    print()

    # D_POBTASI
    total_flops_d_pobtasi = np.zeros((len(p), len(n)))
    for n_i in range(len(n)):
        for p_i in range(len(p)):
            if n[n_i] >= 3 * p[p_i]:
                partition_sizes = get_partition_size(
                    n=n[n_i], p=p[p_i], balancing_ratio=ideal_load_balancing[n_i]
                )

                # The more full central partition is the '1'
                central_partition_size = partition_sizes[1]

                total_flops_d_pobtasi[p_i, n_i] = compute_flops_d_pobtasi(
                    central_partition_size, b, a, p=1
                )
            else:
                total_flops_d_pobtasi[p_i, n_i] = 0

    print("\ntotal_flops_d_pobtasi", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_d_pobtasi[p_i, n_i]:.2e}, ", end="")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    plot_theoretical_parallel_efficiency(
        ax[0], n, b, a, p, total_flops_pobtaf, total_flops_d_pobtaf
    )
    plot_theoretical_parallel_efficiency(
        ax[1], n, b, a, p, total_flops_pobtasi, total_flops_d_pobtasi
    )

    plt.show()


# Strong scaling partition ratio
if __name__ == "__main__":
    exit()

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
