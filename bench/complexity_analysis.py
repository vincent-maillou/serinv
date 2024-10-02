import matplotlib.pyplot as plt
import numpy as np


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
    ax_ref,
    ax_mat,
    n,
    b,
    a,
    p,
    reference,
    flops_matrix,
):
    # Plot the reference
    ax[ax_ref[0], ax_ref[1]].imshow(
        np.reshape(reference, (1, len(n))), cmap="viridis", aspect="auto"
    )

    # Add the reference values
    for i in range(len(n)):
        ax[ax_ref[0], ax_ref[1]].text(
            i,
            0,
            f"{reference[i]:.1e} flops",
            ha="center",
            va="center",
            color="white",
        )

    # remove x and y ticks
    ax[ax_ref[0], ax_ref[1]].set_xticks([])
    ax[ax_ref[0], ax_ref[1]].set_yticks([])

    ideal_d_flops = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ideal_d_flops[p_i, n_i] = reference[n_i] / p[p_i]

    theoretical_parallel_efficiency = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            theoretical_parallel_efficiency[p_i, n_i] = (
                ideal_d_flops[p_i, n_i] / flops_matrix[p_i, n_i]
            ) * 100

    ax[ax_mat[0], ax_mat[1]].imshow(
        theoretical_parallel_efficiency, cmap="viridis", aspect="auto"
    )

    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ax[ax_mat[0], ax_mat[1]].text(
                n_i,
                p_i,
                f"{theoretical_parallel_efficiency[p_i, n_i]:.2f}%\n{flops_matrix[p_i, n_i]:.1e} flops",
                ha="center",
                va="center",
                color="white",
            )

    # Add xtiks and yticks
    ax[ax_mat[0], ax_mat[1]].set_xticks(range(len(n)))
    ax[ax_mat[0], ax_mat[1]].set_xticklabels([str(val) for val in n], fontsize=14)
    ax[ax_mat[0], ax_mat[1]].set_xlabel("Number of diagonal blocks", fontsize=14)

    if ax_mat == [1, 0]:
        ax[ax_mat[0], ax_mat[1]].set_yticks(range(len(p)))
        ax[ax_mat[0], ax_mat[1]].set_yticklabels([str(val) for val in p], fontsize=14)

        # add y label
        ax[ax_ref[0], ax_ref[1]].set_ylabel("Sequential\nreference", fontsize=14)
        ax[ax_mat[0], ax_mat[1]].set_ylabel("Number of processes", fontsize=14)
    else:
        ax[ax_mat[0], ax_mat[1]].set_yticks([])


def plot_theoretical_parallel_efficiency_A2X(
    n,
    b,
    a,
    p,
    reference,
    flops_matrix,
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 5]})

    # Plot the reference
    ax[0].imshow(np.reshape(reference, (1, len(n))), cmap="viridis", aspect="auto")

    # add title
    ax[0].set_title(
        "A2X theoretical parallel efficiency\n(POBTAF + Inv_red + POBTASI)", fontsize=14
    )

    # Add the reference values
    for i in range(len(n)):
        ax[0].text(
            i,
            0,
            f"{reference[i]:.1e} flops",
            ha="center",
            va="center",
            color="white",
        )

    # remove x and y ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylabel("Sequential\nreference", fontsize=14)

    ideal_d_flops = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ideal_d_flops[p_i, n_i] = reference[n_i] / p[p_i]

    theoretical_parallel_efficiency = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            theoretical_parallel_efficiency[p_i, n_i] = (
                ideal_d_flops[p_i, n_i] / flops_matrix[p_i, n_i]
            ) * 100

    # print()
    # # print the parallel efficiency
    # for p_i in range(len(p)):
    #     for n_i in range(len(n)):
    #         print(
    #             f"n={n[n_i]}, p={p[p_i]}: {theoretical_parallel_efficiency[p_i, n_i]:.2f}%"
    # )

    ax[1].imshow(theoretical_parallel_efficiency, cmap="viridis", aspect="auto")

    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ax[1].text(
                n_i,
                p_i,
                f"{theoretical_parallel_efficiency[p_i, n_i]:.2f}%\n{flops_matrix[p_i, n_i]:.1e} flops",
                ha="center",
                va="center",
                color="white",
            )

    # Add xtiks and yticks
    ax[1].set_xticks(range(len(n)))
    ax[1].set_xticklabels([str(val) for val in n], fontsize=14)
    ax[1].set_xlabel("Number of diagonal blocks", fontsize=14)

    ax[1].set_yticks(range(len(p)))
    ax[1].set_yticklabels([str(val) for val in p], fontsize=14)

    # add y label
    ax[1].set_ylabel("Number of processes", fontsize=14)

    fig.tight_layout()

    plt.savefig("theoretical_parallel_efficiency_A2X.png")


def plot_theoretical_speedup_A2X(
    n,
    b,
    a,
    p,
    reference,
    flops_matrix,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    theoretical_speedup = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            theoretical_speedup[p_i, n_i] = reference[n_i] / flops_matrix[p_i, n_i]

    ax.imshow(theoretical_speedup, cmap="viridis", aspect="auto")

    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ax.text(
                n_i,
                p_i,
                f"{theoretical_speedup[p_i, n_i]:.2f}",
                ha="center",
                va="center",
                color="white",
            )

    # Add xtiks and yticks
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels([str(val) for val in n], fontsize=14)
    ax.set_xlabel("Number of diagonal blocks", fontsize=14)

    ax.set_yticks(range(len(p)))
    ax.set_yticklabels([str(val) for val in p], fontsize=14)

    # add y label
    ax.set_ylabel("Number of processes", fontsize=14)

    ax.set_title("A2X theoretical speedup\n(POBTAF + Inv_red + POBTASI)", fontsize=14)
    fig.tight_layout()

    plt.savefig("theoretical_speedup_A2X.png")


def compute_ideal_load_balancing(
    n,
    b,
    a,
):
    # Distributed algorithms
    # load balancing analysis
    load_balancing_d_pobtaf = compute_flops_d_pobtaf(
        n, b, a, p=1
    ) / compute_flops_pobtaf(n, b, a)

    load_balancing_d_pobtasi = compute_flops_d_pobtasi(
        n, b, a, p=1
    ) / compute_flops_pobtasi(n, b, a)

    ratio_d_pobtaf_d_pobtasi = compute_flops_d_pobtaf(
        n, b, a, p=1
    ) / compute_flops_d_pobtasi(n, b, a, p=1)

    ideal_load_balancing = (
        ratio_d_pobtaf_d_pobtasi * load_balancing_d_pobtaf
        + (1 - ratio_d_pobtaf_d_pobtasi) * load_balancing_d_pobtasi
    )

    return ideal_load_balancing


def plot_cost_breakdown(
    n,
    b,
    a,
    p,
    total_flops_d_pobtaf,
    total_flops_d_pobtasi,
    total_flops_reduced_system,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    selected_n = 128
    n_i = n.index(selected_n)

    pct_d_pobtaf = np.zeros((len(p)))
    pct_d_pobtasi = np.zeros((len(p)))
    pct_reduced_system = np.zeros((len(p)))

    for p_i in range(len(p)):
        total = (
            total_flops_d_pobtaf[p_i, n_i]
            + total_flops_d_pobtasi[p_i, n_i]
            + total_flops_reduced_system[p_i, n_i]
        )

        pct_d_pobtaf[p_i] = (total_flops_d_pobtaf[p_i, n_i] / total) * 100
        pct_reduced_system[p_i] = (total_flops_reduced_system[p_i, n_i] / total) * 100
        pct_d_pobtasi[p_i] = (total_flops_d_pobtasi[p_i, n_i] / total) * 100

    width = 0.4
    for p_i in range(len(p)):
        adjusted_width = width * p[p_i]
        # ax.bar(
        #     p[p_i],
        #     pct_d_pobtaf[p_i],
        #     color="b",
        #     width=adjusted_width,
        #     label="d_pobtaf" if p_i == 0 else "",
        # )
        # ax.bar(
        #     p[p_i],
        #     pct_reduced_system[p_i],
        #     color="r",
        #     width=adjusted_width,
        #     bottom=pct_d_pobtaf[p_i],
        #     label="reduced system" if p_i == 0 else "",
        # )
        # ax.bar(
        #     p[p_i],
        #     pct_d_pobtasi[p_i],
        #     color="g",
        #     width=adjusted_width,
        #     bottom=pct_d_pobtaf[p_i] + pct_reduced_system[p_i],
        #     label="d_pobtasi" if p_i == 0 else "",
        # )
        # ax.text(
        #     p[p_i],
        #     pct_d_pobtaf[p_i] / 2,
        #     f"{pct_d_pobtaf[p_i]:.2f}%",
        #     ha="center",
        #     va="center",
        #     color="black",
        # )
        # ax.text(
        #     p[p_i],
        #     pct_d_pobtaf[p_i] + pct_reduced_system[p_i] / 2,
        #     f"{pct_reduced_system[p_i]:.2f}%",
        #     ha="center",
        #     va="center",
        #     color="black",
        # )
        # ax.text(
        #     p[p_i],
        #     pct_d_pobtaf[p_i] + pct_reduced_system[p_i] + pct_d_pobtasi[p_i] / 2,
        #     f"{pct_d_pobtasi[p_i]:.2f}%",
        #     ha="center",
        #     va="center",
        #     color="black",
        # )

        ax.bar(
            p[p_i],
            total_flops_d_pobtaf[p_i, n_i],
            color="b",
            width=adjusted_width,
            label="d_pobtaf" if p_i == 0 else "",
        )
        ax.bar(
            p[p_i],
            total_flops_reduced_system[p_i, n_i],
            color="r",
            width=adjusted_width,
            bottom=total_flops_d_pobtaf[p_i, n_i],
            label="reduced system" if p_i == 0 else "",
        )
        ax.bar(
            p[p_i],
            total_flops_d_pobtasi[p_i, n_i],
            color="g",
            width=adjusted_width,
            bottom=total_flops_d_pobtaf[p_i, n_i]
            + total_flops_reduced_system[p_i, n_i],
            label="d_pobtasi" if p_i == 0 else "",
        )

        ax.text(
            p[p_i],
            total_flops_d_pobtaf[p_i, n_i] / 2,
            f"{total_flops_d_pobtaf[p_i, n_i]:.1e}\n{pct_d_pobtaf[p_i]:.2f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            p[p_i],
            total_flops_d_pobtaf[p_i, n_i] + total_flops_reduced_system[p_i, n_i] / 2,
            f"{total_flops_reduced_system[p_i, n_i]:.1e}\n{pct_reduced_system[p_i]:.2f}%",
            ha="center",
            va="center",
            color="white",
        )
        ax.text(
            p[p_i],
            total_flops_d_pobtaf[p_i, n_i]
            + total_flops_reduced_system[p_i, n_i]
            + total_flops_d_pobtasi[p_i, n_i] / 2,
            f"{total_flops_d_pobtasi[p_i, n_i]:.1e}\n{pct_d_pobtasi[p_i]:.2f}%",
            ha="center",
            va="center",
            color="white",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(p)
    ax.set_xticklabels([str(val) for val in p])
    ax.set_xlabel("Number of processes")

    ax.set_ylabel("Total flops per rank")
    ax.legend(loc="upper right")

    # add title
    ax.set_title(f"n={selected_n}")

    plt.tight_layout()

    plt.savefig("cost_breakdown.png")


def plot_nested_solving(
    np_nested,
    n,
    p,
    sequential_reference,
    flops_matrix_3d,
):
    n_rows = len(p) + 1
    fig, ax = plt.subplots(
        n_rows, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1] + [4] * len(p)}
    )

    # Plot the reference
    ax[0].imshow(
        np.reshape(sequential_reference, (1, len(n))), cmap="viridis", aspect="auto"
    )

    # Add the reference values
    for i in range(len(n)):
        ax[0].text(
            i,
            0,
            f"{sequential_reference[i]:.1e} flops",
            ha="center",
            va="center",
            color="white",
        )

    # remove x and y ticks
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # ax[0].set_ylabel("Sequential\nflops", fontsize=14)

    ideal_d_flops = np.zeros((len(p), len(n)))
    for p_i in range(len(p)):
        for n_i in range(len(n)):
            ideal_d_flops[p_i, n_i] = sequential_reference[n_i] / p[p_i]

    np_nested = [1] + np_nested

    theoretical_parallel_efficiency = np.zeros((len(np_nested), len(p), len(n)))

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    bar_width = 0.22
    offset = np.arange(len(np_nested)) * bar_width

    for np_i in range(len(np_nested)):
        for p_i in range(len(p)):
            if n_reduced >= 3 * np_nested[np_i]:
                for n_i in range(len(n)):
                    theoretical_parallel_efficiency[np_i, p_i, n_i] = (
                        ideal_d_flops[p_i, n_i] / flops_matrix_3d[np_i, p_i, n_i]
                    ) * 100

                    ax[1 + p_i].bar(
                        n_i + offset[np_i],
                        theoretical_parallel_efficiency[np_i, p_i, n_i],
                        width=bar_width,
                        label=f"np_i={np_i}, p_i={p_i}" if n_i == 0 else "",
                        color=colors[np_i],
                    )

                    # add annotation
                    ax[1 + p_i].text(
                        n_i + offset[np_i],
                        theoretical_parallel_efficiency[np_i, p_i, n_i]
                        + 1,  # Adjust the position above the bar
                        f"{theoretical_parallel_efficiency[np_i, p_i, n_i]:.2f}%",
                        ha="center",
                        va="bottom",
                        color="black",
                    )

    # Set axis ranges:
    for ax_i in range(1, n_rows):
        ax[ax_i].set_xlim([-0.2, (len(n) - 1) + 0.8])
        if ax_i == n_rows - 1:
            ax[ax_i].set_xticks(range(len(n)))
            ax[ax_i].set_xticklabels([str(val) for val in n], fontsize=14)
            ax[ax_i].set_xlabel("Number of diagonal blocks", fontsize=14)
        else:
            ax[ax_i].set_xticks([])
            ax[ax_i].set_xticklabels([])

        ax[ax_i].set_ylim([0, 100])

        if ax_i == n_rows // 2:
            ax[ax_i].set_ylabel(f"Parallel efficiency (%)\n{p[ax_i-1]}", fontsize=14)
        else:
            ax[ax_i].set_ylabel(f"{p[ax_i-1]}", fontsize=14)

    handles, labels = [], []
    for np_i in range(len(np_nested)):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[np_i]))
        if np_i == 0:
            labels.append("No nested solving")
        else:
            labels.append(f"np = {np_nested[np_i]}")

    legend = fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.09), fontsize=12
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)

    plt.tight_layout()

    plt.savefig("nested_solving_flops.png")


def cost_breakdown_nested_solving(a, b, n, p):
    ...


if __name__ == "__main__":
    b = 1024
    a = b // 4
    n = 256
    p = [2, 4, 8, 16, 32]

    d_pobtaf = np.zeros(len(p))
    d_pobtasi = np.zeros(len(p))
    d_pobtarssi = np.zeros(len(p))
    total_flops = np.zeros(len(p))

    pct_d_pobtaf = np.zeros(len(p))
    pct_d_pobtasi = np.zeros(len(p))
    pct_d_pobtarssi = np.zeros(len(p))

    for p_i in range(len(p)):
        d_pobtaf[p_i] = compute_flops_d_pobtaf(n, b, a, p[p_i])
        d_pobtarssi[p_i] = compute_flops_reduced_system(b, a, p[p_i])
        d_pobtasi[p_i] = compute_flops_d_pobtasi(n, b, a, p[p_i])
        total_flops[p_i] = d_pobtaf[p_i] + d_pobtarssi[p_i] + d_pobtasi[p_i]

        pct_d_pobtaf[p_i] = (d_pobtaf[p_i] / total_flops[p_i]) * 100
        pct_d_pobtarssi[p_i] = (d_pobtarssi[p_i] / total_flops[p_i]) * 100
        pct_d_pobtasi[p_i] = (d_pobtasi[p_i] / total_flops[p_i]) * 100

    n_ns = np.zeros(len(p))
    p_ns = np.zeros(len(p))
    for i in range(len(p)):
        n_ns[i] = 2 * p[i] - 1  # number of diagonal blocks of the nested solve
        p_ns[i] = p[i] // 2  # number of processes for the nested solve

    d_pobtarssi_nested_solve = np.zeros_like(d_pobtarssi)
    total_flops_nested_solve = np.zeros_like(total_flops)

    pct_d_pobtaf_nested_solve = np.zeros_like(pct_d_pobtaf)
    pct_d_pobtasi_nested_solve = np.zeros_like(pct_d_pobtasi)
    pct_d_pobtarssi_nested_solve = np.zeros_like(pct_d_pobtarssi)

    for p_i in range(len(p_ns)):
        if p_ns[p_i] == 1:
            d_pobtarssi_nested_solve[p_i] = compute_flops_reduced_system(b, a, p[p_i])
        else:
            d_pobtarssi_nested_solve[p_i] += compute_flops_d_pobtaf(
                n_ns[p_i], b, a, p_ns[p_i]
            )

            d_pobtarssi_nested_solve[p_i] += compute_flops_reduced_system(
                b, a, p_ns[p_i]
            )

            d_pobtarssi_nested_solve[p_i] += compute_flops_d_pobtasi(
                n_ns[p_i], b, a, p_ns[p_i]
            )
        total_flops_nested_solve[p_i] = (
            d_pobtaf[p_i] + d_pobtarssi_nested_solve[p_i] + d_pobtasi[p_i]
        )

        pct_d_pobtaf_nested_solve[p_i] = (
            d_pobtaf[p_i] / total_flops_nested_solve[p_i]
        ) * 100

        pct_d_pobtarssi_nested_solve[p_i] = (
            d_pobtarssi_nested_solve[p_i] / total_flops_nested_solve[p_i]
        ) * 100

        pct_d_pobtasi_nested_solve[p_i] = (
            d_pobtasi[p_i] / total_flops_nested_solve[p_i]
        ) * 100

    print("total_flops: ", total_flops)
    print("pct_d_pobtaf: ", pct_d_pobtaf)
    print("pct_d_pobtarssi: ", pct_d_pobtarssi)
    print("pct_d_pobtasi: ", pct_d_pobtasi)
    print("total_flops_nested_solve: ", total_flops_nested_solve)
    print("pct_d_pobtaf_nested_solve: ", pct_d_pobtaf_nested_solve)
    print("pct_d_pobtarssi_nested_solve: ", pct_d_pobtarssi_nested_solve)
    print("pct_d_pobtasi_nested_solve: ", pct_d_pobtasi_nested_solve)

    # Stacked bar plot
    # width = 0.4
    # fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    # for p_i in range(len(p)):
    #     adjusted_width = width * p[p_i]

    #     ax[0].bar(
    #         p[p_i],
    #         d_pobtaf[p_i],
    #         color="b",
    #         width=adjusted_width,
    #         label="d_pobtaf" if p_i == 0 else "",
    #     )
    #     ax[0].bar(
    #         p[p_i],
    #         d_pobtarssi[p_i],
    #         color="r",
    #         width=adjusted_width,
    #         bottom=d_pobtaf[p_i],
    #         label="reduced system" if p_i == 0 else "",
    #     )
    #     ax[0].bar(
    #         p[p_i],
    #         d_pobtasi[p_i],
    #         color="g",
    #         width=adjusted_width,
    #         bottom=d_pobtaf[p_i] + d_pobtarssi[p_i],
    #         label="d_pobtasi" if p_i == 0 else "",
    #     )

    #     ax[0].text(
    #         p[p_i],
    #         d_pobtaf[p_i] / 2,
    #         f"{d_pobtaf[p_i]:.1e}\n{pct_d_pobtaf[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    #     ax[0].text(
    #         p[p_i],
    #         d_pobtaf[p_i] + d_pobtarssi[p_i] / 2,
    #         f"{d_pobtarssi[p_i]:.1e}\n{pct_d_pobtarssi[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    #     ax[0].text(
    #         p[p_i],
    #         d_pobtaf[p_i] + d_pobtarssi[p_i] + d_pobtasi[p_i] / 2,
    #         f"{d_pobtasi[p_i]:.1e}\n{pct_d_pobtasi[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    # ax[0].set_xscale("log", base=2)
    # ax[0].set_xticks(p)
    # ax[0].set_xticklabels([str(val) for val in p])
    # ax[0].set_xlabel("Number of processes")

    # ax[0].set_ylabel("Total flops per rank")
    # ax[0].legend(loc="upper right")

    # ax[0].set_title(f"n={n}")

    # for p_i in range(len(p)):
    #     adjusted_width = width * p[p_i]

    #     ax[1].bar(
    #         p[p_i],
    #         d_pobtaf[p_i],
    #         color="b",
    #         width=adjusted_width,
    #         label="d_pobtaf" if p_i == 0 else "",
    #     )
    #     ax[1].bar(
    #         p[p_i],
    #         d_pobtarssi_nested_solve[p_i],
    #         color="r",
    #         width=adjusted_width,
    #         bottom=d_pobtaf[p_i],
    #         label="reduced system" if p_i == 0 else "",
    #     )
    #     ax[1].bar(
    #         p[p_i],
    #         d_pobtasi[p_i],
    #         color="g",
    #         width=adjusted_width,
    #         bottom=d_pobtaf[p_i] + d_pobtarssi_nested_solve[p_i],
    #         label="d_pobtasi" if p_i == 0 else "",
    #     )

    #     ax[1].text(
    #         p[p_i],
    #         d_pobtaf[p_i] / 2,
    #         f"{d_pobtaf[p_i]:.1e}\n{pct_d_pobtaf_nested_solve[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    #     ax[1].text(
    #         p[p_i],
    #         d_pobtaf[p_i] + d_pobtarssi_nested_solve[p_i] / 2,
    #         f"{d_pobtarssi_nested_solve[p_i]:.1e}\n{pct_d_pobtarssi_nested_solve[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    #     ax[1].text(
    #         p[p_i],
    #         d_pobtaf[p_i] + d_pobtarssi_nested_solve[p_i] + d_pobtasi[p_i] / 2,
    #         f"{d_pobtasi[p_i]:.1e}\n{pct_d_pobtasi_nested_solve[p_i]:.2f}%",
    #         ha="center",
    #         va="center",
    #         color="black",
    #     )
    # ax[1].set_xscale("log", base=2)
    # ax[1].set_xticks(p)
    # ax[1].set_xticklabels([str(val) for val in p])
    # ax[1].set_xlabel("Number of processes")

    # ax[1].set_ylabel("Total flops per rank")
    # ax[1].legend(loc="upper right")

    # ax[1].set_title(f"n={n}")

    # plt.tight_layout()

    # plt.savefig("cost_breakdown_nested_solving.png")

    # plt.show()

    # Make a stack plot
    """ fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    stack_solving = np.vstack((d_pobtaf, d_pobtasi, d_pobtarssi))

    ax[0].stackplot(
        p, stack_solving, labels=["d_pobtaf", "d_pobtasi", "reduced system"]
    )

    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(p)
    ax[0].set_xticklabels([str(val) for val in p])
    ax[0].set_xlabel("Number of processes")

    stack_nested_solving = np.vstack(
        (
            d_pobtaf,
            d_pobtasi,
            d_pobtarssi_nested_solve,
        )
    )

    ax[1].stackplot(
        p, stack_nested_solving, labels=["d_pobtaf", "d_pobtasi", "reduced system"]
    )

    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(p)
    ax[1].set_xticklabels([str(val) for val in p])
    ax[1].set_xlabel("Number of processes")

    plt.show() """

    # Combine data for the first stack plot

    # Combine data for the second stack plot

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the first stack plot
    # stack_solving = np.vstack((d_pobtaf, d_pobtasi, d_pobtarssi))
    upper_limite_stacked_rssi = np.zeros_like(d_pobtarssi)
    lower_limite_stacked_rssi = np.zeros_like(d_pobtarssi)
    for p_i in range(len(p)):
        upper_limite_stacked_rssi[p_i] = (
            d_pobtarssi[p_i] + d_pobtasi[p_i] + d_pobtaf[p_i]
        )
        lower_limite_stacked_rssi[p_i] = (
            d_pobtaf[p_i] + d_pobtasi[p_i] + d_pobtarssi_nested_solve[p_i]
        )

    # fill in between the lower and upper limit with hatches
    ax.fill_between(
        p,
        lower_limite_stacked_rssi,
        upper_limite_stacked_rssi,
        facecolor="gray",  # Background color
        edgecolor="black",
        alpha=1,
        hatch="//",
    )

    ax.plot(p, upper_limite_stacked_rssi, color="gray", alpha=0.5)
    ax.plot(p, lower_limite_stacked_rssi, color="gray", alpha=0.5)

    stack_nested_solving = np.vstack((d_pobtaf, d_pobtasi, d_pobtarssi_nested_solve))

    colors = ["#156082", "#c04f15", "#3b7d23"]

    ax.stackplot(
        p,
        stack_nested_solving,
        labels=["d_pobtaf", "d_pobtasi", "reduced system"],
        colors=colors,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(p)
    ax.set_xticklabels(["" for val in p])
    ax.set_xlim([2, 32])

    # set y ticks
    ax.set_yticks([0, 1e12, 2e12, 3e12, 4e12])
    # ax.set_yticklabels(["0", "1e12", "2e12", "3e12", "4e12"])
    ax.set_yticklabels(["", "", "", "", ""])

    plt.tight_layout()

    plt.savefig("stacked_bar_plot_nested_solving.svg")

    plt.show()


if __name__ == "__main__":
    exit()
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

    if plot:
        plt.rcParams.update({"font.size": 12})
        fig, ax = plt.subplots(
            2, 2, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 5]}
        )

        ax[0, 0].set_title("Factorization", fontsize=18)
        plot_theoretical_parallel_efficiency(
            ax, [0, 0], [1, 0], n, b, a, p, total_flops_pobtaf, total_flops_d_pobtaf
        )

        ax[0, 1].set_title("Selected-inversion", fontsize=18)
        plot_theoretical_parallel_efficiency(
            ax, [0, 1], [1, 1], n, b, a, p, total_flops_pobtasi, total_flops_d_pobtasi
        )

        fig.tight_layout()

        plt.savefig("theoretical_parallel_efficiency.png")

        plt.show()

    if plot:
        # Show reduced system
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(total_flops_reduced_system, cmap="viridis", aspect="auto")

        for p_i in range(len(p)):
            for n_i in range(len(n)):
                if total_flops_reduced_system[p_i, n_i] > 0:
                    ax.text(
                        n_i,
                        p_i,
                        f"{total_flops_reduced_system[p_i, n_i]:.1e}\nflops",
                        ha="center",
                        va="center",
                        color="white",
                    )

        ax.set_xticks(range(len(n)))
        ax.set_xticklabels([str(val) for val in n], fontsize=14)
        ax.set_xlabel("Number of diagonal blocks", fontsize=14)

        ax.set_yticks(range(len(p)))
        ax.set_yticklabels([str(val) for val in p], fontsize=14)
        ax.set_ylabel("Number of processes", fontsize=14)

        ax.set_title("Inversion of the reduced system\n(POBTAF + POBTASI)", fontsize=14)

        fig.tight_layout()
        plt.savefig("reduced_system_inversion_flops.png")

        plt.show()

    # A2X
    total_flops_pobtaA2X = [
        total_flops_pobtaf[i] + total_flops_pobtasi[i] for i in range(len(n))
    ]
    total_flops_d_pobtaA2X = (
        total_flops_d_pobtaf + total_flops_d_pobtasi + total_flops_reduced_system
    )

    print("\ntotal_flops_d_pobtaA2X", end="")
    for p_i in range(len(p)):
        print("\n    ", end="")
        for n_i in range(len(n)):
            print(f"{total_flops_d_pobtaA2X[p_i, n_i]:.2e}, ", end="")

    if plot:
        plot_theoretical_parallel_efficiency_A2X(
            n, b, a, p, total_flops_pobtaA2X, total_flops_d_pobtaA2X
        )

    if True:
        # Pct taken by reduce system
        plot_cost_breakdown(
            n,
            b,
            a,
            p,
            total_flops_d_pobtaf,
            total_flops_d_pobtasi,
            total_flops_reduced_system,
        )

    # Nested solving

    np_nested = [2, 4, 8]
    redistribute_nested_partitions_ideally = True

    total_flops_d_pobtaA2X_nested = np.zeros((len(np_nested) + 1, len(p), len(n)))
    total_flops_d_pobtaA2X_nested[0] = total_flops_d_pobtaA2X

    # plot_theoretical_parallel_efficiency_A2X(
    #     n, b, a, p, total_flops_pobtaA2X, total_flops_d_pobtaA2X_nested[0]
    # )

    for np_i in range(len(np_nested)):
        for p_i in range(len(p)):
            n_reduced = 2 * p[p_i] - 1
            if n_reduced >= 3 * np_nested[np_i]:
                for n_i in range(len(n)):
                    if n[n_i] >= 3 * p[p_i]:
                        # Initialize the base flops
                        total_flops_d_pobtaA2X_nested[np_i + 1, p_i, n_i] = (
                            total_flops_d_pobtaf[p_i, n_i]
                            + total_flops_d_pobtasi[p_i, n_i]
                        )

                        if redistribute_nested_partitions_ideally:
                            load_balancing = compute_ideal_load_balancing(
                                n_reduced, b, a
                            )
                        else:
                            load_balancing = 1

                        partition_sizes = get_partition_size(
                            n=n_reduced,
                            p=np_nested[np_i],
                            balancing_ratio=load_balancing,
                        )

                        # The more full central partition is the '1'
                        central_partition_size = partition_sizes[1]

                        total_flops_d_pobtaA2X_nested[
                            np_i + 1, p_i, n_i
                        ] += compute_flops_d_pobtaf(
                            central_partition_size, b, a, p=1
                        ) + compute_flops_d_pobtasi(
                            central_partition_size, b, a, p=1
                        )

                        total_flops_d_pobtaA2X_nested[
                            np_i + 1, p_i, n_i
                        ] += compute_flops_reduced_system(b, a, np_nested[np_i])

                        print(
                            f"n={n[n_i]}, p={p[p_i]}, np_nested={np_nested[np_i]}: {total_flops_d_pobtaA2X_nested[np_i + 1, p_i, n_i]:.2e}"
                        )

        # plot_theoretical_parallel_efficiency_A2X(
        #     n, b, a, p, total_flops_pobtaA2X, total_flops_d_pobtaA2X_nested[np_i + 1]
        # )

    if plot:
        plot_nested_solving(
            np_nested, n, p, total_flops_pobtaA2X, total_flops_d_pobtaA2X_nested
        )

        plt.show()

    total_flops_reduced_system_nested_solve = np.zeros_like(total_flops_reduced_system)

    if False:
        # Pct taken by reduce system
        plot_cost_breakdown(
            n,
            b,
            a,
            p,
            total_flops_d_pobtaf,
            total_flops_d_pobtasi,
            total_flops_reduced_system,
        )

    """ plot_theoretical_speedup_A2X(
        n, b, a, p, total_flops_pobtaA2X, total_flops_d_pobtaA2X
    ) """

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
