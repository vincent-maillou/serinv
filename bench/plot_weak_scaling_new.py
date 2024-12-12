# from matplotlib import cm
# import seaborn as sns
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_speedup(
    title,
    n_processes,
    ideal_speedup,
    d_x_speedup,
    d_x_strong_scaling_ci,
    x_ref_mean,
    x_ref_ci,
    ax,
):
    speedup_ci_lower = d_x_speedup - (x_ref_mean / d_x_strong_scaling_ci[1, :])
    speedup_ci_upper = (x_ref_mean / d_x_strong_scaling_ci[0, :]) - d_x_speedup
    x_ref_ci_lower = 1 - x_ref_mean / x_ref_ci[1]
    x_ref_ci_upper = x_ref_mean / x_ref_ci[0] - 1

    d_x_speedup *= 100
    speedup_ci_lower *= 100
    speedup_ci_upper *= 100
    x_ref_ci_lower *= 100
    x_ref_ci_upper *= 100

    ax.set_facecolor("#F0F0F0")

    n_processes = [1, *n_processes]
    ideal_speedup = np.array([100 for _ in n_processes])
    d_x_speedup = [100, *d_x_speedup.tolist()]
    speedup_ci_lower = [x_ref_ci_lower, *speedup_ci_lower.tolist()] 
    speedup_ci_upper = [x_ref_ci_upper, *speedup_ci_upper.tolist()]

    # Plot ideal speedup line
    ax.plot(
        n_processes, ideal_speedup, label="Ideal", color="black", linestyle="--"
    )

    # Plot measured speedup with error bars using viridis color scheme
    viridis = plt.get_cmap("viridis")
    ax.errorbar(
        n_processes,
        d_x_speedup,
        yerr=[speedup_ci_lower, speedup_ci_upper],
        fmt="x",
        ls="-",
        color=viridis(0.7),
        label="Measured",
        capsize=10,
        markersize=10,  # Increase the size of the markers
    )

    # Set x and y scales to log base 2
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log", base=2)

    # Set x and y limits
    ax.set_xlim(0.9, 40)
    ax.set_ylim(0, 105)

    # Set x and y axis labels and title
    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=36)
    ax.set_ylabel(f"{title} efficiency", fontsize=36)
    # ax.set_title(title, fontsize=16)

    # Set custom x-axis ticks and labels
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(n_processes)
    ax.set_xticklabels(n_processes)

    # make the x and y tick bigger
    ax.tick_params(axis="both", which="major", labelsize=36)

    # set y_ticks
    # ax.set_yticks([1, 2, 4, 8, 16, 32])

    # Add legend
    ax.legend(loc="upper left")

    # make the legend bigger
    ax.legend(fontsize=32)

    # Add horizontal grid lines for y-axis
    ax.grid(which="both", axis="y", linestyle="--", linewidth=0.5)


def plot_timings(
    title,
    n_processes,
    ideal_timing,
    x_ref_mean,
    x_ref_ci,
    d_x_timing_mean,
    d_x_timing_ci,
    d_x_ns_timing_mean,
    d_x_ns_timing_ci,
    ax,
):
    ref_ci_lower = x_ref_mean - x_ref_ci[0]
    ref_ci_upper = x_ref_ci[1] - x_ref_mean
    timing_ci_lower = d_x_timing_mean - d_x_timing_ci[0, :]
    timing_ci_upper = d_x_timing_ci[1, :] - d_x_timing_mean
    timing_ns_ci_lower = d_x_ns_timing_mean - d_x_ns_timing_ci[0, :]
    timing_ns_ci_upper = d_x_ns_timing_ci[1, :] - d_x_ns_timing_mean

    n_processes = [1, *n_processes]
    ideal_timing = np.array([x_ref_mean for _ in n_processes])
    d_x_timing_mean = [x_ref_mean, *d_x_timing_mean.tolist()]
    timing_ci_lower = [ref_ci_lower, *timing_ci_lower.tolist()]
    timing_ci_upper = [ref_ci_upper, *timing_ci_upper.tolist()]
    d_x_ns_timing_mean = [x_ref_mean, *d_x_ns_timing_mean.tolist()]
    timing_ns_ci_lower = [ref_ci_lower, *timing_ns_ci_lower.tolist()]
    timing_ns_ci_upper = [ref_ci_upper, *timing_ns_ci_upper.tolist()]

    ax.set_facecolor("#F0F0F0")

    # Plot ideal timing line
    ax.plot(
        n_processes,
        ideal_timing,
        label="Ideal",
        color="black",
        linestyle="--",
    )

    # Plot measured timing with error bars using viridis color scheme
    viridis = plt.get_cmap("magma")
    ax.errorbar(
        n_processes,
        d_x_timing_mean,
        yerr=[timing_ci_lower, timing_ci_upper],
        fmt="x",
        ls="-",
        color=viridis(0.7),
        label="Measured",
        capsize=10,
        markersize=10,  # Increase the size of the markers
    )
    ax.errorbar(
        n_processes,
        d_x_ns_timing_mean,
        yerr=[timing_ns_ci_lower, timing_ns_ci_upper],
        fmt="o",
        ls="-",
        color=viridis(0.4),
        label="Measured (nested)",
        capsize=10,
        markersize=10,  # Increase the size of the markers
    )

    # Set x and y axis labels and title
    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=36)
    ax.set_ylabel("Runtime [s]", fontsize=36)
    # ax.set_title(title, fontsize=16)

    ax.set_ylim(0, None)

    # Set custom x-axis ticks and labels
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_processes)
    ax.set_xticklabels(n_processes)

    # make the x and y tick bigger
    ax.tick_params(axis="both", which="major", labelsize=36)

    # Add legend
    ax.legend(loc="upper right")

    # make the legend bigger
    ax.legend(fontsize=27)

    # Add horizontal grid lines for y-axis
    ax.grid(which="both", axis="y", linestyle="--", linewidth=0.5)

    return fig


""" def plot_speedup(
    title, n_processes, ideal_speedup, d_x_speedup, d_x_strong_scaling_ci, x_ref_mean
):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot ideal speedup line
    plt.plot(
        n_processes, ideal_speedup, label="Ideal speedup", color="black", linestyle="--"
    )

    speedup_ci_lower = d_x_speedup - (x_ref_mean / d_x_strong_scaling_ci[1, :])
    speedup_ci_upper = (x_ref_mean / d_x_strong_scaling_ci[0, :]) - d_x_speedup

    # Plot measured speedup with error bars using viridis color scheme
    viridis = plt.get_cmap("viridis")
    plt.errorbar(
        n_processes,
        d_x_speedup,
        yerr=[speedup_ci_lower, speedup_ci_upper],
        fmt="x",
        color=viridis(0.7),
        label="Measured speedup",
        capsize=5,
    )

    # Set x and y scales to log base 2
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)

    # Set x and y limits
    plt.xlim(2, 64)
    plt.ylim(1, 64)

    # Set x and y axis labels and title
    plt.xlabel("Number of processes", fontsize=14)
    plt.ylabel("Speedup", fontsize=14)
    plt.title(title, fontsize=16)

    # Set custom x-axis ticks and labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(n_processes)
    ax.set_xticklabels(n_processes)

    # Add legend
    plt.legend()

    # Add horizontal grid lines for y-axis
    plt.grid(which="both", axis="y", linestyle="--", linewidth=0.5)

    # Add legend in the upper left corner
    plt.legend(loc="upper left")

    # Show the plot
    plt.show()


def plot_timings(
    title,
    n_processes,
    ideal_timing,
    d_x_timing_mean,
    d_x_timing_ci,
):

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot ideal timing line
    plt.plot(
        n_processes, ideal_timing, label="Ideal timing", color="black", linestyle="--"
    )

    timing_ci_lower = d_x_timing_mean - d_x_timing_ci[0, :]
    timing_ci_upper = d_x_timing_ci[1, :] - d_x_timing_mean

    # Plot measured timing with error bars using viridis color scheme
    viridis = plt.get_cmap("viridis")
    plt.errorbar(
        n_processes,
        d_a2x_strong_scaling_mean,
        yerr=[timing_ci_lower, timing_ci_upper],
        fmt="x",
        color=viridis(0.7),
        label="Measured timing",
        capsize=5,
    )

    # Set x and y axis labels and title
    plt.xlabel("Number of processes", fontsize=14)
    plt.ylabel("Time to solution (s)", fontsize=14)
    plt.title(title, fontsize=16)

    # Set custom x-axis ticks and labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(n_processes)
    ax.set_xticklabels(n_processes)

    # Add legend
    plt.legend()

    # Add horizontal grid lines for y-axis
    plt.grid(which="both", axis="y", linestyle="--", linewidth=0.5)

    # Add legend in the upper left corner
    plt.legend(loc="upper right")

    # Show the plot
    plt.show() """


if __name__ == "__main__":
    n = 90
    b = 2865
    a = 4
    nested_solving = False
    n_processes = [2, 4, 8, 16, 32]
    # n_processes = [2, 4, 8, 16]
    ideal_speedup = np.array([1 for _ in n_processes])

    # home = os.environ["HOME"]
    # path = os.path.join(home, "serinv", "bench", "IDPS", "timings_bs2865_as4_nb365")

    # path = "/home/vmaillou/Documents/SDR/bench/IDPS/timings_bs2865_as4_nb365/"
    path = ""

    # Factorization
    print("Factorization")
    # 1. sequential timings
    # file_name = f"save_timings_inlamat_pobtaf_b{b}_a{a}_n{n}.npy"
    file_name = f"timings_synthetic_pobtaf_b{b}_a{a}_n{n}.npy"
    pobtaf_timings = np.load(os.path.join(path, file_name), allow_pickle=True)

    pobtaf_ref_mean = np.mean(pobtaf_timings)
    pobtaf_ref_ci = np.percentile(pobtaf_timings, [5, 95])

    print(f"  sequential, mean: {pobtaf_ref_mean}, CI: {pobtaf_ref_ci}")

    # 2. parallel timings
    d_pobtaf_timings = []
    for p in n_processes:
        file_name = f"timings_d_pobtaf_bs{b}_as{a}_nb{n*p}_np{p}"
        if nested_solving:
            file_name += "_nested_solving"
        file_name += ".npy"
        # file_name = f"timings_d_pobtaf_bs{b}_as{a}_nb{n}_np{p}.npy"
        d_pobtaf_timings.append(
            np.load(os.path.join(path, file_name), allow_pickle=True)
        )

    d_pobtaf_strong_scaling_mean = np.zeros(len(n_processes))
    d_pobtaf_strong_scaling_ci = np.zeros((2, len(n_processes)))

    d_pobtaf_speedup = np.zeros(len(n_processes))

    for i, timings in enumerate(d_pobtaf_timings):
        d_pobtaf_strong_scaling_mean[i] = np.mean(timings)
        d_pobtaf_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])
        d_pobtaf_speedup[i] = pobtaf_ref_mean / d_pobtaf_strong_scaling_mean[i]

        print(
            f"  P: {n_processes[i]}, mean: {d_pobtaf_strong_scaling_mean[i]}, CI: {d_pobtaf_strong_scaling_ci[:, i]}, Speedup: {d_pobtaf_speedup[i]}"
        )

    # Selected inversion
    print("Selected inversion")
    # 1. sequential timings
    # file_name = f"save_timings_inlamat_pobtasi_b{b}_a{a}_n{n}.npy"
    file_name = f"timings_synthetic_pobtasi_b{b}_a{a}_n{n}.npy"
    pobtasi_timings = np.load(os.path.join(path, file_name), allow_pickle=True)

    pobtasi_ref_mean = np.mean(pobtasi_timings)
    pobtasi_ref_ci = np.percentile(pobtasi_timings, [5, 95])

    print(f"    sequential, mean: {pobtasi_ref_mean}, CI: {pobtasi_ref_ci}")

    d_pobtasi_rss_timings = []
    d_pobtasi_rss_ns_timings = []
    d_pobtasi_timings = []
    for p in n_processes:
        file_name = f"timings_d_pobtasi_rss_bs{b}_as{a}_nb{n*p}_np{p}.npy"
        tmp = np.load(os.path.join(path, file_name), allow_pickle=True)
        d_pobtasi_rss_timings.append(tmp)
        file_name = f"timings_d_pobtasi_rss_bs{b}_as{a}_nb{n*p}_np{p}_nested_solving.npy"
        tmp = np.load(os.path.join(path, file_name), allow_pickle=True)
        d_pobtasi_rss_ns_timings.append(tmp)
        file_name = f"timings_d_pobtasi_bs{b}_as{a}_nb{n*p}_np{p}"
        if nested_solving:
            file_name += "_nested_solving"
        file_name += ".npy"
        # file_name = f"timings_d_pobtasi_bs{b}_as{a}_nb{n}_np{p}.npy"
        tmp2 = np.load(os.path.join(path, file_name), allow_pickle=True)
        # d_pobtasi_timings.append(tmp + tmp2)
        d_pobtasi_timings.append(tmp2)

    d_pobtasi_strong_scaling_mean = np.zeros(len(n_processes))
    d_pobtasi_strong_scaling_ci = np.zeros((2, len(n_processes)))

    d_pobtasi_speedup = np.zeros(len(n_processes))

    for i, timings in enumerate(d_pobtasi_timings):
        d_pobtasi_strong_scaling_mean[i] = np.mean(timings)
        d_pobtasi_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])
        d_pobtasi_speedup[i] = pobtasi_ref_mean / d_pobtasi_strong_scaling_mean[i]

        print(
            f"    P: {n_processes[i]}, mean: {d_pobtasi_strong_scaling_mean[i]}, CI: {d_pobtasi_strong_scaling_ci[:, i]}, Speedup: {d_pobtasi_speedup[i]}"
        )

    # A2X
    print("A2X")

    a2x_timings = [
        pobtaf_timings[i] + pobtasi_timings[i] for i in range(len(pobtaf_timings))
    ]

    a2x_ref_mean = np.mean(a2x_timings)
    a2x_ref_ci = np.percentile(a2x_timings, [5, 95])

    print(f"    sequential, mean: {a2x_ref_mean}, CI: {a2x_ref_ci}")

    d_a2x_timings = []
    for i in range(len(n_processes)):
        d_a2x_timings.append(
            [
                d_pobtaf_timings[i][j] + d_pobtasi_timings[i][j] + d_pobtasi_rss_timings[i][j]
                for j in range(len(d_pobtaf_timings[i]))
            ]
        )
    d_a2x_ns_timings = []
    for i in range(len(n_processes)):
        d_a2x_ns_timings.append(
            [
                d_pobtaf_timings[i][j] + d_pobtasi_timings[i][j] + d_pobtasi_rss_ns_timings[i][j]
                for j in range(len(d_pobtaf_timings[i]))
            ]
        )

    d_a2x_strong_scaling_mean = np.zeros(len(n_processes))
    d_a2x_strong_scaling_ci = np.zeros((2, len(n_processes)))
    d_a2x_ns_strong_scaling_mean = np.zeros(len(n_processes))
    d_a2x_ns_strong_scaling_ci = np.zeros((2, len(n_processes)))

    for i, timings in enumerate(d_a2x_timings):
        d_a2x_strong_scaling_mean[i] = np.mean(timings)
        d_a2x_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])

        print(
            f"    P: {n_processes[i]}, mean: {d_a2x_strong_scaling_mean[i]}, CI: {d_a2x_strong_scaling_ci[:, i]}"
        )
    
    for i, timings in enumerate(d_a2x_ns_timings):
        d_a2x_ns_strong_scaling_mean[i] = np.mean(timings)
        d_a2x_ns_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])

        print(
            f"    P: {n_processes[i]}, mean: {d_a2x_ns_strong_scaling_mean[i]}, CI: {d_a2x_ns_strong_scaling_ci[:, i]}"
        )

    ideal_timing = np.array([a2x_ref_mean for _ in n_processes])

    # fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # Generate the speedup plot and add it to the first subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    plot_speedup(
        "PPOBTAF",
        n_processes,
        ideal_speedup,
        d_pobtaf_speedup,
        d_pobtaf_strong_scaling_ci,
        pobtaf_ref_mean,
        pobtaf_ref_ci,
        ax,
    )

    plt.tight_layout()

    file_name = "pobtaf_weak_scaling.pdf"
    # if nested_solving:
    #     file_name += "_nested_solving"
    # file_name += ".png"
    plt.savefig(file_name, dpi=400)

    # Generate the speedup plot and add it to the first subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    plot_speedup(
        "PPOBTASI",
        n_processes,
        ideal_speedup,
        d_pobtasi_speedup,
        d_pobtasi_strong_scaling_ci,
        pobtasi_ref_mean,
        pobtasi_ref_ci,
        ax,
    )

    plt.tight_layout()

    file_name = "pobtasi_weak_scaling.pdf"
    # if nested_solving:
    #     file_name += "_nested_solving"
    # file_name += ".png"
    plt.savefig(file_name, dpi=400)

    # Generate the timings plot and add it to the second subplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    fig_timings = plot_timings(
        "Runtime, weak scaling",
        n_processes,
        ideal_timing,
        a2x_ref_mean,
        a2x_ref_ci,
        d_a2x_strong_scaling_mean,
        d_a2x_strong_scaling_ci,
        d_a2x_ns_strong_scaling_mean,
        d_a2x_ns_strong_scaling_ci,
        ax,
    )

    plt.tight_layout()

    file_name = "a2x_weak_scaling.pdf"
    # if nested_solving:
    #     file_name += "_nested_solving"
    # file_name += ".png"
    plt.savefig(file_name, dpi=400)

    plt.show()
