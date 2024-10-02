import os

import matplotlib.pyplot as plt
import numpy as np


def get_timings(n, a, b, n_processes, nested_solving):
    # Distributed POBTAF
    d_pobtaf_timings = []
    print("Distributed POBTAF")
    for p in n_processes:
        file_name = f"timings_d_pobtaf_bs{b}_as{a}_nb{n}_np{p}"
        if nested_solving:
            file_name += "_nested_solving"
        file_name += ".npy"
        d_pobtaf_timings.append(
            np.load(os.path.join(path, file_name), allow_pickle=True)
        )

    d_pobtaf_strong_scaling_mean = np.zeros(len(n_processes))
    d_pobtaf_strong_scaling_ci = np.zeros((2, len(n_processes)))

    for i, timings in enumerate(d_pobtaf_timings):
        d_pobtaf_strong_scaling_mean[i] = np.mean(timings)
        d_pobtaf_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])

        print(
            f"  P: {n_processes[i]}, mean: {d_pobtaf_strong_scaling_mean[i]}, CI: {d_pobtaf_strong_scaling_ci[:, i]}"
        )

    # Distributed POBTASI
    d_pobtasi_timings = []
    print("Distributed POBTASI")
    for p in n_processes:
        file_name = f"timings_d_pobtasi_bs{b}_as{a}_nb{n}_np{p}"
        if nested_solving:
            file_name += "_nested_solving"
        file_name += ".npy"
        d_pobtasi_timings.append(
            np.load(os.path.join(path, file_name), allow_pickle=True)
        )

    d_pobtasi_strong_scaling_mean = np.zeros(len(n_processes))
    d_pobtasi_strong_scaling_ci = np.zeros((2, len(n_processes)))

    for i, timings in enumerate(d_pobtasi_timings):
        d_pobtasi_strong_scaling_mean[i] = np.mean(timings)
        d_pobtasi_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])

        print(
            f"  P: {n_processes[i]}, mean: {d_pobtasi_strong_scaling_mean[i]}, CI: {d_pobtasi_strong_scaling_ci[:, i]}"
        )

    # Distributed POBTARSSI
    d_pobtarssi_timings = []
    print("Distributed POBTARSSI")
    for p in n_processes:
        file_name = f"timings_d_pobtasi_rss_bs{b}_as{a}_nb{n}_np{p}"
        if nested_solving:
            file_name += "_nested_solving"
        file_name += ".npy"
        d_pobtarssi_timings.append(
            np.load(os.path.join(path, file_name), allow_pickle=True)
        )

    d_pobtarssi_strong_scaling_mean = np.zeros(len(n_processes))
    d_pobtarssi_strong_scaling_ci = np.zeros((2, len(n_processes)))

    for i, timings in enumerate(d_pobtarssi_timings):
        d_pobtarssi_strong_scaling_mean[i] = np.mean(timings)
        d_pobtarssi_strong_scaling_ci[:, i] = np.percentile(timings, [5, 95])

        print(
            f"  P: {n_processes[i]}, mean: {d_pobtarssi_strong_scaling_mean[i]}, CI: {d_pobtarssi_strong_scaling_ci[:, i]}"
        )

    return (
        d_pobtaf_strong_scaling_mean,
        d_pobtasi_strong_scaling_mean,
        d_pobtarssi_strong_scaling_mean,
    )


if __name__ == "__main__":
    n = 365
    b = 2865
    a = 4
    n_processes = [2, 4, 8, 16, 32]

    path = "/home/vmaillou/Documents/SDR/bench/IDPS/timings_bs2865_as4_nb365/"

    d_pobtaf, d_pobtasi, d_pobtarssi = get_timings(
        n, a, b, n_processes, nested_solving=False
    )
    (
        d_pobtaf_nested_solve,
        d_pobtasi_nested_solve,
        d_pobtarssi_nested_solve,
    ) = get_timings(n, a, b, n_processes, nested_solving=True)

    # Stacked plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the first stack plot
    # stack_solving = np.vstack((d_pobtaf, d_pobtasi, d_pobtarssi))
    upper_limite_stacked_rssi = np.zeros_like(d_pobtarssi)
    lower_limite_stacked_rssi = np.zeros_like(d_pobtarssi)
    for p_i in range(len(n_processes)):
        upper_limite_stacked_rssi[p_i] = (
            d_pobtarssi[p_i] + d_pobtasi[p_i] + d_pobtaf[p_i]
        )
        lower_limite_stacked_rssi[p_i] = (
            d_pobtaf[p_i] + d_pobtasi[p_i] + d_pobtarssi_nested_solve[p_i]
        )

    # fill in between the lower and upper limit with hatches
    ax.fill_between(
        n_processes,
        lower_limite_stacked_rssi,
        upper_limite_stacked_rssi,
        facecolor="gray",  # Background color
        edgecolor="black",
        alpha=1,
        hatch="//",
    )

    ax.plot(n_processes, upper_limite_stacked_rssi, color="gray", alpha=0.5)
    ax.plot(n_processes, lower_limite_stacked_rssi, color="gray", alpha=0.5)

    stack_nested_solving = np.vstack((d_pobtaf, d_pobtasi, d_pobtarssi_nested_solve))

    colors = ["#156082", "#c04f15", "#3b7d23"]

    ax.stackplot(
        n_processes,
        stack_nested_solving,
        labels=["d_pobtaf", "d_pobtasi", "reduced system"],
        colors=colors,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_processes)
    ax.set_xticklabels(["" for val in n_processes])
    ax.set_xlim([2, 32])

    # set y ticks
    # ax.set_yticks([0, 1e12, 2e12, 3e12, 4e12])
    # ax.set_yticklabels(["0", "1e12", "2e12", "3e12", "4e12"])
    # ax.set_yticklabels(["", "", "", "", ""])

    plt.tight_layout()

    plt.savefig("experimental_stacked_bar_plot_nested_solving.svg")

    plt.show()
