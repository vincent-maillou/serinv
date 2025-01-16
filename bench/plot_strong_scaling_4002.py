# PRINT COMPARISON FOR B=2865, N=365, A=4

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # Other parameters
    n_processes = [1, 2, 4, 8, 16, 32]
    fontsize = 18
    ticsize = fontsize - 2
    viridis = plt.get_cmap("viridis")
    linewidth = 1

    # SERINV GPU timings
    serinv_gpu_factorization_mean = [
        3.3776,
        2.31842,
        1.43204,
        0.95592,
        0.84885,
        1.20169,
    ]
    serinv_gpu_factorization_ci = [
        [3.18884, 3.56636],
        [2.29015, 2.34669],
        [1.42803, 1.43605],
        [0.95446, 0.95738],
        [0.84684, 0.85086],
        [1.20068, 1.20271],
    ]

    serinv_gpu_sellinv_mean = [5.17384, 3.66254, 2.16145, 1.40028, 1.18536, 1.57982]
    serinv_gpu_sellinv_ci = [
        [5.1689, 5.17877],
        [3.54067, 3.78441],
        [2.15935, 2.16355],
        [1.39858, 1.40198],
        [1.18159, 1.18914],
        [1.57697, 1.58267],
    ]

    serinv_gpu_a2x_mean = [
        serinv_gpu_factorization_mean[i] + serinv_gpu_sellinv_mean[i]
        for i in range(len(serinv_gpu_factorization_mean))
    ]
    serinv_gpu_a2x_ci = [
        [
            serinv_gpu_factorization_ci[i][0] + serinv_gpu_sellinv_ci[i][0],
            serinv_gpu_factorization_ci[i][1] + serinv_gpu_sellinv_ci[i][1],
        ]
        for i in range(len(serinv_gpu_factorization_mean))
    ]

    ideal_scaling = [serinv_gpu_a2x_mean[0] / n for n in n_processes]

    print("ideal_scaling", ideal_scaling)

    # PLOT
    yerr_gpu = np.array(
        [
            [
                serinv_gpu_a2x_mean[i] - serinv_gpu_a2x_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_gpu_a2x_ci[i][1] - serinv_gpu_a2x_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        n_processes,
        serinv_gpu_a2x_mean,
        yerr=yerr_gpu,
        capsize=3,
        capthick=1,
        fmt="x-",
        linewidth=linewidth,
        color=viridis(0.1),
        label="Measured",
    )
    ax.plot(
        n_processes,
        ideal_scaling,
        "--",
        color="black",
        label="Ideal",
        linewidth=linewidth,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_processes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 10)

    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10])

    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)
    # ax.set_ylabel("Time to solution [s]", fontsize=fontsize)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())

    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_ylabel("Time to solution [s]", fontsize=fontsize)

    ax.legend(fontsize=ticsize, loc="upper right")

    ax.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.savefig("strong_Scaling_4002.png", dpi=450)

    plt.show()
