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
    serinv_gpu_factorization_mean = [2.4856, 1.6249, 0.97540, 0.61214, 0.50339, 0.64621]
    serinv_gpu_factorization_ci = [
        [2.41606, 2.55513],
        [1.62393, 1.62587],
        [0.97330, 0.97750],
        [0.62426, 0.62871],
        [0.50311, 0.50367],
        [0.64597, 0.64645],
    ]

    serinv_gpu_sellinv_mean = [2.78955, 1.81765, 1.09637, 0.67208, 0.51710, 0.62597]
    serinv_gpu_sellinv_ci = [
        [2.7893, 2.78979],
        [1.81745, 1.81784],
        [1.09624, 1.09651],
        [0.67204, 0.67212],
        [0.51698, 0.51722],
        [0.62586, 0.62609],
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
    ax.set_ylim(0, 6)

    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels([0, 2, 4])

    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)
    ax.set_ylabel("Time to solution [s]", fontsize=fontsize)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    ax.legend(
        fontsize=ticsize,
        loc="upper right",
    )

    ax.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.savefig("strong_Scaling_2865.png", dpi=450)

    plt.show()
