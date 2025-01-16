# PRINT COMPARISON FOR B=2865, N=365, A=4

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # Other parameters
    n_processes = [1, 2, 4, 8, 16]
    fontsize = 16
    ticsize = fontsize - 2
    viridis = plt.get_cmap("viridis")

    # PARDISO timings
    pardiso_analysis_mean = 17.28936
    pardiso_analysis_ci = [17.13065, 17.44807]
    pardiso_factorization_mean = 31.56015
    pardiso_factorization_ci = [31.35484, 31.76545]
    pardiso_sellinv_mean = 113.58189
    pardiso_sellinv_ci = [103.89863, 123.26515]

    pardiso_a2x_mean = pardiso_factorization_mean + pardiso_sellinv_mean
    pardiso_a2x_ci = [
        pardiso_factorization_ci[0] + pardiso_sellinv_ci[0],
        pardiso_factorization_ci[1] + pardiso_sellinv_ci[1],
    ]

    # MUMPS timings
    mumps_analysis_mean = [15.623940, 17.576883, 10.135503, 7.669187, 6.991071]
    mumps_analysis_ci = [
        [15.619171, 15.643567],
        [17.576883, 17.604513],
        [10.120587, 10.175661],
        [7.668992, 7.714017],
        [6.991071, 7.002190],
    ]
    mumps_factorization_mean = [21.429027, 13.406576, 15.539646, 7.924564, 8.315133]
    mumps_factorization_ci = [
        [21.380414, 21.434369],
        [13.406576, 16.477463],
        [13.528027, 16.536837],
        [7.703853, 8.045787],
        [8.315133, 9.670932],
    ]
    mumps_sellinv_mean = [751.765751, 832.428086, 816.264905, 721.277741, 898.525558]
    mumps_sellinv_ci = [
        [751.765751, 752.416670],
        [832.428086, 832.428086],
        [816.264905, 850.252189],
        [721.277741, 759.372529],
        [898.525558, 898.525558],
    ]

    mumps_a2x_mean = [
        mumps_factorization_mean[i] + mumps_sellinv_mean[i]
        for i in range(len(mumps_factorization_mean))
    ]
    mumps_a2x_ci = [
        [
            mumps_factorization_ci[i][0] + mumps_sellinv_ci[i][0],
            mumps_factorization_ci[i][1] + mumps_sellinv_ci[i][1],
        ]
        for i in range(len(mumps_factorization_mean))
    ]

    # SERINV CPU timings
    serinv_cpu_factorization_mean = [97.91872, 72.91208, 40.75043, 27.39948, 24.25881]
    serinv_cpu_factorization_ci = [
        [97.80991, 98.02753],
        [71.25838, 74.56578],
        [38.40637, 43.09449],
        [26.52501, 28.27396],
        [23.31340, 25.20422],
    ]
    serinv_cpu_sellinv_mean = [125.55661, 87.16977, 55.04744, 36.65297, 30.84092]
    serinv_cpu_sellinv_ci = [
        [125.25415, 125.85907],
        [84.85489, 89.48465],
        [53.15529, 56.93960],
        [35.16313, 38.14281],
        [30.50634, 31.17550],
    ]

    serinv_cpu_a2x_mean = [
        serinv_cpu_factorization_mean[i] + serinv_cpu_sellinv_mean[i]
        for i in range(len(serinv_cpu_factorization_mean))
    ]
    serinv_cpu_a2x_ci = [
        [
            serinv_cpu_factorization_ci[i][0] + serinv_cpu_sellinv_ci[i][0],
            serinv_cpu_factorization_ci[i][1] + serinv_cpu_sellinv_ci[i][1],
        ]
        for i in range(len(serinv_cpu_factorization_mean))
    ]

    # SERINV GPU timings
    serinv_gpu_factorization_mean = [3.3776, 2.31842, 1.43204, 0.95592, 0.84885]
    serinv_gpu_factorization_ci = [
        [3.18884, 3.56636],
        [2.29015, 2.34669],
        [1.42803, 1.43605],
        [0.95446, 0.95738],
        [0.84684, 0.85086],
    ]

    serinv_gpu_sellinv_mean = [5.17384, 3.66254, 2.16145, 1.40028, 1.18536]
    serinv_gpu_sellinv_ci = [
        [5.1689, 5.17877],
        [3.54067, 3.78441],
        [2.15935, 2.16355],
        [1.39858, 1.40198],
        [1.18159, 1.18914],
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

    # Make a groupped bar for factorization
    fig, ax = plt.subplots()

    x_positions = np.arange(len(n_processes))

    bar_width = 0.2

    # PARDISO
    ax.axhline(
        y=pardiso_factorization_mean,
        color="r",
        linestyle="--",
        label="PARDISO",
    )
    # MUMPS
    yerr_mumps = np.array(
        [
            [
                mumps_factorization_mean[i] - mumps_factorization_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                mumps_factorization_ci[i][1] - mumps_factorization_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions,
        mumps_factorization_mean,
        yerr=yerr_mumps,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color=viridis(0.7),
        label="MUMPS",
        zorder=3,
    )

    # SERINV CPU
    yerr_serinv = np.array(
        [
            [
                serinv_cpu_factorization_mean[i] - serinv_cpu_factorization_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_cpu_factorization_ci[i][1] - serinv_cpu_factorization_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions + bar_width,
        serinv_cpu_factorization_mean,
        yerr=yerr_serinv,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color=viridis(0.1),
        label="Serinv-CPU",
        zorder=3,
    )

    # SERINV GPU
    yerr_serinv = np.array(
        [
            [
                serinv_gpu_factorization_mean[i] - serinv_gpu_factorization_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_gpu_factorization_ci[i][1] - serinv_gpu_factorization_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions + 2 * bar_width,
        serinv_gpu_factorization_mean,
        yerr=yerr_serinv,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color="#F0F0F0",
        zorder=2,
    )
    ax.bar(
        x_positions + 2 * bar_width,
        serinv_gpu_factorization_mean,
        capsize=3,
        width=bar_width,
        color=viridis(0.1),
        alpha=0.5,
        label="Serinv-GPU",
        zorder=3,
    )

    # ax.set_title("Factorization time", fontsize=fontsize)

    # Set custom x-ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(n_processes)
    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=ticsize)

    # set y axis in log scale
    ax.set_ylim(0.1, 1000)
    ax.set_yscale("log", base=10)
    ax.set_ylabel("Runtime (s)", fontsize=fontsize)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, zorder=1)

    ax.set_facecolor("#F0F0F0")

    ax.legend(fontsize=ticsize, loc="upper right")

    plt.tight_layout()

    fig.savefig("barchart_4002_factorization.png", dpi=450)

    # plt.show()

    #
    #
    #
    #
    #
    #

    # Make a groupped bar for sellinv
    fig, ax = plt.subplots()

    x_positions = np.arange(len(n_processes))

    bar_width = 0.2

    # PARDISO
    ax.axhline(
        y=pardiso_sellinv_mean,
        color="r",
        linestyle="--",
        label="PARDISO",
    )
    # MUMPS
    yerr_mumps = np.array(
        [
            [
                mumps_sellinv_mean[i] - mumps_sellinv_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                mumps_sellinv_ci[i][1] - mumps_sellinv_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions,
        mumps_sellinv_mean,
        yerr=yerr_mumps,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color=viridis(0.7),
        label="MUMPS",
        zorder=3,
    )

    # SERINV CPU
    yerr_serinv = np.array(
        [
            [
                serinv_cpu_sellinv_mean[i] - serinv_cpu_sellinv_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_cpu_sellinv_ci[i][1] - serinv_cpu_sellinv_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions + bar_width,
        serinv_cpu_sellinv_mean,
        yerr=yerr_serinv,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color=viridis(0.1),
        label="Serinv-CPU",
        zorder=3,
    )

    # SERINV GPU
    yerr_serinv = np.array(
        [
            [
                serinv_gpu_sellinv_mean[i] - serinv_gpu_sellinv_ci[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_gpu_sellinv_ci[i][1] - serinv_gpu_sellinv_mean[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.bar(
        x_positions + 2 * bar_width,
        serinv_gpu_sellinv_mean,
        width=bar_width,
        color="#F0F0F0",
        zorder=2,
    )
    ax.bar(
        x_positions + 2 * bar_width,
        serinv_gpu_sellinv_mean,
        yerr=yerr_serinv,
        capsize=3,
        error_kw=dict(ecolor="black", capsize=5, capthick=1, alpha=0.75),
        width=bar_width,
        color=viridis(0.1),
        alpha=0.5,
        label="Serinv-GPU",
        zorder=3,
    )

    # ax.set_title("sellinv time", fontsize=fontsize)

    # Set custom x-ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(n_processes)
    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)

    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, zorder=1)

    ax.tick_params(axis="both", which="major", labelsize=ticsize)

    # set y axis in log scale
    ax.set_ylim(0.1, 1000)
    ax.set_yscale("log", base=10)
    # ax.set_ylabel("Time (s)", fontsize=fontsize)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    ax.set_facecolor("#F0F0F0")

    # ax.legend(fontsize=ticsize, loc="lower left")

    plt.tight_layout()

    fig.savefig("barchart_4002_sellinv.png", dpi=450)

    # plt.show()

    #
    #
    #
    #
    #
    #

    # Make a stacked bar for A2X
    fig, ax = plt.subplots()

    x_positions = np.arange(len(n_processes))

    bar_width = 0.2
    bar_adjustment = 0.02

    # hatches_factorization = "oo"
    # hatches_sellinv = "//"

    # PARDISO
    ax.axhline(y=pardiso_a2x_mean, color="r", linestyle="--", label="PARDISO")

    # Fake bars for legend
    # ax.bar(
    #     1,
    #     0,
    #     color="white",
    #     label="Selected inversion",
    #     # hatch=hatches_sellinv,
    # )
    # ax.bar(
    #     1,
    #     0,
    #     color="white",
    #     label="Factorization",
    #     # hatch=hatches_factorization,
    # )

    # MUMPS
    ax.bar(
        x_positions,
        mumps_factorization_mean,
        width=bar_width,
        color=viridis(0.7),
        zorder=3,
        # hatch=hatches_factorization,
    )
    ax.bar(
        x_positions,
        mumps_sellinv_mean,
        width=bar_width,
        bottom=mumps_factorization_mean,
        color=viridis(0.7),
        zorder=3,
        # hatch=hatches_sellinv,
        label="MUMPS",
    )

    # Serinv-CPU
    ax.bar(
        x_positions + bar_width + bar_adjustment,
        serinv_cpu_factorization_mean,
        width=bar_width,
        color=viridis(0.1),
        zorder=3,
        # hatch=hatches_factorization,
    )
    ax.bar(
        x_positions + bar_width + bar_adjustment,
        serinv_cpu_sellinv_mean,
        width=bar_width,
        bottom=serinv_cpu_factorization_mean,
        color=viridis(0.1),
        zorder=3,
        # hatch=hatches_sellinv,
        label="Serinv-CPU",
    )

    # Serinv-GPU
    ax.bar(
        x_positions + 2 * (bar_width + bar_adjustment),
        serinv_gpu_factorization_mean,
        width=bar_width,
        color="#F0F0F0",
        zorder=2,
    )
    ax.bar(
        x_positions + 2 * (bar_width + bar_adjustment),
        serinv_gpu_sellinv_mean,
        width=bar_width,
        bottom=serinv_gpu_factorization_mean,
        color="#F0F0F0",
        zorder=2,
    )
    ax.bar(
        x_positions + 2 * (bar_width + bar_adjustment),
        serinv_gpu_factorization_mean,
        width=bar_width,
        color=viridis(0.1),
        alpha=0.5,
        zorder=3,
        # hatch=hatches_factorization,
    )
    ax.bar(
        x_positions + 2 * (bar_width + bar_adjustment),
        serinv_gpu_sellinv_mean,
        width=bar_width,
        bottom=serinv_gpu_factorization_mean,
        color=viridis(0.1),
        alpha=0.5,
        zorder=3,
        # hatch=hatches_sellinv,
        label="Serinv-GPU",
    )

    # ax.set_title("A2X time", fontsize=fontsize)

    # Set custom x-ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(n_processes)
    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)

    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, zorder=1)

    ax.tick_params(axis="both", which="major", labelsize=ticsize)

    # set y axis in log scale
    ax.set_ylim(0.1, 1000)
    ax.set_yscale("log", base=10)
    # ax.set_ylabel("Time (s)", fontsize=fontsize)
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    ax.set_facecolor("#F0F0F0")

    # put legend on bottom left and with an alpha of 1
    # ax.legend(loc="lower left", fontsize=fontsize, framealpha=0.95)

    # ax.legend(fontsize=ticsize, loc="lower left")

    plt.tight_layout()

    fig.savefig("barchart_4002_a2x.png", dpi=450)

    plt.show()  #
