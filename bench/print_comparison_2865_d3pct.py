# PRINT COMPARISON FOR B=2865, N=365, A=4

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # Other parameters
    n_processes = [1, 2, 4, 8, 16]
    fontsize = 12
    viridis = plt.get_cmap("viridis")

    # PARDISO timings
    pardiso_analysis_mean = 112.44209
    pardiso_analysis_ci = [111.65188, 113.23230]
    pardiso_factorization_mean = 67.93375
    pardiso_factorization_ci = [67.24529, 68.62220]
    pardiso_sellinv_mean = 258.69864
    pardiso_sellinv_ci = [221.60335, 295.79393]

    pardiso_a2x_mean = pardiso_factorization_mean + pardiso_sellinv_mean
    pardiso_a2x_ci = [
        pardiso_factorization_ci[0] + pardiso_sellinv_ci[0],
        pardiso_factorization_ci[1] + pardiso_sellinv_ci[1],
    ]

    # MUMPS timings
    mumps_analysis_mean = [100.483038, 92.670997, 58.637949, 36.134608, 32.612968]
    mumps_analysis_ci = [
        [100.483038, 100.509227],
        [92.605942, 92.714337],
        [58.578062, 58.876641],
        [36.042327, 36.305006],
        [32.518000, 62.325532],
    ]
    mumps_factorization_mean = [39.225085, 34.435948, 23.922578, 17.898905, 15.841664]
    mumps_factorization_ci = [
        [39.225085, 39.378294],
        [34.357458, 36.195732],
        [23.288814, 24.129296],
        [17.712703, 18.470056],
        [15.819302, 20.931893],
    ]
    mumps_sellinv_mean = [1682.169842, 1264.773926, 715.862319, 632.781570, 752.983719]
    mumps_sellinv_ci = [
        [1682.169842, 1682.169842],
        [1264.773926, 1271.240979],
        [715.159370, 717.532285],
        [624.410655, 640.204100],
        [735.803012, 753.103131],
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
    serinv_cpu_factorization_mean = [73.99500, 51.15493, 32.98155, 19.84690, 16.29430]
    serinv_cpu_factorization_ci = [
        [73.85720, 74.13281],
        [49.90798, 52.40188],
        [31.42778, 34.53532],
        [18.60800, 21.08580],
        [15.12700, 17.46160],
    ]
    serinv_cpu_sellinv_mean = [86.84368, 60.91100, 39.48085, 24.56537, 18.41315]
    serinv_cpu_sellinv_ci = [
        [86.73152, 86.95583],
        [60.25982, 61.56218],
        [37.52429, 41.43740],
        [23.22687, 25.90387],
        [17.36571, 19.46058],
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
    serinv_gpu_factorization_mean = [
        2.34213,
        1.55993,
        0.93301,
        0.49508,
        0.23080,
    ]
    serinv_gpu_factorization_ci = [
        [2.34195, 2.34231],
        [1.55977, 1.56010],
        [0.93294, 0.93309],
        [0.49471, 0.49545],
        [0.23068, 0.23093],
    ]

    serinv_gpu_sellinv_mean = [
        3.07213,
        2.00457,
        1.19332,
        0.63325,
        0.30589,
    ]
    serinv_gpu_sellinv_ci = [
        [3.05504, 3.08921],
        [2.00362, 2.00551],
        [1.19257, 1.19407],
        [0.63276, 0.63375],
        [0.30571, 0.30607],
    ]

    serinv_gpu_rss_mean = [
        0,
        0.03913,
        0.09712,
        0.23255,
        0.49242,
    ]
    serinv_gpu_rss_ci = [
        [0, 0],
        [0.03910, 0.03915],
        [0.09700, 0.09724],
        [0.23224, 0.23286],
        [0.49184, 0.49300],
    ]

    serinv_gpu_a2x_mean = [
        serinv_gpu_factorization_mean[i]
        + serinv_gpu_sellinv_mean[i]
        + serinv_gpu_rss_mean[i]
        for i in range(len(serinv_gpu_factorization_mean))
    ]
    serinv_gpu_a2x_ci = [
        [
            serinv_gpu_factorization_ci[i][0]
            + serinv_gpu_sellinv_ci[i][0]
            + serinv_gpu_rss_ci[i][0],
            serinv_gpu_factorization_ci[i][1]
            + serinv_gpu_sellinv_ci[i][1]
            + serinv_gpu_rss_ci[i][1],
        ]
        for i in range(len(serinv_gpu_factorization_mean))
    ]

    # We look at relative performance w.r.t fastest PARDISO
    # MUMPS RELATIVE PERFORMANCE
    mumps_factorization_mean_rel = [
        pardiso_factorization_mean / mumps_factorization_mean[i]
        for i in range(len(n_processes))
    ]
    mumps_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / mumps_factorization_ci[i][1],
            pardiso_factorization_ci[1] / mumps_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    mumps_sellinv_mean_rel = [
        pardiso_sellinv_mean / mumps_sellinv_mean[i] for i in range(len(n_processes))
    ]
    mumps_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / mumps_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / mumps_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    mumps_a2x_mean_rel = [
        pardiso_a2x_mean / mumps_a2x_mean[i] for i in range(len(n_processes))
    ]
    mumps_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / mumps_a2x_ci[i][1],
            pardiso_a2x_ci[1] / mumps_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]

    # SERINV CPU RELATIVE PERFORMANCE
    serinv_cpu_factorization_mean_rel = [
        pardiso_factorization_mean / serinv_cpu_factorization_mean[i]
        for i in range(len(n_processes))
    ]
    serinv_cpu_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / serinv_cpu_factorization_ci[i][1],
            pardiso_factorization_ci[1] / serinv_cpu_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    serinv_cpu_sellinv_mean_rel = [
        pardiso_sellinv_mean / serinv_cpu_sellinv_mean[i]
        for i in range(len(n_processes))
    ]
    serinv_cpu_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / serinv_cpu_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / serinv_cpu_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    serinv_cpu_a2x_mean_rel = [
        pardiso_a2x_mean / serinv_cpu_a2x_mean[i] for i in range(len(n_processes))
    ]
    serinv_cpu_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / serinv_cpu_a2x_ci[i][1],
            pardiso_a2x_ci[1] / serinv_cpu_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]

    print("SERINV CPU factorization relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_cpu_factorization_mean_rel[i]:.5f}, ci=[{serinv_cpu_factorization_ci_rel[i][0]:.5f}, {serinv_cpu_factorization_ci_rel[i][1]:.5f}]"
        )

    print("SERINV CPU sellinv relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_cpu_sellinv_mean_rel[i]:.5f}, ci=[{serinv_cpu_sellinv_ci_rel[i][0]:.5f}, {serinv_cpu_sellinv_ci_rel[i][1]:.5f}]"
        )

    print("SERINV CPU a2x relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_cpu_a2x_mean_rel[i]:.5f}, ci=[{serinv_cpu_a2x_ci_rel[i][0]:.5f}, {serinv_cpu_a2x_ci_rel[i][1]:.5f}]"
        )

    # SERINV GPU RELATIVE PERFORMANCE
    serinv_gpu_factorization_mean_rel = [
        pardiso_factorization_mean / serinv_gpu_factorization_mean[i]
        for i in range(len(n_processes))
    ]
    serinv_gpu_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / serinv_gpu_factorization_ci[i][1],
            pardiso_factorization_ci[1] / serinv_gpu_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    serinv_gpu_sellinv_mean_rel = [
        pardiso_sellinv_mean / serinv_gpu_sellinv_mean[i]
        for i in range(len(n_processes))
    ]
    serinv_gpu_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / serinv_gpu_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / serinv_gpu_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    serinv_gpu_a2x_mean_rel = [
        pardiso_a2x_mean / serinv_gpu_a2x_mean[i] for i in range(len(n_processes))
    ]
    serinv_gpu_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / serinv_gpu_a2x_ci[i][1],
            pardiso_a2x_ci[1] / serinv_gpu_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]

    print("SERINV gpu factorization relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_gpu_factorization_mean_rel[i]:.5f}, ci=[{serinv_gpu_factorization_ci_rel[i][0]:.5f}, {serinv_gpu_factorization_ci_rel[i][1]:.5f}]"
        )

    print("SERINV gpu sellinv relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_gpu_sellinv_mean_rel[i]:.5f}, ci=[{serinv_gpu_sellinv_ci_rel[i][0]:.5f}, {serinv_gpu_sellinv_ci_rel[i][1]:.5f}]"
        )

    print("SERINV gpu a2x relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={serinv_gpu_a2x_mean_rel[i]:.5f}, ci=[{serinv_gpu_a2x_ci_rel[i][0]:.5f}, {serinv_gpu_a2x_ci_rel[i][1]:.5f}]"
        )

    # MUMPS RELATIVE PERFORMANCE
    mumps_factorization_mean_rel = [
        pardiso_factorization_mean / mumps_factorization_mean[i]
        for i in range(len(n_processes))
    ]
    mumps_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / mumps_factorization_ci[i][1],
            pardiso_factorization_ci[1] / mumps_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    mumps_sellinv_mean_rel = [
        pardiso_sellinv_mean / mumps_sellinv_mean[i] for i in range(len(n_processes))
    ]
    mumps_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / mumps_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / mumps_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]
    mumps_a2x_mean_rel = [
        pardiso_a2x_mean / mumps_a2x_mean[i] for i in range(len(n_processes))
    ]
    mumps_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / mumps_a2x_ci[i][1],
            pardiso_a2x_ci[1] / mumps_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ]

    print("MUMPS factorization relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={mumps_factorization_mean_rel[i]:.5f}, ci=[{mumps_factorization_ci_rel[i][0]:.5f}, {mumps_factorization_ci_rel[i][1]:.5f}]"
        )

    print("MUMPS sellinv relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={mumps_sellinv_mean_rel[i]:.5f}, ci=[{mumps_sellinv_ci_rel[i][0]:.5f}, {mumps_sellinv_ci_rel[i][1]:.5f}]"
        )

    print("MUMPS a2x relative performance")
    for i in range(len(n_processes)):
        print(
            f"n_processes={n_processes[i]}, mean={mumps_a2x_mean_rel[i]:.5f}, ci=[{mumps_a2x_ci_rel[i][0]:.5f}, {mumps_a2x_ci_rel[i][1]:.5f}]"
        )

    # Make an error bar plot for relative performances of serinv. 100% is pardiso
    fig, ax = plt.subplots()
    # Plot pardiso reference line at 1
    ax.axhline(y=1, color="r", linestyle="--", label="PARDISO Reference")
    # Plot MUMPS
    yerr_mumps = np.array(
        [
            [
                mumps_a2x_mean_rel[i] - mumps_a2x_ci_rel[i][0]
                for i in range(len(n_processes))
            ],
            [
                mumps_a2x_ci_rel[i][1] - mumps_a2x_mean_rel[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.errorbar(
        n_processes,
        mumps_a2x_mean_rel,
        yerr=yerr_mumps,
        capsize=3,
        capthick=1,
        fmt="x-",
        label="MUMPS",
        color=viridis(0.7),
    )
    # Plot SERINV CPU
    yerr_cpu = np.array(
        [
            [
                serinv_cpu_a2x_mean_rel[i] - serinv_cpu_a2x_ci_rel[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_cpu_a2x_ci_rel[i][1] - serinv_cpu_a2x_mean_rel[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.errorbar(
        n_processes,
        serinv_cpu_a2x_mean_rel,
        yerr=yerr_cpu,
        capsize=3,
        capthick=1,
        fmt="x-",
        label="Serinv-CPU",
        color=viridis(0.1),
    )
    # Plot SERINV GPU
    yerr_gpu = np.array(
        [
            [
                serinv_gpu_a2x_mean_rel[i] - serinv_gpu_a2x_ci_rel[i][0]
                for i in range(len(n_processes))
            ],
            [
                serinv_gpu_a2x_ci_rel[i][1] - serinv_gpu_a2x_mean_rel[i]
                for i in range(len(n_processes))
            ],
        ]
    )
    ax.errorbar(
        n_processes,
        serinv_gpu_a2x_mean_rel,
        yerr=yerr_gpu,
        capsize=3,
        capthick=1,
        fmt="x--",
        label="Serinv-GPU",
        color=viridis(0.1),
        alpha=0.5,
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(n_processes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    # set the y scale from 0 to 2
    ax.set_ylim(0.1, 400)

    ax.set_xlabel("Number of processes $\\mathit{P}$", fontsize=fontsize)
    # ax.set_ylabel("Performances comparison to PARDISO-multithreaded", fontsize=fontsize)
    # ax.set_title("Relative performance of SERINV and MUMPS \n compared to PARDISO")
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    # Modify the y ticks by 0.1, 1, 10, 100
    ax.set_yticks([0.1, 1, 10, 100])

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    # ax.legend(fontsize=fontsize, loc="upper left")

    ax.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.savefig("relative_performance_2865_d3.png", dpi=450)

    plt.show()
