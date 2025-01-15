# PRINT COMPARISON FOR B=2865, N=365, A=4

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # PARDISO timings
    pardiso_analysis_mean = 22.52604
    pardiso_analysis_ci = [22.45823, 22.59385]
    pardiso_factorization_mean = 9.60548
    pardiso_factorization_ci = [9.55302, 9.65794]
    pardiso_sellinv_mean = 56.78813
    pardiso_sellinv_ci = [52.39210, 61.18416]

    pardiso_a2x_mean = pardiso_factorization_mean + pardiso_sellinv_mean
    pardiso_a2x_ci = [
        pardiso_factorization_ci[0] + pardiso_sellinv_ci[0],
        pardiso_factorization_ci[1] + pardiso_sellinv_ci[1],
    ]

    # MUMPS timings
    mumps_analysis_mean = [22, 18, 10.7, 7.7, 6.5]
    mumps_analysis_ci = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    mumps_factorization_mean = [11.8, 6.5, 7, 3.6, 4.2]
    mumps_factorization_ci = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    mumps_sellinv_mean = [450, 497, 413, 362, 483]
    mumps_sellinv_ci = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
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

    # Other parameters
    n_processes = [1, 2, 4, 8, 16]

    # We look at relative performance w.r.t fastest PARDISO
    # MUMPS RELATIVE PERFORMANCE
    mumps_factorization_mean_rel = [
        pardiso_factorization_mean / mumps_factorization_mean[i]
        for i in range(len(n_processes))
    ]
    """ mumps_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / mumps_factorization_ci[i][1],
            pardiso_factorization_ci[1] / mumps_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """
    mumps_sellinv_mean_rel = [
        pardiso_sellinv_mean / mumps_sellinv_mean[i] for i in range(len(n_processes))
    ]
    """ mumps_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / mumps_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / mumps_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """
    mumps_a2x_mean_rel = [
        pardiso_a2x_mean / mumps_a2x_mean[i] for i in range(len(n_processes))
    ]
    """ mumps_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / mumps_a2x_ci[i][1],
            pardiso_a2x_ci[1] / mumps_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """

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
    """ mumps_factorization_ci_rel = [
        [
            pardiso_factorization_ci[0] / mumps_factorization_ci[i][1],
            pardiso_factorization_ci[1] / mumps_factorization_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """
    mumps_sellinv_mean_rel = [
        pardiso_sellinv_mean / mumps_sellinv_mean[i] for i in range(len(n_processes))
    ]
    """ mumps_sellinv_ci_rel = [
        [
            pardiso_sellinv_ci[0] / mumps_sellinv_ci[i][1],
            pardiso_sellinv_ci[1] / mumps_sellinv_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """
    mumps_a2x_mean_rel = [
        pardiso_a2x_mean / mumps_a2x_mean[i] for i in range(len(n_processes))
    ]
    """ mumps_a2x_ci_rel = [
        [
            pardiso_a2x_ci[0] / mumps_a2x_ci[i][1],
            pardiso_a2x_ci[1] / mumps_a2x_ci[i][0],
        ]
        for i in range(len(n_processes))
    ] """

    """ print("MUMPS factorization relative performance")
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
        ) """

    # Make an error bar plot for relative performances of serinv. 100% is pardiso
    fig, ax = plt.subplots()
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
    )
    # Plot MUMPS

    ax.errorbar(
        n_processes,
        mumps_a2x_mean_rel,
        capsize=3,
        capthick=1,
        fmt="x--",
        label="MUMPS",
    )

    # Plot pardiso reference line at 1
    ax.axhline(y=1, color="r", linestyle="--", label="PARDISO")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(n_processes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # set the y scale from 0 to 2
    # ax.set_ylim(0, 100)

    ax.set_xlabel("Number of processes")
    ax.set_ylabel("Relative performance to PARDISO")
    ax.set_title("Relative performance of SERINV and MUMPS \n compared to PARDISO")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.show()
