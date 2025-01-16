# PRINT COMPARISON FOR B=2865, N=365, A=4

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    # Other parameters
    n_processes = [1, 2, 4, 8, 16]
    fontsize = 18
    ticsize = fontsize - 2
    viridis = plt.get_cmap("viridis")

    # PARDISO timings
    pardiso_analysis_mean = 154.06736
    pardiso_analysis_ci = [152.68448, 155.45025]
    pardiso_factorization_mean = 61.60185
    pardiso_factorization_ci = [60.53969, 62.66400]
    pardiso_sellinv_mean = 225.16645
    pardiso_sellinv_ci = [196.68438, 253.64853]

    pardiso_a2x_mean = pardiso_factorization_mean + pardiso_sellinv_mean
    pardiso_a2x_ci = [
        pardiso_factorization_ci[0] + pardiso_sellinv_ci[0],
        pardiso_factorization_ci[1] + pardiso_sellinv_ci[1],
    ]

    # MUMPS timings
    mumps_analysis_mean = [137.359827, 126.326724, 81.934029, 51.536856, 51.082372]
    mumps_analysis_ci = [
        [137.359827, 137.458724],
        [126.262346, 126.431899],
        [81.923481, 82.130227],
        [51.435951, 51.53853],
        [50.993882, 51.225979],
    ]
    mumps_factorization_mean = [38.581357, 40.081410, 27.049463, 17.963536, 18.624930]
    mumps_factorization_ci = [
        [38.581357, 38.755972],
        [39.393201, 40.369052],
        [27.006295, 29.001502],
        [17.842745, 18.364568],
        [18.276191, 18.645041],
    ]
    mumps_sellinv_mean = [1766.510641, 897.651438, 841.216716, 622.927269, 723.880556]
    mumps_sellinv_ci = [
        [1766.510641, 1766.510641],
        [895.383524, 901.799116],
        [841.026371, 844.019168],
        [617.23098, 631.621537],
        [722.243672, 772.270123],
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
    serinv_gpu_factorization_mean = [2.4856, 1.6249, 0.97540, 0.61214, 0.50339]
    serinv_gpu_factorization_ci = [
        [2.41606, 2.55513],
        [1.62393, 1.62587],
        [0.97330, 0.97750],
        [0.62426, 0.62871],
        [0.50311, 0.50367],
    ]

    serinv_gpu_sellinv_mean = [2.78955, 1.81765, 1.09637, 0.67208, 0.51710]
    serinv_gpu_sellinv_ci = [
        [2.7893, 2.78979],
        [1.81745, 1.81784],
        [1.09624, 1.09651],
        [0.67204, 0.67212],
        [0.51698, 0.51722],
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

    fig.savefig("relative_performance_2865_d5.png", dpi=450)

    plt.show()
