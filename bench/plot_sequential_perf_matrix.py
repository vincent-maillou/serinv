import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

l_b = [256, 512, 1024, 2048]
l_a = [l_b[i] // 4 for i in range(len(l_b))]
n = 16

l_flops_pobtaf_potrf = np.zeros_like(l_b)
l_flops_pobtaf_trsm = np.zeros_like(l_b)
l_flops_pobtaf_gemm = np.zeros_like(l_b)
l_sutained_flops_achieved_pobtaf = np.zeros_like(l_b)

l_peak_pct_pobtaf_potrf = np.zeros_like(l_b)
l_peak_pct_pobtaf_trsm = np.zeros_like(l_b)
l_peak_pct_pobtaf_gemm = np.zeros_like(l_b)
l_sutained_peak_pct_achieved_pobtaf = np.zeros_like(l_b)

PATH = "/home/vincent-maillou/Documents/Repository/serinv/bench/sequential_perf_matrix/"

PEAK_FLOPS = 19.5

for i in range(len(l_b)):
    diagonal_blocksize = l_b[i]
    arrowhead_blocksize = l_a[i]
    n_diag_blocks = n

    n_ops_conjugate_diag = (1 / 2) * pow(diagonal_blocksize, 2)
    n_ops_conjugate_arrow = (1 / 2) * diagonal_blocksize * arrowhead_blocksize

    FILE = f"{PATH}dict_timings_synthetic_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy"

    dict_timings_pobtaf = np.load(
        FILE,
        allow_pickle=True,
    ).item()

    potrf_avg = np.mean(dict_timings_pobtaf["potrf"] / 1000)
    trsm_avg = np.mean(dict_timings_pobtaf["trsm"] / 1000)
    gemm_avg = np.mean(dict_timings_pobtaf["gemm"] / 1000)

    FILE = f"{PATH}timings_synthetic_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy"

    timings_pobtaf = np.load(
        FILE,
        allow_pickle=True,
    )

    total_avg = np.mean(timings_pobtaf)

    potrf_ratio = potrf_avg / total_avg
    trsm_ratio = trsm_avg / total_avg
    gemm_ratio = gemm_avg / total_avg

    # Compute FLOPS
    potrf_single_kernel_avg = potrf_avg / n_diag_blocks
    n_ops_potrf = (
        (1 / 3) * pow(diagonal_blocksize, 3)
        + (1 / 2) * pow(diagonal_blocksize, 2)
        + (1 / 6) * diagonal_blocksize
    )
    flops_potrf = (n_ops_potrf / potrf_single_kernel_avg) / 1e12

    trsm_single_iteration_avg = trsm_avg / n_diag_blocks
    n_ops_trsm_diag = pow(diagonal_blocksize, 3)
    n_ops_trsm_arrow = pow(diagonal_blocksize, 2) * arrowhead_blocksize
    n_ops_trsm = (
        n_ops_trsm_diag
        + n_ops_trsm_arrow
        + 2 * n_ops_conjugate_diag
        + 2 * n_ops_conjugate_arrow
    )
    flops_trsm = (n_ops_trsm / trsm_single_iteration_avg) / 1e12

    gemm_avg = np.mean(dict_timings_pobtaf["gemm"] / 1000)
    gemm_single_iteration_avg = gemm_avg / n_diag_blocks
    n_ops_gemm_diag = 2 * pow(diagonal_blocksize, 3)
    n_ops_gemm_diad_arrow = 2 * pow(diagonal_blocksize, 2) * arrowhead_blocksize
    n_ops_gemm_arrow = pow(arrowhead_blocksize, 2) * diagonal_blocksize
    n_ops_gemm = (
        n_ops_gemm_diag
        + n_ops_gemm_diad_arrow
        + n_ops_gemm_arrow
        + 2 * n_ops_conjugate_diag
        + n_ops_conjugate_arrow
    )
    flops_gemm = (n_ops_gemm / gemm_single_iteration_avg) / 1e12

    sustained_flops_achieved = (
        potrf_ratio * flops_potrf + trsm_ratio * flops_trsm + gemm_ratio * flops_gemm
    )

    print("diagonal_blocksize: ", diagonal_blocksize)
    print(" flops_potrf: ", flops_potrf)
    print(" flops_trsm: ", flops_trsm)
    print(" flops_gemm: ", flops_gemm)
    print(" sustained_flops_achieved: ", sustained_flops_achieved)

    l_flops_pobtaf_potrf[i] = flops_potrf
    l_flops_pobtaf_trsm[i] = flops_trsm
    l_flops_pobtaf_gemm[i] = flops_gemm
    l_sutained_flops_achieved_pobtaf[i] = sustained_flops_achieved

    l_peak_pct_pobtaf_potrf[i] = (flops_potrf / PEAK_FLOPS) * 100.0
    l_peak_pct_pobtaf_trsm[i] = (flops_trsm / PEAK_FLOPS) * 100.0
    l_peak_pct_pobtaf_gemm[i] = (flops_gemm / PEAK_FLOPS) * 100.0
    l_sutained_peak_pct_achieved_pobtaf[i] = (
        sustained_flops_achieved / PEAK_FLOPS
    ) * 100.0

    print(" peak_pct_pobtaf_potrf: ", l_peak_pct_pobtaf_potrf[i])
    print(" peak_pct_pobtaf_trsm: ", l_peak_pct_pobtaf_trsm[i])
    print(" peak_pct_pobtaf_gemm: ", l_peak_pct_pobtaf_gemm[i])
    print(
        " sutained_peak_pct_achieved_pobtaf: ", l_sutained_peak_pct_achieved_pobtaf[i]
    )

matrix_peak_pct = np.zeros((len(l_b), 4))
matrix_peak_pct[0, :] = l_peak_pct_pobtaf_potrf
matrix_peak_pct[1, :] = l_peak_pct_pobtaf_trsm
matrix_peak_pct[2, :] = l_peak_pct_pobtaf_gemm
matrix_peak_pct[3, :] = l_sutained_peak_pct_achieved_pobtaf

matrix_flops = np.zeros((len(l_b), 4))
matrix_flops[0, :] = l_flops_pobtaf_potrf
matrix_flops[1, :] = l_flops_pobtaf_trsm
matrix_flops[2, :] = l_flops_pobtaf_gemm
matrix_flops[3, :] = l_sutained_flops_achieved_pobtaf

# Plot the matrix
fig, ax = plt.subplots()
cax = ax.imshow(
    matrix_peak_pct, cmap=cm.viridis, interpolation="nearest", vmin=0, vmax=100
)

# Add color bar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label("Percentage of Peak Performance\n (19.5 TFLOPS Tensor Core FP64)")

# Set axis labels
ax.set_xticks(np.arange(len(l_b)))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(l_b)
ax.set_yticklabels(["POTRF", "TRSM", "GEMM", "Average"])

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(4):
    for j in range(len(l_b)):
        text = ax.text(
            j,
            i,
            f"{matrix_peak_pct[i, j]:.1f}%\n{matrix_flops[i, j]:.1f} TFLOPS",
            ha="center",
            va="center",
            color="w",
        )

fig.tight_layout()

# Save the figure
plt.savefig("sequential_perf_matrix_pobtaf.png")

plt.show()


# --- POBTASI ---
for i in range(len(l_b)):
    diagonal_blocksize = l_b[i]
    arrowhead_blocksize = l_a[i]
    n_diag_blocks = n

    n_ops_conjugate_diag = (1 / 2) * pow(diagonal_blocksize, 2)
    n_ops_conjugate_arrow = (1 / 2) * diagonal_blocksize * arrowhead_blocksize

    FILE = f"{PATH}dict_timings_synthetic_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy"

    dict_timings_pobtaf = np.load(
        FILE,
        allow_pickle=True,
    ).item()

    trsm_avg = np.mean(dict_timings_pobtaf["trsm"] / 1000)
    gemm_avg = np.mean(dict_timings_pobtaf["gemm"] / 1000)

    FILE = f"{PATH}timings_synthetic_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy"

    timings_pobtaf = np.load(
        FILE,
        allow_pickle=True,
    )

    total_avg = np.mean(timings_pobtaf)

    trsm_ratio = trsm_avg / total_avg
    gemm_ratio = gemm_avg / total_avg

    # Compute FLOPS
    trsm_single_iteration_avg = trsm_avg / n_diag_blocks
    n_ops_trsm_diag = pow(diagonal_blocksize, 3)
    n_ops_trsm = n_ops_trsm_diag
    flops_trsm = (n_ops_trsm / trsm_single_iteration_avg) / 1e12

    gemm_single_iteration_avg = gemm_avg / n_diag_blocks
    n_ops_gemm_diag = 2 * pow(diagonal_blocksize, 3)
    n_ops_gemm_diad_arrow = 2 * pow(diagonal_blocksize, 2) * arrowhead_blocksize
    n_ops_gemm_arrow = pow(arrowhead_blocksize, 2) * diagonal_blocksize
    n_ops_gemm = (
        4 * n_ops_gemm_diag
        + 2 * n_ops_gemm_diad_arrow
        + 3 * n_ops_gemm_arrow
        + 2 * n_ops_conjugate_diag
        + 2 * n_ops_conjugate_arrow
    )
    flops_gemm = (n_ops_gemm / gemm_single_iteration_avg) / 1e12

    sustained_flops_achieved = trsm_ratio * flops_trsm + gemm_ratio * flops_gemm

    print("diagonal_blocksize: ", diagonal_blocksize)
    print(" flops_trsm: ", flops_trsm)
    print(" flops_gemm: ", flops_gemm)
    print(" sustained_flops_achieved: ", sustained_flops_achieved)

    l_flops_pobtaf_trsm[i] = flops_trsm
    l_flops_pobtaf_gemm[i] = flops_gemm
    l_sutained_flops_achieved_pobtaf[i] = sustained_flops_achieved

    l_peak_pct_pobtaf_trsm[i] = (flops_trsm / PEAK_FLOPS) * 100.0
    l_peak_pct_pobtaf_gemm[i] = (flops_gemm / PEAK_FLOPS) * 100.0
    l_sutained_peak_pct_achieved_pobtaf[i] = (
        sustained_flops_achieved / PEAK_FLOPS
    ) * 100.0

    print(" peak_pct_pobtaf_trsm: ", l_peak_pct_pobtaf_trsm[i])
    print(" peak_pct_pobtaf_gemm: ", l_peak_pct_pobtaf_gemm[i])
    print(
        " sutained_peak_pct_achieved_pobtaf: ", l_sutained_peak_pct_achieved_pobtaf[i]
    )

matrix_peak_pct = np.zeros((len(l_b), 3))
matrix_peak_pct[0, :] = l_peak_pct_pobtaf_trsm
matrix_peak_pct[1, :] = l_peak_pct_pobtaf_gemm
matrix_peak_pct[2, :] = l_sutained_peak_pct_achieved_pobtaf

matrix_flops = np.zeros((len(l_b), 3))
matrix_flops[0, :] = l_flops_pobtaf_trsm
matrix_flops[1, :] = l_flops_pobtaf_gemm
matrix_flops[2, :] = l_sutained_flops_achieved_pobtaf

# Plot the matrix
fig, ax = plt.subplots()
cax = ax.imshow(
    matrix_peak_pct, cmap=cm.viridis, interpolation="nearest", vmin=0, vmax=100
)

# Add color bar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label("Percentage of Peak Performance\n (19.5 TFLOPS Tensor Core FP64)")

# Set axis labels
ax.set_xticks(np.arange(len(l_b)))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(l_b)
ax.set_yticklabels(["TRSM", "GEMM", "Average"])

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(3):
    for j in range(len(l_b)):
        text = ax.text(
            j,
            i,
            f"{matrix_peak_pct[i, j]:.1f}%\n{matrix_flops[i, j]:.1f} TFLOPS",
            ha="center",
            va="center",
            color="w",
        )

fig.tight_layout()

# Save the figure
plt.savefig("sequential_perf_matrix_pobtasi.png")

plt.show()
