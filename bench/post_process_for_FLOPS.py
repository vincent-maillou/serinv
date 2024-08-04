import numpy as np

diagonal_blocksize = 2865
arrowhead_blocksize = 4
n_diag_blocks = 365

n_ops_conjugate_diag = (1 / 2) * pow(diagonal_blocksize, 2)
n_ops_conjugate_arrow = (1 / 2) * diagonal_blocksize * arrowhead_blocksize

# --- POBTAF Pie Chart section ---
print(" --- POBTAF --- ")
dict_timings_pobtaf = np.load(
    f"dict_timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
).item()

potrf_avg = np.mean(dict_timings_pobtaf["potrf"] / 1000)
potrf_single_kernel_avg = potrf_avg / n_diag_blocks
n_ops_potrf = (
    (1 / 3) * pow(diagonal_blocksize, 3)
    + (1 / 2) * pow(diagonal_blocksize, 2)
    + (1 / 6) * diagonal_blocksize
)
flops_potrf = (n_ops_potrf / potrf_single_kernel_avg) / 1e12

print(f"potrf_avg: {potrf_single_kernel_avg} s")
print(f"n_ops_potrf: {n_ops_potrf}")
print(f"flops_potrf: {flops_potrf} TFLOPS")

trsm_avg = np.mean(dict_timings_pobtaf["trsm"] / 1000)
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

print(f"trsm_avg: {trsm_single_iteration_avg} s")
print(f"n_ops_trsm: {n_ops_trsm}")
print(f"flops_trsm: {flops_trsm} TFLOPS")

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

print(f"gemm_avg: {gemm_single_iteration_avg} s")
print(f"n_ops_gemm: {n_ops_gemm}")
print(f"flops_gemm: {flops_gemm} TFLOPS")


# # --- POBTASI Pie Chart section ---
print(" --- POBTASI --- ")
dict_timings_pobtasi = np.load(
    f"dict_timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
).item()

trsm_avg = np.mean(dict_timings_pobtasi["trsm"] / 1000)
trsm_single_iteration_avg = trsm_avg / n_diag_blocks
n_ops_trsm_diag = pow(diagonal_blocksize, 3)

n_ops_trsm = n_ops_trsm_diag
flops_trsm = (n_ops_trsm / trsm_single_iteration_avg) / 1e12

print(f"trsm_avg: {trsm_single_iteration_avg} s")
print(f"n_ops_trsm: {n_ops_trsm}")
print(f"flops_trsm: {flops_trsm} TFLOPS")

gemm_avg = np.mean(dict_timings_pobtasi["gemm"] / 1000)
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

print(f"gemm_avg: {gemm_single_iteration_avg} s")
print(f"n_ops_gemm: {n_ops_gemm}")
print(f"flops_gemm: {flops_gemm} TFLOPS")
