import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

DEBUG = False
NDIGITS = 5


def make_autopct(sizes, flops, labels):
    def my_autopct(pct):
        total = sum(sizes)
        val = round(pct * total / 100.0, ndigits=NDIGITS)
        if DEBUG:
            print(f"pct: {pct}, total: {total}, val: {val}", flush=True)
            print(f"sizes: {sizes}", flush=True)
        idx = sizes.index(val)
        flops_val = flops[idx]
        label = labels[idx]
        if flops_val > 0:
            return f"{label}\n{pct:.1f}%\n({flops_val:.1f} TFLOPS)"
        else:
            return f"{label}\n{pct:.1f}%"

    return my_autopct


plt.rcParams.update({"font.size": 12})

# Load the timings
diagonal_blocksize = 2865
arrowhead_blocksize = 4
n_diag_blocks = 365


# --- POBTAF Pie Chart section ---
dict_timings_pobtaf = np.load(
    f"dict_timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
).item()


timings_pobtaf = np.load(
    f"timings_inlamat_pobtaf_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
)

# Average times
potrf_avg = round(np.mean(dict_timings_pobtaf["potrf"] / 1000), ndigits=NDIGITS)
trsm_avg = round(np.mean(dict_timings_pobtaf["trsm"] / 1000), ndigits=NDIGITS)
gemm_avg = round(np.mean(dict_timings_pobtaf["gemm"] / 1000), ndigits=NDIGITS)
total_avg = round(np.mean(timings_pobtaf), ndigits=NDIGITS)
other_avg = round(total_avg - (potrf_avg + trsm_avg + gemm_avg), ndigits=NDIGITS)

print(
    f"potrf_avg: {potrf_avg}, trsm_avg: {trsm_avg}, gemm_avg: {gemm_avg}, total_avg: {total_avg}, other_avg: {other_avg}"
)

# FLOPS as measured from nsys-profile
potrf_flops = 3.53
trsm_flops = 3.54
gemm_flops = 16.67

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = "POTRF", "TRSM", "GEMM", "OTHER"
sizes = [potrf_avg, trsm_avg, gemm_avg, other_avg]
flops = [
    potrf_flops,
    trsm_flops,
    gemm_flops,
    0,
]  # Assuming 'OTHER' has 0 FLOPS

shadow = False
labeldistance = 0
startangle = 90


# Generate colors from the viridis colormap
colors = cm.Pastel1(np.linspace(0, 1, len(labels)))
colors = cm.Dark2(np.linspace(0, 1, len(labels)))


fig1, ax1 = plt.subplots()
ax1.pie(
    sizes,
    colors=colors,
    autopct=make_autopct(sizes, flops, labels),
    shadow=shadow,
    startangle=startangle,
)
ax1.axis("equal")

plt.savefig(
    f"pobtaf_pie_chart_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.png",
    dpi=400,
)
plt.show()


# # --- POBTASI Pie Chart section ---
dict_timings_pobtasi = np.load(
    f"dict_timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
).item()

timings_pobtasi = np.load(
    f"timings_inlamat_pobtasi_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.npy",
    allow_pickle=True,
)

# Average times
trsm_avg = round(np.mean(dict_timings_pobtasi["trsm"] / 1000), ndigits=NDIGITS)
gemm_avg = round(np.mean(dict_timings_pobtasi["gemm"] / 1000), ndigits=NDIGITS)
total_avg = round(np.mean(timings_pobtasi), ndigits=NDIGITS)
other_avg = round(total_avg - (trsm_avg + gemm_avg), ndigits=NDIGITS)

# FLOPS as measured from nsys-profile
trsm_flops = 4.53
gemm_flops = 17.27

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = "TRSM", "GEMM", "OTHER"
sizes = [trsm_avg, gemm_avg, other_avg]
flops = [
    trsm_flops,
    gemm_flops,
    0,
]  # Assuming 'OTHER' has 0 FLOPS

shadow = False
labeldistance = 0
startangle = 90

# Generate colors from the viridis colormap
# colors = cm.Pastel1(np.linspace(0, 1, len(labels)))
colors = cm.Dark2(np.linspace(0, 1, len(labels)))

fig1, ax1 = plt.subplots()
ax1.pie(
    sizes,
    colors=colors,
    autopct=make_autopct(sizes, flops, labels),
    shadow=shadow,
    startangle=startangle,
)
ax1.axis("equal")

plt.savefig(
    f"pobtasi_pie_chart_b{diagonal_blocksize}_a{arrowhead_blocksize}_n{n_diag_blocks}.png",
    dpi=400,
)
plt.show()
