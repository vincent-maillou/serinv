import numpy as np
import matplotlib.pyplot as plt

# Load the timings
diagonal_blocksize = 256
arrowhead_blocksize = 256
n_diag_blocks = 128


# --- POBTAF Pie Chart section ---
dict_timings_pobtaf = np.load(
    f"dict_timings_inlamat_pobtaf_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}.npy",
    allow_pickle=True,
).item()

# vectors times
potrf = dict_timings_pobtaf["potrf"]
trsm = dict_timings_pobtaf["trsm"]
gemm = dict_timings_pobtaf["gemm"]
total = dict_timings_pobtaf["total"]

# Average times
potrf_avg = np.mean(potrf)
trsm_avg = np.mean(trsm)
gemm_avg = np.mean(gemm)
total_avg = np.mean(total)

# Other times
other_avg = total_avg - (potrf_avg + trsm_avg + gemm_avg)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'POTRF', 'TRSM', 'GEMM', 'OTHER'
sizes = [potrf_avg, trsm_avg, gemm_avg, other_avg]
explode = (0.1, 0, 0, 0)  # only "explode" the 1st slice (i.e. 'Potrf')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(f"POBTAF bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}")
plt.savefig(f"pobtaf_pie_chart_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}.png")
plt.show()



# --- POBTASI Pie Chart section ---
dict_timings_pobtasi = np.load(
    f"dict_timings_inlamat_pobtasi_bs{diagonal_blocksize}_as{arrowhead_blocksize}_nb{n_diag_blocks}.npy",
    allow_pickle=True,
).item()



