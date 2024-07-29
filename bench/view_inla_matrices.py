import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from load_datmat import read_sym_CSC
import time 

tic = time.perf_counter()
file_path = "/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples/"

diagonal_blocksize = 2865
n_diag_blocks = 365
arrowhead_blocksize = 4
n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

file = (
        "Qxy_ns"
        + str(diagonal_blocksize)
        + "_nt"
        + str(n_diag_blocks)
        + "_nss0_nb"
        + str(arrowhead_blocksize)
        + "_n"
        + str(n)
        + ".dat"
    )

filename = file_path + file

A = read_sym_CSC(filename)

print(f"Matrix shape: {A.shape}", flush=True)

n_days = 7

A_some_days = A[-(n_days*diagonal_blocksize+arrowhead_blocksize):, -(n_days*diagonal_blocksize+arrowhead_blocksize):]
# A_some_days = A

toc = time.perf_counter()
print(f"(1/3) Finished loading the sparse matrix. [{toc-tic}s]", flush=True)    

tic = time.perf_counter()
markersize = 0.001
plt.spy(A_some_days, markersize=markersize, marker='.', color='blue')

plt.rcParams.update({'font.size': 2})
plt.xticks(np.arange(0, n_days*diagonal_blocksize, diagonal_blocksize) + diagonal_blocksize//2, np.arange(1, n_days+1))
plt.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

plt.yticks([])
plt.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

linewidth = 0.1
# Add vertical and horizontal lines to indicate block structure
for i in range(1, n_diag_blocks):
    plt.axhline(y=i * diagonal_blocksize, color='r', linestyle='--', linewidth=linewidth)
    plt.axvline(x=i * diagonal_blocksize, color='r', linestyle='--', linewidth=linewidth)

# Add lines for the arrowhead block
plt.axhline(y=n_days*diagonal_blocksize, color='b', linestyle='--', linewidth=linewidth)
plt.axvline(x=n_days*diagonal_blocksize, color='b', linestyle='--', linewidth=linewidth)


# Add double-sided arrow and legend
plt.annotate('', xy=(0, 0), xytext=(0, diagonal_blocksize), 
             arrowprops=dict(arrowstyle='<->, head_length=3, head_width=1.5', color='black', lw=1))

plt.text(0, diagonal_blocksize/2, 'b', va='center', ha='right', fontsize=5, rotation=90)


toc = time.perf_counter()
print(f"(2/3) Finished making the spy. [{toc-tic}s]", flush=True)    

tic = time.perf_counter()
format = "png"
plt.savefig(f"spy_with_blocks.{format}", format=format, dpi=300)
toc = time.perf_counter()
print(f"(3/3) Finished saving. [{toc-tic}s]", flush=True)    
