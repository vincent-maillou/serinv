import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from load_datmat import read_sym_CSC
import time 

tic = time.perf_counter()
# inla_mat = 'examples'
inla_mat = 'temperature'

if inla_mat == 'examples':
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
    
elif inla_mat == 'temperature':
    file_path = "/home/vault/j101df/j101df10/inla_matrices/temperature_examples/"

    ns = 607 # diagonal_blocksize
    nt = 1095 # n_diag_blocks
    nss = 153 # do know
    nb = 4 # do know
    arrowhead_blocksize = nss + nb
    n = ns * nt + arrowhead_blocksize

    file = (
            "Qxy_ns"
            + str(ns)
            + "_nt"
            + str(nt)
            + "_nss"
            + str(nss)
            + "_nb"
            + str(nb)
            + "_n"
            + str(n)
            + ".dat"
        )

    n_diag_blocks = nt
    diagonal_blocksize = ns

filename = file_path + file

A = read_sym_CSC(filename)

print(f"Matrix shape: {A.shape}", flush=True)

n_days = 7

A_some_days = A[-(n_days*diagonal_blocksize+arrowhead_blocksize):, -(n_days*diagonal_blocksize+arrowhead_blocksize):]
# A_some_days = A

toc = time.perf_counter()
print(f"(1/3) Finished loading the sparse matrix. [{toc-tic}s]", flush=True)    

tic = time.perf_counter()
markersize = 0.01
plt.spy(A_some_days, markersize=markersize, marker='.', color='blue')

linewidth = 1
for i in range(1, n_days):
    # Horizontal lines
    plt.plot(
        [(i - 2) * diagonal_blocksize, (i + 1) * diagonal_blocksize],
        [i * diagonal_blocksize, i * diagonal_blocksize],
        color="black",
        linestyle="--",
        linewidth=linewidth,
    )

    # Vertical lines
    plt.plot(
        [i * diagonal_blocksize, i * diagonal_blocksize],
        [(i - 1) * diagonal_blocksize, (i + 2) * diagonal_blocksize],
        color="black",
        linestyle="--",
        linewidth=linewidth,
    )

    plt.plot(
        [i * diagonal_blocksize, i * diagonal_blocksize],
        [n_days * diagonal_blocksize, n_days * diagonal_blocksize + arrowhead_blocksize],
        color="black",
        linestyle="--",
        linewidth=linewidth,
    )

# Arrowhead
plt.plot(
    [0, n_days * diagonal_blocksize + arrowhead_blocksize],
    [n_days * diagonal_blocksize, n_days * diagonal_blocksize],
    color="black",
    linestyle="--",
    linewidth=linewidth,
)

plt.plot(
    [n_days * diagonal_blocksize, n_days * diagonal_blocksize],
    [(n_days-1) * diagonal_blocksize, n_days * diagonal_blocksize + arrowhead_blocksize],
    color="black",
    linestyle="--",
    linewidth=linewidth,
)

plt.tick_params(axis='x', left=False, right=False, labelleft=False, labelright=False)
plt.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
plt.yticks([])
plt.xticks([])

toc = time.perf_counter()
print(f"(2/3) Finished making the spy. [{toc-tic}s]", flush=True)    

tic = time.perf_counter()
format = "png"
plt.savefig(f"spy_inla_{n_days}days.{format}", format=format, dpi=400)
toc = time.perf_counter()
print(f"(3/3) Finished saving. [{toc-tic}s]", flush=True)    
