import scipy.sparse as sp
import matplotlib.pyplot as plt

file_path = "/home/vincent-maillou/Documents/Repository/serinv/bench/CNT_DEVICE/"
file_name = "H00.npz"
file = file_path + file_name

device_hamiltonian_coo = sp.load_npz(file)

device_hamiltonian_csc = device_hamiltonian_coo.tocsc()

size = device_hamiltonian_csc.shape[0]
n_diag_blocks = 21
diagonal_blocksize = size // n_diag_blocks

n_cells = 10
slice = diagonal_blocksize * n_cells

device_hamiltonian_slice = device_hamiltonian_csc[-slice:, -slice:]

markersize = 0.01
plt.spy(
    device_hamiltonian_slice,
    markersize=markersize,
    marker=".",
    color="blue",
)

plt.tick_params(axis="x", left=False, right=False, labelleft=False, labelright=False)
plt.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
plt.yticks([])
plt.xticks([])

linewidth = 1
for i in range(1, n_cells):
    # Horizontal lines
    plt.plot(
        [(i - 2) * diagonal_blocksize, (i + 2) * diagonal_blocksize],
        [i * diagonal_blocksize, i * diagonal_blocksize],
        color="black",
        linestyle="--",
        linewidth=linewidth,
    )

    # Vertical lines
    plt.plot(
        [i * diagonal_blocksize, i * diagonal_blocksize],
        [(i - 2) * diagonal_blocksize, (i + 2) * diagonal_blocksize],
        color="black",
        linestyle="--",
        linewidth=linewidth,
    )

format = "png"
plt.savefig(f"spy_inla_{n_cells}days.{format}", format=format, dpi=400)
