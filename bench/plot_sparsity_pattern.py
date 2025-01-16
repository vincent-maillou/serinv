import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import coo_matrix

def plot_and_save_block(matrix, title, filename):
    plt.figure()
    plt.spy(matrix.toarray())
    plt.title(title)
    plt.savefig(filename)
    plt.close()


diagonal_blocksize = 2865
n_diag_blocks = 365
arrowhead_blocksize = 4
density = 0.05

n = diagonal_blocksize * n_diag_blocks + arrowhead_blocksize

#path = '/home/vault/j101df/j101df10/inla_matrices/INLA_paper_examples'
path = '/home/vault/ihpc/ihpc060h/matrices/serinv'

matrix_filename = f'Qxy_ns{diagonal_blocksize}_nt{n_diag_blocks}_nss0_nb{arrowhead_blocksize}_n{n}_density{density}_with_offdiag.mtx'
matrix_path = path + '/' + matrix_filename
print(matrix_path)
# Load the matrix from a Matrix Market file
A_coo = mmread(matrix_path)

A_csr = A_coo.tocsr()

# Plot and save the first diagonal, off-diagonal, and arrowhead blocks
A_diagonal_block = A_csr[:diagonal_blocksize, :diagonal_blocksize]
A_off_diagonal_block = A_csr[diagonal_blocksize:2*diagonal_blocksize, :diagonal_blocksize]
A_arrowhead_block = A_csr[-arrowhead_blocksize:, :diagonal_blocksize]

plot_and_save_block(A_diagonal_block, "First Diagonal Block", "diagonal_block.png")
plot_and_save_block(A_off_diagonal_block, "First Off-Diagonal Block", "off_diagonal_block.png")
plot_and_save_block(A_arrowhead_block, "Arrowhead Block", "arrowhead_block.png")