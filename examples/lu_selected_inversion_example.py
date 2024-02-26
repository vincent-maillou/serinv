import copy
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from sdr.utils import matrix_generation


from sdr.utils.matrix_transform import cut_to_blocktridiag
from sdr.lu.lu_decompose import lu_dcmp_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag, sinv_ndiags_greg

nblocks = 5
blocksize = 2
symmetric = False
diagonal_dominant = True
seed = 63

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

A = matrix_generation.generate_tridiag_dense(
    nblocks, blocksize, symmetric, diagonal_dominant, seed
)

in_A = copy.deepcopy(A)
# --- Inversion ---

X_ref = la.inv(A)
X_ref = cut_to_blocktridiag(X_ref, blocksize)

L_sdr, U_sdr = lu_dcmp_tridiag(A, blocksize)

fig, ax = plt.subplots(1, 3)
ax[0].set_title("X_ref: Scipy reference inversion")
ax[0].matshow(X_ref)

X_sdr = lu_sinv_tridiag(L_sdr, U_sdr, blocksize)
ax[1].set_title("X_sdr: LU selected inversion")
ax[1].matshow(X_sdr)

width_in_blocks = 3
ndiags = width_in_blocks * blocksize // 2

P, L, U = la.lu(copy.deepcopy(in_A))
LU = L + U - np.eye(in_A.shape[0])
LU_inv = la.inv(L) + la.inv(U) - np.eye(in_A.shape[0])
UL_inv = la.inv(U) @ la.inv(L)

print(f"\n\ncorrect: {UL_inv[0,0]}\n\n")

print("full\n\n")
X_greg = sinv_ndiags_greg(copy.deepcopy(in_A), 100000)
print("diag\n\n")
X_greg = sinv_ndiags_greg(copy.deepcopy(in_A), ndiags)

X_diff = X_ref - X_sdr
ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
ax[2].matshow(X_diff)
fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)




# arrowhead


from sdr.utils.matrix_transform import cut_to_blockndiags_arrowhead
from sdr.lu.lu_decompose import lu_dcmp_ndiags_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_ndiags_arrowhead

nblocks = 7
ndiags = 5
diag_blocksize = 3
arrow_blocksize = 2
symmetric = False
diagonal_dominant = True
seed = 63

A = matrix_generation.generate_ndiags_arrowhead(
    nblocks, ndiags, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
    seed
)

# --- Inversion ---

X_ref = la.inv(A)
X_ref = cut_to_blockndiags_arrowhead(X_ref, ndiags, diag_blocksize, arrow_blocksize)

L_sdr, U_sdr = lu_dcmp_ndiags_arrowhead(A, ndiags, diag_blocksize, arrow_blocksize)

fig, ax = plt.subplots(1, 3)
ax[0].set_title("X_ref: Scipy reference inversion")
ax[0].matshow(X_ref)

X_sdr = lu_sinv_ndiags_arrowhead(L_sdr, U_sdr, ndiags, diag_blocksize, arrow_blocksize)
ax[1].set_title("X_sdr: LU selected inversion")
ax[1].matshow(X_sdr)

X_diff = X_ref - X_sdr
ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
ax[2].matshow(X_diff)
fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)


# n diagonal

from sdr.utils.matrix_transform import cut_to_blockndiags
from sdr.lu.lu_decompose import lu_dcmp_ndiags
from sdr.lu.lu_selected_inversion import lu_sinv_ndiags

nblocks = 8
ndiags = 7
blocksize = 2
symmetric = False
diagonal_dominant = True
seed = 63

A = matrix_generation.generate_block_ndiags(
    nblocks, ndiags, blocksize, symmetric, diagonal_dominant, seed
)

# --- Inversion ---

X_ref = la.inv(A)
X_ref = cut_to_blockndiags(X_ref, ndiags, blocksize)

L_sdr, U_sdr = lu_dcmp_ndiags(A, ndiags, blocksize)

fig, ax = plt.subplots(1, 3)
ax[0].set_title("X_ref: Scipy reference inversion")
ax[0].matshow(X_ref)

X_sdr = lu_sinv_ndiags(L_sdr, U_sdr, ndiags, blocksize)
ax[1].set_title("X_sdr: LU selected inversion")
ax[1].matshow(X_sdr)

X_diff = X_ref - X_sdr
ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
ax[2].matshow(X_diff)
fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)

# n diagonal arrowhead


from sdr.utils.matrix_transform import cut_to_blockndiags_arrowhead
from sdr.lu.lu_decompose import lu_dcmp_ndiags_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_ndiags_arrowhead

nblocks = 7
ndiags = 5
diag_blocksize = 3
arrow_blocksize = 2
symmetric = False
diagonal_dominant = True
seed = 63

A = matrix_generation.generate_ndiags_arrowhead(
    nblocks, ndiags, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
    seed
)


# --- Inversion ---

X_ref = la.inv(A)
X_ref = cut_to_blockndiags_arrowhead(X_ref, ndiags, diag_blocksize, arrow_blocksize)

L_sdr, U_sdr = lu_dcmp_ndiags_arrowhead(A, ndiags, diag_blocksize, arrow_blocksize)

fig, ax = plt.subplots(1, 3)
ax[0].set_title("X_ref: Scipy reference inversion")
ax[0].matshow(X_ref)

X_sdr = lu_sinv_ndiags_arrowhead(L_sdr, U_sdr, ndiags, diag_blocksize, arrow_blocksize)
ax[1].set_title("X_sdr: LU selected inversion")
ax[1].matshow(X_sdr)

X_diff = X_ref - X_sdr
ax[2].set_title("X_diff: Difference between X_ref and X_sdr")
ax[2].matshow(X_diff)
fig.colorbar(ax[2].matshow(X_diff), ax=ax[2], label="Relative error", shrink=0.4)
