"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for lu selected inversion routines.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.utils.matrix_transform import cut_to_blocktridiag_arrowhead, from_dense_to_arrowhead_arrays, from_arrowhead_arrays_to_dense
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead
from sdr.lu.lu_factorize import lu_factorize_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import pytest



# @pytest.mark.parametrize(
#     "nblocks, diag_blocksize, arrow_blocksize", 
#     [
#         (2, 2, 2),
#         (2, 3, 2),
#         (2, 2, 3),
#         (10, 2, 2),
#         (10, 3, 2),
#         (10, 2, 3),
#         (10, 10, 2),
#         (10, 2, 10),
#     ]
# )
# def test_lu_sinv_tridiag_arrowhead(
#     nblocks: int, 
#     diag_blocksize: int, 
#     arrow_blocksize: int, 
# ):
#     symmetric = False
#     diagonal_dominant = True
#     seed = 63

#     A = matrix_generation.generate_tridiag_arrowhead_dense(
#         nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, 
#         seed
#     )

#     # --- Inversion ---

#     X_ref = la.inv(A)
#     X_ref = cut_to_blocktridiag_arrowhead(X_ref, diag_blocksize, arrow_blocksize)

#     L_sdr, U_sdr = lu_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
#     X_sdr = lu_sinv_tridiag_arrowhead(L_sdr, U_sdr, diag_blocksize, arrow_blocksize)

#     assert np.allclose(X_ref, X_sdr)
    