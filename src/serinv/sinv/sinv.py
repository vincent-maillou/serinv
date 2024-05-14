# Copyright 2023-2024 ETH Zurich and USI. All rights reserved.

import numpy as np
import scipy.linalg as la

from serinv.cholesky.cholesky_factorize import (
    cholesky_factorize_block_tridiagonal_arrowhead,
)
from serinv.cholesky.cholesky_selected_inversion import (
    cholesky_sinv_block_tridiagonal_arrowhead,
)

from serinv.lu.lu_factorize import lu_factorize_tridiag_arrowhead
from serinv.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead


def sinv_tridiagonal_arrowhead(
    A_diagonal_blocks: np.ndarray,
    A_lower_diagonal_blocks: np.ndarray,
    A_arrow_bottom_blocks: np.ndarray,
    A_arrow_tip_block: np.ndarray,
    A_upper_diagonal_blocks: np.ndarray = None,
    A_arrow_right_blocks: np.ndarray = None,
):
    """Direct interface for selected inversion of tridiagonal arrowhead matrices.

    - Handles potential pivoting of arrowhead tip block.
    """

    if A_upper_diagonal_blocks is None or A_arrow_right_blocks is None:
        # Perform cholesky factorization and selected inversion
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
        ) = cholesky_factorize_block_tridiagonal_arrowhead(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
        )

        return cholesky_sinv_block_tridiagonal_arrowhead(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            L_arrow_tip_block,
        )
    else:
        # Perform LU factorization and selected inversion
        (
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            P_arrow_tip_block,
        ) = lu_factorize_tridiag_arrowhead(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )

        return lu_sinv_tridiag_arrowhead(
            L_diagonal_blocks,
            L_lower_diagonal_blocks,
            L_arrow_bottom_blocks,
            U_diagonal_blocks,
            U_upper_diagonal_blocks,
            U_arrow_right_blocks,
            P_arrow_tip_block,
        )
