# Copyright 2023-2024 ETH Zurich. All rights reserved.
from serinv import SolverConfig

try:
    import cupy as cp
    import cupyx as cpx
    import cupyx.scipy.linalg as cu_la

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
import scipy.linalg as np_la
from mpi4py import MPI
from numpy.typing import ArrayLike

comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


def d_ddbtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_upper_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_right_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Perform the distributed LU factorization of a block tridiagonal
    with arrowhead matrix.

    Note:
    -----
    - The matrix is assumed to be block-diagonally dominant.
    - The given matrix will be overwritten.
    - If a device array is given, the algorithm will run on the GPU.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The blocks on the diagonal of the matrix.
    A_lower_diagonal_blocks : ArrayLike
        The blocks on the lower diagonal of the matrix.
    A_upper_diagonal_blocks : ArrayLike
        The blocks on the upper diagonal of the matrix.
    A_arrow_bottom_blocks : ArrayLike
        The blocks on the bottom arrow of the matrix.
    A_arrow_right_blocks : ArrayLike
        The blocks on the right arrow of the matrix.
    A_arrow_tip_block : ArrayLike
        The block at the tip of the arrowhead.
    solver_config : SolverConfig, optional
        Configuration of the solver.

    Returns
    -------
    LU_diagonal_blocks : ArrayLike
        LU factors of the diagonal blocks.
    LU_lower_diagonal_blocks : ArrayLike
        LU factors of the lower diagonal blocks.
    LU_upper_diagonal_blocks : ArrayLike
        LU factors of the upper diagonal blocks.
    LU_arrow_bottom_blocks : ArrayLike
        LU factors of the bottom arrow blocks.
    LU_arrow_right_blocks : ArrayLike
        LU factors of the right arrow blocks.
    LU_arrow_tip_block : ArrayLike
        LU factors of the tip block of the arrowhead.
    B_permutation_upper : ArrayLike, optional
        Local upper buffer used in the nested dissection factorization. None for
        uppermost process.
    B_permutation_lower : ArrayLike, optional
        Local lower buffer used in the nested dissection factorization. None for
        uppermost process.
    """

    array_module = cp.get_array_module(A_diagonal_blocks_local)

    if array_module == cp:
        # Device computation
        return _device_d_ddbtaf(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_arrow_tip_block_global,
            solver_config,
        )
    else:
        # Host computation
        return _host_d_ddbtaf(
            A_diagonal_blocks_local,
            A_lower_diagonal_blocks_local,
            A_upper_diagonal_blocks_local,
            A_arrow_bottom_blocks_local,
            A_arrow_right_blocks_local,
            A_arrow_tip_block_global,
        )


def _host_d_ddbtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_upper_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_right_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    n_diag_blocks_local = A_diagonal_blocks_local.shape[0]

    # LU aliases
    LU_diagonal_blocks_local = A_diagonal_blocks_local
    LU_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local
    LU_upper_diagonal_blocks_local = A_upper_diagonal_blocks_local
    LU_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local
    LU_arrow_right_blocks_local = A_arrow_right_blocks_local

    # Pivots array
    P_diag = np.zeros(
        (n_diag_blocks_local, A_diagonal_blocks_local.shape[1]), dtype=np.int32
    )

    B_permutation_upper = None
    B_permutation_lower = None

    Update_arrow_tip_block = np.zeros_like(A_arrow_tip_block_global)

    if comm_rank == 0:
        # Forward block-LU, performed by a "top" process
        for i in range(0, n_diag_blocks_local - 1):
            # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
            (
                LU_diagonal_blocks_local[i, :, :],
                P_diag[i, :],
            ) = np_la.lu_factor(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
            LU_lower_diagonal_blocks_local[i, :, :] = (
                np_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                    lower=False,
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
            LU_arrow_bottom_blocks_local[i, :, :] = (
                np_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                    lower=False,
                )
                .conj()
                .T
            )

            # Compute upper factors
            # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
            LU_upper_diagonal_blocks_local[i, :, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_upper_diagonal_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
            LU_arrow_right_blocks_local[i, :, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_arrow_right_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # Update next upper/lower blocks of the arrowhead
            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[i + 1, :, :] = (
                A_arrow_right_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )
    else:
        B_permutation_upper = np.empty_like(A_diagonal_blocks_local)
        B_permutation_lower = np.empty_like(A_diagonal_blocks_local)

        B_permutation_upper[1, :, :] = A_upper_diagonal_blocks_local[0, :, :]
        B_permutation_lower[1, :, :] = A_lower_diagonal_blocks_local[0, :, :]

        # Forward block-LU, performed by a "middle" process
        for i in range(1, n_diag_blocks_local - 1):
            # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
            (
                LU_diagonal_blocks_local[i, :, :],
                P_diag[i, :],
            ) = np_la.lu_factor(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
            LU_lower_diagonal_blocks_local[i, :, :] = (
                np_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
            B_permutation_upper[i, :, :] = (
                np_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    B_permutation_upper[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
            LU_arrow_bottom_blocks_local[i, :, :] = (
                np_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # Compute upper factors
            # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
            LU_upper_diagonal_blocks_local[i, :, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_upper_diagonal_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
            B_permutation_lower[i, :, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                B_permutation_lower[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
            LU_arrow_right_blocks_local[i, :, :] = np_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_arrow_right_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[i + 1, :, :] = (
                A_arrow_right_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update top and next upper/lower blocks of 2-sided factorization pattern
            # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
            A_diagonal_blocks_local[0, :, :] = (
                A_diagonal_blocks_local[0, :, :]
                - B_permutation_upper[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # A_{top, i+1} = - L{top, i} @ U_{i, i+1}
            B_permutation_upper[i + 1, :, :] = (
                -B_permutation_upper[i, :, :] @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, top} = - L{i+1, i} @ U_{i, top}
            B_permutation_lower[i + 1, :, :] = (
                -LU_lower_diagonal_blocks_local[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # Update the top (first blocks) of the arrowhead
            # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
            A_arrow_bottom_blocks_local[0, :, :] = (
                A_arrow_bottom_blocks_local[0, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[0, :, :] = (
                A_arrow_right_blocks_local[0, :, :]
                - B_permutation_upper[i, :, :] @ LU_arrow_right_blocks_local[i, :, :]
            )

    Update_arrow_tip_block_host = Update_arrow_tip_block

    # Accumulate the distributed update of the arrow tip block
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        Update_arrow_tip_block_host,
        op=MPI.SUM,
    )

    A_arrow_tip_block_global[:, :] += Update_arrow_tip_block[:, :]
    LU_arrow_tip_block_global = A_arrow_tip_block_global

    return (
        LU_diagonal_blocks_local,
        LU_lower_diagonal_blocks_local,
        LU_upper_diagonal_blocks_local,
        LU_arrow_bottom_blocks_local,
        LU_arrow_right_blocks_local,
        LU_arrow_tip_block_global,
        B_permutation_upper,
        B_permutation_lower,
    )


def _device_d_ddbtaf(
    A_diagonal_blocks_local: ArrayLike,
    A_lower_diagonal_blocks_local: ArrayLike,
    A_upper_diagonal_blocks_local: ArrayLike,
    A_arrow_bottom_blocks_local: ArrayLike,
    A_arrow_right_blocks_local: ArrayLike,
    A_arrow_tip_block_global: ArrayLike,
    solver_config: SolverConfig,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    n_diag_blocks_local = A_diagonal_blocks_local.shape[0]

    # LU aliases
    LU_diagonal_blocks_local = A_diagonal_blocks_local
    LU_lower_diagonal_blocks_local = A_lower_diagonal_blocks_local
    LU_upper_diagonal_blocks_local = A_upper_diagonal_blocks_local
    LU_arrow_bottom_blocks_local = A_arrow_bottom_blocks_local
    LU_arrow_right_blocks_local = A_arrow_right_blocks_local

    # Pivots array
    P_diag = cp.zeros(
        (n_diag_blocks_local, A_diagonal_blocks_local.shape[1]), dtype=cp.int32
    )

    B_permutation_upper = None
    B_permutation_lower = None

    Update_arrow_tip_block = cp.zeros_like(A_arrow_tip_block_global)

    if comm_rank == 0:
        # Forward block-LU, performed by a "top" process
        for i in range(0, n_diag_blocks_local - 1):
            # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
            (
                LU_diagonal_blocks_local[i, :, :],
                P_diag[i, :],
            ) = cu_la.lu_factor(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ U{i, i}^{-1}
            LU_lower_diagonal_blocks_local[i, :, :] = (
                cu_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                    lower=False,
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ U{i, i}^{-1}
            LU_arrow_bottom_blocks_local[i, :, :] = (
                cu_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                    lower=False,
                )
                .conj()
                .T
            )

            # Compute upper factors
            # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
            LU_upper_diagonal_blocks_local[i, :, :] = cu_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_upper_diagonal_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
            LU_arrow_right_blocks_local[i, :, :] = cu_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_arrow_right_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # Update next upper/lower blocks of the arrowhead
            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[i + 1, :, :] = (
                A_arrow_right_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )
    else:
        B_permutation_upper = cp.empty_like(A_diagonal_blocks_local)
        B_permutation_lower = cp.empty_like(A_diagonal_blocks_local)

        B_permutation_upper[1, :, :] = A_upper_diagonal_blocks_local[0, :, :]
        B_permutation_lower[1, :, :] = A_lower_diagonal_blocks_local[0, :, :]

        # Forward block-LU, performed by a "middle" process
        for i in range(1, n_diag_blocks_local - 1):
            # L_{i, i}, U_{i, i} = lu_dcmp(A_{i, i})
            (
                LU_diagonal_blocks_local[i, :, :],
                P_diag[i, :],
            ) = cu_la.lu_factor(
                A_diagonal_blocks_local[i, :, :],
            )

            # Compute lower factors
            # L_{i+1, i} = A_{i+1, i} @ U_{i, i}^{-T}
            LU_lower_diagonal_blocks_local[i, :, :] = (
                cu_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_lower_diagonal_blocks_local[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # L_{top, i} = A_{top, i} @ U{i, i}^{-1}
            B_permutation_upper[i, :, :] = (
                cu_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    B_permutation_upper[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # L_{ndb+1, i} = A_{ndb+1, i} @ U_{i, i}^{-T}
            LU_arrow_bottom_blocks_local[i, :, :] = (
                cu_la.solve_triangular(
                    LU_diagonal_blocks_local[i, :, :],
                    A_arrow_bottom_blocks_local[i, :, :].conj().T,
                    lower=False,
                    trans="T" if LU_diagonal_blocks_local.dtype.char == "f" else "C",
                )
                .conj()
                .T
            )

            # Compute upper factors
            # U_{i, i+1} = L{i, i}^{-1} @ A_{i, i+1}
            LU_upper_diagonal_blocks_local[i, :, :] = cu_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_upper_diagonal_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, top} = L{i, i}^{-1} @ A_{i, top}
            B_permutation_lower[i, :, :] = cu_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                B_permutation_lower[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # U_{i, ndb+1} = L{i, i}^{-1} @ A_{i, ndb+1}
            LU_arrow_right_blocks_local[i, :, :] = cu_la.solve_triangular(
                LU_diagonal_blocks_local[i, :, :],
                A_arrow_right_blocks_local[i, :, :],
                lower=True,
                unit_diagonal=True,
            )

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ U_{i, i+1}
            A_diagonal_blocks_local[i + 1, :, :] = (
                A_diagonal_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ U_{i, i+1}
            A_arrow_bottom_blocks_local[i + 1, :, :] = (
                A_arrow_bottom_blocks_local[i + 1, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[i + 1, :, :] = (
                A_arrow_right_blocks_local[i + 1, :, :]
                - LU_lower_diagonal_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update the block at the tip of the arrowhead
            # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}
            Update_arrow_tip_block[:, :] = (
                Update_arrow_tip_block[:, :]
                - LU_arrow_bottom_blocks_local[i, :, :]
                @ LU_arrow_right_blocks_local[i, :, :]
            )

            # Update top and next upper/lower blocks of 2-sided factorization pattern
            # A_{top, top} = A_{top, top} - L_{top, i} @ U_{i, top}
            A_diagonal_blocks_local[0, :, :] = (
                A_diagonal_blocks_local[0, :, :]
                - B_permutation_upper[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # A_{top, i+1} = - L{top, i} @ U_{i, i+1}
            B_permutation_upper[i + 1, :, :] = (
                -B_permutation_upper[i, :, :] @ LU_upper_diagonal_blocks_local[i, :, :]
            )

            # A_{i+1, top} = - L{i+1, i} @ U_{i, top}
            B_permutation_lower[i + 1, :, :] = (
                -LU_lower_diagonal_blocks_local[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # Update the top (first blocks) of the arrowhead
            # A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} @ U_{i, top}
            A_arrow_bottom_blocks_local[0, :, :] = (
                A_arrow_bottom_blocks_local[0, :, :]
                - LU_arrow_bottom_blocks_local[i, :, :] @ B_permutation_lower[i, :, :]
            )

            # A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} @ U_{i, ndb+1}
            A_arrow_right_blocks_local[0, :, :] = (
                A_arrow_right_blocks_local[0, :, :]
                - B_permutation_upper[i, :, :] @ LU_arrow_right_blocks_local[i, :, :]
            )

    # Get the tip blocks back on the host to perform the accumulation through MPI.
    Update_arrow_tip_block_host = cpx.empty_like_pinned(Update_arrow_tip_block)
    Update_arrow_tip_block.get(out=Update_arrow_tip_block_host)

    # Accumulate the distributed update of the arrow tip block
    MPI.COMM_WORLD.Allreduce(
        MPI.IN_PLACE,
        Update_arrow_tip_block_host,
        op=MPI.SUM,
    )

    Update_arrow_tip_block.set(arr=Update_arrow_tip_block_host)

    A_arrow_tip_block_global[:, :] += Update_arrow_tip_block[:, :]
    LU_arrow_tip_block_global = A_arrow_tip_block_global

    return (
        LU_diagonal_blocks_local,
        LU_lower_diagonal_blocks_local,
        LU_upper_diagonal_blocks_local,
        LU_arrow_bottom_blocks_local,
        LU_arrow_right_blocks_local,
        LU_arrow_tip_block_global,
        B_permutation_upper,
        B_permutation_lower,
    )
