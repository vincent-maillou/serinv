# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
)



def ddbtsci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    **kwargs,
):
    """Perform the selected-inversion of the Schur-complement of a block tridiagonal with arrowhead matrix.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal with arrowhead matrix.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal with arrowhead matrix.
    
    Keyword Arguments
    -----------------
    rhs : dict
        The right-hand side of the equation to solve. If given, the rhs dictionary
        must contain the following arrays:
        - B_diagonal_blocks : ArrayLike
            The diagonal blocks of the right-hand side.
        - B_lower_diagonal_blocks : ArrayLike
            The lower diagonal blocks of the right-hand side.
        - B_upper_diagonal_blocks : ArrayLike
            The upper diagonal blocks of the right-hand side.
    quadratic : bool
        If True, and a rhs is given, the Schur-complement is performed for the equation AXA^T=B.
        If False, and a rhs is given, the Schur-complement is performed for the equation AX=B.

    """
    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)

    if rhs is None:
        return _ddbtsci(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
        )
    else:
        # Check that rhs contains the correct arrays
        B_diagonal_blocks = rhs.get("B_diagonal_blocks", None)
        B_lower_diagonal_blocks = rhs.get("B_lower_diagonal_blocks", None)
        B_upper_diagonal_blocks = rhs.get("B_upper_diagonal_blocks", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        if quadratic:
            # Perform the schur-complement selected-inversion for AXA^T=B
            _ddbtsci_quadratic(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
            )
        else:
            # Perform the schur-complement selected-inversion for AX=B
            ...

def _ddbtsci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        temp_lower = xp.empty_like(A_lower_diagonal_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        temp_lower[:, :] = A_lower_diagonal_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i + 1]
            @ A_lower_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i]
        )

        A_upper_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ A_upper_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i + 1]
        )

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            - A_upper_diagonal_blocks[n_i] @ temp_lower @ A_diagonal_blocks[n_i]
        )
    

def _ddbtsci_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        temp_lower_retarded = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_lower_lesser = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_upper_lesser = xp.empty_like(A_upper_diagonal_blocks[0])

    temp_1 = xp.empty_like(A_diagonal_blocks[0])
    temp_2 = xp.empty_like(A_diagonal_blocks[0])
    temp_3 = xp.empty_like(A_diagonal_blocks[0])
    temp_4 = xp.empty_like(A_diagonal_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        temp_upper_lesser[:, :] = B_upper_diagonal_blocks[n_i]
        temp_1[:, :] = A_diagonal_blocks[n_i] @ A_upper_diagonal_blocks[n_i]
        temp_4[:, :] = A_lower_diagonal_blocks[n_i].T @ A_diagonal_blocks[n_i + 1].T

        B_upper_diagonal_blocks[n_i] = (
            -temp_1[:, :] @ B_diagonal_blocks[n_i + 1]
            - B_diagonal_blocks[n_i] @ temp_4[:, :]
            + A_diagonal_blocks[n_i]
            @ B_upper_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i + 1].T
        )

        temp_lower_lesser[:, :] = B_lower_diagonal_blocks[n_i]
        temp_2[:, :] = A_upper_diagonal_blocks[n_i].T @ A_diagonal_blocks[n_i].T
        temp_3[:, :] = A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]

        B_lower_diagonal_blocks[n_i] = (
            -B_diagonal_blocks[n_i + 1] @ temp_2[:, :]
            - temp_3[:, :] @ B_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i + 1]
            @ B_lower_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].T
        )

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + temp_1[:, :] @ B_diagonal_blocks[n_i + 1] @ temp_2[:, :]
            + temp_1[:, :] @ temp_3[:, :] @ B_diagonal_blocks[n_i]
            + B_diagonal_blocks[n_i].T @ temp_4[:, :] @ temp_2[:, :]
            - temp_1[:, :]
            @ A_diagonal_blocks[n_i + 1]
            @ temp_lower_lesser[:, :]
            @ A_diagonal_blocks[n_i].T
            - A_diagonal_blocks[n_i]
            @ temp_upper_lesser[:, :]
            @ A_diagonal_blocks[n_i + 1].T
            @ temp_2[:, :]
        )

        temp_lower_retarded[:, :] = A_lower_diagonal_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -temp_3[:, :] @ A_diagonal_blocks[n_i]
        A_upper_diagonal_blocks[n_i] = -temp_1[:, :] @ A_diagonal_blocks[n_i + 1]
        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            - A_upper_diagonal_blocks[n_i]
            @ temp_lower_retarded[:, :]
            @ A_diagonal_blocks[n_i]
        )