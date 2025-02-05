# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
)



def ddbtsc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    **kwargs,
):
    """Perform the Schur-complement for a block tridiagonal with arrowhead matrix.
    
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
        return _ddbtsc(
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
            # Perform the schur-complement for AXA^T=B
            _ddbtsc_quadratic(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
            )
        else:
            # Perform the schur-complement for AX=B
            ...

def _ddbtsc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(0, A_diagonal_blocks.shape[0] - 1):
        A_diagonal_blocks[n_i] = xp.linalg.inv(A_diagonal_blocks[n_i])

        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1]
            - A_lower_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i]
            @ A_upper_diagonal_blocks[n_i]
        )

    A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])

def _ddbtsc_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    temp_1 = xp.empty_like(A_diagonal_blocks[0])
    temp_2 = xp.empty_like(A_diagonal_blocks[0])

    for n_i in range(0, A_diagonal_blocks.shape[0] - 1):
        A_diagonal_blocks[n_i] = xp.linalg.inv(A_diagonal_blocks[n_i])
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i].T
        )

        temp_1[:, :] = A_lower_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i]
        temp_2[:, :] = A_diagonal_blocks[n_i].T @ A_lower_diagonal_blocks[n_i].T

        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1] - temp_1[:, :] @ A_upper_diagonal_blocks[n_i]
        )

        B_diagonal_blocks[n_i + 1] = (
            B_diagonal_blocks[n_i + 1]
            + A_lower_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_lower_diagonal_blocks[n_i].T
            - B_lower_diagonal_blocks[n_i] @ temp_2[:, :]
            - temp_1[:, :] @ B_upper_diagonal_blocks[n_i]
        )

    A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])
    B_diagonal_blocks[-1] = (
        A_diagonal_blocks[-1] @ B_diagonal_blocks[-1] @ A_diagonal_blocks[-1].T
    )