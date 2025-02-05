# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
)



def ddbtasc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
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
    A_lower_arrow_blocks : ArrayLike
        The arrow bottom blocks of the block tridiagonal with arrowhead matrix.
    A_upper_arrow_blocks : ArrayLike
        The upper arrow blocks of the block tridiagonal with arrowhead matrix.
    A_arrow_tip_block : ArrayLike
        The arrow tip block of the block tridiagonal with arrowhead matrix.
    
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
        - B_lower_arrow_blocks : ArrayLike
            The arrow bottom blocks of the right-hand side.
        - B_upper_arrow_blocks : ArrayLike
            The upper arrow blocks of the right-hand side.
        - B_arrow_tip_block : ArrayLike
            The arrow tip block of the right-hand side.
    quadratic : bool
        If True, and a rhs is given, the Schur-complement is performed for the equation AXA^T=B.
        If False, and a rhs is given, the Schur-complement is performed for the equation AX=B.

    """
    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)

    if rhs is None:
        return _ddbtasc(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_lower_arrow_blocks,
            A_upper_arrow_blocks,
            A_arrow_tip_block,
        )
    else:
        # Check that rhs contains the correct arrays
        B_diagonal_blocks = rhs.get("B_diagonal_blocks", None)
        B_lower_diagonal_blocks = rhs.get("B_lower_diagonal_blocks", None)
        B_upper_diagonal_blocks = rhs.get("B_upper_diagonal_blocks", None)
        B_lower_arrow_blocks = rhs.get("B_lower_arrow_blocks", None)
        B_upper_arrow_blocks = rhs.get("B_upper_arrow_blocks", None)
        B_arrow_tip_block = rhs.get("B_arrow_tip_block", None)
        if any(
            x is None
            for x in [
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
                B_lower_arrow_blocks,
                B_upper_arrow_blocks,
                B_arrow_tip_block,
            ]
        ):
            raise ValueError("rhs does not contain the correct arrays")
        if quadratic:
            # Perform the schur-complement for AXA^T=B
            _ddbtasc_quadratic(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_lower_arrow_blocks,
                A_upper_arrow_blocks,
                A_arrow_tip_block,
                B_diagonal_blocks,
                B_lower_diagonal_blocks,
                B_upper_diagonal_blocks,
                B_lower_arrow_blocks,
                B_upper_arrow_blocks,
                B_arrow_tip_block,
            )
        else:
            # Perform the schur-complement for AX=B
            ...

def _ddbtasc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(1, A_diagonal_blocks.shape[0]):
        # Invert previous diagonal block
        A_diagonal_blocks[n_i - 1] = xp.linalg.inv(A_diagonal_blocks[n_i - 1])

        # Update next diagonal block
        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            - A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_diagonal_blocks[n_i - 1]
        )

        # Update next lower arrow block
        A_lower_arrow_blocks[n_i] = (
            A_lower_arrow_blocks[n_i]
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_diagonal_blocks[n_i - 1]
        )

        # Update next upper arrow block
        A_upper_arrow_blocks[n_i] = (
            A_upper_arrow_blocks[n_i]
            - A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_arrow_blocks[n_i - 1]
        )

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:]
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_arrow_blocks[n_i - 1]
        )

    A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])
    A_arrow_tip_block[:] = xp.linalg.inv(
        A_arrow_tip_block[:]
        - A_lower_arrow_blocks[-1]
        @ A_diagonal_blocks[-1]
        @ A_upper_arrow_blocks[-1]
    )

def _ddbtasc_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
    B_lower_arrow_blocks: ArrayLike,
    B_upper_arrow_blocks: ArrayLike,
    B_arrow_tip_block: ArrayLike,
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    A_diagonal_blocks[0] = xp.linalg.inv(A_diagonal_blocks[0])
    B_diagonal_blocks[0] = (
        A_diagonal_blocks[0] @ B_diagonal_blocks[0] @ A_diagonal_blocks[0].T
    )

    for n_i in range(1, A_diagonal_blocks.shape[0]):
        # --- Xr ---
        # Update next diagonal block
        A_diagonal_blocks[n_i] = xp.linalg.inv(
            A_diagonal_blocks[n_i]
            - A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_diagonal_blocks[n_i - 1]
        )

        # Update next lower arrow block
        A_lower_arrow_blocks[n_i] = (
            A_lower_arrow_blocks[n_i]
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_diagonal_blocks[n_i - 1]
        )

        # Update next upper arrow block
        A_upper_arrow_blocks[n_i] = (
            A_upper_arrow_blocks[n_i]
            - A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_arrow_blocks[n_i - 1]
        )

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:]
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ A_upper_arrow_blocks[n_i - 1]
        )

        # --- Xl ---
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            @ (
                B_diagonal_blocks[n_i]
                + A_lower_diagonal_blocks[n_i - 1]
                @ B_diagonal_blocks[n_i - 1]
                @ A_lower_diagonal_blocks[n_i - 1].T
                - B_lower_diagonal_blocks[n_i - 1]
                @ A_diagonal_blocks[n_i - 1].T
                @ A_lower_diagonal_blocks[n_i - 1].T
                - A_lower_diagonal_blocks[n_i - 1]
                @ A_diagonal_blocks[n_i - 1]
                @ B_upper_diagonal_blocks[n_i - 1]
            )
            @ A_diagonal_blocks[n_i].T
        )

        B_upper_arrow_blocks[n_i] = (
            B_upper_arrow_blocks[n_i]
            + A_lower_diagonal_blocks[n_i - 1]
            @ B_diagonal_blocks[n_i - 1]
            @ A_lower_arrow_blocks[n_i - 1].T
            - B_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1].T
            @ A_lower_arrow_blocks[n_i - 1].T
            - A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ B_upper_arrow_blocks[n_i - 1]
        )

        B_lower_arrow_blocks[n_i] = (
            B_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[n_i - 1]
            @ B_diagonal_blocks[n_i - 1]
            @ A_lower_diagonal_blocks[n_i - 1].T
            - B_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1].T
            @ A_lower_diagonal_blocks[n_i - 1].T
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ B_upper_diagonal_blocks[n_i - 1]
        )

        B_arrow_tip_block[:, :] = (
            B_arrow_tip_block
            + A_lower_arrow_blocks[n_i - 1]
            @ B_diagonal_blocks[n_i - 1]
            @ A_lower_arrow_blocks[n_i - 1].T
            - B_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1].T
            @ A_lower_arrow_blocks[n_i - 1].T
            - A_lower_arrow_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
            @ B_upper_arrow_blocks[n_i - 1]
        )

    A_arrow_tip_block[:] = xp.linalg.inv(
        A_arrow_tip_block[:]
        - A_lower_arrow_blocks[-1]
        @ A_diagonal_blocks[-1]
        @ A_upper_arrow_blocks[-1]
    )
    B_arrow_tip_block[:] = (
        A_arrow_tip_block[:]
        @ (
            B_arrow_tip_block[:]
            + A_lower_arrow_blocks[-1]
            @ B_diagonal_blocks[-1]
            @ A_lower_arrow_blocks[-1].T
            - B_lower_arrow_blocks[-1]
            @ A_diagonal_blocks[-1].T
            @ A_lower_arrow_blocks[-1].T
            - A_lower_arrow_blocks[-1]
            @ A_diagonal_blocks[-1]
            @ B_upper_arrow_blocks[-1]
        )
        @ A_arrow_tip_block[:].T
    )