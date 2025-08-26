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
    buffers : dict
        The buffers to use in the permuted Schur-complement algorithm. If buffers are given, the
        Schur-complement is performed using the permuted Schur-complement algorithm.
        In the case of the `AX=I` equation the following buffers are required:
        - A_lower_buffer_blocks : ArrayLike
            The lower buffer blocks of the matrix A.
        - A_upper_buffer_blocks : ArrayLike
            The upper buffer blocks of the matrix A.
        In the case of `AXA^T=B` equation the following buffers are required:
        - B_lower_buffer_blocks : ArrayLike
            The lower buffer blocks of the matrix B.
        - B_upper_buffer_blocks : ArrayLike
            The upper buffer blocks of the matrix B.
    direction : str
        The direction in which to perform the Schur-complement. Can be either "downward" (default) or "upward".
        This doesn't matter for performances but it is usefull in the case of the parallel implementation which last
        process needs to perform the Schur complement upward.
    """
    expected_kwargs = {"rhs", "quadratic", "buffers", "invert_last_block", "direction"}
    unexpected_kwargs = set(kwargs) - expected_kwargs
    if unexpected_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    invert_last_block: bool = kwargs.get("invert_last_block", True)
    direction: str = kwargs.get("direction", "downward")

    if rhs is None:
        if buffers is None:
            # Perform a regular Schur-complement
            if direction == "downward":
                _ddbtsc(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                    invert_last_block=invert_last_block,
                )
            elif direction == "upward":
                _ddbtsc_upward(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                    invert_last_block=invert_last_block,
                )
            else:
                raise ValueError(
                    "Optional keyword argument `direction` must be either 'downward' or 'upward'"
                )
        else:
            # Perform a permuted Schur-complement
            A_lower_buffer_blocks = buffers.get("A_lower_buffer_blocks", None)
            A_upper_buffer_blocks = buffers.get("A_upper_buffer_blocks", None)
            if any(x is None for x in [A_lower_buffer_blocks, A_upper_buffer_blocks]):
                raise ValueError("buffers does not contain the correct arrays for A")
            _ddbtsc_permuted(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_lower_buffer_blocks,
                A_upper_buffer_blocks,
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
            if buffers is None:
                # Perform a regular Schur-complement ("quadratic")
                if direction == "downward":
                    _ddbtsc_quadratic(
                        A_diagonal_blocks,
                        A_lower_diagonal_blocks,
                        A_upper_diagonal_blocks,
                        B_diagonal_blocks,
                        B_lower_diagonal_blocks,
                        B_upper_diagonal_blocks,
                        invert_last_block=invert_last_block,
                    )
                elif direction == "upward":
                    _ddbtsc_upward_quadratic(
                        A_diagonal_blocks,
                        A_lower_diagonal_blocks,
                        A_upper_diagonal_blocks,
                        B_diagonal_blocks,
                        B_lower_diagonal_blocks,
                        B_upper_diagonal_blocks,
                        invert_last_block=invert_last_block,
                    )
                else:
                    raise ValueError(
                        "Optional keyword argument `direction` must be either 'downward' or 'upward'"
                    )
            else:
                # Perform a permuted Schur-complement ("quadratic")
                A_lower_buffer_blocks = buffers.get("A_lower_buffer_blocks", None)
                A_upper_buffer_blocks = buffers.get("A_upper_buffer_blocks", None)
                if any(
                    x is None for x in [A_lower_buffer_blocks, A_upper_buffer_blocks]
                ):
                    raise ValueError(
                        "buffers does not contain the correct arrays for A"
                    )
                B_lower_buffer_blocks = buffers.get("B_lower_buffer_blocks", None)
                B_upper_buffer_blocks = buffers.get("B_upper_buffer_blocks", None)
                if any(
                    x is None for x in [B_lower_buffer_blocks, B_upper_buffer_blocks]
                ):
                    raise ValueError(
                        "buffers does not contain the correct arrays for B"
                    )
                _ddbtsc_quadratic_permuted(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                    A_lower_buffer_blocks,
                    A_upper_buffer_blocks,
                    B_diagonal_blocks,
                    B_lower_diagonal_blocks,
                    B_upper_diagonal_blocks,
                    B_lower_buffer_blocks,
                    B_upper_buffer_blocks,
                )
        else:
            # Perform the schur-complement for AX=B
            ...


def _ddbtsc(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    2xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(0, A_diagonal_blocks.shape[0] - 1):
        # C: LU(b) + 2xTRSM(b)
        A_diagonal_blocks[n_i] = xp.linalg.inv(A_diagonal_blocks[n_i])

        # C: 2xMM(bbb)
        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1]
            - A_lower_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i]
            @ A_upper_diagonal_blocks[n_i]
        )

    if invert_last_block:
        A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])


def _ddbtsc_upward(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    2xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(A_diagonal_blocks.shape[0] - 1, 0, -1):
        # C: LU(b) + 2xTRSM(b)
        A_diagonal_blocks[n_i] = xp.linalg.inv(A_diagonal_blocks[n_i])

        # C: 2xMM(bbb)
        A_diagonal_blocks[n_i - 1] = (
            A_diagonal_blocks[n_i - 1]
            - A_upper_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i]
            @ A_lower_diagonal_blocks[n_i - 1]
        )

    if invert_last_block:
        A_diagonal_blocks[0] = xp.linalg.inv(A_diagonal_blocks[0])


def _ddbtsc_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_buffer_blocks: ArrayLike,
    A_upper_buffer_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    6xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    A_lower_buffer_blocks[0] = A_upper_diagonal_blocks[0]
    A_upper_buffer_blocks[0] = A_lower_diagonal_blocks[0]

    for n_i in range(1, A_diagonal_blocks.shape[0] - 1):
        # Inverse current diagonal block
        A_diagonal_blocks[n_i] = xp.linalg.inv(
            A_diagonal_blocks[n_i]
        )  # C: LU(b) + 2xTRSM(b)

        # Update next diagonal block
        temp_1 = A_lower_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1] - temp_1 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Update lower buffer block
        temp_2 = A_lower_buffer_blocks[n_i - 1] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i] = (
            -temp_2 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Update upper buffer block
        A_upper_buffer_blocks[n_i] = (
            -temp_1 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # Update 0-block (first)
        A_diagonal_blocks[0] = (
            A_diagonal_blocks[0] - temp_2 @ A_upper_buffer_blocks[n_i - 1]  # C: MM(bbb)
        )


def _ddbtsc_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    8xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(0, A_diagonal_blocks.shape[0] - 1):
        A_diagonal_blocks[n_i] = xp.linalg.inv(
            A_diagonal_blocks[n_i]
        )  # C: LU(b) + 2xTRSM(b)
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        temp_1 = A_lower_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        temp_2 = (
            temp_1.conj().T
        )  # A_diagonal_blocks[n_i].conj().T @ A_lower_diagonal_blocks[n_i].conj().T

        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1] - temp_1 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        B_diagonal_blocks[n_i + 1] = (
            B_diagonal_blocks[n_i + 1]
            + A_lower_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_2
            - temp_1 @ B_upper_diagonal_blocks[n_i]
        )  # C: 4xMM(bbb)

    if invert_last_block:
        A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])
        B_diagonal_blocks[-1] = (
            A_diagonal_blocks[-1]
            @ B_diagonal_blocks[-1]
            @ A_diagonal_blocks[-1].conj().T
        )


def _ddbtsc_upward_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    8xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(A_diagonal_blocks.shape[0] - 1, 0, -1):
        A_diagonal_blocks[n_i] = xp.linalg.inv(
            A_diagonal_blocks[n_i]
        )  # C: LU(b) + 2xTRSM(b)
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        temp_1 = A_upper_diagonal_blocks[n_i - 1] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        temp_2 = temp_1.conj().T

        A_diagonal_blocks[n_i - 1] = (
            A_diagonal_blocks[n_i - 1] - temp_1 @ A_lower_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)

        B_diagonal_blocks[n_i - 1] = (
            B_diagonal_blocks[n_i - 1]
            + A_upper_diagonal_blocks[n_i - 1]
            @ B_diagonal_blocks[n_i]
            @ A_upper_diagonal_blocks[n_i - 1].conj().T
            - B_upper_diagonal_blocks[n_i - 1] @ temp_2
            - temp_1 @ B_lower_diagonal_blocks[n_i - 1]
        )  # C: 4xMM(bbb)

    if invert_last_block:
        A_diagonal_blocks[0] = xp.linalg.inv(A_diagonal_blocks[0])
        B_diagonal_blocks[0] = (
            A_diagonal_blocks[0] @ B_diagonal_blocks[0] @ A_diagonal_blocks[0].conj().T
        )


def _ddbtsc_quadratic_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_buffer_blocks: ArrayLike,
    A_upper_buffer_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
    B_lower_buffer_blocks: ArrayLike,
    B_upper_buffer_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    22xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    A_lower_buffer_blocks[0] = A_upper_diagonal_blocks[0]
    A_upper_buffer_blocks[0] = A_lower_diagonal_blocks[0]

    B_lower_buffer_blocks[0] = B_upper_diagonal_blocks[0]
    B_upper_buffer_blocks[0] = B_lower_diagonal_blocks[0]

    for n_i in range(1, A_diagonal_blocks.shape[0] - 1):
        # Inverse current diagonal block
        A_diagonal_blocks[n_i] = xp.linalg.inv(
            A_diagonal_blocks[n_i]
        )  # C: LU(b) + 2xTRSM(b)

        # Update next diagonal block
        temp_1 = A_lower_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1] - temp_1 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Update lower buffer block
        temp_2 = A_lower_buffer_blocks[n_i - 1] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i] = (
            -temp_2 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Update upper buffer block
        A_upper_buffer_blocks[n_i] = (
            -temp_1 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # Update 0-block (first)
        A_diagonal_blocks[0] = (
            A_diagonal_blocks[0] - temp_2 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # --- Xl ---
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        temp_1_conjt = temp_1.conj().T
        temp_3 = A_lower_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i]  # C: MM(bbb)
        B_diagonal_blocks[n_i + 1] = (
            B_diagonal_blocks[n_i + 1]
            + temp_3 @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_1_conjt
            - temp_1 @ B_upper_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb)

        temp_2_conjt = temp_2.conj().T
        B_upper_buffer_blocks[n_i] = (
            B_upper_buffer_blocks[n_i]
            + temp_3 @ A_lower_buffer_blocks[n_i - 1].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_2_conjt
            - temp_1 @ B_upper_buffer_blocks[n_i - 1]
        )  # C: 3xMM(bbb)

        temp_4 = A_lower_buffer_blocks[n_i - 1] @ B_diagonal_blocks[n_i]  # C: MM(bbb)
        B_lower_buffer_blocks[n_i] = (
            B_lower_buffer_blocks[n_i]
            + temp_4 @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_buffer_blocks[n_i - 1] @ temp_1_conjt
            - temp_2 @ B_upper_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb)

        B_diagonal_blocks[0] = (
            B_diagonal_blocks[0]
            + temp_4 @ A_lower_buffer_blocks[n_i - 1].conj().T
            - B_lower_buffer_blocks[n_i - 1] @ temp_2_conjt
            - temp_2 @ B_upper_buffer_blocks[n_i - 1]
        )  # C: 3xMM(bbb)
