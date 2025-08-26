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
    """
    expected_kwargs = {"rhs", "quadratic", "buffers", "invert_last_block"}
    unexpected_kwargs = set(kwargs) - expected_kwargs
    if unexpected_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    invert_last_block: bool = kwargs.get("invert_last_block", True)

    if rhs is None:
        if buffers is None:
            # Perform a regular Schur-complement
            return _ddbtasc(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_lower_arrow_blocks,
                A_upper_arrow_blocks,
                A_arrow_tip_block,
                invert_last_block=invert_last_block,
            )
        else:
            # Perform a permuted Schur-complement
            A_lower_buffer_blocks = buffers.get("A_lower_buffer_blocks", None)
            A_upper_buffer_blocks = buffers.get("A_upper_buffer_blocks", None)
            if any(x is None for x in [A_lower_buffer_blocks, A_upper_buffer_blocks]):
                raise ValueError("buffers does not contain the correct arrays for A")
            return _ddbtasc_permuted(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_lower_arrow_blocks,
                A_upper_arrow_blocks,
                A_arrow_tip_block,
                A_lower_buffer_blocks,
                A_upper_buffer_blocks,
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
            if buffers is None:
                # Perform a regular Schur-complement ("quadratic")
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
                    invert_last_block=invert_last_block,
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
                _ddbtasc_quadratic_permuted(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                    A_lower_arrow_blocks,
                    A_upper_arrow_blocks,
                    A_arrow_tip_block,
                    A_lower_buffer_blocks,
                    A_upper_buffer_blocks,
                    B_diagonal_blocks,
                    B_lower_diagonal_blocks,
                    B_upper_diagonal_blocks,
                    B_lower_arrow_blocks,
                    B_upper_arrow_blocks,
                    B_arrow_tip_block,
                    B_lower_buffer_blocks,
                    B_upper_buffer_blocks,
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
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    2xMM(bbb)
    1xMM(abb)
    2xMM(bba)
    1xMM(aba)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(1, A_diagonal_blocks.shape[0]):
        # Invert previous diagonal block
        A_diagonal_blocks[n_i - 1] = xp.linalg.inv(
            A_diagonal_blocks[n_i - 1]
        )  # C: LU(b) + 2xTRSM(b)

        temp = (
            A_diagonal_blocks[n_i - 1] @ A_upper_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)
        # Update next diagonal block
        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i] - A_lower_diagonal_blocks[n_i - 1] @ temp
        )  # C: MM(bbb)

        # Update next lower arrow block
        A_lower_arrow_blocks[n_i] = (
            A_lower_arrow_blocks[n_i] - A_lower_arrow_blocks[n_i - 1] @ temp
        )  # C: MM(abb)

        temp = A_diagonal_blocks[n_i - 1] @ A_upper_arrow_blocks[n_i - 1]  # C: MM(bba)
        # Update next upper arrow block
        A_upper_arrow_blocks[n_i] = (
            A_upper_arrow_blocks[n_i] - A_lower_diagonal_blocks[n_i - 1] @ temp
        )  # C: MM(bba)

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:] - A_lower_arrow_blocks[n_i - 1] @ temp
        )  # C: MM(aba)

    if invert_last_block:
        A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])
        A_arrow_tip_block[:] = xp.linalg.inv(
            A_arrow_tip_block[:]
            - A_lower_arrow_blocks[-1]
            @ A_diagonal_blocks[-1]
            @ A_upper_arrow_blocks[-1]
        )


def _ddbtasc_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    A_lower_buffer_blocks: ArrayLike,
    A_upper_buffer_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    6xMM(bbb)
    3xMM(abb)
    2xMM(bba)
    1xMM(aba)
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

        # Update next lower arrow block
        temp_2 = A_lower_arrow_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(abb)
        A_lower_arrow_blocks[n_i + 1] = (
            A_lower_arrow_blocks[n_i + 1] - temp_2 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(abb)

        # Update next upper arrow block
        A_upper_arrow_blocks[n_i + 1] = (
            A_upper_arrow_blocks[n_i + 1] - temp_1 @ A_upper_arrow_blocks[n_i]
        )  # C: MM(bba)

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:] - temp_2 @ A_upper_arrow_blocks[n_i]
        )  # C: MM(aba)

        # --- Update of working buffer linked to permuted partition
        # Lower buffer block
        temp_3 = A_lower_buffer_blocks[n_i - 1] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i] = (
            -temp_3 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Upper buffer block
        A_upper_buffer_blocks[n_i] = (
            -temp_1 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # 0-diagonal block (first)
        A_diagonal_blocks[0] = (
            A_diagonal_blocks[0] - temp_3 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # 0-lower arrow block (first)
        A_lower_arrow_blocks[0] = (
            A_lower_arrow_blocks[0] - temp_2 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(abb)

        # 0-upper arrow block (first)
        A_upper_arrow_blocks[0] = (
            A_upper_arrow_blocks[0] - temp_3 @ A_upper_arrow_blocks[n_i]
        )  # C: MM(bba)


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
    invert_last_block: bool,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    8xMM(bbb)
    5xMM(abb)
    5xMM(bba)
    4xMM(aba)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    for n_i in range(1, A_diagonal_blocks.shape[0]):
        # Invert previous diagonal block of A
        A_diagonal_blocks[n_i - 1] = xp.linalg.inv(
            A_diagonal_blocks[n_i - 1]
        )  # C: LU(b) + 2xTRSM(b)

        # Update next diagonal block
        temp_a_diag = (
            A_lower_diagonal_blocks[n_i - 1] @ A_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)
        temp_a_diag_conjt = (
            temp_a_diag.conj().T
        )  # A_diagonal_blocks[n_i - 1].conj().T @ A_lower_diagonal_blocks[n_i - 1].conj().T

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i] - temp_a_diag @ A_upper_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)

        # Update next lower arrow block
        temp_a_arrow = (
            A_lower_arrow_blocks[n_i - 1] @ A_diagonal_blocks[n_i - 1]
        )  # C: MM(abb)
        temp_a_arrow_conjt = (
            temp_a_arrow.conj().T
        )  # A_diagonal_blocks[n_i - 1].conj().T @ A_lower_arrow_blocks[n_i - 1].conj().T

        A_lower_arrow_blocks[n_i] = (
            A_lower_arrow_blocks[n_i] - temp_a_arrow @ A_upper_diagonal_blocks[n_i - 1]
        )  # C: MM(abb)

        # Update next upper arrow block
        A_upper_arrow_blocks[n_i] = (
            A_upper_arrow_blocks[n_i] - temp_a_diag @ A_upper_arrow_blocks[n_i - 1]
        )  # C: MM(bba)

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:] - temp_a_arrow @ A_upper_arrow_blocks[n_i - 1]
        )  # C: MM(aba)

        # --- Xl ---
        # Inverse previous diagonal block of B
        B_diagonal_blocks[n_i - 1] = (
            A_diagonal_blocks[n_i - 1]
            @ B_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1].conj().T
        )  # C: 2xMM(bbb)

        temp_b_diag = (
            B_diagonal_blocks[n_i - 1] @ A_lower_diagonal_blocks[n_i - 1].conj().T
        )  # C: MM(bbb)
        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + A_lower_diagonal_blocks[n_i - 1] @ temp_b_diag
            - B_lower_diagonal_blocks[n_i - 1] @ temp_a_diag_conjt
            - temp_a_diag @ B_upper_diagonal_blocks[n_i - 1]
        )  # C: 3xMM(bbb)

        temp_b_arrow = (
            B_diagonal_blocks[n_i - 1] @ A_lower_arrow_blocks[n_i - 1].conj().T
        )  # C: MM(bba)
        B_upper_arrow_blocks[n_i] = (
            B_upper_arrow_blocks[n_i]
            + A_lower_diagonal_blocks[n_i - 1] @ temp_b_arrow
            - B_lower_diagonal_blocks[n_i - 1] @ temp_a_arrow_conjt
            - temp_a_diag @ B_upper_arrow_blocks[n_i - 1]
        )  # C: 3xMM(bba)

        B_lower_arrow_blocks[n_i] = (
            B_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[n_i - 1] @ temp_b_diag
            - B_lower_arrow_blocks[n_i - 1] @ temp_a_diag_conjt
            - temp_a_arrow @ B_upper_diagonal_blocks[n_i - 1]
        )  # C: 3xMM(abb)

        B_arrow_tip_block[:, :] = (
            B_arrow_tip_block
            + A_lower_arrow_blocks[n_i - 1] @ temp_b_arrow
            - B_lower_arrow_blocks[n_i - 1] @ temp_a_arrow_conjt
            - temp_a_arrow @ B_upper_arrow_blocks[n_i - 1]
        )  # C: 3xMM(aba)

    if invert_last_block:
        A_diagonal_blocks[-1] = xp.linalg.inv(A_diagonal_blocks[-1])
        B_diagonal_blocks[-1] = (
            A_diagonal_blocks[-1] @ B_diagonal_blocks[-1] @ A_diagonal_blocks[-1].conj().T
        )

        temp_a_arrow = A_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1]
        A_arrow_tip_block[:] = xp.linalg.inv(
            A_arrow_tip_block[:] - temp_a_arrow @ A_upper_arrow_blocks[-1]
        )
        B_arrow_tip_block[:] = (
            A_arrow_tip_block[:]
            @ (
                B_arrow_tip_block[:]
                + A_lower_arrow_blocks[-1]
                @ B_diagonal_blocks[-1]
                @ A_lower_arrow_blocks[-1].conj().T
                - B_lower_arrow_blocks[-1] @ temp_a_arrow.conj().T
                - temp_a_arrow @ B_upper_arrow_blocks[-1]
            )
            @ A_arrow_tip_block[:].conj().T
        )


def _ddbtasc_quadratic_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    A_lower_buffer_blocks: ArrayLike,
    A_upper_buffer_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
    B_lower_arrow_blocks: ArrayLike,
    B_upper_arrow_blocks: ArrayLike,
    B_arrow_tip_block: ArrayLike,
    B_lower_buffer_blocks: ArrayLike,
    B_upper_buffer_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    1xLU(b)
    2xTRSM(b)
    22xMM(bbb)
    10xMM(abb)
    8xMM(bba)
    4xMM(aba)
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
        temp_1_conjt = temp_1.conj().T
        A_diagonal_blocks[n_i + 1] = (
            A_diagonal_blocks[n_i + 1] - temp_1 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Update next lower arrow block
        temp_a_arrow = A_lower_arrow_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(abb)
        temp_a_arrow_conjt = (
            temp_a_arrow.conj().T
        )  # A_diagonal_blocks[n_i].conj().T @ A_lower_arrow_blocks[n_i].conj().T

        A_lower_arrow_blocks[n_i + 1] = (
            A_lower_arrow_blocks[n_i + 1] - temp_a_arrow @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(abb)

        # Update next upper arrow block
        A_upper_arrow_blocks[n_i + 1] = (
            A_upper_arrow_blocks[n_i + 1] - temp_1 @ A_upper_arrow_blocks[n_i]
        )  # C: MM(bba)

        # Update tip arrow block
        A_arrow_tip_block[:] = (
            A_arrow_tip_block[:] - temp_a_arrow @ A_upper_arrow_blocks[n_i]
        )  # C: MM(aba)

        # --- Update of working buffer linked to permuted partition
        # Lower buffer block
        temp_2 = A_lower_buffer_blocks[n_i - 1] @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        temp_2_conjt = temp_2.conj().T
        A_lower_buffer_blocks[n_i] = (
            -temp_2 @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        # Upper buffer block
        A_upper_buffer_blocks[n_i] = (
            -temp_1 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # 0-diagonal block (first)
        A_diagonal_blocks[0] = (
            A_diagonal_blocks[0] - temp_2 @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(bbb)

        # 0-lower arrow block (first)
        A_lower_arrow_blocks[0] = (
            A_lower_arrow_blocks[0] - temp_a_arrow @ A_upper_buffer_blocks[n_i - 1]
        )  # C: MM(abb)

        # 0-upper arrow block (first)
        A_upper_arrow_blocks[0] = (
            A_upper_arrow_blocks[0] - temp_2 @ A_upper_arrow_blocks[n_i]
        )  # C: MM(bba)

        # --- Xl ---
        # Inverse current diagonal block
        B_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            @ B_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        # Update next diagonal block
        temp_3 = A_lower_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i]  # C: MM(bbb)
        B_diagonal_blocks[n_i + 1] = (
            B_diagonal_blocks[n_i + 1]
            + temp_3 @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_1_conjt
            - temp_1 @ B_upper_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb)

        # Update next lower arrow block
        temp_4 = A_lower_arrow_blocks[n_i] @ B_diagonal_blocks[n_i]  # C: MM(abb)
        B_lower_arrow_blocks[n_i + 1] = (
            B_lower_arrow_blocks[n_i + 1]
            + temp_4 @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_arrow_blocks[n_i] @ temp_1_conjt
            - temp_a_arrow @ B_upper_diagonal_blocks[n_i]
        )  # C: 3xMM(abb)

        # Update next upper arrow block
        B_upper_arrow_blocks[n_i + 1] = (
            B_upper_arrow_blocks[n_i + 1]
            + temp_3 @ A_lower_arrow_blocks[n_i].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_a_arrow_conjt
            - temp_1 @ B_upper_arrow_blocks[n_i]
        )  # C: 3xMM(bba)

        # Update tip arrow block
        B_arrow_tip_block[:, :] = (
            B_arrow_tip_block
            + temp_4 @ A_lower_arrow_blocks[n_i].conj().T
            - B_lower_arrow_blocks[n_i] @ temp_a_arrow_conjt
            - temp_a_arrow @ B_upper_arrow_blocks[n_i]
        )  # C: 3xMM(aba)

        # --- Update of working buffer linked to permuted partition
        # Lower buffer block
        temp_5 = A_lower_buffer_blocks[n_i - 1] @ B_diagonal_blocks[n_i]  # C: MM(bbb)
        B_lower_buffer_blocks[n_i] = (
            B_lower_buffer_blocks[n_i]
            + temp_5 @ A_lower_diagonal_blocks[n_i].conj().T
            - B_lower_buffer_blocks[n_i - 1] @ temp_1_conjt
            - temp_2 @ B_upper_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb)

        # Upper buffer block
        B_upper_buffer_blocks[n_i] = (
            B_upper_buffer_blocks[n_i]
            + temp_3 @ A_lower_buffer_blocks[n_i - 1].conj().T
            - B_lower_diagonal_blocks[n_i] @ temp_2_conjt
            - temp_1 @ B_upper_buffer_blocks[n_i - 1]
        )  # C: 3xMM(bbb)

        # 0-diagonal block (first)
        B_diagonal_blocks[0] = (
            B_diagonal_blocks[0]
            + temp_5 @ A_lower_buffer_blocks[n_i - 1].conj().T
            - B_lower_buffer_blocks[n_i - 1] @ temp_2_conjt
            - temp_2 @ B_upper_buffer_blocks[n_i - 1]
        )  # C: 3xMM(bbb)

        # 0-lower arrow block (first)
        B_lower_arrow_blocks[0] = (
            B_lower_arrow_blocks[0]
            + temp_4 @ A_lower_buffer_blocks[n_i - 1].conj().T
            - B_lower_arrow_blocks[n_i] @ temp_2_conjt
            - temp_a_arrow @ B_upper_buffer_blocks[n_i - 1]
        )  # C: 3xMM(abb)

        # 0-upper arrow block (first)
        B_upper_arrow_blocks[0] = (
            B_upper_arrow_blocks[0]
            + temp_5 @ A_lower_arrow_blocks[n_i].conj().T
            - B_lower_buffer_blocks[n_i - 1] @ temp_a_arrow_conjt
            - temp_2 @ B_upper_arrow_blocks[n_i]
        )  # C: 3xMM(bba)
