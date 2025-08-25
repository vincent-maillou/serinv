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
    """Perform the selected-inversion of the Schur-complement of a block tridiagonal.
    This routine is refered to as Schur-complement Selected Inversion (SCI).

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        The diagonal blocks of the block tridiagonal.
    A_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the block tridiagonal.
    A_upper_diagonal_blocks : ArrayLike
        The upper diagonal blocks of the block tridiagonal.

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
        If True, and a rhs is given, the SCI is performed for the equation AXA^T=B.
        If False, and a rhs is given, the SCI is performed for the equation AX=B.
    buffers : dict
        The buffers to use in the permuted SCI algorithm.
        If buffers are given, the SCI is performed using the permuted SCI algorithm.
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
        The direction in which to perform the SCI. Can be either "downward" (default) or "upward".
        This doesn't matter for performances but it is usefull in the case of the parallel implementation which last
        process needs to perform the Schur complement upward.
    """
    expected_kwargs = {"rhs", "quadratic", "buffers", "direction"}
    unexpected_kwargs = set(kwargs) - expected_kwargs
    if unexpected_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_kwargs}")

    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)
    direction: str = kwargs.get("direction", "downward")

    if rhs is None:
        if buffers is None:
            # Perform a regular SCI
            if direction == "downward":
                _ddbtsci(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                )
            elif direction == "upward":
                _ddbtsci_upward(
                    A_diagonal_blocks,
                    A_lower_diagonal_blocks,
                    A_upper_diagonal_blocks,
                )
            else:
                raise ValueError(
                    "Optional keyword argument `direction` must be either 'downward' or 'upward'"
                )
        else:
            # Perform a permuted SCI
            A_lower_buffer_blocks = buffers.get("A_lower_buffer_blocks", None)
            A_upper_buffer_blocks = buffers.get("A_upper_buffer_blocks", None)
            if any(x is None for x in [A_lower_buffer_blocks, A_upper_buffer_blocks]):
                raise ValueError("buffers does not contain the correct arrays for A")
            _ddbtsci_permuted(
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
            # Perform the SCI for AXA^T=B
            if buffers is None:
                # Perform a regular SCI ("quadratic")
                if direction == "downward":
                    _ddbtsci_quadratic(
                        A_diagonal_blocks,
                        A_lower_diagonal_blocks,
                        A_upper_diagonal_blocks,
                        B_diagonal_blocks,
                        B_lower_diagonal_blocks,
                        B_upper_diagonal_blocks,
                    )
                elif direction == "upward":
                    _ddbtsci_upward_quadratic(
                        A_diagonal_blocks,
                        A_lower_diagonal_blocks,
                        A_upper_diagonal_blocks,
                        B_diagonal_blocks,
                        B_lower_diagonal_blocks,
                        B_upper_diagonal_blocks,
                    )
                else:
                    raise ValueError(
                        "Optional keyword argument `direction` must be either 'downward' or 'upward'"
                    )
            else:
                # Perform a permuted SCI ("quadratic")
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
                _ddbtsci_quadratic_permuted(
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
            # Perform the SCI for AX=B
            ...


def _ddbtsci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    5xMM(bbb)
    """
    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        temp_lower = A_lower_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i]  # C: MM(bbb)

        A_lower_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i + 1] @ temp_lower
        )  # C: MM(bbb)

        A_upper_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ A_upper_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i + 1]
        )  # C: 2xMM(bbb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i] - A_upper_diagonal_blocks[n_i] @ temp_lower
        )  # C: MM(bbb)


def _ddbtsci_upward(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    5xMM(bbb)
    """
    for n_i in range(1, A_diagonal_blocks.shape[0]):
        temp_upper = (
            A_upper_diagonal_blocks[n_i - 1] @ A_diagonal_blocks[n_i]
        )  # C: MM(bbb)

        A_lower_diagonal_blocks[n_i - 1] = (
            -A_diagonal_blocks[n_i]
            @ A_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1]
        )  # C: 2xMM(bbb)

        A_upper_diagonal_blocks[n_i - 1] = (
            -A_diagonal_blocks[n_i - 1] @ temp_upper
        )  # C: MM(bbb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i] - A_lower_diagonal_blocks[n_i - 1] @ temp_upper
        )  # C: MM(bbb)


def _ddbtsci_permuted(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_buffer_blocks: ArrayLike,
    A_upper_buffer_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    16xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    B1 = xp.empty_like(A_lower_diagonal_blocks[0])
    B2 = xp.empty_like(A_upper_buffer_blocks[0])

    C1 = xp.empty_like(A_upper_diagonal_blocks[0])
    C2 = xp.empty_like(A_lower_buffer_blocks[0])

    D1 = xp.empty_like(A_lower_diagonal_blocks[0])
    D2 = xp.empty_like(A_lower_buffer_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, 0, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_buffer_blocks[n_i - 1] @ A_lower_buffer_blocks[n_i]
        )  # C: 2xMM(bbb)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
        )  # C: 2xMM(bbb)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb)

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb)

        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bbb)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i - 1] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(bbb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i] @ (B1 @ D1 + B2 @ D2) @ A_diagonal_blocks[n_i]
        )  # C: 4xMM(bbb)

    A_lower_diagonal_blocks[0] = A_upper_buffer_blocks[0]
    A_upper_diagonal_blocks[0] = A_lower_buffer_blocks[0]


def _ddbtsci_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    19xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        temp_lower_retarded = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_lower_lesser = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_upper_lesser = xp.empty_like(A_upper_diagonal_blocks[0])

    temp_1 = xp.empty_like(A_diagonal_blocks[0])
    temp_2 = xp.empty_like(A_diagonal_blocks[0])
    temp_3 = xp.empty_like(A_diagonal_blocks[0])
    temp_3_b = xp.empty_like(A_diagonal_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        temp_1[:, :] = (
            A_diagonal_blocks[n_i] @ A_upper_diagonal_blocks[n_i]
        )  # C: MM(bbb)
        temp_2[:, :] = temp_1.conj().T
        temp_2_b = B_diagonal_blocks[n_i + 1] @ temp_2  # C: MM(bbb)
        temp_3[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
        )  # C: MM(bbb)
        temp_3_b[:, :] = B_diagonal_blocks[n_i] @ temp_3.conj().T  # C: MM(bbb)
        temp_4 = temp_3 @ B_diagonal_blocks[n_i]  # C: MM(bbb)

        temp_upper_lesser[:, :] = (
            A_diagonal_blocks[n_i]
            @ B_upper_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i + 1].conj().T
        )  # C: 2xMM(bbb)

        B_upper_diagonal_blocks[n_i] = (
            -temp_1 @ B_diagonal_blocks[n_i + 1] - temp_3_b + temp_upper_lesser
        )  # C: MM(bbb)

        temp_lower_lesser[:, :] = (
            A_diagonal_blocks[n_i + 1]
            @ B_lower_diagonal_blocks[n_i]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        B_lower_diagonal_blocks[n_i] = -temp_2_b - temp_4 + temp_lower_lesser

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + temp_1 @ temp_2_b
            + temp_1 @ temp_4
            + temp_3_b @ temp_2
            - temp_1 @ temp_lower_lesser
            - temp_upper_lesser @ temp_2
        )  # C: 5xMM(bbb)

        temp_lower_retarded[:, :] = A_lower_diagonal_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -temp_3 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_upper_diagonal_blocks[n_i] = (
            -temp_1 @ A_diagonal_blocks[n_i + 1]
        )  # C: MM(bbb)
        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            - A_upper_diagonal_blocks[n_i]
            @ temp_lower_retarded
            @ A_diagonal_blocks[n_i]
        )  # C: 2xMM(bbb)


def _ddbtsci_upward_quadratic(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    B_diagonal_blocks: ArrayLike,
    B_lower_diagonal_blocks: ArrayLike,
    B_upper_diagonal_blocks: ArrayLike,
):
    """
    Operations Counts:
    ------------------
    19xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        temp_upper_retarded = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_lower_lesser = xp.empty_like(A_lower_diagonal_blocks[0])
        temp_upper_lesser = xp.empty_like(A_upper_diagonal_blocks[0])

    temp_1 = xp.empty_like(A_diagonal_blocks[0])
    temp_2 = xp.empty_like(A_diagonal_blocks[0])
    temp_3 = xp.empty_like(A_diagonal_blocks[0])
    temp_4 = xp.empty_like(A_diagonal_blocks[0])

    for n_i in range(1, A_diagonal_blocks.shape[0]):
        temp_1[:, :] = (
            A_diagonal_blocks[n_i - 1] @ A_upper_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)
        temp_1_b = temp_1 @ B_diagonal_blocks[n_i]  # C: MM(bbb)
        temp_2[:, :] = temp_1.conj().T
        temp_2_b = B_diagonal_blocks[n_i] @ temp_2  # C: MM(bbb)
        temp_3[:, :] = (
            A_diagonal_blocks[n_i] @ A_lower_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)
        temp_4[:, :] = temp_3.conj().T
        temp_4_b = B_diagonal_blocks[n_i - 1] @ temp_4  # C: MM(bbb)

        temp_upper_lesser[:, :] = (
            A_diagonal_blocks[n_i - 1]
            @ B_upper_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 2xMM(bbb)

        B_upper_diagonal_blocks[n_i - 1] = -temp_1_b - temp_4_b + temp_upper_lesser

        temp_lower_lesser[:, :] = (
            A_diagonal_blocks[n_i]
            @ B_lower_diagonal_blocks[n_i - 1]
            @ A_diagonal_blocks[n_i - 1].conj().T
        )  # C: 2xMM(bbb)

        B_lower_diagonal_blocks[n_i - 1] = (
            -temp_2_b - temp_3 @ B_diagonal_blocks[n_i - 1] + temp_lower_lesser
        )  # C: MM(bbb)

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + temp_3 @ temp_4_b
            + temp_3 @ temp_1_b
            + temp_2_b @ temp_4
            - temp_3 @ temp_upper_lesser
            - temp_lower_lesser @ temp_4
        )  # C: 5xMM(bbb)

        temp_upper_retarded[:, :] = A_upper_diagonal_blocks[n_i - 1]

        A_lower_diagonal_blocks[n_i - 1] = (
            -temp_3 @ A_diagonal_blocks[n_i - 1]
        )  # C: MM(bbb)
        A_upper_diagonal_blocks[n_i - 1] = (
            -temp_1 @ A_diagonal_blocks[n_i]
        )  # C: MM(bbb)
        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            - A_lower_diagonal_blocks[n_i - 1]
            @ temp_upper_retarded
            @ A_diagonal_blocks[n_i]
        )  # C: 2xMM(bbb)


def _ddbtsci_quadratic_permuted(
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
    66xMM(bbb)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    B1 = xp.empty_like(A_lower_diagonal_blocks[0])
    B2 = xp.empty_like(A_upper_buffer_blocks[0])

    C1 = xp.empty_like(A_upper_diagonal_blocks[0])
    C2 = xp.empty_like(A_lower_buffer_blocks[0])

    D1 = xp.empty_like(A_lower_diagonal_blocks[0])
    D2 = xp.empty_like(A_lower_buffer_blocks[0])

    temp_B_12 = xp.empty_like(A_upper_diagonal_blocks[0])
    temp_B_13 = xp.empty_like(A_upper_buffer_blocks[0])
    temp_B_21 = xp.empty_like(A_lower_diagonal_blocks[0])
    temp_B_31 = xp.empty_like(A_lower_buffer_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, 0, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_buffer_blocks[n_i - 1] @ A_lower_buffer_blocks[n_i]
        )  # C: 2xMM(bbb)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
        )  # C: 2xMM(bbb)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb)

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb)

        # --- Xl ---
        temp_B_12[:, :] = B_upper_diagonal_blocks[n_i]
        temp_B_12_buffer = temp_B_12 @ A_lower_buffer_blocks[n_i].conj().T  # C: MM(bbb)
        temp_B_13[:, :] = B_upper_buffer_blocks[n_i - 1]
        temp_B_21[:, :] = B_lower_diagonal_blocks[n_i]
        temp_B_31[:, :] = B_lower_buffer_blocks[n_i - 1]

        D1 = (
            A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
            + A_upper_buffer_blocks[n_i - 1] @ B_lower_buffer_blocks[n_i]
        )  # C: 2xMM(bbb)

        D2 = (
            A_upper_diagonal_blocks[n_i] @ B_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ B_diagonal_blocks[0]
        )  # C: 2xMM(bbb)

        B_upper_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i] @ D1
            - B_diagonal_blocks[n_i] @ C1.conj().T
            + A_diagonal_blocks[n_i]
            @ (
                B_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1].conj().T
                + B_upper_buffer_blocks[n_i - 1] @ A_upper_buffer_blocks[n_i].conj().T
            )
        )  # C: 5xMM(bbb)

        B_upper_buffer_blocks[n_i - 1] = (
            -A_diagonal_blocks[n_i] @ D2
            - B_diagonal_blocks[n_i] @ C2.conj().T
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12_buffer
                + B_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0].conj().T
            )
        )  # C: 4xMM(bbb)

        B_lower_diagonal_blocks[n_i] = (
            -(
                B_diagonal_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_upper_buffer_blocks[n_i] @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T
            - C1 @ B_diagonal_blocks[n_i]
            + (
                A_diagonal_blocks[n_i + 1] @ B_lower_diagonal_blocks[n_i]
                + A_upper_buffer_blocks[n_i] @ B_lower_buffer_blocks[n_i - 1]
            )
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 7xMM(bbb)

        B_lower_buffer_blocks[n_i - 1] = (
            -(
                B_lower_buffer_blocks[n_i] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_diagonal_blocks[0] @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T
            - C2 @ B_diagonal_blocks[n_i]
            + (
                A_lower_buffer_blocks[n_i] @ temp_B_21
                + A_diagonal_blocks[0] @ B_lower_buffer_blocks[n_i - 1]
            )
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 7xMM(bbb)

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (
                D1 @ A_upper_diagonal_blocks[n_i].conj().T
                + D2 @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T
            + A_diagonal_blocks[n_i]
            @ (B1 @ A_lower_diagonal_blocks[n_i] + B2 @ A_lower_buffer_blocks[n_i - 1])
            @ B_diagonal_blocks[n_i]
            + B_diagonal_blocks[n_i]
            @ (
                C1.conj().T @ A_upper_diagonal_blocks[n_i].conj().T
                + C2.conj().T @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T
            - A_diagonal_blocks[n_i]
            @ (B1 @ temp_B_21 + B2 @ temp_B_31)
            @ A_diagonal_blocks[n_i].conj().T
            - A_diagonal_blocks[n_i]
            @ (
                (
                    temp_B_12 @ A_diagonal_blocks[n_i + 1].conj().T
                    + temp_B_13 @ A_upper_buffer_blocks[n_i].conj().T
                )
                @ A_upper_diagonal_blocks[n_i].conj().T
                + (temp_B_12_buffer + temp_B_13 @ A_diagonal_blocks[0].conj().T)
                @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T
        )  # C: 23xMM(bbb)

        # --- Xr ---
        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bbb)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i - 1] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(bbb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i] @ (B1 @ D1 + B2 @ D2) @ A_diagonal_blocks[n_i]
        )  # C: 4xMM(bbb)

    A_lower_diagonal_blocks[0] = A_upper_buffer_blocks[0]
    A_upper_diagonal_blocks[0] = A_lower_buffer_blocks[0]

    B_lower_diagonal_blocks[0] = B_upper_buffer_blocks[0]
    B_upper_diagonal_blocks[0] = B_lower_buffer_blocks[0]
