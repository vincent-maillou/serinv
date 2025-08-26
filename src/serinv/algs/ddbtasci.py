# Copyright 2023-2025 ETH Zurich. All rights reserved.

from serinv import (
    ArrayLike,
    _get_module_from_array,
)


def ddbtasci(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_upper_diagonal_blocks: ArrayLike,
    A_lower_arrow_blocks: ArrayLike,
    A_upper_arrow_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
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
        The buffers to use in the permuted Schur-complement Selected Inversion (SCI) algorithm.
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
            # Perform a regular SCI
            return _ddbtasci(
                A_diagonal_blocks,
                A_lower_diagonal_blocks,
                A_upper_diagonal_blocks,
                A_lower_arrow_blocks,
                A_upper_arrow_blocks,
                A_arrow_tip_block,
                invert_last_block=invert_last_block,
            )
        else:
            # Perform a permuted SCI
            A_lower_buffer_blocks = buffers.get("A_lower_buffer_blocks", None)
            A_upper_buffer_blocks = buffers.get("A_upper_buffer_blocks", None)
            if any(x is None for x in [A_lower_buffer_blocks, A_upper_buffer_blocks]):
                raise ValueError("buffers does not contain the correct arrays for A")
            _ddbtasci_permuted(
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
            # Perform the SCI for AXA^T=B
            if buffers is None:
                # Perform a regular SCI ("quadratic")
                _ddbtasci_quadratic(
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
                _ddbtasci_quadratic_permuted(
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
            # Perform the SCI for AX=B
            ...


def _ddbtasci(
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
    7xMM(bbb)
    2xMM(bba)
    1xMM(baa)
    2xMM(abb)
    1xMM(aab)
    3xMM(bab)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        B1 = xp.empty_like(A_lower_diagonal_blocks[0])
        C1 = xp.empty_like(A_upper_diagonal_blocks[0])
        D1 = xp.empty_like(A_lower_diagonal_blocks[0])

    B2 = xp.empty_like(A_upper_arrow_blocks[0])
    C2 = xp.empty_like(A_lower_arrow_blocks[0])
    D2 = xp.empty_like(A_lower_arrow_blocks[0])

    if invert_last_block:
        B2[:, :] = A_diagonal_blocks[-1] @ A_upper_arrow_blocks[-1]

        A_lower_arrow_blocks[-1] = (
            -A_arrow_tip_block[:] @ A_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1]
        )

        A_upper_arrow_blocks[-1] = -B2[:, :] @ A_arrow_tip_block[:]

        A_diagonal_blocks[-1] = (
            A_diagonal_blocks[-1] - B2[:, :] @ A_lower_arrow_blocks[-1]
        )

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
        )  # C: MM(bbb) + MM(bab)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block
        )  # C: MM(bba) + MM(baa)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
        )  # C: MM(bbb) + MM(bab)

        C2[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block @ A_lower_arrow_blocks[n_i]
        )  # C: MM(abb) + MM(aab)

        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bba)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_arrow_blocks[n_i] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(abb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i] @ (B1 @ D1 + B2 @ D2) @ A_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb) + MM(bab)


def _ddbtasci_permuted(
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
    14xMM(bbb)
    5xMM(bba)
    1xMM(baa)
    3xMM(abb)
    1xMM(aab)
    5xMM(bab)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    B1 = xp.empty_like(A_lower_diagonal_blocks[0])
    B2 = xp.empty_like(A_upper_buffer_blocks[0])
    B3 = xp.empty_like(A_upper_arrow_blocks[0])

    C1 = xp.empty_like(A_upper_diagonal_blocks[0])
    C2 = xp.empty_like(A_lower_buffer_blocks[0])
    C3 = xp.empty_like(A_lower_arrow_blocks[0])

    D1 = xp.empty_like(A_lower_diagonal_blocks[0])
    D2 = xp.empty_like(A_lower_buffer_blocks[0])
    D3 = xp.empty_like(A_lower_arrow_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, 0, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
            + A_upper_buffer_blocks[n_i - 1] @ A_lower_buffer_blocks[n_i]
        )  # C: 2xMM(bbb) + MM(bab)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[0]
        )  # C: 2xMM(bbb) + MM(bab)

        B3[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :]
            + A_upper_buffer_blocks[n_i - 1] @ A_upper_arrow_blocks[0]
        )  # C: 2xMM(bba) + MM(baa)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb) + MM(bab)

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
            + A_upper_arrow_blocks[0] @ A_lower_arrow_blocks[n_i]
        )  # C: 2xMM(bba) + MM(bab)

        C3[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block @ A_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(abb) + MM(aab)

        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bbb)
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B3  # C: MM(bba)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]
        D3[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i - 1] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_arrow_blocks[n_i] = -C3 @ A_diagonal_blocks[n_i]  # C: MM(abb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1 @ D1 + B2 @ D2 + B3 @ D3)
            @ A_diagonal_blocks[n_i]
        )  # C: 4xMM(bbb) + MM(bab)

    A_lower_diagonal_blocks[0] = A_upper_buffer_blocks[0]
    A_upper_diagonal_blocks[0] = A_lower_buffer_blocks[0]


def _ddbtasci_quadratic(
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
    34xMM(bbb)
    9xMM(bba)
    4xMM(baa)
    6xMM(abb)
    3xMM(aab)
    13xMM(bab)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        B1 = xp.empty_like(A_lower_diagonal_blocks[0])
        C1 = xp.empty_like(A_upper_diagonal_blocks[0])
        D1 = xp.empty_like(A_lower_diagonal_blocks[0])

        temp_B_12 = xp.empty_like(A_upper_diagonal_blocks[0])
        temp_B_21 = xp.empty_like(A_lower_diagonal_blocks[0])

    B2 = xp.empty_like(A_upper_arrow_blocks[0])
    C2 = xp.empty_like(A_lower_arrow_blocks[0])
    D2 = xp.empty_like(A_lower_arrow_blocks[0])

    temp_B_13 = xp.empty_like(A_upper_arrow_blocks[0])
    temp_B_31 = xp.empty_like(A_lower_arrow_blocks[0])

    # --- Xl ---
    if invert_last_block:
        B2[:, :] = A_diagonal_blocks[-1] @ A_upper_arrow_blocks[-1]
        C2[:, :] = B2.conj().T
        D2[:, :] = A_arrow_tip_block @ A_lower_arrow_blocks[-1] @ B_diagonal_blocks[-1]

        temp_B_13[:, :] = B_upper_arrow_blocks[-1]
        temp_B_31[:, :] = B_lower_arrow_blocks[-1]

        B_upper_arrow_blocks[-1] = (
            -B2 @ B_arrow_tip_block
            - B_diagonal_blocks[-1]
            @ A_lower_arrow_blocks[-1].conj().T
            @ A_arrow_tip_block.conj().T
            + A_diagonal_blocks[-1]
            @ B_upper_arrow_blocks[-1]
            @ A_arrow_tip_block.conj().T
        )

        B_lower_arrow_blocks[-1] = (
            -B_arrow_tip_block @ C2
            - D2
            + A_arrow_tip_block
            @ B_lower_arrow_blocks[-1]
            @ A_diagonal_blocks[-1].conj().T
        )

        B_diagonal_blocks[-1] = (
            B_diagonal_blocks[-1]
            + B2 @ B_arrow_tip_block @ C2
            + B2 @ D2
            + B_diagonal_blocks[-1]
            @ A_lower_arrow_blocks[-1].conj().T
            @ A_arrow_tip_block.conj().T
            @ C2
            - B2 @ A_arrow_tip_block @ temp_B_31 @ A_diagonal_blocks[-1].conj().T
            - A_diagonal_blocks[-1] @ temp_B_13 @ A_arrow_tip_block.conj().T @ C2
        )

        # --- Xr ---
        A_lower_arrow_blocks[-1] = (
            -A_arrow_tip_block[:] @ A_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1]
        )
        A_upper_arrow_blocks[-1] = -B2 @ A_arrow_tip_block[:]
        A_diagonal_blocks[-1] = A_diagonal_blocks[-1] - B2 @ A_lower_arrow_blocks[-1]

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
        )  # C: MM(bbb) + MM(bab)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block
        )  # C: MM(bba) + MM(baa)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
        )  # C: MM(bbb) + MM(bab)

        C2[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block @ A_lower_arrow_blocks[n_i]
        )  # C: MM(abb) + MM(aab)

        # --- Xl ---
        temp_B_12[:, :] = B_upper_diagonal_blocks[n_i]
        temp_B_13[:, :] = B_upper_arrow_blocks[n_i]
        temp_B_21[:, :] = B_lower_diagonal_blocks[n_i]
        temp_B_31[:, :] = B_lower_arrow_blocks[n_i]

        B_upper_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[n_i + 1]
            )  # C: 2xMM(bbb) + MM(bab)
            - B_diagonal_blocks[n_i] @ C1.conj().T  # C: MM(bbb)
            + A_diagonal_blocks[n_i]
            @ (
                B_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1].conj().T
                + B_upper_arrow_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1].conj().T
            )  # C: 2xMM(bbb) + MM(bab)
        )

        B_upper_arrow_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block
            )  # C: 2xMM(bba) + MM(baa)
            - B_diagonal_blocks[n_i] @ C2.conj().T  # C: MM(bba)
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12 @ A_lower_arrow_blocks[n_i + 1].conj().T
                + B_upper_arrow_blocks[n_i] @ A_arrow_tip_block.conj().T
            )  # C: 2xMM(bba) + MM(baa)
        )

        B_lower_diagonal_blocks[n_i] = (
            -(
                B_diagonal_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_upper_arrow_blocks[n_i + 1] @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 2xMM(bbb) + MM(bab)
            - C1 @ B_diagonal_blocks[n_i]  # C: MM(bbb)
            + (
                A_diagonal_blocks[n_i + 1] @ B_lower_diagonal_blocks[n_i]
                + A_upper_arrow_blocks[n_i + 1] @ B_lower_arrow_blocks[n_i]
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 2xMM(bbb) + MM(bab)
        )

        B_lower_arrow_blocks[n_i] = (
            -(
                B_lower_arrow_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_arrow_tip_block @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: MM(abb) + MM(aab)
            - C2 @ B_diagonal_blocks[n_i]  # C: MM(abb)
            + (
                A_lower_arrow_blocks[n_i + 1] @ temp_B_21
                + A_arrow_tip_block @ B_lower_arrow_blocks[n_i]
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 2xMM(abb) + MM(aab)
        )

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (
                (
                    A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[n_i + 1]
                )
                @ A_upper_diagonal_blocks[n_i].conj().T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block
                )
                @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i]
            .conj()
            .T  # C: 4xMM(bbb) + 2xMM(bab) + MM(bba) + M(baa)
            + A_diagonal_blocks[n_i]
            @ (B1 @ A_lower_diagonal_blocks[n_i] + B2 @ A_lower_arrow_blocks[n_i])
            @ B_diagonal_blocks[n_i]  # C: 3xMM(bbb) + MM(bab)
            + B_diagonal_blocks[n_i]
            @ (
                C1.conj().T @ A_upper_diagonal_blocks[n_i].conj().T
                + C2.conj().T @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
            - A_diagonal_blocks[n_i]
            @ (B1 @ temp_B_21 + B2 @ temp_B_31)
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
            - A_diagonal_blocks[n_i]
            @ (
                (
                    temp_B_12 @ A_diagonal_blocks[n_i + 1].conj().T
                    + temp_B_13 @ A_upper_arrow_blocks[n_i + 1].conj().T
                )
                @ A_upper_diagonal_blocks[n_i].conj().T
                + (
                    temp_B_12 @ A_lower_arrow_blocks[n_i + 1].conj().T
                    + temp_B_13 @ A_arrow_tip_block.conj().T
                )
                @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i]
            .conj()
            .T  # C: 4xMM(bbb) + 2xMM(bab) + MM(bba) + MM(baa)
        )

        # --- Xr ---
        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bba)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_arrow_blocks[n_i] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(abb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i] @ (B1 @ D1 + B2 @ D2) @ A_diagonal_blocks[n_i]
        )  # C: 3xMM(bbb) + MM(bab)


def _ddbtasci_quadratic_permuted(
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
    72xMM(bbb)
    14xMM(bba)
    5xMM(baa)
    10xMM(abb)
    3xMM(aab)
    22xMM(bab)
    """
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    B1 = xp.empty_like(A_lower_diagonal_blocks[0])
    B2 = xp.empty_like(A_upper_buffer_blocks[0])
    B3 = xp.empty_like(A_upper_arrow_blocks[0])

    C1 = xp.empty_like(A_upper_diagonal_blocks[0])
    C2 = xp.empty_like(A_lower_buffer_blocks[0])
    C3 = xp.empty_like(A_lower_arrow_blocks[0])

    D1 = xp.empty_like(A_lower_diagonal_blocks[0])
    D2 = xp.empty_like(A_lower_buffer_blocks[0])
    D3 = xp.empty_like(A_lower_arrow_blocks[0])

    temp_B_12 = xp.empty_like(A_upper_diagonal_blocks[0])
    temp_B_13 = xp.empty_like(A_upper_arrow_blocks[0])
    temp_B_14 = xp.empty_like(A_upper_buffer_blocks[0])

    temp_B_21 = xp.empty_like(A_lower_diagonal_blocks[0])
    temp_B_31 = xp.empty_like(A_lower_arrow_blocks[0])
    temp_B_41 = xp.empty_like(A_lower_buffer_blocks[0])

    for n_i in range(A_diagonal_blocks.shape[0] - 2, 0, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
            + A_upper_buffer_blocks[n_i - 1] @ A_lower_buffer_blocks[n_i]
        )  # C: 2xMM(bbb) + MM(bab)

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[0]
        )  # C: 2xMM(bbb) + MM(bab)

        B3[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block
            + A_upper_buffer_blocks[n_i - 1] @ A_upper_arrow_blocks[0]
        )  # C: 2xMM(bba) + MM(baa)

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(bbb) + MM(bab)

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
            + A_upper_arrow_blocks[0] @ A_lower_arrow_blocks[n_i]
        )  # C: 2xMM(bbb) + MM(bab)

        C3[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block @ A_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )  # C: 2xMM(abb) + MM(aab)

        # --- Xl ---
        temp_B_12[:, :] = B_upper_diagonal_blocks[n_i]
        temp_B_13[:, :] = B_upper_arrow_blocks[n_i]
        temp_B_14[:, :] = B_upper_buffer_blocks[n_i - 1]

        temp_B_21[:, :] = B_lower_diagonal_blocks[n_i]
        temp_B_31[:, :] = B_lower_arrow_blocks[n_i]
        temp_B_41[:, :] = B_lower_buffer_blocks[n_i - 1]

        B_upper_diagonal_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[n_i + 1]
                + A_upper_buffer_blocks[n_i - 1] @ B_lower_buffer_blocks[n_i]
            )  # C: 3xMM(bbb) + MM(bab)
            - B_diagonal_blocks[n_i] @ C1.conj().T  # C: MM(bbb)
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12 @ A_diagonal_blocks[n_i + 1].conj().T
                + temp_B_13 @ A_upper_arrow_blocks[n_i + 1].conj().T
                + temp_B_14 @ A_upper_buffer_blocks[n_i].conj().T
            )  # C: 3xMM(bbb) + MM(bab)
        )

        B_upper_buffer_blocks[n_i - 1] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_buffer_blocks[n_i]
                + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[0]
                + A_upper_buffer_blocks[n_i - 1] @ B_diagonal_blocks[0]
            )  # C: 3xMM(bbb) + MM(bab)
            - B_diagonal_blocks[n_i] @ C2.conj().T  # C: MM(bbb)
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12 @ A_lower_buffer_blocks[n_i].conj().T
                + temp_B_13 @ A_upper_arrow_blocks[0].conj().T
                + temp_B_14 @ A_diagonal_blocks[0].conj().T
            )  # C: 3xMM(bbb) + MM(bab)
        )

        B_upper_arrow_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block
                + A_upper_buffer_blocks[n_i - 1] @ B_upper_arrow_blocks[0]
            )  # C: 3xMM(bba) + MM(baa)
            - B_diagonal_blocks[n_i] @ C3.conj().T  # C: MM(bba)
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12 @ A_lower_arrow_blocks[n_i + 1].conj().T
                + temp_B_13 @ A_arrow_tip_block.conj().T
                + temp_B_14 @ A_lower_arrow_blocks[0].conj().T
            )  # C: 3xMM(bba) + MM(baa)
        )

        B_lower_diagonal_blocks[n_i] = (
            -(
                B_diagonal_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_upper_arrow_blocks[n_i + 1] @ A_upper_arrow_blocks[n_i].conj().T
                + B_upper_buffer_blocks[n_i] @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
            - C1 @ B_diagonal_blocks[n_i]  # C: MM(bbb)
            + (
                A_diagonal_blocks[n_i + 1] @ temp_B_21
                + A_upper_arrow_blocks[n_i + 1] @ temp_B_31
                + A_upper_buffer_blocks[n_i] @ temp_B_41
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
        )

        B_lower_buffer_blocks[n_i - 1] = (
            -(
                B_lower_buffer_blocks[n_i] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_diagonal_blocks[0] @ A_upper_buffer_blocks[n_i - 1].conj().T
                + B_upper_arrow_blocks[0] @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
            - C2 @ B_diagonal_blocks[n_i]  # C: MM(bbb)
            + (
                A_lower_buffer_blocks[n_i] @ temp_B_21
                + A_upper_arrow_blocks[0] @ temp_B_31
                + A_diagonal_blocks[0] @ temp_B_41
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(bbb) + MM(bab)
        )

        B_lower_arrow_blocks[n_i] = (
            -(
                B_lower_arrow_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].conj().T
                + B_arrow_tip_block @ A_upper_arrow_blocks[n_i].conj().T
                + B_lower_arrow_blocks[0] @ A_upper_buffer_blocks[n_i - 1].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(abb) + MM(aab)
            - C3 @ B_diagonal_blocks[n_i]  # C: MM(abb)
            + (
                A_lower_arrow_blocks[n_i + 1] @ temp_B_21
                + A_arrow_tip_block @ temp_B_31
                + A_lower_arrow_blocks[0] @ temp_B_41
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 3xMM(abb) + MM(aab)
        )

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (
                (
                    A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[n_i + 1]
                    + A_upper_buffer_blocks[n_i - 1] @ B_lower_buffer_blocks[n_i]
                )
                @ A_upper_diagonal_blocks[n_i].conj().T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_buffer_blocks[n_i]
                    + A_upper_buffer_blocks[n_i - 1] @ B_diagonal_blocks[0]
                    + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[0]
                )
                @ A_upper_buffer_blocks[n_i - 1].conj().T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block
                    + A_upper_buffer_blocks[n_i - 1] @ B_upper_arrow_blocks[0]
                )
                @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i]
            .conj()
            .T  # C: 8xMM(bbb) + 3xMM(bab) + 2xMM(bba) + MM(baa)
            + A_diagonal_blocks[n_i]
            @ (
                B1 @ A_lower_diagonal_blocks[n_i]
                + B2 @ A_lower_buffer_blocks[n_i - 1]
                + B3 @ A_lower_arrow_blocks[n_i]
            )
            @ B_diagonal_blocks[n_i]  # C: 4xMM(bbb) + MM(bab)
            + B_diagonal_blocks[n_i]
            @ (
                C1.conj().T @ A_upper_diagonal_blocks[n_i].conj().T
                + C2.conj().T @ A_upper_buffer_blocks[n_i - 1].conj().T
                + C3.conj().T @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i].conj().T  # C: 4xMM(bbb) + MM(bab)
            - A_diagonal_blocks[n_i]
            @ (B1 @ temp_B_21 + B2 @ temp_B_41 + B3 @ temp_B_31)
            @ A_diagonal_blocks[n_i].conj().T  # C: 4xMM(bbb) + MM(bab)
            - A_diagonal_blocks[n_i]
            @ (
                (
                    temp_B_12 @ A_diagonal_blocks[n_i + 1].conj().T
                    + temp_B_13 @ A_upper_arrow_blocks[n_i + 1].conj().T
                    + temp_B_14 @ A_upper_buffer_blocks[n_i].conj().T
                )
                @ A_upper_diagonal_blocks[n_i].conj().T
                + (
                    temp_B_12 @ A_lower_buffer_blocks[n_i].conj().T
                    + temp_B_14 @ A_diagonal_blocks[0].conj().T
                    + temp_B_13 @ A_upper_arrow_blocks[0].conj().T
                )
                @ A_upper_buffer_blocks[n_i - 1].conj().T
                + (
                    temp_B_12 @ A_lower_arrow_blocks[n_i + 1].conj().T
                    + temp_B_13 @ A_arrow_tip_block.conj().T
                    + temp_B_14 @ A_lower_arrow_blocks[0].conj().T
                )
                @ A_upper_arrow_blocks[n_i].conj().T
            )
            @ A_diagonal_blocks[n_i]
            .conj()
            .T  # C: 8xMM(bbb) + 3xMM(bab) + 2xMM(bba) + MM(baa)
        )

        # --- Xr ---
        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1  # C: MM(bbb)
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2  # C: MM(bbb)
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B3  # C: MM(bba)

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]
        D3[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_buffer_blocks[n_i - 1] = -C2 @ A_diagonal_blocks[n_i]  # C: MM(bbb)
        A_lower_arrow_blocks[n_i] = -C3 @ A_diagonal_blocks[n_i]  # C: MM(abb)

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1 @ D1 + B2 @ D2 + B3 @ D3)
            @ A_diagonal_blocks[n_i]
        )  # C: 4xMM(bbb) + MM(bab)

    A_lower_diagonal_blocks[0] = A_upper_buffer_blocks[0]
    A_upper_diagonal_blocks[0] = A_lower_buffer_blocks[0]

    B_lower_diagonal_blocks[0] = B_upper_buffer_blocks[0]
    B_upper_diagonal_blocks[0] = B_lower_buffer_blocks[0]
