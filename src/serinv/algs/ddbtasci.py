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
    rhs: dict = kwargs.get("rhs", None)
    quadratic: bool = kwargs.get("quadratic", False)
    buffers: dict = kwargs.get("buffers", None)

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
):
    xp, _ = _get_module_from_array(A_diagonal_blocks)

    if A_diagonal_blocks.shape[0] > 1:
        # If there is only a single diagonal block, we don't need these buffers.
        B1 = xp.empty_like(A_lower_diagonal_blocks[0])
        C1 = xp.empty_like(A_upper_diagonal_blocks[0])
        D1 = xp.empty_like(A_lower_diagonal_blocks[0])

    B2 = xp.empty_like(A_upper_arrow_blocks[0])
    C2 = xp.empty_like(A_lower_arrow_blocks[0])
    D2 = xp.empty_like(A_lower_arrow_blocks[0])

    B2[:, :] = A_diagonal_blocks[-1] @ A_upper_arrow_blocks[-1]

    A_lower_arrow_blocks[-1] = (
        -A_arrow_tip_block[:] @ A_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1]
    )

    A_upper_arrow_blocks[-1] = -B2[:, :] @ A_arrow_tip_block[:]

    A_diagonal_blocks[-1] = A_diagonal_blocks[-1] - B2[:, :] @ A_lower_arrow_blocks[-1]

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        print(n_i, flush=True)
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
        )

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :]
        )

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
        )

        C2[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block[:, :] @ A_lower_arrow_blocks[n_i]
        )

        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1[:, :]
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B2[:, :]

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1[:, :] @ A_diagonal_blocks[n_i]
        A_lower_arrow_blocks[n_i] = -C2[:, :] @ A_diagonal_blocks[n_i]

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1[:, :] @ D1[:, :] + B2[:, :] @ D2[:, :])
            @ A_diagonal_blocks[n_i]
        )


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
        )

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[0]
        )

        B3[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :]
            + A_upper_buffer_blocks[n_i - 1] @ A_upper_arrow_blocks[0]
        )

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
            + A_upper_arrow_blocks[0] @ A_lower_arrow_blocks[n_i]
        )

        C3[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block[:, :] @ A_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )

        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1[:, :]
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2[:, :]
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B3[:, :]

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]
        D3[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1[:, :] @ A_diagonal_blocks[n_i]
        A_lower_buffer_blocks[n_i - 1] = -C2[:, :] @ A_diagonal_blocks[n_i]
        A_lower_arrow_blocks[n_i] = -C3[:, :] @ A_diagonal_blocks[n_i]

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1[:, :] @ D1[:, :] + B2[:, :] @ D2[:, :] + B3[:, :] @ D3[:, :])
            @ A_diagonal_blocks[n_i]
        )

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
):
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
    B2[:, :] = A_diagonal_blocks[-1] @ A_upper_arrow_blocks[-1]
    C2[:, :] = A_upper_arrow_blocks[-1].T @ A_diagonal_blocks[-1].T
    D2[:, :] = (
        A_arrow_tip_block[:, :] @ A_lower_arrow_blocks[-1] @ B_diagonal_blocks[-1]
    )

    temp_B_13[:, :] = B_upper_arrow_blocks[-1]
    temp_B_31[:, :] = B_lower_arrow_blocks[-1]

    B_upper_arrow_blocks[-1] = (
        -B2[:, :] @ B_arrow_tip_block[:, :]
        - B_diagonal_blocks[-1] @ A_lower_arrow_blocks[-1].T @ A_arrow_tip_block[:, :].T
        + A_diagonal_blocks[-1] @ B_upper_arrow_blocks[-1] @ A_arrow_tip_block[:, :].T
    )

    B_lower_arrow_blocks[-1] = (
        -B_arrow_tip_block[:, :] @ C2[:, :]
        - D2[:, :]
        + A_arrow_tip_block[:, :] @ B_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1].T
    )

    B_diagonal_blocks[-1] = (
        B_diagonal_blocks[-1]
        + B2[:, :] @ B_arrow_tip_block[:, :] @ C2[:, :]
        + B2[:, :] @ D2[:, :]
        + B_diagonal_blocks[-1].T
        @ A_lower_arrow_blocks[-1].T
        @ A_arrow_tip_block[:, :].T
        @ C2[:, :]
        - B2[:, :] @ A_arrow_tip_block[:, :] @ temp_B_31[:, :] @ A_diagonal_blocks[-1].T
        - A_diagonal_blocks[-1] @ temp_B_13[:, :] @ A_arrow_tip_block[:, :].T @ C2[:, :]
    )

    # --- Xr ---
    A_lower_arrow_blocks[-1] = (
        -A_arrow_tip_block[:] @ A_lower_arrow_blocks[-1] @ A_diagonal_blocks[-1]
    )
    A_upper_arrow_blocks[-1] = -B2[:, :] @ A_arrow_tip_block[:]
    A_diagonal_blocks[-1] = A_diagonal_blocks[-1] - B2[:, :] @ A_lower_arrow_blocks[-1]

    for n_i in range(A_diagonal_blocks.shape[0] - 2, -1, -1):
        B1[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[n_i + 1]
        )

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :]
        )

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
        )

        C2[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block[:, :] @ A_lower_arrow_blocks[n_i]
        )

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
            )
            - B_diagonal_blocks[n_i] @ C1[:, :].T
            + A_diagonal_blocks[n_i]
            @ (
                B_upper_diagonal_blocks[n_i] @ A_diagonal_blocks[n_i + 1].T
                + B_upper_arrow_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1].T
            )
        )
        B_upper_arrow_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block[:, :]
            )
            - B_diagonal_blocks[n_i] @ C2[:, :].T
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12[:, :] @ A_lower_arrow_blocks[n_i + 1].T
                + B_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :].T
            )
        )

        B_lower_diagonal_blocks[n_i] = (
            -(
                B_diagonal_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].T
                + B_upper_arrow_blocks[n_i + 1] @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            - (C1[:, :]) @ B_diagonal_blocks[n_i]
            + (
                A_diagonal_blocks[n_i + 1] @ B_lower_diagonal_blocks[n_i]
                + A_upper_arrow_blocks[n_i + 1] @ B_lower_arrow_blocks[n_i]
            )
            @ A_diagonal_blocks[n_i].T
        )
        B_lower_arrow_blocks[n_i] = (
            -(
                B_lower_arrow_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].T
                + B_arrow_tip_block[:, :] @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            - (C2[:, :]) @ B_diagonal_blocks[n_i]
            + (
                A_lower_arrow_blocks[n_i + 1] @ temp_B_21[:, :]
                + A_arrow_tip_block[:, :] @ B_lower_arrow_blocks[n_i]
            )
            @ A_diagonal_blocks[n_i].T
        )

        B_diagonal_blocks[n_i] = (
            B_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (
                (
                    A_upper_diagonal_blocks[n_i] @ B_diagonal_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[n_i + 1]
                )
                @ A_upper_diagonal_blocks[n_i].T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block[:, :]
                )
                @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            + A_diagonal_blocks[n_i]
            @ (
                (B1[:, :]) @ A_lower_diagonal_blocks[n_i]
                + (B2[:, :]) @ A_lower_arrow_blocks[n_i]
            )
            @ B_diagonal_blocks[n_i]
            + B_diagonal_blocks[n_i].T
            @ (
                C1[:, :].T @ A_upper_diagonal_blocks[n_i].T
                + C2[:, :].T @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            - A_diagonal_blocks[n_i]
            @ ((B1[:, :]) @ temp_B_21 + (B2[:, :]) @ temp_B_31)
            @ A_diagonal_blocks[n_i].T
            - A_diagonal_blocks[n_i]
            @ (
                (
                    temp_B_12 @ A_diagonal_blocks[n_i + 1].T
                    + temp_B_13 @ A_upper_arrow_blocks[n_i + 1].T
                )
                @ A_upper_diagonal_blocks[n_i].T
                + (
                    temp_B_12 @ A_lower_arrow_blocks[n_i + 1].T
                    + temp_B_13 @ A_arrow_tip_block[:, :].T
                )
                @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
        )

        # --- Xr ---
        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1[:, :]
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B2[:, :]

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1[:, :] @ A_diagonal_blocks[n_i]
        A_lower_arrow_blocks[n_i] = -C2[:, :] @ A_diagonal_blocks[n_i]

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1[:, :] @ D1[:, :] + B2[:, :] @ D2[:, :])
            @ A_diagonal_blocks[n_i]
        )


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
        )

        B2[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_buffer_blocks[n_i]
            + A_upper_buffer_blocks[n_i - 1] @ A_diagonal_blocks[0]
            + A_upper_arrow_blocks[n_i] @ A_lower_arrow_blocks[0]
        )

        B3[:, :] = (
            A_upper_diagonal_blocks[n_i] @ A_upper_arrow_blocks[n_i + 1]
            + A_upper_arrow_blocks[n_i] @ A_arrow_tip_block[:, :]
            + A_upper_buffer_blocks[n_i - 1] @ A_upper_arrow_blocks[0]
        )

        C1[:, :] = (
            A_diagonal_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_upper_arrow_blocks[n_i + 1] @ A_lower_arrow_blocks[n_i]
            + A_upper_buffer_blocks[n_i] @ A_lower_buffer_blocks[n_i - 1]
        )

        C2[:, :] = (
            A_lower_buffer_blocks[n_i] @ A_lower_diagonal_blocks[n_i]
            + A_diagonal_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
            + A_upper_arrow_blocks[0] @ A_lower_arrow_blocks[n_i]
        )

        C3[:, :] = (
            A_lower_arrow_blocks[n_i + 1] @ A_lower_diagonal_blocks[n_i]
            + A_arrow_tip_block[:, :] @ A_lower_arrow_blocks[n_i]
            + A_lower_arrow_blocks[0] @ A_lower_buffer_blocks[n_i - 1]
        )

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
            )
            - B_diagonal_blocks[n_i] @ C1[:, :].T
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12[:, :] @ A_diagonal_blocks[n_i + 1].T
                + temp_B_13[:, :] @ A_upper_arrow_blocks[n_i + 1].T
                + temp_B_14[:, :] @ A_upper_buffer_blocks[n_i].T
            )
        )
        B_upper_buffer_blocks[n_i - 1] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_buffer_blocks[n_i]
                + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[0]
                + A_upper_buffer_blocks[n_i - 1] @ B_diagonal_blocks[0]
            )
            - B_diagonal_blocks[n_i] @ C2[:, :].T
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12[:, :] @ A_lower_buffer_blocks[n_i].T
                + temp_B_13[:, :] @ A_upper_arrow_blocks[0].T
                + temp_B_14[:, :] @ A_diagonal_blocks[0].T
            )
        )
        B_upper_arrow_blocks[n_i] = (
            -A_diagonal_blocks[n_i]
            @ (
                A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block[:, :]
                + A_upper_buffer_blocks[n_i - 1] @ B_upper_arrow_blocks[0]
            )
            - B_diagonal_blocks[n_i] @ C3[:, :].T
            + A_diagonal_blocks[n_i]
            @ (
                temp_B_12[:, :] @ A_lower_arrow_blocks[n_i + 1].T
                + temp_B_13[:, :] @ A_arrow_tip_block[:, :].T
                + temp_B_14[:, :] @ A_lower_arrow_blocks[0].T
            )
        )

        B_lower_diagonal_blocks[n_i] = (
            -(
                B_diagonal_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].T
                + B_upper_arrow_blocks[n_i + 1] @ A_upper_arrow_blocks[n_i].T
                + B_upper_buffer_blocks[n_i] @ A_upper_buffer_blocks[n_i - 1].T
            )
            @ A_diagonal_blocks[n_i].T
            - C1[:, :] @ B_diagonal_blocks[n_i]
            + (
                A_diagonal_blocks[n_i + 1] @ temp_B_21[:, :]
                + A_upper_arrow_blocks[n_i + 1] @ temp_B_31[:, :]
                + A_upper_buffer_blocks[n_i] @ temp_B_41[:, :]
            )
            @ A_diagonal_blocks[n_i].T
        )
        B_lower_buffer_blocks[n_i - 1] = (
            -(
                B_lower_buffer_blocks[n_i] @ A_upper_diagonal_blocks[n_i].T
                + B_diagonal_blocks[0] @ A_upper_buffer_blocks[n_i - 1].T
                + B_upper_arrow_blocks[0] @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            - C2[:, :] @ B_diagonal_blocks[n_i]
            + (
                A_lower_buffer_blocks[n_i] @ temp_B_21[:, :]
                + A_upper_arrow_blocks[0] @ temp_B_31[:, :]
                + A_diagonal_blocks[0] @ temp_B_41[:, :]
            )
            @ A_diagonal_blocks[n_i].T
        )
        B_lower_arrow_blocks[n_i] = (
            -(
                B_lower_arrow_blocks[n_i + 1] @ A_upper_diagonal_blocks[n_i].T
                + B_arrow_tip_block[:, :] @ A_upper_arrow_blocks[n_i].T
                + B_lower_arrow_blocks[0] @ A_upper_buffer_blocks[n_i - 1].T
            )
            @ A_diagonal_blocks[n_i].T
            - C3[:, :] @ B_diagonal_blocks[n_i]
            + (
                A_lower_arrow_blocks[n_i + 1] @ temp_B_21[:, :]
                + A_arrow_tip_block[:, :] @ temp_B_31[:, :]
                + A_lower_arrow_blocks[0] @ temp_B_41[:, :]
            )
            @ A_diagonal_blocks[n_i].T
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
                @ A_upper_diagonal_blocks[n_i].T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_buffer_blocks[n_i]
                    + A_upper_buffer_blocks[n_i - 1] @ B_diagonal_blocks[0]
                    + A_upper_arrow_blocks[n_i] @ B_lower_arrow_blocks[0]
                )
                @ A_upper_buffer_blocks[n_i - 1].T
                + (
                    A_upper_diagonal_blocks[n_i] @ B_upper_arrow_blocks[n_i + 1]
                    + A_upper_arrow_blocks[n_i] @ B_arrow_tip_block[:, :]
                    + A_upper_buffer_blocks[n_i - 1] @ B_upper_arrow_blocks[0]
                )
                @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            + A_diagonal_blocks[n_i]
            @ (
                B1[:, :] @ A_lower_diagonal_blocks[n_i]
                + B2[:, :] @ A_lower_buffer_blocks[n_i - 1]
                + B3[:, :] @ A_lower_arrow_blocks[n_i]
            )
            @ B_diagonal_blocks[n_i]
            + B_diagonal_blocks[n_i].T
            @ (
                C1[:, :].T @ A_upper_diagonal_blocks[n_i].T
                + C2[:, :].T @ A_upper_buffer_blocks[n_i - 1].T
                + C3[:, :].T @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
            - A_diagonal_blocks[n_i]
            @ (B1[:, :] @ temp_B_21 + B2[:, :] @ temp_B_41 + B3[:, :] @ temp_B_31)
            @ A_diagonal_blocks[n_i].T
            - A_diagonal_blocks[n_i]
            @ (
                (
                    temp_B_12[:, :] @ A_diagonal_blocks[n_i + 1].T
                    + temp_B_13[:, :] @ A_upper_arrow_blocks[n_i + 1].T
                    + temp_B_14[:, :] @ A_upper_buffer_blocks[n_i].T
                )
                @ A_upper_diagonal_blocks[n_i].T
                + (
                    temp_B_12[:, :] @ A_lower_buffer_blocks[n_i].T
                    + temp_B_14[:, :] @ A_diagonal_blocks[0].T
                    + temp_B_13[:, :] @ A_upper_arrow_blocks[0].T
                )
                @ A_upper_buffer_blocks[n_i - 1].T
                + (
                    temp_B_12[:, :] @ A_lower_arrow_blocks[n_i + 1].T
                    + temp_B_13[:, :] @ A_arrow_tip_block[:, :].T
                    + temp_B_14[:, :] @ A_lower_arrow_blocks[0].T
                )
                @ A_upper_arrow_blocks[n_i].T
            )
            @ A_diagonal_blocks[n_i].T
        )

        # --- Xr ---
        A_upper_diagonal_blocks[n_i] = -A_diagonal_blocks[n_i] @ B1[:, :]
        A_upper_buffer_blocks[n_i - 1] = -A_diagonal_blocks[n_i] @ B2[:, :]
        A_upper_arrow_blocks[n_i] = -A_diagonal_blocks[n_i] @ B3[:, :]

        D1[:, :] = A_lower_diagonal_blocks[n_i]
        D2[:, :] = A_lower_buffer_blocks[n_i - 1]
        D3[:, :] = A_lower_arrow_blocks[n_i]

        A_lower_diagonal_blocks[n_i] = -C1[:, :] @ A_diagonal_blocks[n_i]
        A_lower_buffer_blocks[n_i - 1] = -C2[:, :] @ A_diagonal_blocks[n_i]
        A_lower_arrow_blocks[n_i] = -C3[:, :] @ A_diagonal_blocks[n_i]

        A_diagonal_blocks[n_i] = (
            A_diagonal_blocks[n_i]
            + A_diagonal_blocks[n_i]
            @ (B1[:, :] @ D1[:, :] + B2[:, :] @ D2[:, :] + B3[:, :] @ D3[:, :])
            @ A_diagonal_blocks[n_i]
        )

    A_lower_diagonal_blocks[0] = A_upper_buffer_blocks[0]
    A_upper_diagonal_blocks[0] = A_lower_buffer_blocks[0]

    B_lower_diagonal_blocks[0] = B_upper_buffer_blocks[0]
    B_upper_diagonal_blocks[0] = B_lower_buffer_blocks[0]
