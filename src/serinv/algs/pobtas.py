# Copyright 2023-2025 ETH Zurich. All rights reserved.


from serinv import (
    ArrayLike,
    _get_module_from_array,
)


def pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    **kwargs,
) -> ArrayLike:
    """Solve a block tridiagonal arrowhead linear system given its Cholesky factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

    Currently implemented:
    ----------------------
    |              | Natural | Permuted |
    | ------------ | ------- | -------- |
    | Direct-array | x       |          |
    | Streaming    |         |          |
    """
    device_streaming: bool = kwargs.get("device_streaming", False)
    buffer = kwargs.get("buffer", None)
    solve_last_rhs = kwargs.get("solve_last_rhs", True)

    if buffer is not None:
        # Permuted arrowhead
        if device_streaming:
            raise NotImplementedError(
                "Streaming is not implemented for the permuted arrowhead."
            )
        else:
            _pobtas_permuted(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_arrow_bottom_blocks,
                L_arrow_tip_block,
                B,
                buffer,
            )
    else:
        # Natural arrowhead
        if device_streaming:
            raise NotImplementedError(
                "Streaming is not implemented for the natural arrowhead."
            )
        else:
            _pobtas(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                L_arrow_bottom_blocks,
                L_arrow_tip_block,
                B,
            )


def _pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
):

    xp, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_arrow_bottom_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    # ----- Forward substitution -----
    B[0:diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[0],
        B[0:diag_blocksize],
        lower=True,
    )

    for i in range(1, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[i],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i - 1]
            @ B[(i - 1) * diag_blocksize : (i) * diag_blocksize],
            lower=True,
        )

    # Accumulation of the arrowhead blocks
    B_tip_rhs = B[-arrow_blocksize:]
    for i in range(0, n_diag_blocks):
        B_tip_rhs -= (
            L_arrow_bottom_blocks[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    B[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:], B_tip_rhs[:], lower=True
    )

    # ----- Backward substitution -----
    # X_{ndb+1} = L_{ndb+1,ndb+1}^{-T} (Y_{ndb+1})
    B[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:],
        B[-arrow_blocksize:],
        lower=True,
        trans="C",
    )

    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    B[-arrow_blocksize - diag_blocksize : -arrow_blocksize] = la.solve_triangular(
        L_diagonal_blocks[-1],
        B[-arrow_blocksize - diag_blocksize : -arrow_blocksize]
        - L_arrow_bottom_blocks[-1].conj().T @ B[-arrow_blocksize:],
        lower=True,
        trans="C",
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[i],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i].conj().T
            @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize]
            - L_arrow_bottom_blocks[i].conj().T @ B[-arrow_blocksize:],
            lower=True,
            trans="C",
        )


def _pobtas_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    buffer: ArrayLike,
):
    xp, la = _get_module_from_array(arr=L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]
