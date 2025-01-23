# Copyright 2023-2025 ETH Zurich. All rights reserved.


from serinv import (
    ArrayLike,
    _get_module_from_array,
)


def pobts(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
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
            raise NotImplementedError("Permuted arrowhead is not implemented.")
        else:
            raise NotImplementedError("Permuted arrowhead is not implemented.")
    else:
        # Natural arrowhead
        if device_streaming:
            raise NotImplementedError(
                "Streaming is not implemented for the natural arrowhead."
            )
        else:
            return _pobts(
                L_diagonal_blocks,
                L_lower_diagonal_blocks,
                B,
            )


def _pobts(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    B: ArrayLike,
):
    xp, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
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

    # ----- Backward substitution -----
    # X_{ndb} = L_{ndb,ndb}^{-T} (Y_{ndb} - L_{ndb+1,ndb}^{T} X_{ndb+1})
    B[-diag_blocksize:] = la.solve_triangular(
        L_diagonal_blocks[-1],
        B[-diag_blocksize:],
        lower=True,
        trans="C",
    )

    for i in range(n_diag_blocks - 2, -1, -1):
        # X_{i} = L_{i,i}^{-T} (Y_{i} - L_{i+1,i}^{T} X_{i+1}) - L_{ndb+1,i}^T X_{ndb+1}
        B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[i],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i].conj().T
            @ B[(i + 1) * diag_blocksize : (i + 2) * diag_blocksize],
            lower=True,
            trans="C",
        )
