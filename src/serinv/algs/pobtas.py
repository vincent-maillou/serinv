# Copyright 2023-2025 ETH Zurich. All rights reserved.


from serinv import (
    ArrayLike,
    _get_module_from_array,
)


def pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    **kwargs,
) -> ArrayLike:
    """Solve a block tridiagonal arrowhead linear system given its Cholesky factorization
    using a sequential block algorithm.

    Note:
    -----
    - If a device array is given, the algorithm will run on the GPU.

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
                L_lower_arrow_blocks,
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
                L_lower_arrow_blocks,
                L_arrow_tip_block,
                B,
            )


def _pobtas(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
):
    xp, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
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
            L_lower_arrow_blocks[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
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
        - L_lower_arrow_blocks[-1].conj().T @ B[-arrow_blocksize:],
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
            - L_lower_arrow_blocks[i].conj().T @ B[-arrow_blocksize:],
            lower=True,
            trans="C",
        )


def _pobtas_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    buffer: ArrayLike,
):
    xp, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    B1_update = xp.zeros_like(B[0:diag_blocksize])
    # ----- Forward substitution -----
    B[diag_blocksize : 2 * diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[1],
        B[diag_blocksize : 2 * diag_blocksize],
        lower=True,
    )

    B1_update -= buffer[1] @ B[diag_blocksize : 2 * diag_blocksize]

    for i in range(2, n_diag_blocks):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[i],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i - 1]
            @ B[(i - 1) * diag_blocksize : (i) * diag_blocksize],
            lower=True,
        )

        # Accumulation of the first block (permutation-linked)
        # if i < n_diag_blocks - 1:
        B1_update -= buffer[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]

    # B[0:diag_blocksize] += B1_update
    B1_update[:] += B[0:diag_blocksize]
    B[0:diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[0],
        B1_update,
        lower=True,
    )

    # Accumulation of the arrowhead blocks
    # Y_{ndb+1} = L_{ndb+1,ndb+1}^{-1} (B_{ndb+1} - \Sigma_{i=1}^{ndb} L_{ndb+1,i} Y_{i)
    Btip_update = xp.zeros_like(B[-arrow_blocksize:])
    for i in range(0, n_diag_blocks):
        Btip_update -= (
            L_lower_arrow_blocks[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    # B[-arrow_blocksize:] += Btip_update
    Btip_update[:] += B[-arrow_blocksize:]
    B[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:],
        Btip_update,
        lower=True,
    )

    # ----- Backward substitution -----
    B[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:],
        B[-arrow_blocksize:],
        lower=True,
        trans="C",
    )

    B[0:diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[0],
        B[0:diag_blocksize] - L_lower_arrow_blocks[0].conj().T @ B[-arrow_blocksize:],
        lower=True,
    )


def _v0_pobtas_permuted(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_lower_arrow_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    B: ArrayLike,
    buffer: ArrayLike,
):
    xp, la = _get_module_from_array(L_diagonal_blocks)

    diag_blocksize = L_diagonal_blocks.shape[1]
    arrow_blocksize = L_lower_arrow_blocks.shape[1]
    n_diag_blocks = L_diagonal_blocks.shape[0]

    # ----- Forward substitution -----
    B[diag_blocksize : 2 * diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[1],
        B[diag_blocksize : 2 * diag_blocksize],
        lower=True,
    )

    B[0:diag_blocksize] -= buffer[1] @ B[diag_blocksize : 2 * diag_blocksize]

    for i in range(2, n_diag_blocks - 1):
        # Y_{i} = L_{i,i}^{-1} (B_{i} - L_{i,i-1} Y_{i-1})
        B[i * diag_blocksize : (i + 1) * diag_blocksize] = la.solve_triangular(
            L_diagonal_blocks[i],
            B[i * diag_blocksize : (i + 1) * diag_blocksize]
            - L_lower_diagonal_blocks[i - 1]
            @ B[(i - 1) * diag_blocksize : (i) * diag_blocksize],
            lower=True,
        )

        # Accumulation of the first block (permutation-linked)
        B[0:diag_blocksize] -= (
            buffer[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

        B[-arrow_blocksize:] -= (
            L_lower_arrow_blocks[i] @ B[i * diag_blocksize : (i + 1) * diag_blocksize]
        )

    _B_0 = B[0:diag_blocksize].copy()
    _B_1 = B[-diag_blocksize - arrow_blocksize : -arrow_blocksize].copy()
    _B_t = B[-arrow_blocksize:].copy()

    _L_00 = L_diagonal_blocks[0].copy()
    _L_11 = L_diagonal_blocks[-1].copy()
    _L_tt = L_arrow_tip_block.copy()
    _L_10 = buffer[-1].conj().T.copy()
    _L_t0 = L_lower_arrow_blocks[0].copy()
    _L_t1 = L_lower_arrow_blocks[-1].copy()

    # --- forward substitution ---
    _B_0 = la.solve_triangular(
        _L_00,
        _B_0,
        lower=True,
    )

    _B_1 = la.solve_triangular(
        _L_11,
        _B_1 - _L_10 @ _B_0,
        lower=True,
    )

    _B_t = la.solve_triangular(
        _L_tt,
        _B_t - _L_t0 @ _B_0 - _L_t1 @ _B_1,
        lower=True,
    )

    # --- backward substitution ---
    _B_t = la.solve_triangular(
        _L_tt,
        _B_t,
        lower=True,
        trans="C",
    )

    B[0:diag_blocksize] = _B_0
    B[-diag_blocksize - arrow_blocksize : -arrow_blocksize] = _B_1
    B[-arrow_blocksize:] = _B_t

    """ # ----- Backward substitution -----
    B[-arrow_blocksize:] = la.solve_triangular(
        L_arrow_tip_block[:],
        B[-arrow_blocksize:],
        lower=True,
        trans="C",
    )

    B[0:diag_blocksize] = la.solve_triangular(
        L_diagonal_blocks[0],
        B[0:diag_blocksize] - L_lower_arrow_blocks[0].conj().T @ B[-arrow_blocksize:],
        lower=True,
    ) """
