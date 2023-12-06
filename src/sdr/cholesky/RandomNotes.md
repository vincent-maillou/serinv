# Notes on Cholesky Factorization

## chol_dcmp_ndiags_arrowhead()

#### Parameters
A : np.ndarray
    Input matrix to decompose.
    
ndiags : int
    Number of diagonals of the matrix.

diag_blocksize : int
    Blocksize of the diagonals blocks of the matrix.

arrow_blocksize : int
    Blocksize of the blocks composing the arrowhead.


### **How is this implemented?**

    for i in range(0, n_diag_blocks-1):
        dense Cholesky of diagonal block(i,i)
        # L_{i, i} = chol(A_{i, i}) 
        
        # Temporary storage of re-used triangular solving
        explicit computation of inv(L{i,i})

        for j in range(1, min(ndiags, n_diag_blocks-i)):
            # update of diagonal blocks below
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}

            for k in range(1, j+1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                update of coupled entries using newly computed L_{i+j, i} 

        # Part of the decomposition for the arrowhead structure
        update of arrowhead structure below
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}

        for k in range(1, min(ndiags, n_diag_blocks-i)):
            # A_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            apply arrowhead update to coupled terms -> have mixed terms with off-diagonal blocks ...

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        apply update to last dense arrowhead block

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    cholesky factorization of last diagonal block that is not part of arrowhead structure

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    update arrowhead block below

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    update last dense arrowhead block

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    dense cholesky diagonal arrowhead block


    