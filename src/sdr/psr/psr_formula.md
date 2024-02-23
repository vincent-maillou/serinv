# Theoretical analysis of the BTA-sequential and BTA-distributed algorithms

## 1. Analysis parameters

| ID          | Operation   | Complexity  | Description  |
| ----------- | ----------- | ----------- | ----------- |
| OpA(n) | Block LU factorization | $O(n^3)$ | ----------- |
| OpB(n) | Block solve | $O(n^2)$ | ----------- |
| OpC(n, m, p) | Block Matrix multiplication | $O(n*m*p)$ | Defined for $(n, m) \cdot (m, p)$ matrices |
| OpD(n, m) | Block add/sub | $O(n*m)$ | ----------- |

## 2. BTA-sequential algorithm analysis

### BTA-seq problem parameters

    n_blocks = "ndb" : number of diagonals blocks in the matrix
    diag_blocksize : size of the diagonal blocks
    arrowhead_blocksize : size of the arrowhead blocks

### BTA-seq factorization algorithm

$$
\begin{align*}
    &\text{for i in range(0, nblocks - 1):} \\
    &\:\:\:\:\:\:\:\:\: L_{i, i}, U_{i, i} = ludcmp(A_{i, i}) \\

    &\:\:\:\:\:\:\:\:\: L_{i+1, i} = A_{i+1, i} \cdot U_{i, i}^{-1} \\
    &\:\:\:\:\:\:\:\:\: L_{ndb+1, i} = A_{ndb+1, i} \cdot U_{i, i}^{-1} \\

    &\:\:\:\:\:\:\:\:\: U_{i, i+1} = L_{i, i}^{-1} \cdot A_{i, i+1} \\
    &\:\:\:\:\:\:\:\:\: U_{i, ndb+1} = L_{i, i}^{-1} \cdot A_{i, ndb+1} \\

    &\:\:\:\:\:\:\:\:\: A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} \cdot U_{i, i+1} \\

    &\:\:\:\:\:\:\:\:\: A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} \cdot U_{i, i+1} \\
    &\:\:\:\:\:\:\:\:\: A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} \cdot U_{i, ndb+1} \\

    &\:\:\:\:\:\:\:\:\: A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} \cdot U_{i, ndb+1} \\

    & L_{nblocks, nblocks}, U_{nblocks, nblocks} = ludcmp(A_{nblocks, nblocks}) \\

    & L_{ndb+1, ndb} = A_{ndb+1, ndb} \cdot U_{ndb, ndb}^{-1} \\

    & U_{ndb, ndb+1} = L_{ndb, ndb}^{-1} \cdot A_{ndb, ndb+1} \\

    & A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} \cdot U_{ndb, ndb+1} \\

    & L_{ndb+1, ndb+1}, U_{ndb+1, ndb+1} = ludcmp(A_{ndb+1, ndb+1}) \\

\end{align*}
$$

### BTA-seq factorization operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpA(diag\_blocksize) | $ n\_blocks $ | $ O(n\_blocks * diag\_blocksize^3) $ |
| OpA(arrowhead\_blocksize) | $ 1 $ | $ O(arrowhead\_blocksize) $ |
| OpB(diag\_blocksize) | $ 2 * n\_blocks $ | $ O(2 * n\_blocks * diag\_blocksize^2) $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 3 * (n\_blocks - 1) $ | $ O(3 * (n\_blocks - 1) * diag\_blocksize^3) $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 4 * (n\_blocks - 1) $ | $ O(4 * (n\_blocks - 1) * diag\_blocksize^2 * arrowhead\_blocksize) $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ (n\_blocks - 1) $ | $ O((n\_blocks - 1) * diag\_blocksize * arrowhead\_blocksize^2) $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ (n\_blocks - 1) $ | $ O((n\_blocks - 1) * diag\_blocksize^2) $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 2 * (n\_blocks - 1) $ | $ O((n\_blocks - 1) * diag\_blocksize * arrowhead\_blocksize) $ |
| OpD(arrowhead\_blocksize, arrowhead\_blocksize) | $ n\_blocks $ | $ O((n\_blocks - 1) * arrowhead\_blocksize^2) $ |

### BTA-seq selected inversion

$$
\begin{align*}

    & X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1} \\

    & X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1} \\

    & X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1} \\

    &\text{for i in range(nblocks - 2, 0, -1):} \\
    &\:\:\:\:\:\:\:\:\: X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
    &\:\:\:\:\:\:\:\:\: X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1}) \\

    &\:\:\:\:\:\:\:\:\: X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
    &\:\:\:\:\:\:\:\:\: X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1}) \\

    &\:\:\:\:\:\:\:\:\: X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
\end{align*}
$$

### BTA-seq selected inversion operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpB(diag\_blocksize) | $ 2 * n\_blocks $ | $ O(2 * n\_blocks * diag\_blocksize^2) $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 6 * (n\_blocks - 1) + 3 $ | $ O((6 * (n\_blocks - 1) + 3) * diag\_blocksize^3) $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 7 * (n\_blocks - 1) $ | $ O(7 * (n\_blocks - 1) * diag\_blocksize^2 * arrowhead\_blocksize) $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ 2 * n\_blocks $ | $ O(2 * n\_blocks * diag\_blocksize * arrowhead\_blocksize^2) $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ 4 * (n\_blocks - 1) + 1 $ | $ O((4 * (n\_blocks - 1) + 1) * diag\_blocksize^2) $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 2 * (n\_blocks - 1) $ | $ O(2 * (n\_blocks - 1) * diag\_blocksize * arrowhead\_blocksize) $ |

## 3. BTA-distributed algorithm analysis

### BTA-dist problem parameters

    n_blocks = "ndb" : number of diagonals blocks in the matrix
    diag_blocksize : size of the diagonal blocks
    arrowhead_blocksize : size of the arrowhead blocks

    n_processes = MPI.COMM_WORLD : number of processes
    n_partitions = n_processes : number of distributed partitions
    n_blocks_partition = "ndb\_p" = n_blocks / n_partitions : number of blocks per partition

    reduced_system_n_blocks = 2 * (n_processes - 1) : number of blocks in the reduced system

### BTA-dist partitions distribution

### BTA-dist factorization

#### Top process factorization

$$
\begin{align*}
    &\text{for i in range(0, $n\_blocks\_partition - 1$):} \\
    &\:\:\:\:\:\:\:\:\: L_{i, i}, U_{i, i} = ludcmp(A_{i, i}) \\

    &\:\:\:\:\:\:\:\:\: L_{i+1, i} = A_{i+1, i} \cdot U_{i, i}^{-1} \\
    &\:\:\:\:\:\:\:\:\: L_{ndb+1, i} = A_{ndb+1, i} \cdot U_{i, i}^{-1} \\

    &\:\:\:\:\:\:\:\:\: L_{i, i+1} = L_{i, i}^{-1} \cdot A_{i, i+1} \\
    &\:\:\:\:\:\:\:\:\: U_{i, ndb+1} = L_{i, i}^{-1} \cdot A_{i, ndb+1} \\

    &\:\:\:\:\:\:\:\:\: A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} \cdot U_{i, i+1} \\

    &\:\:\:\:\:\:\:\:\: A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} \cdot U_{i, i+1} \\
    &\:\:\:\:\:\:\:\:\: A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} \cdot U_{i, ndb+1} \\

    &\:\:\:\:\:\:\:\:\: A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} \cdot U_{i, ndb+1} \\

    & L_{ndb\_p, ndb\_p}, U_{ndb\_p, ndb\_p} = ludcmp(A_{ndb\_p, ndb\_p}) \\
\end{align*}
$$

### BTA-dist top process factorization operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpA(diag\_blocksize) | $ n\_blocks\_partition $ | $ O(n\_blocks\_partition * diag\_blocksize^3) $ |
| OpB(diag\_blocksize) | $ 2 * n\_blocks\_partition $ | $ O(2 * n\_blocks\_partition * diag\_blocksize^2) $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 3 * (n\_blocks\_partition - 1) $ | $ O(3 * (n\_blocks\_partition - 1) * diag\_blocksize^3) $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 4 * (n\_blocks\_partition - 1) $ | $ O(4 * (n\_blocks\_partition - 1) * diag\_blocksize^2 * arrowhead\_blocksize) $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ n\_blocks\_partition $ | $ O(n\_blocks\_partition * diag\_blocksize * arrowhead\_blocksize^2) $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ (n\_blocks\_partition - 1) $ | $ O((n\_blocks\_partition - 1) * diag\_blocksize^2) $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 2 * (n\_blocks\_partition - 1) $ | $ O(2 * (n\_blocks\_partition - 1) * diag\_blocksize * arrowhead\_blocksize) $ |
| OpD(arrowhead\_blocksize, arrowhead\_blocksize) | $ n\_blocks\_partition $ | $ O(n\_blocks\_partition * arrowhead\_blocksize^2) $ |

#### Middle processes factorization

$$
\begin{align*}
    &\text{for i in range(1, $n\_blocks\_partition - 1$):} \\
        &\:\:\:\:\:\:\:\:\: L_{i, i}, U_{i, i} = ludcmp(A_{i, i}) \\

        &\:\:\:\:\:\:\:\:\: L_{i+1, i} = A_{i+1, i} \cdot U_{i, i}^{-1} \\
        &\:\:\:\:\:\:\:\:\: L_{top, i} = A_{top, i} \cdot U_{i, i}^{-1} \\
        &\:\:\:\:\:\:\:\:\: L_{ndb+1, i} = A_{ndb+1, i} \cdot U_{i, i}^{-1} \\

        &\:\:\:\:\:\:\:\:\: L_{i, i+1} = L_{i, i}^{-1} \cdot A_{i, i+1} \\
        &\:\:\:\:\:\:\:\:\: U_{i, top} = L{i, i}^{-1} \cdot A_{i, top} \\ 
        &\:\:\:\:\:\:\:\:\: U_{i, ndb+1} = L_{i, i}^{-1} \cdot A_{i, ndb+1} \\

        &\:\:\:\:\:\:\:\:\: A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} \cdot U_{i, i+1} \\

        &\:\:\:\:\:\:\:\:\: A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} \cdot U_{i, i+1} \\
        &\:\:\:\:\:\:\:\:\: A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} \cdot U_{i, ndb+1} \\

        &\:\:\:\:\:\:\:\:\: A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} \cdot U_{i, ndb+1} \\

        &\:\:\:\:\:\:\:\:\: A_{top, top} = A_{top, top} - L_{top, i} \cdot U_{i, top} \\

        &\:\:\:\:\:\:\:\:\: A_{i+1, top} = - L_{i+1, i} \cdot U_{i, top} \\

        &\:\:\:\:\:\:\:\:\: A_{top, i+1} = - L_{top, i} \cdot U_{i, i+1} \\

        &\:\:\:\:\:\:\:\:\: A_{ndb+1, top} = A_{ndb+1, top} - L_{ndb+1, i} \cdot U_{i, top} \\

        &\:\:\:\:\:\:\:\:\: A_{top, ndb+1} = A_{top, ndb+1} - L_{top, i} \cdot U_{i, ndb+1} \\


    & L_{ndb\_p, ndb\_p}, U_{ndb\_p, ndb\_p} = ludcmp(A_{ndb\_p, ndb\_p}) \\

    & L_{top, ndb\_p} = A_{top, ndb\_p} \cdot U_{ndb\_p, ndb\_p}^{-1} \\

    & L_{ndb+1, ndb\_p} = A_{ndb+1, ndb\_p} \cdot U_{ndb\_p, ndb\_p}^{-1} \\

    & U_{ndb\_p, top} = L_{ndb\_p, ndb\_p}^{-1} \cdot A_{ndb\_p, top} \\

    & U_{ndb\_p, ndb+1} = L_{ndb\_p, ndb\_p}^{-1} \cdot A_{ndb\_p, ndb+1} \\

    & L_{top, top}, U_{top, top} = ludcmp(A_{top, top}) \\

    & L_{top+1, top} = A_{top+1, top} \cdot U_{top, top}^{-1} \\

    & L_{ndb+1, top} = A_{ndb+1, top} \cdot U_{top, top}^{-1} \\

    & U_{top, top+1} = L_{top, top}^{-1} \cdot A_{top, top+1} \\

    & U_{top, ndb+1} = L_{top, top}^{-1} \cdot A_{top, ndb+1} \\

\end{align*}
$$

### BTA-dist middle processes factorization operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpA(diag\_blocksize) | $ n\_blocks\_partition $ | $ O(n\_blocks\_partition * diag\_blocksize^3) $ |
| OpB(diag\_blocksize) | $ n\_blocks\_partition $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 8 * (n\_blocks\_partition - 2) + 4 $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 6 * (n\_blocks\_partition - 2) + 4 $ | $  $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ (n\_blocks\_partition - 2) $ | $  $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ 2 * (n\_blocks\_partition - 2) $ | $  $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 4 * (n\_blocks\_partition - 2) $ | $  $ |
| OpD(arrowhead\_blocksize, arrowhead\_blocksize) | $ (n\_blocks\_partition - 2) $ | $  $ |

### Reduced system communication

### Inversion of reduced system

Cost of one selected inversion of the reduced system using the BTA-sequential algorithm.

### BTA-dist selected inversion

#### Top process selected inversion

$$
\begin{align*}
    &\text{for i in range($n\_blocks\_partition - 2$, 0, -1):} \\
    &\:\:\:\:\:\:\:\:\: X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
    &\:\:\:\:\:\:\:\:\: X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\

    &\:\:\:\:\:\:\:\:\: X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, ndb+1} X_{ndb+1, i+1}) \\
    &\:\:\:\:\:\:\:\:\: X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1}) \\

    &\:\:\:\:\:\:\:\:\: X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
\end{align*}
$$

### BTA-dist top process selected inversion operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpB(diag\_blocksize) | $ 2 * (n\_blocks\_partition - 1) $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 6 * (n\_blocks\_partition - 1) + 3 $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 7 * (n\_blocks\_partition - 1) $ | $  $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ 2 * n\_blocks\_partition $ | $  $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ 4 * (n\_blocks\_partition - 1) + 1 $ | $  $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 2 * (n\_blocks\_partition - 1) $ | $  $ |

#### Middle processes selected inversion

$$
\begin{align*}
    &\text{for i in range($n\_blocks\_partition - 2$, 0, -1):} \\
        &\:\:\:\:\:\:\:\:\: X_{i+1, i} = (- X_{i+1, top} L_{top, i} - X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
        &\:\:\:\:\:\:\:\:\: X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, top} X_{top, i+1} - U_{i, ndb+1} X_{ndb+1, i+1}) \\

        &\:\:\:\:\:\:\:\:\: X_{top, i} = (- X_{top, i+1} L_{i+1, i} - X_{top, top} L_{top, i} - X_{top, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
        &\:\:\:\:\:\:\:\:\: X_{i, top} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, top} - U_{i, top} X_{top, top} - U_{i, ndb+1} X_{ndb+1, top}) \\

        &\:\:\:\:\:\:\:\:\: X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, top} L_{top, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
        &\:\:\:\:\:\:\:\:\: X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, top} X_{top, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1}) \\ 

        &\:\:\:\:\:\:\:\:\: X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, top} L_{top, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1} \\
\end{align*}
$$

### BTA-dist middle processes selected inversion operations count

| Operation   | Count       | Complexity  |
| ----------- | ----------- | ----------- |
| OpB(diag\_blocksize) | $ 2 * (n\_blocks\_partition - 1) $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, diag\_blocksize) | $ 15 * (n\_blocks\_partition - 1) $ | $  $ |
| OpC(diag\_blocksize, diag\_blocksize, arrowhead\_blocksize) | $ 11 * (n\_blocks\_partition - 1) $ | $  $ |
| OpC(diag\_blocksize, arrowhead\_blocksize, arrowhead\_blocksize) | $ 2 * (n\_blocks\_partition - 1) $ | $  $ |
| OpD(diag\_blocksize, diag\_blocksize) | $ 11 * (n\_blocks\_partition - 1) $ | $  $ |
| OpD(diag\_blocksize, arrowhead\_blocksize) | $ 4 * (n\_blocks\_partition - 1) $ | $  $ |
