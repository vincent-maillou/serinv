"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Tests for cholesky selected decompositions routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""

from sdr.utils import matrix_generation
from sdr.cholesky.cholesky_decompose import chol_dcmp_tridiag_arrowhead

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt



if __name__ == "__main__":

    nblocks = [8, 16, 32, 64, 128, 256]
    diag_blocksize = 30
    arrow_blocksize = 4

    n_runs = 10

    relative_error = np.ndarray((len(nblocks), n_runs))

    for i in range(len(nblocks)):
        print("Starting test with nblocks = ", nblocks[i], ", diag_blocksize = ", diag_blocksize, ", arrow_blocksize = ", arrow_blocksize,":")

        for run in range(n_runs):
            print("     Run ", run, " of ", n_runs, "...")

            A = matrix_generation.generate_blocktridiag_arrowhead(
                nblocks[i], diag_blocksize, arrow_blocksize, symmetric=True, 
                diagonal_dominant=True, seed=None
            )
            L_ref = la.cholesky(A, lower=True)
            L_sdr = chol_dcmp_tridiag_arrowhead(A, diag_blocksize, arrow_blocksize)
            relative_error[i, run] = la.norm(L_ref - L_sdr) / la.norm(L_ref)

    # Save results in a file
    np.save(f"chol_dcmp_tridiag_arrowhead_accuracy_dbs{diag_blocksize}_abs{arrow_blocksize}.npy", relative_error)

    # --- Plotting ---
    # Compute the mean and standard deviation of the relative error
    mean = np.mean(relative_error, axis=1)
    std_dev = np.std(relative_error, axis=1)

    # Plot the mean and standard deviation
    fig, ax = plt.subplots(1, 1)
    ax.bar(nblocks, mean, yerr=std_dev, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xlabel("Number of blocks\diag_blocksize: " + str(diag_blocksize) + "\narrow_blocksize: " + str(arrow_blocksize))
    ax.set_ylabel("Relative error")
    ax.set_title("Relative error of selected cholesky decomposition")
    plt.show()