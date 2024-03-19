"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2024-03

Distributed implementation of lu factorization and selected inversion for 
block tridiagonal arrowhead matrices.

Copyright 2023-2024 ETH Zurich and USI. All rights reserved.
"""

try:
    import cupy as cp
except ImportError:
    pass


def set_device(
    rank_id: int,
    n_gpus_per_node: int,
    debug: bool = False,
):
    """
    Set the device for the current process.

    Parameters
    ----------
    rank_id : int
        The rank of the current process.
    n_gpus_per_node int: int
        The number of GPUs per node.
    debug : bool, optional
        Whether to print debug information.
    """
    device_id = rank_id % n_gpus_per_node
    cp.cuda.Device(device_id).use()
    if debug:
        print(f"Rank {rank_id} uses GPU {device_id}")

    return device_id
