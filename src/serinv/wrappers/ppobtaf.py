# Copyright 2023-2025 ETH Zurich. All rights reserved.

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la
except:
    ...

from serinv import (
    ArrayLike,
    CUPY_AVAIL,
    DEVICE_STREAMING,
    _get_array_module,
)


def ppobtaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    arrow_direction: str = "downward",
) -> ArrayLike:
    """"""
