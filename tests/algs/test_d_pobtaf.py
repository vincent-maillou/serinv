# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx as cpx

    CUPY_AVAIL = True

except:
    CUPY_AVAIL = False

import numpy as np

import pytest

from serinv.algs import d_pobtaf


def test_d_pobtaf():
    pass
