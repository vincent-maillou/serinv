# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Global pytest fixtures for the Serinv tests.

import os
import pytest

from serinv import backend_flags

ARRAY_TYPE = [
    pytest.param("host", id="host"),
]
if backend_flags["cupy_avail"]:
    ARRAY_TYPE.extend(
        [
            pytest.param("device", id="device"),
        ]
    )


DTYPE = [
    pytest.param("float64", id="float64"),
    pytest.param("complex128", id="complex128"),
]

DIAGONAL_BLOCKSIZE = [
    pytest.param(2, id="diagonal_blocksize=2"),
    pytest.param(3, id="diagonal_blocksize=3"),
]

if os.environ.get("LARGE_TESTS", "0") == "1":
    DIAGONAL_BLOCKSIZE.extend(
        [
            pytest.param(512, id="diagonal_blocksize=512"),
            pytest.param(1024, id="diagonal_blocksize=1024"),
            pytest.param(2048, id="diagonal_blocksize=2048"),
        ]
    )

@pytest.fixture(params=ARRAY_TYPE, autouse=True)
def array_type(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=DTYPE, autouse=True)
def dtype(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=DIAGONAL_BLOCKSIZE, autouse=True)
def diagonal_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param
