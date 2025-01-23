# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the sequential codes of Serinv.

import pytest


ARROWHEAD_BLOCKSIZE = [
    pytest.param(2, id="arrowhead_blocksize=2"),
    pytest.param(3, id="arrowhead_blocksize=3"),
]


@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param
