# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the utils of Serinv.

import pytest


N_DIAG_BLOCKS = [
    pytest.param(1, id="n_diag_blocks=1"),
    pytest.param(2, id="n_diag_blocks=2"),
    pytest.param(3, id="n_diag_blocks=3"),
    pytest.param(4, id="n_diag_blocks=4"),
]


ARROWHEAD_BLOCKSIZE = [
    pytest.param(2, id="arrowhead_blocksize=2"),
    pytest.param(3, id="arrowhead_blocksize=3"),
]


@pytest.fixture(params=N_DIAG_BLOCKS, autouse=True)
def n_diag_blocks(request: pytest.FixtureRequest) -> int:
    return request.param

@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
    return request.param