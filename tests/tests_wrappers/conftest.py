# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the distributed codes of Serinv.

import pytest

PARTITION_SIZE = [
    pytest.param(3, id="partition_size=3"),
    pytest.param(4, id="partition_size=4"),
    pytest.param(5, id="partition_size=5"),
]

@pytest.fixture(params=PARTITION_SIZE, autouse=True)
def partition_size(request: pytest.FixtureRequest) -> int:
    return request.param
