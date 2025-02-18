# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the sequential codes of Serinv.

import pytest

COMM_STRATEGY = [
    pytest.param("allgather", id="comm_strategy=allgather"),
]


@pytest.fixture(params=COMM_STRATEGY, autouse=True)
def comm_strategy(request: pytest.FixtureRequest) -> str:
    return request.param
