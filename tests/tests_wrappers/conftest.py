# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the distributed codes of Serinv.

import pytest

PARTITION_SIZE = [
    pytest.param(3, id="partition_size=3"),
    pytest.param(4, id="partition_size=4"),
    pytest.param(5, id="partition_size=5"),
]

PREALLOCATE_PERMUTATION_BUFFER = [
    pytest.param(True, id="preallocate_permutation_buffer=True"),
    pytest.param(False, id="preallocate_permutation_buffer=False"),
]

PREALLOCATE_REDUCED_SYSTEM = [
    pytest.param(True, id="preallocate_reduced_system=True"),
    pytest.param(False, id="preallocate_reduced_system=False"),
]

COMM_STRATEGY = [
    pytest.param("allreduce", id="comm_strategy=allreduce"),
    pytest.param("allgather", id="comm_strategy=allgather"),
    pytest.param("gather-scatter", id="comm_strategy=gather-scatter"),
]


@pytest.fixture(params=PARTITION_SIZE, autouse=True)
def partition_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=PREALLOCATE_PERMUTATION_BUFFER, autouse=True)
def preallocate_permutation_buffer(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=PREALLOCATE_REDUCED_SYSTEM, autouse=True)
def preallocate_reduced_system(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=COMM_STRATEGY, autouse=True)
def comm_strategy(request: pytest.FixtureRequest) -> str:
    return request.param
