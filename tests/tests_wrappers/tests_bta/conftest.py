# Copyright 2023-2025 ETH Zurich. All rights reserved.
# Pytest fixtures for the tests of the sequential codes of Serinv.

import pytest


ARROWHEAD_BLOCKSIZE = [
    pytest.param(2, id="arrowhead_blocksize=2"),
    pytest.param(3, id="arrowhead_blocksize=3"),
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

@pytest.fixture(params=ARROWHEAD_BLOCKSIZE, autouse=True)
def arrowhead_blocksize(request: pytest.FixtureRequest) -> int:
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