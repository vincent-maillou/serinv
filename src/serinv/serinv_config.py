# Copyright 2023-2024 ETH Zurich and the QuaTrEx authors. All rights reserved.

from pydantic import BaseModel


class SolverConfig(BaseModel):
    device_streaming: bool = True
    cuda_aware_mpi: bool = False
