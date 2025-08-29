"""
Utilities for MuP implementation and shape management.
"""

from .shapes import InfDim, InfShape, make_base_shapes, set_base_shapes
from .init import uniform_, normal_, kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_
from .distributed import setup_distributed, cleanup_distributed, get_world_size, get_rank

__all__ = [
    "InfDim",
    "InfShape", 
    "make_base_shapes",
    "set_base_shapes",
    "uniform_",
    "normal_",
    "kaiming_normal_",
    "kaiming_uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "setup_distributed",
    "cleanup_distributed", 
    "get_world_size",
    "get_rank",
]