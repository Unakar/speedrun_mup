"""
Distributed training utilities for multi-GPU MuP experiments.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(backend: str = 'nccl') -> None:
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"Initialized distributed training: rank {rank}/{world_size}")
    else:
        print("No distributed environment detected, using single GPU")


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> None:
    """All-reduce tensor across processes."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather tensor from all processes."""
    if not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)