"""
Utilities for speedrun-mup including logging, metrics, and distributed training support.
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, Deque
from collections import deque, defaultdict
import json


# Distributed training utilities

def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE']) 
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        print(f"Distributed training: rank {rank}/{world_size}")
        return True
    return False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Check if current process is main."""
    return get_rank() == 0


# Logging and metrics

class MetricsLogger:
    """Simple metrics logger with W&B support."""
    
    def __init__(self, project_name: str = "speedrun-mup", use_wandb: bool = True,
                 log_dir: Optional[str] = None):
        self.use_wandb = use_wandb
        self.log_dir = log_dir
        self.step_count = 0
        self.metrics_buffer = {}
        
        # Rolling averages
        self.rolling_metrics: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize wandb if available and requested
        if use_wandb and is_main_process():
            try:
                import wandb
                wandb.init(project=project_name, config={})
                self.wandb = wandb
                print("Initialized W&B logging")
            except ImportError:
                print("wandb not available, using local logging only")
                self.wandb = None
                self.use_wandb = False
        else:
            self.wandb = None
            
        # Create log directory
        if log_dir and is_main_process():
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, 'metrics.jsonl'), 'a')
        else:
            self.log_file = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if not is_main_process():
            return
            
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Add to rolling averages
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.rolling_metrics[key].append(value)
        
        # Log to wandb
        if self.wandb:
            self.wandb.log(metrics, step=step)
        
        # Log to file
        if self.log_file:
            log_entry = {'step': step, 'timestamp': time.time(), **metrics}
            self.log_file.write(json.dumps(log_entry) + '\n')
            self.log_file.flush()
        
        # Store in buffer for console output
        self.metrics_buffer.update(metrics)
    
    def get_avg(self, key: str, default: float = 0.0) -> float:
        """Get rolling average of metric."""
        values = self.rolling_metrics.get(key, [])
        return sum(values) / len(values) if values else default
    
    def print_metrics(self, keys: Optional[list] = None):
        """Print current metrics to console."""
        if not is_main_process():
            return
            
        if keys is None:
            keys = ['loss', 'lr', 'grad_norm', 'tokens_per_sec']
        
        metrics_str = []
        for key in keys:
            if key in self.metrics_buffer:
                value = self.metrics_buffer[key]
                if isinstance(value, float):
                    metrics_str.append(f"{key}={value:.4f}")
                else:
                    metrics_str.append(f"{key}={value}")
            elif key in self.rolling_metrics:
                avg = self.get_avg(key)
                metrics_str.append(f"{key}_avg={avg:.4f}")
        
        if metrics_str:
            print(f"Step {self.step_count}: {', '.join(metrics_str)}")
    
    def close(self):
        """Clean up logging."""
        if self.log_file:
            self.log_file.close()
        if self.wandb:
            self.wandb.finish()


class Timer:
    """Simple timer for measuring step times."""
    
    def __init__(self):
        self.start_time = None
        self.step_times = deque(maxlen=100)
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop timing and return step time."""
        if self.start_time is None:
            return 0.0
        
        step_time = time.time() - self.start_time
        self.step_times.append(step_time)
        self.start_time = None
        return step_time
    
    def avg_time(self) -> float:
        """Get average step time."""
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
    
    def throughput(self, batch_size: int, seq_len: int) -> float:
        """Calculate tokens per second."""
        avg_time = self.avg_time()
        if avg_time > 0:
            return (batch_size * seq_len) / avg_time
        return 0.0


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute global gradient norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_model_info(model: torch.nn.Module) -> Dict[str, int]:
    """Get basic model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming fp32
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   step: int, loss: float, config: Dict[str, Any], 
                   checkpoint_path: str):
    """Save training checkpoint."""
    if not is_main_process():
        return
        
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    return checkpoint


# System utilities

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_mfu(model_params: int, batch_size: int, seq_len: int, step_time: float) -> float:
    """
    Estimate model flops utilization (MFU).
    Simplified calculation based on transformer forward pass.
    """
    # Approximate FLOPs per token (6 * model_params for forward + backward)
    flops_per_token = 6 * model_params
    flops_per_step = flops_per_token * batch_size * seq_len
    
    # Theoretical peak FLOPS (A100: 312 TFLOPS for bfloat16)
    peak_flops = 312e12
    
    # Actual FLOPS achieved
    if step_time > 0:
        actual_flops = flops_per_step / step_time
        mfu = actual_flops / peak_flops
        return min(mfu, 1.0)  # Cap at 100%
    
    return 0.0