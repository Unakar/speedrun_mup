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

def print0(*args, console=False, **kwargs):
    """Print only on main process (rank 0), modded-nanogpt style."""
    if is_main_process():
        if console:
            print(*args, **kwargs, flush=True)
        else:
            print(*args, **kwargs)


def generate_experiment_name(prefix: str = "", config: Dict[str, Any] = None) -> str:
    """Generate timestamp-based experiment name."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add config-based suffix
    suffix_parts = []
    if config:
        if config.get('mup', False):
            suffix_parts.append("mup")
        if 'width' in config:
            suffix_parts.append(f"w{config['width']}")
        if 'base_width' in config and config.get('mup', False):
            suffix_parts.append(f"base{config['base_width']}")
    
    # Combine parts
    name_parts = [p for p in [prefix, timestamp] + suffix_parts if p]
    return "_".join(name_parts)


class SimpleLogger:
    """Simple logger aligned with modded-nanogpt style."""
    
    def __init__(self, use_wandb: bool = False, project_name: str = "speedrun-mup",
                 experiment_name: str = None, config: Dict[str, Any] = None):
        self.use_wandb = use_wandb
        self.wandb = None
        self.start_time = time.time()
        self.experiment_name = experiment_name or generate_experiment_name("speedrun", config)
        
        # Create logs directory with experiment name
        self.log_dir = f"logs/{self.experiment_name}"
        if is_main_process():
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize wandb if available and requested
        if use_wandb and is_main_process():
            try:
                import wandb
                wandb.init(
                    project=project_name, 
                    name=self.experiment_name,
                    config=config or {}
                )
                self.wandb = wandb
                print0(f"Initialized W&B logging: {project_name}/{self.experiment_name}")
            except ImportError:
                print0("wandb not available, skipping")
                self.use_wandb = False
        
        # Initialize log file
        self.log_file = None
        if is_main_process():
            log_file_path = os.path.join(self.log_dir, "training.log")
            self.log_file = open(log_file_path, 'w')
        
        # Log experiment info
        self.log_experiment_start(config)
    
    def log_experiment_start(self, config: Dict[str, Any] = None):
        """Log experiment start information."""
        if not is_main_process():
            return
        
        import sys
        import datetime
        
        start_msg = [
            f"=" * 80,
            f"Experiment: {self.experiment_name}",
            f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Log directory: {self.log_dir}",
            f"Python: {sys.version}",
            f"PyTorch: {torch.__version__}",
            f"Hardware: {get_gpu_name()}",
            f"=" * 80,
        ]
        
        if config:
            start_msg.extend([
                "Configuration:",
                json.dumps(config, indent=2, default=str),
                "=" * 80,
            ])
        
        for line in start_msg:
            print0(line, console=True)
            if self.log_file:
                self.log_file.write(line + '\n')
                self.log_file.flush()
    
    def log_step(self, step: int, total_steps: int, training_time_ms: float, 
                 val_loss: Optional[float] = None, **kwargs):
        """Log training step in modded-nanogpt style."""
        step_avg = training_time_ms / max(step, 1)
        
        if val_loss is not None:
            msg = f"step:{step}/{total_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{step_avg:.2f}ms"
        else:
            msg = f"step:{step}/{total_steps} train_time:{training_time_ms:.0f}ms step_avg:{step_avg:.2f}ms"
        
        print0(msg, console=True)
        
        # Also log to file with timestamp
        if self.log_file:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_file.write(f"[{timestamp}] {msg}\n")
            self.log_file.flush()
        
        # Log to wandb if available
        if self.wandb:
            log_dict = {
                'step': step,
                'training_time_ms': training_time_ms,
                'step_avg_ms': step_avg,
                **kwargs
            }
            if val_loss is not None:
                log_dict['val_loss'] = val_loss
            self.wandb.log(log_dict, step=step)
    
    def log_final_stats(self, peak_memory_mb: int, reserved_memory_mb: int, 
                       total_training_time: float):
        """Log final statistics."""
        final_msg = [
            f"peak memory allocated: {peak_memory_mb} MiB reserved: {reserved_memory_mb} MiB",
            f"total training time: {format_time(total_training_time)}",
            f"experiment completed: {self.experiment_name}",
            "=" * 80
        ]
        
        for line in final_msg:
            print0(line, console=True)
            if self.log_file:
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_file.write(f"[{timestamp}] {line}\n")
                self.log_file.flush()
        
        if self.wandb:
            self.wandb.log({
                'peak_memory_mb': peak_memory_mb,
                'reserved_memory_mb': reserved_memory_mb,
                'total_training_time_s': total_training_time
            })
    
    def log(self, metrics: Dict[str, Any]):
        """Generic log method for arbitrary metrics."""
        if not is_main_process():
            return
            
        # Log to wandb with organized grouping if available
        if self.wandb:
            grouped_metrics = self._group_metrics_for_wandb(metrics)
            self.wandb.log(grouped_metrics)
        
        # Log to file with timestamp
        if self.log_file:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics_str = ', '.join([f"{k}={v}" for k, v in metrics.items()])
            self.log_file.write(f"[{timestamp}] {metrics_str}\n")
            self.log_file.flush()
    
    def _group_metrics_for_wandb(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Group metrics by category for better wandb organization."""
        grouped = {}
        
        # Define metric groups
        time_metrics = {'approx_train_time_ms', 'step_avg_ms', 'training_time_ms', 'total_time_s'}
        loss_metrics = {'train_loss', 'val_loss'}
        optimization_metrics = {'lr', 'grad_norm', 'momentum'}
        model_metrics = {'param_norm', 'spectral_norm_max', 'spectral_norm_mean', 'spectral_norm_std'}
        hardware_metrics = {'peak_memory_mb', 'reserved_memory_mb'}
        activation_metrics = set()  # For future activation stats
        
        for key, value in metrics.items():
            # Determine which group this metric belongs to
            if key in time_metrics:
                grouped[f"Time/{key}"] = value
            elif key in loss_metrics:
                grouped[f"Loss/{key}"] = value
            elif key in optimization_metrics:
                grouped[f"Optimization/{key}"] = value
            elif key in model_metrics:
                grouped[f"Model/{key}"] = value
            elif key in hardware_metrics:
                grouped[f"Hardware/{key}"] = value
            elif key.endswith(('_mean', '_std', '_max', '_min', '_l2_norm')):
                # All activation statistics end with these suffixes
                grouped[f"Activations/{key}"] = value
            elif key == 'step':
                # Keep step at root level for x-axis
                grouped[key] = value
            else:
                # Default group for unknown metrics
                grouped[f"Other/{key}"] = value
        
        return grouped
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration to file."""
        if not is_main_process():
            return
        
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dumps(config, f, indent=2, default=str)
        print0(f"Saved config to: {config_path}")
    
    def close(self):
        """Clean up logging."""
        if self.log_file:
            self.log_file.close()
        if self.wandb:
            self.wandb.finish()


class Timer:
    """Simple timer aligned with modded-nanogpt approach."""
    
    def __init__(self):
        self.start_time = None
        self.total_time = 0.0
        self.step_count = 0
    
    def start(self):
        """Start timing with CUDA sync."""
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        self.total_time += elapsed
        self.step_count += 1
        self.start_time = None
        return elapsed
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds without stopping."""
        if self.start_time is None:
            return 0.0
        return 1000 * (time.perf_counter() - self.start_time)
    
    def avg_time(self) -> float:
        """Get average time per step in seconds."""
        if self.step_count == 0:
            return 0.0
        return self.total_time / self.step_count


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute global gradient norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_param_norm(model: torch.nn.Module) -> float:
    """Compute global parameter norm."""
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# ---------- Newton–Schulz building blocks (bf16-friendly, GPU matmul only) ----------

@torch.no_grad()
def _ns5_zeroth_power(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Quintic Newton–Schulz used in Muon to approx the orthogonal polar factor.
    Uses the (a,b,c) tuned for large slope at 0 (non-convergent to 1 but fast).
    Ref: Keller Jordan's Muon writeup & modded-nanogpt README.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Muon coefficients
    # Normalize to keep singular values in [0,1]-ish
    X = G.to(dtype=torch.bfloat16)
    # Use Fro norm; for tall/flat handling we follow Muon trick (transpose to tall)
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X  # keep bf16; caller may cast


@torch.no_grad()
def _ns3_polar(G: torch.Tensor, steps: int = 8, eps: float = 1e-7) -> torch.Tensor:
    """
    Classic convergent Newton–Schulz for polar factor:
      X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    After pre-scaling. Slower but better accuracy for polar(Q).
    """
    assert G.ndim == 2
    X = G.to(dtype=torch.bfloat16)
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + eps)
    I = torch.eye(X.size(-1), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        XtX = X.mT @ X
        X = 0.5 * (X @ (3*I - XtX))
    if transposed:
        X = X.mT
    return X


@torch.no_grad()
def _polar_factor(G: torch.Tensor, *, mode: str = "ns5-fast") -> torch.Tensor:
    if mode == "ns5-fast":
        return _ns5_zeroth_power(G, steps=5)
    elif mode == "ns3-accurate":
        return _ns3_polar(G, steps=8)
    else:
        raise ValueError(f"unknown mode={mode}")


# ---------- Spectral norm via polar + power iteration on H ≈ (W^T W)^{1/2} ----------

@torch.no_grad()
def _spectral_norm_via_polar_power(W: torch.Tensor, *, mode: str, power_iters: int = 7) -> float:
    """
    Estimate sigma_max(W) using:
      1) Q ≈ polar(W) by Newton–Schulz (bf16-friendly)
      2) H = Q^T W ≈ (W^T W)^{1/2}; run power-iteration on linear op v -> Q^T (W v)
    Memory-lean: never explicitly forms H.
    """
    device = W.device
    # Flatten conv/ND to (out_features, in_features)
    if W.ndim > 2:
        W2 = W.reshape(W.size(0), -1)
    else:
        W2 = W

    # Build Q in bf16 (GPU-friendly). Keep W matmuls in bf16 as well.
    Q = _polar_factor(W2, mode=mode)  # bf16
    n = W2.size(1)
    # Power iteration on symmetric PSD linear operator H(v) = Q^T (W v)
    # Work vectors in bf16 for matmul, but norms/inner-products in fp32
    v = torch.randn(n, device=device, dtype=torch.bfloat16)
    v = v / (v.float().norm() + 1e-12)

    for _ in range(max(1, power_iters)):
        y = W2.to(torch.bfloat16) @ v              # shape (m,)
        z = Q.mT @ y                               # shape (n,) == H @ v
        v = z / (z.float().norm() + 1e-12)

    # Rayleigh quotient on the last (v, Hv)
    y = W2.to(torch.bfloat16) @ v
    Hv = Q.mT @ y
    num = (v.float() * Hv.float()).sum()
    den = (v.float() * v.float()).sum()
    lam = (num / (den + 1e-20)).item()            # eigenvalue of H
    # H ≈ (W^T W)^{1/2} => eigenvalues of H are singular values of W
    sigma = max(lam, 0.0)
    return float(sigma)


# ---------- Public API ----------

def compute_weight_spectral_norms(
    model: torch.nn.Module, 
    target_params: list = None,
    mode: str = "ns5-fast",          # or "ns3-accurate"
    power_iters: int = 7
) -> Dict[str, float]:
    """
    Compute per-weight spectral norms during bf16/fp16 mixed-precision training
    using efficient Newton-Schulz polar decomposition + power iteration.
    
    This method is bf16-friendly and compatible with Muon optimizer's approach.
    Heavy ops done in bf16; reductions in fp32.

    Args:
        model: The model to analyze
        target_params: If provided, only compute spectral norms for parameters in this list
        mode: "ns5-fast" (Muon-style, fast but slight bias) or "ns3-accurate" (slower, more accurate)
        power_iters: Number of power iteration steps

    Returns:
        {parameter_name: approx_sigma_max}
    """
    import contextlib
    
    spectral_norms: Dict[str, float] = {}
    
    # If target_params provided, create a set of parameter objects for fast lookup
    target_param_set = set(target_params) if target_params else None

    # ensure we don't get autocast surprises from the training region
    maybe_no_autocast = (
        torch.amp.autocast(device_type='cuda', enabled=False)
        if torch.cuda.is_available() else contextlib.nullcontext()
    )
    
    with maybe_no_autocast:
        for name, param in model.named_parameters():
            # Skip if target_params specified and this param is not in the list
            if target_param_set and param not in target_param_set:
                continue
            if param.ndim < 2 or 'weight' not in name:
                continue
                
            W = param.detach()  # never touches autograd graph
            try:
                sigma = _spectral_norm_via_polar_power(W, mode=mode, power_iters=power_iters)
            except Exception:
                # Robust fallback: plain power iteration on W^T W (still bf16 matmuls)
                # This matches PyTorch spectral_norm's math, but stays off matrix_norm().
                if W.ndim > 2:
                    W2 = W.reshape(W.size(0), -1)
                else:
                    W2 = W
                m, n = W2.shape
                v = torch.randn(n, device=W2.device, dtype=torch.bfloat16)
                v = v / (v.float().norm() + 1e-12)
                for _ in range(max(3, power_iters)):
                    z = W2.mT @ (W2.to(torch.bfloat16) @ v)  # (n,)
                    v = z / (z.float().norm() + 1e-12)
                # Rayleigh: v^T (W^T W) v
                z = W2.mT @ (W2.to(torch.bfloat16) @ v)
                num = (v.float() * z.float()).sum()
                den = (v.float() * v.float()).sum()
                sigma = float(torch.sqrt((num / (den + 1e-20)).clamp_min(0)).item())
            spectral_norms[name] = sigma
    
    return spectral_norms


def compute_activation_stats(activations: torch.Tensor, name: str = "activation") -> Dict[str, float]:
    """Compute activation distribution statistics."""
    with torch.no_grad():
        flat_act = activations.view(-1)
        return {
            f"{name}_mean": flat_act.mean().item(),
            f"{name}_std": flat_act.std().item(),
            f"{name}_max": flat_act.max().item(),
            f"{name}_min": flat_act.min().item(),
            f"{name}_l2_norm": flat_act.norm(2).item(),
        }


class ActivationHook:
    """Hook to capture activation statistics during forward pass."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model: torch.nn.Module, layer_names: list = None):
        """Register hooks on specified layers or all ReLU/GELU layers."""
        if layer_names is None:
            # Auto-register on activation layers
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)
        else:
            for name in layer_names:
                module = dict(model.named_modules())[name]
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
    
    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = compute_activation_stats(output, name)
        return hook
    
    def get_stats(self) -> Dict[str, float]:
        """Get accumulated activation statistics."""
        stats = {}
        for layer_stats in self.activations.values():
            stats.update(layer_stats)
        return stats
    
    def clear(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class TrainingMonitor:
    """Performance-aware training monitor for advanced metrics."""
    
    def __init__(self, model: torch.nn.Module, monitor_interval: int = 100,
                 enable_spectral_norms: bool = False, enable_activation_stats: bool = False):
        self.model = model
        self.monitor_interval = monitor_interval
        self.enable_spectral_norms = enable_spectral_norms
        self.enable_activation_stats = enable_activation_stats
        self.step_count = 0
        
        # Activation hooks (expensive, use sparingly)
        self.activation_hook = None
        if enable_activation_stats:
            self.activation_hook = ActivationHook()
            self.activation_hook.register_hooks(model)
    
    def should_monitor(self, step: int = None) -> bool:
        """Check if we should collect expensive metrics this step."""
        if step is not None:
            return step % self.monitor_interval == 0
        self.step_count += 1
        return self.step_count % self.monitor_interval == 0
    
    def get_basic_metrics(self) -> Dict[str, float]:
        """Get lightweight metrics (always computed)."""
        return {
            'grad_norm': compute_grad_norm(self.model),
            'param_norm': compute_param_norm(self.model)
        }
    
    def get_advanced_metrics(self) -> Dict[str, float]:
        """Get expensive metrics (use sparingly)."""
        metrics = {}
        
        if self.enable_spectral_norms:
            spectral_norms = compute_weight_spectral_norms(self.model)
            # Log max, mean, and specific layers
            if spectral_norms:
                values = list(spectral_norms.values())
                metrics.update({
                    'spectral_norm_max': max(values),
                    'spectral_norm_mean': sum(values) / len(values),
                    'spectral_norm_std': torch.tensor(values).std().item()
                })
        
        if self.enable_activation_stats and self.activation_hook:
            activation_stats = self.activation_hook.get_stats()
            metrics.update(activation_stats)
            self.activation_hook.clear()
        
        return metrics
    
    def get_all_metrics(self, step: int = None) -> Dict[str, float]:
        """Get appropriate metrics based on monitoring schedule."""
        metrics = self.get_basic_metrics()
        
        if self.should_monitor(step):
            advanced = self.get_advanced_metrics()
            metrics.update(advanced)
        
        return metrics
    
    def close(self):
        """Clean up hooks."""
        if self.activation_hook:
            self.activation_hook.remove_hooks()


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


def get_gpu_name() -> str:
    """Get GPU name for hardware detection."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return "unknown"


def get_peak_flops(gpu_name: str = None) -> float:
    """Get theoretical peak FLOPS for different hardware."""
    if gpu_name is None:
        gpu_name = get_gpu_name().lower()
    else:
        gpu_name = gpu_name.lower()
    
    # Hardware-specific peak FLOPS (bfloat16/fp16)
    if "h100" in gpu_name:
        return 989e12  # H100 SXM: 989 TFLOPS
    elif "b200" in gpu_name or "blackwell" in gpu_name:
        return 2500e12  # B200: ~2500 TFLOPS (estimated)
    elif "a100" in gpu_name:
        return 312e12  # A100: 312 TFLOPS
    elif "v100" in gpu_name:
        return 125e12  # V100: 125 TFLOPS
    else:
        # Default to H100 since modded-nanogpt targets 8xH100
        return 989e12


def estimate_mfu(model_params: int, batch_size: int, seq_len: int, step_time: float) -> float:
    """
    Estimate model flops utilization (MFU).
    Auto-detects hardware for appropriate peak FLOPS.
    """
    if step_time <= 0:
        return 0.0
    
    # Approximate FLOPs per token (6 * model_params for forward + backward)
    flops_per_token = 6 * model_params
    flops_per_step = flops_per_token * batch_size * seq_len
    
    # Get hardware-appropriate peak FLOPS
    peak_flops = get_peak_flops()
    
    # Calculate MFU
    actual_flops = flops_per_step / step_time
    mfu = actual_flops / peak_flops
    return min(mfu, 1.0)  # Cap at 100%