"""
Metrics collection for MuP experiments.

Comprehensive metrics tracking including standard training metrics
and MuP-specific validation metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque
import time

from ..utils.shapes import get_infshape


@dataclass
class MuPMetrics:
    """Container for MuP-specific metrics."""
    
    # Activation statistics
    activation_means: Dict[str, float] = field(default_factory=dict)
    activation_stds: Dict[str, float] = field(default_factory=dict)
    activation_maxs: Dict[str, float] = field(default_factory=dict)
    
    # Gradient statistics
    grad_norms: Dict[str, float] = field(default_factory=dict)
    param_norms: Dict[str, float] = field(default_factory=dict)
    update_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Width scaling metrics
    width_multiplier: float = 1.0
    coordinate_scaling: Dict[str, float] = field(default_factory=dict)
    
    # Learning rates per parameter group
    learning_rates: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to flat dictionary for logging."""
        metrics = {}
        
        # Flatten nested dictionaries with prefixes
        for key, value in self.activation_means.items():
            metrics[f"activation_mean/{key}"] = value
            
        for key, value in self.activation_stds.items():
            metrics[f"activation_std/{key}"] = value
            
        for key, value in self.activation_maxs.items():
            metrics[f"activation_max/{key}"] = value
            
        for key, value in self.grad_norms.items():
            metrics[f"grad_norm/{key}"] = value
            
        for key, value in self.param_norms.items():
            metrics[f"param_norm/{key}"] = value
            
        for key, value in self.update_ratios.items():
            metrics[f"update_ratio/{key}"] = value
            
        for key, value in self.coordinate_scaling.items():
            metrics[f"coordinate_scaling/{key}"] = value
            
        for key, value in self.learning_rates.items():
            metrics[f"lr/{key}"] = value
        
        metrics["width_multiplier"] = self.width_multiplier
        
        return metrics


class ActivationHook:
    """Hook to collect activation statistics."""
    
    def __init__(self, name: str):
        self.name = name
        self.activations = []
    
    def __call__(self, module, input, output):
        """Hook function to collect activations."""
        if isinstance(output, tuple):
            output = output[0]
        
        if torch.is_tensor(output):
            # Compute statistics
            flat_output = output.detach().flatten()
            stats = {
                'mean': torch.mean(torch.abs(flat_output)).item(),
                'std': torch.std(flat_output).item(),
                'max': torch.max(torch.abs(flat_output)).item(),
            }
            self.activations.append(stats)
    
    def get_latest_stats(self) -> Optional[Dict[str, float]]:
        """Get most recent activation statistics."""
        return self.activations[-1] if self.activations else None
    
    def clear(self):
        """Clear collected statistics."""
        self.activations.clear()


class MetricsCollector:
    """
    Comprehensive metrics collection for MuP experiments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        collect_activations: bool = True,
        activation_layers: Optional[List[str]] = None,
        activation_frequency: int = 100
    ):
        self.model = model
        self.collect_activations = collect_activations
        self.activation_frequency = activation_frequency
        
        # Metrics storage
        self.step_count = 0
        self.activation_hooks = {}
        self.hooks = []
        
        # Rolling averages for smoothed metrics
        self.rolling_losses = deque(maxlen=100)
        self.rolling_grad_norms = deque(maxlen=100)
        
        # Timing
        self.step_start_time = None
        self.step_times = deque(maxlen=100)
        
        # Set up activation collection
        if collect_activations:
            self._setup_activation_hooks(activation_layers)
    
    def _setup_activation_hooks(self, layer_names: Optional[List[str]] = None):
        """Set up hooks to collect activation statistics."""
        if layer_names is None:
            # Default layers to monitor
            layer_names = self._get_default_layers()
        
        for layer_name in layer_names:
            try:
                layer = self._get_layer_by_name(layer_name)
                hook_fn = ActivationHook(layer_name)
                handle = layer.register_forward_hook(hook_fn)
                
                self.activation_hooks[layer_name] = hook_fn
                self.hooks.append(handle)
                
            except (AttributeError, KeyError):
                print(f"Warning: Layer '{layer_name}' not found, skipping")
    
    def _get_default_layers(self) -> List[str]:
        """Get default layers to monitor based on model architecture."""
        layers = []
        
        # Try to identify common layer patterns
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in ['attn', 'mlp', 'embed', 'ln_f']):
                if not any(skip in name for skip in ['weight', 'bias', 'norm']):
                    layers.append(name)
        
        # Limit to avoid too many hooks
        return layers[:10]
    
    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by dot-separated name."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def step_start(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()
    
    def step_end(self):
        """Mark the end of a training step."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            self.step_start_time = None
        
        self.step_count += 1
    
    def collect_metrics(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        collect_detailed: bool = None
    ) -> MuPMetrics:
        """
        Collect comprehensive metrics for current training state.
        
        Args:
            loss: Current training loss
            optimizer: Optimizer being used
            collect_detailed: Whether to collect detailed metrics (uses default frequency if None)
            
        Returns:
            MuPMetrics object with collected statistics
        """
        if collect_detailed is None:
            collect_detailed = self.step_count % self.activation_frequency == 0
        
        metrics = MuPMetrics()
        
        # Basic metrics
        self.rolling_losses.append(loss.item())
        
        # Gradient and parameter norms
        self._collect_grad_and_param_metrics(metrics)
        
        # Learning rates
        self._collect_lr_metrics(optimizer, metrics)
        
        # Activation statistics (if enabled and due)
        if collect_detailed and self.collect_activations:
            self._collect_activation_metrics(metrics)
        
        # Width scaling info
        self._collect_width_metrics(metrics)
        
        return metrics
    
    def _collect_grad_and_param_metrics(self, metrics: MuPMetrics):
        """Collect gradient and parameter norm metrics."""
        total_grad_norm = 0.0
        total_param_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                
                # Per-parameter metrics
                metrics.grad_norms[name] = grad_norm
                metrics.param_norms[name] = param_norm
                
                # Update ratio (relative size of update)
                if param_norm > 0:
                    # Estimate update size (simplified)
                    lr = 1e-4  # This should come from optimizer
                    update_size = lr * grad_norm
                    metrics.update_ratios[name] = update_size / param_norm
                
                # Global norms
                total_grad_norm += grad_norm ** 2
                total_param_norm += param_norm ** 2
        
        # Global norms
        metrics.grad_norms['global'] = total_grad_norm ** 0.5
        metrics.param_norms['global'] = total_param_norm ** 0.5
        
        self.rolling_grad_norms.append(metrics.grad_norms['global'])
    
    def _collect_lr_metrics(self, optimizer: torch.optim.Optimizer, metrics: MuPMetrics):
        """Collect learning rate metrics."""
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get('name', f'group_{i}')
            metrics.learning_rates[group_name] = param_group['lr']
    
    def _collect_activation_metrics(self, metrics: MuPMetrics):
        """Collect activation statistics from hooks."""
        for layer_name, hook in self.activation_hooks.items():
            stats = hook.get_latest_stats()
            if stats:
                metrics.activation_means[layer_name] = stats['mean']
                metrics.activation_stds[layer_name] = stats['std']
                metrics.activation_maxs[layer_name] = stats['max']
        
        # Clear hook data to avoid memory buildup
        for hook in self.activation_hooks.values():
            hook.clear()
    
    def _collect_width_metrics(self, metrics: MuPMetrics):
        """Collect width scaling related metrics."""
        # Find a representative parameter with infshape to get width multiplier
        for param in self.model.parameters():
            infshape = get_infshape(param)
            if infshape is not None:
                metrics.width_multiplier = infshape.fanin_mult()
                break
        
        # Coordinate scaling analysis
        for name, param in self.model.named_parameters():
            infshape = get_infshape(param)
            if infshape is not None:
                # Simple coordinate scaling check
                expected_scale = 1.0 / (infshape.fanin_mult() ** 0.5)
                actual_scale = param.norm().item()
                metrics.coordinate_scaling[name] = actual_scale / expected_scale
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics for logging."""
        summary = {}
        
        # Smoothed loss
        if self.rolling_losses:
            summary['loss_smooth'] = np.mean(list(self.rolling_losses))
            
        # Smoothed gradient norm
        if self.rolling_grad_norms:
            summary['grad_norm_smooth'] = np.mean(list(self.rolling_grad_norms))
        
        # Timing metrics
        if self.step_times:
            summary['step_time'] = np.mean(list(self.step_times))
            summary['steps_per_sec'] = 1.0 / summary['step_time']
        
        # Step count
        summary['step'] = self.step_count
        
        return summary
    
    def cleanup(self):
        """Remove all hooks and clean up resources."""
        for handle in self.hooks:
            handle.remove()
        
        self.hooks.clear()
        self.activation_hooks.clear()


def compute_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for a model given input shape.
    
    This is a simplified estimation - more sophisticated tools like
    fvcore or thop could be used for precise measurements.
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimate: 2 FLOPs per parameter per token
    # (1 for forward, 1 for backward pass)
    batch_size, seq_len = input_shape[:2]
    estimated_flops = total_params * seq_len * 2
    
    return estimated_flops


def compute_tokens_per_second(
    batch_size: int,
    seq_len: int,
    step_time: float
) -> float:
    """Compute tokens processed per second."""
    tokens_per_batch = batch_size * seq_len
    return tokens_per_batch / step_time if step_time > 0 else 0.0