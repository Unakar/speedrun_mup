"""
Coordinate checking for MuP validation.

Implements the empirical validation method from the original MuP paper
to verify that activations remain O(1) across different model widths.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

from ..models.mup_integration import apply_mup
from ..utils.shapes import get_infshape


@dataclass
class CoordinateStats:
    """Statistics for coordinate checking."""
    mean: float
    std: float
    max_abs: float
    width_mult: float
    layer_name: str
    activation_type: str  # 'input', 'output', 'hidden'


class ActivationCollector:
    """Collects activations from specified layers during forward pass."""
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        
        def make_hook(name: str):
            def hook(module, input, output):
                # Store both input and output activations
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                self.activations[f"{name}_input"] = input_tensor.detach()
                self.activations[f"{name}_output"] = output_tensor.detach()
            
            return hook
        
        # Register hooks on specified layers
        for layer_name in self.layer_names:
            try:
                layer = self._get_layer_by_name(layer_name)
                hook = layer.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)
            except AttributeError:
                print(f"Warning: Layer {layer_name} not found in model")
    
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
    
    def clear(self):
        """Clear collected activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def coord_check(
    model_factory: Callable[[int], nn.Module],
    widths: List[int],
    input_shape: Tuple[int, ...] = (8, 512),
    n_steps: int = 3,
    n_seeds: int = 1,
    device: str = 'cuda',
    layer_names: Optional[List[str]] = None
) -> Dict[str, List[CoordinateStats]]:
    """
    Perform coordinate checking across different model widths.
    
    Args:
        model_factory: Function that creates model given width
        widths: List of model widths to test
        input_shape: Shape of input tensor (batch_size, seq_len)
        n_steps: Number of forward/backward steps
        n_seeds: Number of random seeds to average over
        device: Device to run on
        layer_names: Specific layers to monitor (if None, uses defaults)
        
    Returns:
        Dictionary mapping layer names to coordinate statistics
    """
    if layer_names is None:
        # Default layers to check - these should be updated based on model architecture
        layer_names = [
            'transformer.h.0.attn',
            'transformer.h.0.mlp', 
            f'transformer.h.{6}.attn',  # Middle layer
            f'transformer.h.{6}.mlp',
            'transformer.ln_f',
        ]
    
    results = defaultdict(list)
    
    for seed in range(n_seeds):
        torch.manual_seed(42 + seed)  # Deterministic but different seeds
        
        for width in widths:
            print(f"Checking width {width} (seed {seed})")
            
            # Create model
            model = model_factory(width).to(device)
            model.train()
            
            # Set up activation collection
            collector = ActivationCollector(model, layer_names)
            
            # Create random input
            batch_size, seq_len = input_shape
            input_ids = torch.randint(0, model.config.vocab_size, 
                                    (batch_size, seq_len), device=device)
            
            # Run forward/backward steps
            for step in range(n_steps):
                collector.clear()
                
                # Forward pass
                outputs = model(input_ids)
                loss = outputs['loss'] if 'loss' in outputs else outputs['logits'].sum()
                
                # Backward pass
                loss.backward()
                
                # Collect statistics from activations
                for act_name, activation in collector.activations.items():
                    stats = _compute_activation_stats(activation, width, act_name)
                    results[f"{act_name}_seed{seed}"].append(stats)
                
                # Clear gradients for next step
                model.zero_grad()
            
            # Clean up
            collector.remove_hooks()
            del model
            torch.cuda.empty_cache()
    
    # Average statistics across seeds
    averaged_results = _average_coordinate_stats(results, n_seeds)
    return averaged_results


def _compute_activation_stats(
    activation: torch.Tensor, 
    width: int, 
    layer_name: str
) -> CoordinateStats:
    """Compute coordinate statistics for an activation tensor."""
    # Flatten spatial dimensions but keep batch dimension
    flat_act = activation.view(activation.size(0), -1)
    
    # Compute statistics across batch and feature dimensions
    mean_abs = torch.mean(torch.abs(flat_act)).item()
    std = torch.std(flat_act).item()
    max_abs = torch.max(torch.abs(flat_act)).item()
    
    return CoordinateStats(
        mean=mean_abs,
        std=std,
        max_abs=max_abs,
        width_mult=width / 256.0,  # Normalize by base width
        layer_name=layer_name,
        activation_type=_get_activation_type(layer_name)
    )


def _get_activation_type(layer_name: str) -> str:
    """Determine activation type from layer name."""
    if 'input' in layer_name:
        return 'input'
    elif 'output' in layer_name:
        return 'output'
    else:
        return 'hidden'


def _average_coordinate_stats(
    results: Dict[str, List[CoordinateStats]], 
    n_seeds: int
) -> Dict[str, List[CoordinateStats]]:
    """Average coordinate statistics across seeds."""
    if n_seeds == 1:
        # Remove seed suffix from keys
        return {k.replace('_seed0', ''): v for k, v in results.items()}
    
    # Group by layer name and average
    averaged = defaultdict(list)
    layer_names = set(k.split('_seed')[0] for k in results.keys())
    
    for layer_name in layer_names:
        # Collect all stats for this layer across seeds
        all_stats = []
        for seed in range(n_seeds):
            seed_key = f"{layer_name}_seed{seed}"
            if seed_key in results:
                all_stats.extend(results[seed_key])
        
        # Group by width and average
        width_groups = defaultdict(list)
        for stat in all_stats:
            width_groups[stat.width_mult].append(stat)
        
        for width_mult, stat_group in width_groups.items():
            avg_stat = CoordinateStats(
                mean=np.mean([s.mean for s in stat_group]),
                std=np.mean([s.std for s in stat_group]),
                max_abs=np.mean([s.max_abs for s in stat_group]),
                width_mult=width_mult,
                layer_name=stat_group[0].layer_name,
                activation_type=stat_group[0].activation_type
            )
            averaged[layer_name].append(avg_stat)
    
    return dict(averaged)


def plot_coord_data(
    coord_stats: Dict[str, List[CoordinateStats]],
    save_path: Optional[str] = None,
    show_plot: bool = True,
    title: str = "MuP Coordinate Check"
) -> None:
    """
    Plot coordinate checking results.
    
    Args:
        coord_stats: Results from coord_check
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    metrics = ['mean', 'std', 'max_abs']
    metric_titles = ['Mean Absolute Activation', 'Standard Deviation', 'Max Absolute Activation']
    
    for i, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot each layer
        for layer_name, stats_list in coord_stats.items():
            if not stats_list:
                continue
                
            widths = [s.width_mult for s in stats_list]
            values = [getattr(s, metric) for s in stats_list]
            
            ax.loglog(widths, values, 'o-', label=layer_name, alpha=0.7)
        
        ax.set_xlabel('Width Multiplier')
        ax.set_ylabel(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add reference lines for ideal MuP behavior (constant across widths)
        if len(coord_stats) > 0:
            widths_range = [min(s.width_mult for stats in coord_stats.values() for s in stats),
                          max(s.width_mult for stats in coord_stats.values() for s in stats)]
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (O(1))')
    
    # Use remaining subplot for summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Add summary text
    summary_text = []
    summary_text.append("MuP Validation Summary:")
    summary_text.append("=" * 25)
    
    for layer_name, stats_list in coord_stats.items():
        if not stats_list:
            continue
            
        # Check if activations are roughly O(1) across widths
        means = [s.mean for s in stats_list]
        width_variation = max(means) / min(means) if min(means) > 0 else float('inf')
        
        status = "✓ PASS" if width_variation < 3.0 else "✗ FAIL"
        summary_text.append(f"{layer_name}: {status}")
        summary_text.append(f"  Width variation: {width_variation:.2f}x")
    
    ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Coordinate check plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def validate_mup_coordinates(
    coord_stats: Dict[str, List[CoordinateStats]],
    tolerance: float = 3.0
) -> Dict[str, bool]:
    """
    Validate that coordinate statistics satisfy MuP assumptions.
    
    Args:
        coord_stats: Results from coord_check
        tolerance: Maximum allowed variation across widths
        
    Returns:
        Dictionary mapping layer names to pass/fail status
    """
    results = {}
    
    for layer_name, stats_list in coord_stats.items():
        if not stats_list:
            results[layer_name] = False
            continue
        
        # Check variation in mean activation across widths
        means = [s.mean for s in stats_list]
        if len(means) < 2:
            results[layer_name] = True  # Single width always passes
            continue
        
        variation = max(means) / min(means) if min(means) > 0 else float('inf')
        results[layer_name] = variation <= tolerance
    
    return results