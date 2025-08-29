"""
MuP (Maximal Update Parameterization) integration for speedrun architecture.

Implements the core MuP scaling rules:
1. Init scaling: σ² ∝ 1/width for matrix parameters  
2. LR scaling: η ∝ 1/width for matrix parameters
3. Output scaling: 1/width for readout layer
4. Coordinate checking for validation
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math


@dataclass 
class MuPConfig:
    """Configuration for MuP scaling."""
    base_width: int = 768     # Base model width for scaling reference
    target_width: int = 768   # Target model width
    init_std: float = 0.02    # Base initialization std
    use_mup: bool = True      # Enable MuP scaling
    
    @property
    def width_mult(self) -> float:
        """Width multiplier from base to target."""
        return self.target_width / self.base_width


def apply_mup_init(module: nn.Module, mup_config: MuPConfig):
    """Apply MuP initialization scaling to a module."""
    if not mup_config.use_mup:
        return
        
    width_mult = mup_config.width_mult
    
    for name, param in module.named_parameters():
        if param.dim() >= 2:  # Matrix-like parameters
            if 'embed' not in name.lower() and 'lm_head' not in name.lower():
                # Hidden layer matrices: scale by sqrt(1/width_mult)
                scale_factor = (1.0 / width_mult) ** 0.5
                with torch.no_grad():
                    param.data.mul_(scale_factor)
        
        # Attach MuP metadata for learning rate scaling
        param.mup_width_mult = width_mult
        param.mup_param_type = _get_param_type(name, param)


def _get_param_type(name: str, param: torch.Tensor) -> str:
    """Determine parameter type for MuP scaling."""
    if 'embed' in name.lower():
        return 'embedding'
    elif 'lm_head' in name.lower() or 'c_proj' in name.lower():
        return 'output'
    elif param.dim() >= 2:
        return 'matrix'
    else:
        return 'vector'


def create_mup_optimizer(model: nn.Module, base_lr: float, weight_decay: float, 
                        mup_config: MuPConfig):
    """Create optimizer with MuP-scaled learning rates."""
    if not mup_config.use_mup:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, 
                               weight_decay=weight_decay, betas=(0.9, 0.95))
    
    width_mult = mup_config.width_mult
    param_groups = []
    
    for name, param in model.named_parameters():
        # Get existing lr_mul from modded-nanogpt
        existing_lr_mul = getattr(param, 'lr_mul', 1.0) 
        existing_wd_mul = getattr(param, 'wd_mul', 1.0)
        
        # Apply MuP scaling on top of existing scaling
        param_type = getattr(param, 'mup_param_type', 'matrix')
        
        if param_type == 'matrix':
            # Matrix parameters: LR scales as 1/width
            mup_lr_scale = 1.0 / width_mult
            mup_wd_scale = width_mult  # WD scales opposite to LR
        elif param_type == 'vector':
            # Vector parameters: LR scales with width  
            mup_lr_scale = width_mult
            mup_wd_scale = 1.0
        else:
            # Embeddings and output: no additional MuP scaling
            mup_lr_scale = 1.0
            mup_wd_scale = 1.0
            
        final_lr = base_lr * existing_lr_mul * mup_lr_scale
        final_wd = weight_decay * existing_wd_mul * mup_wd_scale
        
        param_groups.append({
            'params': [param],
            'lr': final_lr,
            'weight_decay': final_wd
        })
    
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)


def apply_mup(model: nn.Module, mup_config: MuPConfig):
    """Apply full MuP parameterization to model."""
    # Apply initialization scaling
    apply_mup_init(model, mup_config)
    
    # Apply output layer scaling for inference
    if hasattr(model, 'lm_head') and mup_config.use_mup:
        original_forward = model.lm_head.forward
        width_mult = mup_config.width_mult
        
        def scaled_forward(x):
            output = original_forward(x)
            return output / width_mult  # MuP readout scaling
            
        model.lm_head.forward = scaled_forward
    
    return model


# Coordinate checking utilities

def coord_check(model_factory, widths: List[int], input_shape: Tuple[int, ...] = (1024,),
                n_steps: int = 3, device: str = 'cuda'):
    """
    Perform coordinate checking to validate MuP implementation.
    
    Args:
        model_factory: Function to create model given width
        widths: List of widths to test
        input_shape: Shape of input sequence 
        n_steps: Number of forward/backward steps
        device: Device to run on
        
    Returns:
        Dictionary with activation statistics per width
    """
    results = {}
    
    for width in widths:
        print(f"Testing width {width}...")
        
        # Create model and apply MuP
        model = model_factory(width).to(device)
        mup_config = MuPConfig(base_width=widths[0], target_width=width)
        apply_mup(model, mup_config)
        model.train()
        
        # Run forward/backward passes
        activations = []
        
        for step in range(n_steps):
            # Create random input
            input_seq = torch.randint(0, model.config.vocab_size, input_shape, device=device)
            
            # Forward pass
            loss = model(input_seq, input_seq)  # Use same seq as target
            
            # Collect activation statistics
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # Get activation magnitude
                        if hasattr(module, 'weight'):
                            act_norm = module.weight.norm().item()
                            activations.append((name, act_norm))
            
            # Backward pass
            loss.backward()
            model.zero_grad()
        
        results[width] = activations
        del model
        torch.cuda.empty_cache()
    
    return results


def plot_coord_check(results: Dict[int, List], save_path: Optional[str] = None):
    """Plot coordinate checking results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        widths = sorted(results.keys())
        
        # Extract layer statistics
        layer_stats = {}
        for width, activations in results.items():
            for name, norm in activations[-10:]:  # Last 10 for brevity
                if name not in layer_stats:
                    layer_stats[name] = {}
                layer_stats[name][width] = norm
        
        # Plot each layer
        for layer_name, width_norms in layer_stats.items():
            if len(width_norms) == len(widths):  # Only plot if all widths present
                norms = [width_norms[w] for w in widths]
                ax.loglog(widths, norms, 'o-', label=layer_name[:20], alpha=0.7)
        
        ax.set_xlabel('Model Width')
        ax.set_ylabel('Parameter Norm')
        ax.set_title('MuP Coordinate Check: Parameter Norms vs Width')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add reference line for ideal MuP behavior (constant)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal (O(1))')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coordinate check plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available, skipping plot")
        
    # Print validation summary  
    print("\nCoordinate Check Summary:")
    print("=" * 40)
    
    widths = sorted(results.keys())
    if len(widths) >= 2:
        base_width = widths[0]
        for width in widths[1:]:
            base_norms = [norm for _, norm in results[base_width][-5:]]
            target_norms = [norm for _, norm in results[width][-5:]]
            
            if base_norms and target_norms:
                ratio = np.mean(target_norms) / np.mean(base_norms)
                status = "✅ PASS" if 0.5 < ratio < 2.0 else "❌ FAIL"
                print(f"Width {width}: {ratio:.2f}x vs base, {status}")


# Simple shape tracking for compatibility with existing MuP libraries

class InfShape:
    """Simplified shape tracking for MuP."""
    def __init__(self, base_shape: Tuple[int, ...], target_shape: Tuple[int, ...]):
        self.base_shape = base_shape
        self.target_shape = target_shape
    
    def width_mult(self, dim: int = -1) -> float:
        return self.target_shape[dim] / self.base_shape[dim]