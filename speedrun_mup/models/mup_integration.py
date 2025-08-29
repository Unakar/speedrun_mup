"""
Core MuP integration utilities for applying MuP parameterization to models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import warnings

from ..utils.shapes import set_base_shapes, get_infshape
from ..utils import init as mup_init


def apply_mup(
    model: nn.Module,
    base_model: nn.Module,
    delta_model: Optional[nn.Module] = None,
    base_shapes: Optional[Dict[str, Any]] = None,
    rescale_params: bool = True,
    replace_layers: bool = True
) -> nn.Module:
    """
    Apply MuP parameterization to an existing model.
    
    Args:
        model: Target model to apply MuP to
        base_model: Reference model with base dimensions
        delta_model: Model with different dimensions (if base_shapes not provided)
        base_shapes: Pre-computed base shapes (if available)
        rescale_params: Whether to rescale existing parameters
        replace_layers: Whether to replace layers with MuP variants
        
    Returns:
        Model with MuP parameterization applied
    """
    # Set base shapes on all parameters
    set_base_shapes(
        model, 
        base_model, 
        delta_model=delta_model,
        base_shapes=base_shapes,
        rescale_params=rescale_params
    )
    
    # Replace initialization functions with MuP versions
    _replace_initialization(model)
    
    # Optionally replace specific layers with MuP variants
    if replace_layers:
        _replace_layers_with_mup(model)
    
    return model


def make_mup_model(
    model_factory: Callable[[], nn.Module],
    base_config: Dict[str, Any],
    target_config: Dict[str, Any],
    delta_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Create a model with MuP parameterization from config specifications.
    
    Args:
        model_factory: Function that creates model from config
        base_config: Configuration for base model
        target_config: Configuration for target model
        delta_config: Configuration for delta model (if None, use target_config)
        
    Returns:
        Target model with MuP parameterization
    """
    if delta_config is None:
        delta_config = target_config.copy()
    
    # Create models
    base_model = model_factory(**base_config)
    delta_model = model_factory(**delta_config)
    target_model = model_factory(**target_config)
    
    # Apply MuP
    apply_mup(target_model, base_model, delta_model)
    
    return target_model


def _replace_initialization(model: nn.Module) -> None:
    """Replace parameter initialization with MuP-aware versions."""
    for name, param in model.named_parameters():
        infshape = get_infshape(param)
        if infshape is None:
            continue
            
        # Apply MuP-aware initialization based on parameter type
        if 'weight' in name:
            if param.dim() >= 2:  # Matrix-like parameters
                if 'embed' in name.lower():
                    # Embedding layers use normal initialization
                    mup_init.normal_(param, mean=0.0, std=0.02)
                elif 'proj' in name.lower() or 'fc' in name.lower():
                    # Projection layers use Kaiming
                    mup_init.kaiming_normal_(param)
                else:
                    # Default to Kaiming for matrix parameters
                    mup_init.kaiming_normal_(param)
            else:
                # Vector parameters usually stay as standard initialization
                with torch.no_grad():
                    if param.dim() == 1:
                        param.zero_()  # Bias vectors to zero
        elif 'bias' in name:
            # Biases to zero
            mup_init.zero_(param)


def _replace_layers_with_mup(model: nn.Module) -> None:
    """Replace specific layers with MuP-aware variants where beneficial."""
    # This is model-specific and would need to be customized
    # For now, we'll add a warning that this should be implemented
    warnings.warn(
        "Layer replacement not implemented yet. "
        "Use MuP-aware layers directly in model definition for best results."
    )


def validate_mup_setup(model: nn.Module) -> Dict[str, Any]:
    """
    Validate that MuP has been properly set up on a model.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_params': 0,
        'mup_params': 0,
        'missing_infshape': [],
        'scaling_info': {},
        'warnings': []
    }
    
    for name, param in model.named_parameters():
        results['total_params'] += param.numel()
        
        infshape = get_infshape(param)
        if infshape is None:
            results['missing_infshape'].append(name)
        else:
            results['mup_params'] += param.numel()
            
            # Check scaling
            fanin_mult = infshape.fanin_mult()
            fanout_mult = infshape.fanout_mult()
            
            results['scaling_info'][name] = {
                'fanin_mult': fanin_mult,
                'fanout_mult': fanout_mult,
                'shape': tuple(param.shape)
            }
            
            # Check for potential issues
            if fanin_mult > 10 or fanout_mult > 10:
                results['warnings'].append(
                    f"Large scaling factor for {name}: "
                    f"fanin={fanin_mult:.2f}, fanout={fanout_mult:.2f}"
                )
    
    results['mup_coverage'] = results['mup_params'] / results['total_params']
    
    return results


def get_mup_learning_rates(model: nn.Module, base_lr: float) -> Dict[str, float]:
    """
    Compute MuP-appropriate learning rates for different parameter groups.
    
    Args:
        model: Model with MuP parameterization
        base_lr: Base learning rate
        
    Returns:
        Dictionary mapping parameter names to learning rates
    """
    lr_dict = {}
    
    for name, param in model.named_parameters():
        infshape = get_infshape(param)
        if infshape is None:
            # No MuP info, use base learning rate
            lr_dict[name] = base_lr
            continue
        
        # Determine parameter type and scaling
        if param.dim() <= 1:
            # Vector-like parameters (bias, layernorm): LR scales with width
            width_mult = infshape.fanout_mult()
            lr_dict[name] = base_lr * width_mult
        else:
            # Matrix-like parameters: LR scales as 1/width
            if 'embed' in name.lower():
                # Embedding layers: special handling
                lr_dict[name] = base_lr
            else:
                # Standard matrix parameters
                fanin_mult = infshape.fanin_mult()
                lr_dict[name] = base_lr / fanin_mult
    
    return lr_dict


def create_mup_param_groups(model: nn.Module, base_lr: float) -> list:
    """
    Create parameter groups with MuP-appropriate learning rates.
    
    Args:
        model: Model with MuP parameterization
        base_lr: Base learning rate
        
    Returns:
        List of parameter groups for optimizer
    """
    lr_dict = get_mup_learning_rates(model, base_lr)
    
    # Group parameters by learning rate
    lr_groups = {}
    for name, param in model.named_parameters():
        lr = lr_dict[name]
        if lr not in lr_groups:
            lr_groups[lr] = {'params': [], 'lr': lr, 'names': []}
        lr_groups[lr]['params'].append(param)
        lr_groups[lr]['names'].append(name)
    
    # Convert to list format expected by optimizers
    param_groups = []
    for lr, group in lr_groups.items():
        param_groups.append({
            'params': group['params'],
            'lr': lr,
            'names': group['names']  # For debugging
        })
    
    return param_groups