"""
Shape tracking and management for MuP parameterization.

Adapted from Microsoft's mup library for integration with modded-nanogpt architecture.
"""

from typing import Dict, Any, Tuple, Optional, Union, Callable
import torch
import torch.nn as nn
from dataclasses import dataclass
import copy
import warnings


@dataclass
class InfDim:
    """Represents a dimension that can be infinite (scale with width) or finite."""
    size: int
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.size <= 0:
            raise ValueError(f"Dimension size must be positive, got {self.size}")
    
    def __str__(self):
        if self.name:
            return f"{self.name}({self.size})"
        return str(self.size)
    
    def __repr__(self):
        return f"InfDim(size={self.size}, name={self.name!r})"


@dataclass 
class InfShape:
    """Shape information for MuP parameters with base and target dimensions."""
    base_shape: Tuple[InfDim, ...]
    target_shape: Tuple[InfDim, ...]
    
    def __post_init__(self):
        if len(self.base_shape) != len(self.target_shape):
            raise ValueError(
                f"Base and target shapes must have same rank: "
                f"{len(self.base_shape)} vs {len(self.target_shape)}"
            )
    
    @property
    def ndim(self) -> int:
        return len(self.base_shape)
    
    def width_mult(self, dim: int = -1) -> float:
        """Get width multiplier for a specific dimension."""
        if dim < 0:
            dim = self.ndim + dim
        if not (0 <= dim < self.ndim):
            raise ValueError(f"Dimension {dim} out of range for {self.ndim}D shape")
        
        base_size = self.base_shape[dim].size
        target_size = self.target_shape[dim].size
        return target_size / base_size
    
    def fanin_mult(self) -> float:
        """Get fan-in width multiplier (typically the second-to-last dimension)."""
        if self.ndim < 2:
            return 1.0
        return self.width_mult(-2)  # Fan-in is typically second-to-last dimension
    
    def fanout_mult(self) -> float:
        """Get fan-out width multiplier (typically the last dimension)."""
        return self.width_mult(-1)
    
    def __str__(self):
        base_str = "×".join(str(d) for d in self.base_shape)
        target_str = "×".join(str(d) for d in self.target_shape)
        return f"InfShape({base_str} → {target_str})"


def make_base_shapes(
    base_model: nn.Module,
    delta_model: nn.Module,
    savefile: Optional[str] = None
) -> Dict[str, InfShape]:
    """
    Create base shape definitions by comparing base and delta models.
    
    Args:
        base_model: Reference model with base dimensions
        delta_model: Model with different dimensions (typically wider)
        savefile: Optional path to save base shapes
        
    Returns:
        Dictionary mapping parameter names to InfShape objects
    """
    base_shapes = {}
    
    base_params = dict(base_model.named_parameters())
    delta_params = dict(delta_model.named_parameters())
    
    for name, base_param in base_params.items():
        if name not in delta_params:
            warnings.warn(f"Parameter {name} not found in delta model, skipping")
            continue
            
        delta_param = delta_params[name]
        
        if base_param.shape != delta_param.shape:
            # Create InfShape for parameters that differ between models
            base_dims = tuple(InfDim(size=s, name=f"dim_{i}") 
                            for i, s in enumerate(base_param.shape))
            delta_dims = tuple(InfDim(size=s, name=f"dim_{i}") 
                             for i, s in enumerate(delta_param.shape))
            
            base_shapes[name] = InfShape(base_shape=base_dims, target_shape=delta_dims)
    
    if savefile:
        torch.save(base_shapes, savefile)
        print(f"Saved base shapes to {savefile}")
    
    return base_shapes


def set_base_shapes(
    target_model: nn.Module,
    base_model: nn.Module,
    delta_model: Optional[nn.Module] = None,
    base_shapes: Optional[Dict[str, InfShape]] = None,
    rescale_params: bool = True
) -> None:
    """
    Set base shape information on target model parameters.
    
    Args:
        target_model: Model to attach shape information to
        base_model: Reference model with base dimensions
        delta_model: Model with different dimensions (if base_shapes not provided)
        base_shapes: Pre-computed base shapes (if available)
        rescale_params: Whether to rescale existing parameters based on MuP rules
    """
    if base_shapes is None:
        if delta_model is None:
            raise ValueError("Either delta_model or base_shapes must be provided")
        base_shapes = make_base_shapes(base_model, delta_model)
    
    base_params = dict(base_model.named_parameters())
    target_params = dict(target_model.named_parameters())
    
    for name, target_param in target_params.items():
        if name in base_shapes:
            # Use pre-computed base shape
            infshape = base_shapes[name]
            # Update target shape to match actual target parameter
            target_dims = tuple(InfDim(size=s, name=f"dim_{i}") 
                              for i, s in enumerate(target_param.shape))
            infshape = InfShape(base_shape=infshape.base_shape, target_shape=target_dims)
        elif name in base_params:
            # Create identity InfShape for parameters that don't change
            base_param = base_params[name]
            dims = tuple(InfDim(size=s, name=f"dim_{i}") 
                        for i, s in enumerate(base_param.shape))
            infshape = InfShape(base_shape=dims, target_shape=dims)
        else:
            warnings.warn(f"Parameter {name} not found in base model, skipping")
            continue
        
        # Attach InfShape to parameter
        target_param.infshape = infshape
        
        # Rescale existing parameters if requested
        if rescale_params and name in base_shapes:
            _rescale_parameter(target_param, infshape)


def _rescale_parameter(param: torch.Tensor, infshape: InfShape) -> None:
    """Rescale parameter based on MuP rules."""
    if infshape.ndim == 0:
        return  # Scalar parameters don't need rescaling
    
    fanin_mult = infshape.fanin_mult()
    
    if fanin_mult != 1.0:
        # Apply standard MuP rescaling: scale by sqrt(1/fanin_mult)
        scale_factor = (1.0 / fanin_mult) ** 0.5
        with torch.no_grad():
            param.data.mul_(scale_factor)


def load_base_shapes(filename: str) -> Dict[str, InfShape]:
    """Load base shapes from file."""
    return torch.load(filename, map_location='cpu')


def save_base_shapes(base_shapes: Dict[str, InfShape], filename: str) -> None:
    """Save base shapes to file."""
    torch.save(base_shapes, filename)
    print(f"Saved base shapes to {filename}")


def get_infshape(param: torch.Tensor) -> Optional[InfShape]:
    """Get InfShape from parameter if it exists."""
    return getattr(param, 'infshape', None)


def has_infshape(param: torch.Tensor) -> bool:
    """Check if parameter has InfShape information."""
    return hasattr(param, 'infshape')