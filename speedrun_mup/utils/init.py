"""
MuP-aware parameter initialization functions.

Drop-in replacements for torch.nn.init functions that automatically
scale based on infshape information attached to parameters.
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from .shapes import get_infshape


def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    """Calculate fan_in and fan_out for a tensor."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, using manual calculation
        for s in tensor.shape[2:]:
            receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _get_mup_scaling(tensor: torch.Tensor) -> float:
    """Get MuP scaling factor based on infshape information."""
    infshape = get_infshape(tensor)
    if infshape is None:
        return 1.0
    
    # MuP scaling: scale by sqrt(1/fanin_mult)
    fanin_mult = infshape.fanin_mult()
    return (1.0 / fanin_mult) ** 0.5 if fanin_mult > 0 else 1.0


def uniform_(tensor: torch.Tensor, a: float = 0.0, b: float = 1.0) -> torch.Tensor:
    """MuP-aware uniform initialization."""
    scale = _get_mup_scaling(tensor)
    with torch.no_grad():
        tensor.uniform_(a * scale, b * scale)
    return tensor


def normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """MuP-aware normal initialization."""
    scale = _get_mup_scaling(tensor)
    with torch.no_grad():
        tensor.normal_(mean, std * scale)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> torch.Tensor:
    """MuP-aware Kaiming uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    
    if mode == 'fan_in':
        num = fan_in
    elif mode == 'fan_out':
        num = fan_out
    else:
        num = (fan_in + fan_out) / 2
    
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(num)
    
    # Apply MuP scaling
    mup_scale = _get_mup_scaling(tensor)
    std *= mup_scale
    
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu'
) -> torch.Tensor:
    """MuP-aware Kaiming normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    
    if mode == 'fan_in':
        num = fan_in
    elif mode == 'fan_out':
        num = fan_out
    else:
        num = (fan_in + fan_out) / 2
    
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(num)
    
    # Apply MuP scaling
    mup_scale = _get_mup_scaling(tensor)
    std *= mup_scale
    
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """MuP-aware Xavier uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    # Apply MuP scaling
    mup_scale = _get_mup_scaling(tensor)
    std *= mup_scale
    
    a = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-a, a)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """MuP-aware Xavier normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    # Apply MuP scaling
    mup_scale = _get_mup_scaling(tensor)
    std *= mup_scale
    
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


def zero_(tensor: torch.Tensor) -> torch.Tensor:
    """Zero initialization (MuP-invariant)."""
    with torch.no_grad():
        tensor.zero_()
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    """Ones initialization (MuP-invariant)."""
    with torch.no_grad():
        tensor.fill_(1.0)
    return tensor