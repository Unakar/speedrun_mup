"""
MuP (Maximal Update Parameterization) integration for speedrun architecture.

This implementation follows the standard mup library patterns:
- InfShape dimension tracking system
- Proper MuReadout layer for output scaling
- MuP-aware optimizers (MuAdam, MuSGD)
- Coordinate checking for validation
- Zero-shot hyperparameter transfer

Key MuP principles implemented:
1. Init scaling: σ² ∝ 1/width for matrix parameters  
2. LR scaling: η ∝ 1/width for matrix parameters (Adam), η ∝ 1/√width for vector parameters
3. Output scaling: 1/width for readout layer
4. Attention scaling: 1/d_head instead of 1/√d_head
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any, Union
import math
from collections import defaultdict
from copy import deepcopy
import yaml


# -----------------------------------------------------------------------------
# Core MuP dimension tracking system (based on mup library)

class InfDim:
    """A dimension with a base dimension, used for calculating μP scaling."""
    
    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        """Check if this is an infinite (width) dimension."""
        return self.base_dim is not None

    def width_mult(self):
        """Width multiplier used for calculating μP scaling."""
        if self.isinf():
            return self.dim / self.base_dim
        return 1

    def __repr__(self):
        return f'InfDim({self.base_dim}, {self.dim})'

    def __str__(self):
        if self.isinf():
            return repr(self)
        return f'FinDim({self.dim})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, InfDim):
            return False
        return self.base_dim == other.base_dim and self.dim == other.dim


class InfShape(tuple):
    """A tuple of InfDims, attached to each parameter tensor as p.infshape."""
    
    def __init__(self, *args, **kwargs):
        tuple.__init__(*args, **kwargs)
        for dim in self:
            if not isinstance(dim, InfDim):
                raise ValueError('Elements of InfShape need to be of class InfDim')
        
        # Set main to be the last dimension that is infinite
        # For inf x inf this is fanin, for inf x fin or fin x inf it's the unique inf dim
        self.main_idx = self.main = None
        for i, dim in list(enumerate(self))[::-1]:
            if dim.isinf():
                self.main_idx = i
                self.main = dim
                break

    def fanin_fanout(self):
        """Get fanin and fanout dimensions."""
        assert len(self) >= 2, 'fanin, fanout undefined for 1-dimensional weights'
        return self[1], self[0]
    
    def fanin_fanout_mult_ratio(self):
        """Ratio of fanin to fanout width multipliers."""
        fanin, fanout = self.fanin_fanout()
        return fanin.width_mult() / fanout.width_mult()

    def ninf(self):
        """Number of infinite dimensions."""
        return sum(1 for dim in self if dim.isinf())

    def width_mult(self):
        """Main width multiplier."""
        if self.main is not None:
            return self.main.width_mult()
        return 1
    
    def base_shape(self):
        """Base shape as list."""
        return [d.base_dim for d in self]

    def shape(self):
        """Current shape as list."""
        return [d.dim for d in self]

    @classmethod
    def from_base_shape(cls, base_shape, shape=None):
        """Create InfShape from base shape."""
        if shape is None:
            shape = base_shape
        return cls([InfDim(b, s) for b, s in zip(base_shape, shape)])

    def __repr__(self):
        r = tuple.__repr__(self)[1:-1]
        return f'InfShape([{r}])'


def zip_infshape(base_shape, shape):
    """Create InfShape from base and target shapes."""
    return InfShape([InfDim(b, s) for b, s in zip(base_shape, shape)])


# -----------------------------------------------------------------------------
# Shape management utilities

def get_shapes(model):
    """Get parameter shapes from model."""
    if hasattr(model, "get_shapes"):
        return model.get_shapes()
    return {name: param.shape for name, param in model.named_parameters()}


def get_infshapes(model):
    """Get InfShapes from model parameters."""
    return {name: param.infshape for name, param in model.named_parameters()}


def save_base_shapes(model_or_shapes, file):
    """Save base shapes to YAML file."""
    if isinstance(model_or_shapes, nn.Module):
        sh = get_infshapes(model_or_shapes)
    elif isinstance(model_or_shapes, dict):
        sh = deepcopy(model_or_shapes)
    else:
        raise ValueError("model_or_shapes must be nn.Module or dict")
    
    sh = {k: s.base_shape() for k, s in sh.items()}
    comment = '''# Base shape file for MuP scaling
# - `null` indicates a finite dimension
# - a number indicates the base dimension of an infinite dimension
'''
    with open(file, 'w') as f:
        f.write(comment)
        yaml.dump(sh, f, indent=4)


def load_base_shapes(filename):
    """Load base shapes from YAML file."""
    with open(filename, 'r') as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}


def _dataparallel_hack(base_shapes, shapes):
    """Fix module name discrepancy caused by DataParallel."""
    if all(k.startswith('module.') for k in shapes) and \
        all(not k.startswith('module.') for k in base_shapes):
        return {'module.' + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith('module.') for k in shapes) and \
        all(k.startswith('module.') for k in base_shapes):
        return {k.strip('module.'): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes


def set_base_shapes(model, base, delta=None, savefile=None):
    """Set base shapes for MuP scaling."""
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = make_base_shapes(base_shapes, delta_shapes)
    
    shapes = get_shapes(model)
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    
    # Attach InfShapes to parameters
    for name, param in model.named_parameters():
        if name in base_shapes:
            param.infshape = zip_infshape(base_shapes[name], param.shape)
        else:
            # Create finite InfShape for parameters not in base_shapes
            param.infshape = InfShape([InfDim(None, d) for d in param.shape])
    
    if savefile is not None:
        save_base_shapes(model, savefile)


def make_base_shapes(base_shapes, delta_shapes):
    """Create base shapes by comparing base and delta models."""
    result = {}
    for name in base_shapes:
        if name in delta_shapes:
            base_shape = base_shapes[name]
            delta_shape = delta_shapes[name]
            # Mark dimensions that differ as infinite
            result[name] = [b if b == d else b for b, d in zip(base_shape, delta_shape)]
        else:
            result[name] = base_shapes[name]
    return result


def _extract_shapes(x):
    """Extract shapes from model, dict, or file."""
    if isinstance(x, nn.Module):
        return get_shapes(x)
    elif isinstance(x, dict):
        return deepcopy(x)
    elif isinstance(x, str):
        return load_base_shapes(x)
    else:
        raise ValueError(f'Unsupported type: {type(x)}')


# -----------------------------------------------------------------------------
# MuP-specific layers

class MuReadout(nn.Linear):
    """Readout layer with MuP scaling."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 readout_zero_init: bool = False, output_mult: float = 1.0):
        super().__init__(in_features, out_features, bias)
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        
        if readout_zero_init:
            nn.init.zeros_(self.weight)
            if bias:
                nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        # Apply standard linear layer
        output = super().forward(x)
        
        # Apply MuP output scaling
        if hasattr(self.weight, 'infshape') and self.weight.infshape.ninf() > 0:
            width_mult = self.weight.infshape.width_mult()
            output = output / width_mult
        
        return self.output_mult * output


class MuSharedReadout(MuReadout):
    """Shared readout layer (e.g., tied embeddings)."""
    
    def __init__(self, weight: Tensor, output_mult: float = 1.0):
        # Don't call super().__init__() to avoid creating new parameters
        nn.Module.__init__(self)
        self.weight = weight
        self.bias = None
        self.output_mult = output_mult


def rescale_linear_bias(linear: nn.Linear):
    """Rescale linear layer bias for MuP."""
    if linear.bias is not None and hasattr(linear.weight, 'infshape'):
        if linear.weight.infshape.ninf() == 1:  # Vector-like parameter
            width_mult = linear.weight.infshape.width_mult()
            linear.bias.data.div_(width_mult)


# -----------------------------------------------------------------------------  
# MuP initialization functions

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> Tensor:
    """MuP-aware uniform initialization."""
    with torch.no_grad():
        tensor.uniform_(a, b)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                width_mult = tensor.infshape.width_mult()
                tensor.div_(width_mult ** 0.5)
        return tensor


def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    """MuP-aware normal initialization."""
    with torch.no_grad():
        tensor.normal_(mean, std)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                width_mult = tensor.infshape.width_mult()
                tensor.div_(width_mult ** 0.5)
        return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', 
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    """MuP-aware Kaiming uniform initialization."""
    with torch.no_grad():
        nn.init.kaiming_uniform_(tensor, a, mode, nonlinearity)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                fanin, fanout = tensor.infshape.fanin_fanout()
                if mode == 'fan_in' and fanin.isinf():
                    width_mult = fanin.width_mult()
                    tensor.div_(width_mult ** 0.5)
                elif mode == 'fan_out' and fanout.isinf():
                    width_mult = fanout.width_mult()
                    tensor.div_(width_mult ** 0.5)
        return tensor


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', 
                    nonlinearity: str = 'leaky_relu') -> Tensor:
    """MuP-aware Kaiming normal initialization."""
    with torch.no_grad():
        nn.init.kaiming_normal_(tensor, a, mode, nonlinearity)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                fanin, fanout = tensor.infshape.fanin_fanout()
                if mode == 'fan_in' and fanin.isinf():
                    width_mult = fanin.width_mult()
                    tensor.div_(width_mult ** 0.5)
                elif mode == 'fan_out' and fanout.isinf():
                    width_mult = fanout.width_mult()
                    tensor.div_(width_mult ** 0.5)
        return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    """MuP-aware Xavier uniform initialization."""
    with torch.no_grad():
        nn.init.xavier_uniform_(tensor, gain)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                fanin, fanout = tensor.infshape.fanin_fanout()
                if fanin.isinf() and fanout.isinf():
                    # Both dimensions infinite - use geometric mean
                    width_mult = (fanin.width_mult() * fanout.width_mult()) ** 0.5
                    tensor.div_(width_mult ** 0.5)
        return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    """MuP-aware Xavier normal initialization."""
    with torch.no_grad():
        nn.init.xavier_normal_(tensor, gain)
        if hasattr(tensor, 'infshape'):
            if tensor.infshape.ninf() == 2:  # Matrix-like parameter
                fanin, fanout = tensor.infshape.fanin_fanout()
                if fanin.isinf() and fanout.isinf():
                    # Both dimensions infinite - use geometric mean
                    width_mult = (fanin.width_mult() * fanout.width_mult()) ** 0.5
                    tensor.div_(width_mult ** 0.5)
        return tensor


# -----------------------------------------------------------------------------
# MuP-aware optimizers

def process_param_groups(params, **kwargs):
    """Process parameter groups for MuP optimizers."""
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        if 'lr' not in param_group:
            param_group['lr'] = kwargs['lr']
        if 'weight_decay' not in param_group:
            param_group['weight_decay'] = kwargs.get('weight_decay', 0.)
    return param_groups


def MuAdam(params, impl=torch.optim.Adam, decoupled_wd=False, **kwargs):
    """Adam with μP scaling."""
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g
        
        # Matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = new_group()
        
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (
                f'Parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `set_base_shapes`?')
            
            if p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('More than 2 inf dimensions not supported')
            else:
                vector_like_p['params'].append(p)
        
        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] /= width_mult
            if not decoupled_wd:
                group['weight_decay'] *= width_mult
        
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    
    return impl(new_param_groups, **kwargs)


def MuAdamW(params, **kwargs):
    """AdamW with μP scaling."""
    return MuAdam(params, impl=torch.optim.AdamW, **kwargs)


def MuSGD(params, impl=torch.optim.SGD, decoupled_wd=False, **kwargs):
    """SGD with μP scaling."""
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g
        
        matrix_like_p = defaultdict(new_group)
        vector_like_p = new_group()
        
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (
                f'Parameter with shape {p.shape} does not have `infshape` attribute.')
            
            if p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('More than 2 inf dimensions not supported')
            else:
                vector_like_p['params'].append(p)
        
        # For SGD: matrix parameters get lr/√width, vector parameters get lr*√width  
        for width_mult, group in matrix_like_p.items():
            group['lr'] /= width_mult ** 0.5
            if not decoupled_wd:
                group['weight_decay'] *= width_mult ** 0.5
        
        # Vector parameters
        if vector_like_p['params']:
            # Take the first parameter to get width multiplier
            p = vector_like_p['params'][0]
            if p.infshape.ninf() == 1:
                width_mult = p.infshape.width_mult()
                vector_like_p['lr'] *= width_mult ** 0.5
                if not decoupled_wd:
                    vector_like_p['weight_decay'] /= width_mult ** 0.5
        
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    
    return impl(new_param_groups, **kwargs)


# -----------------------------------------------------------------------------
# Configuration and utilities

@dataclass 
class MuPConfig:
    """Configuration for MuP scaling."""
    use_mup: bool = True
    base_width: int = 768
    target_width: int = 768
    init_std: float = 0.02
    readout_zero_init: bool = True
    output_mult: float = 1.0
    
    # Attention scaling: use 1/d instead of 1/√d
    attn_scale_by_d: bool = True
    
    @property
    def width_mult(self) -> float:
        """Width multiplier from base to target."""
        return self.target_width / self.base_width


def apply_mup_to_model(model: nn.Module, config: MuPConfig):
    """Apply MuP modifications to an existing model."""
    if not config.use_mup:
        return model
    
    # Replace output layer with MuReadout if needed
    if hasattr(model, 'lm_head') and not isinstance(model.lm_head, MuReadout):
        old_head = model.lm_head
        model.lm_head = MuReadout(
            old_head.in_features, 
            old_head.out_features, 
            bias=old_head.bias is not None,
            readout_zero_init=config.readout_zero_init,
            output_mult=config.output_mult
        )
        # Copy weights if not zero-initializing
        if not config.readout_zero_init:
            model.lm_head.weight.data.copy_(old_head.weight.data)
            if old_head.bias is not None:
                model.lm_head.bias.data.copy_(old_head.bias.data)
    
    # Adjust attention scaling if requested
    if config.attn_scale_by_d:
        for module in model.modules():
            if hasattr(module, 'attn_scale') and hasattr(module, 'head_dim'):
                # Change from 1/√d to 1/d
                module.attn_scale = 1.0 / module.head_dim
    
    return model


def create_model_factory(config_class, mup_config: MuPConfig):
    """Create a model factory for coordinate checking."""
    def factory(width: int):
        config = config_class()
        config.model_dim = width
        # Adjust other dimensions proportionally
        config.num_heads = max(1, width // 128)  # Keep head_dim around 128
        return config
    return factory