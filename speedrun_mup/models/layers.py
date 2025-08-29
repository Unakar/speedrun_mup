"""
MuP-aware transformer layers with modded-nanogpt optimizations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..utils.shapes import get_infshape
from ..utils import init as mup_init


class MuPReadout(nn.Module):
    """
    MuP-aware readout layer that applies 1/width scaling to outputs.
    
    This replaces the standard language modeling head with proper MuP scaling.
    """
    
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Create the linear layer but don't initialize yet (will be done by MuP)
        self.linear = nn.Linear(d_model, vocab_size, bias=bias)
        
        # Initialize with zeros (MuP pattern for output layer)
        mup_init.zero_(self.linear.weight)
        if bias:
            mup_init.zero_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MuP scaling."""
        logits = self.linear(x)
        
        # Apply MuP readout scaling if infshape is available
        infshape = get_infshape(self.linear.weight)
        if infshape is not None:
            # Scale by 1/width_mult for MuP
            width_mult = infshape.fanin_mult()
            if width_mult != 1.0:
                logits = logits / width_mult
        
        return logits


class MuPAttention(nn.Module):
    """
    MuP-aware multi-head attention with FlexAttention support.
    
    Integrates MuP scaling with modded-nanogpt's performance optimizations.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        use_flex_attention: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flex_attention = use_flex_attention
        self.qk_norm = qk_norm
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection  
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # QK normalization layers
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_head, bias=False)
            self.k_norm = nn.LayerNorm(self.d_head, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with MuP patterns
        mup_init.kaiming_normal_(self.qkv.weight)
        mup_init.zero_(self.c_proj.weight)  # Zero init for residual connection
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh) 
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        
        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Attention computation with MuP scaling
        if self.use_flex_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized attention with MuP scaling
            # MuP uses 1/d_head scaling instead of 1/sqrt(d_head)
            scale = 1.0 / self.d_head  # MuP scaling
            
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=scale
            )
        else:
            # Manual attention computation
            scale = 1.0 / self.d_head  # MuP scaling
            
            att = (q @ k.transpose(-2, -1)) * scale
            
            if mask is not None:
                att = att.masked_fill(mask == 0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            
            y = att @ v
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y


class MuPMLP(nn.Module):
    """
    MuP-aware MLP with ReLU² activation from modded-nanogpt.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two linear layers
        self.c_fc = nn.Linear(d_model, d_ff, bias=bias)
        self.c_proj = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with MuP patterns
        mup_init.kaiming_normal_(self.c_fc.weight)
        mup_init.zero_(self.c_proj.weight)  # Zero init for residual connection
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x) ** 2  # ReLU² activation from modded-nanogpt
        x = self.dropout(x)
        x = self.c_proj(x)
        return x


class MuPBlock(nn.Module):
    """
    MuP-aware transformer block with optional skip connections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_skip_connections: bool = True,
        **attention_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_skip_connections = use_skip_connections
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        
        # Attention and MLP
        self.attn = MuPAttention(d_model, n_heads, bias=bias, dropout=dropout, **attention_kwargs)
        self.mlp = MuPMLP(d_model, d_ff, bias=bias, dropout=dropout)
        
        # Skip connection weights (if enabled)
        if use_skip_connections:
            self.skip_attn = nn.Parameter(torch.ones(1) * 0.1)
            self.skip_mlp = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        skip_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block
        attn_input = self.ln_1(x)
        if skip_input is not None and self.use_skip_connections:
            attn_input = attn_input + self.skip_attn * skip_input
            
        attn_output = self.attn(attn_input, mask=mask)
        x = x + attn_output
        
        # MLP block  
        mlp_input = self.ln_2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output
        
        # Return output and residual for skip connections
        skip_output = attn_input if self.use_skip_connections else None
        return x, skip_output


class RoPE(nn.Module):
    """
    Rotary Position Embedding with half-truncated frequencies.
    From modded-nanogpt architecture.
    """
    
    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Only use half the dimensions for rotation (half-truncated)
        self.d_rope = d_head // 2
        
        # Precompute frequency inverse
        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_rope, 2).float() / self.d_rope))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)
        
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(-2)
            
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)
        
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        
        return self._apply_rope(x, cos, sin)
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        # Split into rotation and non-rotation parts
        x_rot = x[..., :self.d_rope * 2]  # First d_rope*2 dimensions for rotation
        x_pass = x[..., self.d_rope * 2:]  # Remaining dimensions pass through
        
        # Reshape for rotation
        x1 = x_rot[..., 0::2]  # Even indices
        x2 = x_rot[..., 1::2]  # Odd indices
        
        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_rope)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_rope)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Recombine
        x_rot_out = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
        
        return torch.cat([x_rot_out, x_pass], dim=-1)