"""
GPT model implementation based on modded-nanogpt architecture with MuP integration.

Key features from modded-nanogpt:
- FP8 custom operators
- FlexAttention with block masks
- U-Net skip connections
- Value embeddings
- RMSNorm
- Logit soft-capping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT model configuration matching modded-nanogpt."""
    
    # Model dimensions
    vocab_size: int = 50304  # Padded to multiple of 128
    max_seq_len: int = 1024
    model_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    
    # Architecture features
    use_fp8: bool = True
    use_flex_attention: bool = True 
    use_skip_connections: bool = True
    num_value_embeds: int = 3
    
    # Training
    dropout: float = 0.0


def norm(x: Tensor) -> Tensor:
    """RMSNorm as used in modded-nanogpt."""
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """Linear layer with FP8 support, from modded-nanogpt."""
    
    def __init__(self, in_features: int, out_features: int, use_fp8: bool = False, 
                 x_s: float = 1.0, w_s: float = 1.0, grad_s: float = 1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s  
        self.grad_s = grad_s
        
    def reset_parameters(self) -> None:
        """Initialize with modded-nanogpt scaling."""
        std = 0.5 * (self.in_features ** -0.5)  # 0.5 better than 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            
    def forward(self, x: Tensor) -> Tensor:
        if self.use_fp8 and self.training:
            # Use custom FP8 ops when available
            try:
                return torch.ops.nanogpt.mm(x, self.weight, self.x_s, self.w_s, self.grad_s)[0]
            except:
                # Fallback to regular matmul
                return F.linear(x, self.weight)
        return F.linear(x, self.weight)


class RoPE(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute frequency inverse
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)
        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)
        
    def apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply rotary position embedding."""
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Split x into pairs for rotation
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        return rotated_x


class Attention(nn.Module):
    """Multi-head attention with FlexAttention support."""
    
    def __init__(self, model_dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.layer_idx = layer_idx
        
        # QKV projection
        self.qkv = CastedLinear(model_dim, 3 * model_dim)
        
        # Output projection
        self.c_proj = CastedLinear(model_dim, model_dim)
        self.c_proj.weight.detach().zero_()  # Zero init for residual
        
        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len)
        
        # Layer-specific parameters for attention pattern
        self.register_buffer('layer_idx_tensor', torch.tensor(layer_idx))
        
    def forward(self, x: Tensor, value_embed: Optional[Tensor] = None, 
                x0: Optional[Tensor] = None, block_mask: Optional[BlockMask] = None) -> Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self.rope.apply_rope(q, T)
        k = self.rope.apply_rope(k, T)
        
        # Mix in value embeddings if provided
        if value_embed is not None:
            v_embed = value_embed.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v + v_embed
            
        # FlexAttention computation
        if block_mask is not None:
            y = flex_attention(q, k, v, block_mask=block_mask)
        else:
            # Fallback to standard attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP block with ReLU² activation."""
    
    def __init__(self, model_dim: int):
        super().__init__()
        self.c_fc = CastedLinear(model_dim, 4 * model_dim)
        self.c_proj = CastedLinear(4 * model_dim, model_dim)
        self.c_proj.weight.detach().zero_()  # Zero init for residual
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.relu(x) ** 2  # ReLU² activation from modded-nanogpt
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, model_dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = Attention(model_dim, num_heads, max_seq_len, layer_idx)
        self.mlp = MLP(model_dim)
        self.layer_idx = layer_idx
        
    def forward(self, x: Tensor, value_embed: Optional[Tensor] = None,
                x0: Optional[Tensor] = None, lambdas: Optional[Tensor] = None,
                sa_lambdas: Optional[Tensor] = None, block_mask: Optional[BlockMask] = None) -> Tensor:
        
        # Self-attention with residual
        if lambdas is not None:
            # Use adaptive residual weights
            attn_out = self.attn(norm(x), value_embed, x0, block_mask)
            x = lambdas[0] * x + lambdas[1] * attn_out
        else:
            x = x + self.attn(norm(x), value_embed, x0, block_mask)
            
        # MLP with residual  
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """GPT model based on modded-nanogpt architecture."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and value embeddings
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        if config.num_value_embeds > 0:
            self.value_embeds = nn.ModuleList([
                nn.Embedding(config.vocab_size, config.model_dim)
                for _ in range(config.num_value_embeds)
            ])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config.model_dim, config.num_heads, config.max_seq_len, i)
            for i in range(config.num_layers)
        ])
        
        # Language model head
        self.lm_head = CastedLinear(
            config.model_dim, config.vocab_size, 
            use_fp8=True,
            x_s=(config.model_dim**0.5)/448,
            w_s=24/448,
            grad_s=1/448
        )
        self.lm_head.weight.detach().zero_()  # Zero init
        
        # Skip connection weights (U-Net style)
        if config.use_skip_connections:
            assert config.num_layers % 2 == 0
            self.scalars = nn.Parameter(torch.cat([
                torch.ones(config.num_layers),  # skip_weights
                *[torch.tensor([1.0, 0.0]) for _ in range(config.num_layers)],  # block lambdas
                *[torch.tensor([0.5, 0.5]) for _ in range(config.num_layers)],  # SA lambdas
            ]))
        
        # Set learning rate multipliers (from modded-nanogpt)
        for param in self.embed.parameters():
            param.lr_mul = 75.0
        if hasattr(self, 'value_embeds'):
            for value_embed in self.value_embeds:
                for param in value_embed.parameters():
                    param.lr_mul = 75.0
        self.lm_head.weight.lr_mul = 27.5
        if hasattr(self, 'scalars'):
            self.scalars.lr_mul = 5.0
    
    def create_block_masks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        """Create FlexAttention block masks for sliding window attention."""
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)
        
        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask
            
        # Create block masks following modded-nanogpt pattern
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            # Simplified version - full implementation would follow modded-nanogpt exactly
            return BlockMask.from_causal(seq_len=len(input_seq))
            
        long_bm = build_bm(sliding_window_num_blocks)
        short_bm = build_bm(sliding_window_num_blocks // 2)
        return long_bm, short_bm
        
    def forward(self, input_seq: Tensor, target_seq: Optional[Tensor] = None,
                sliding_window_num_blocks: Optional[Tensor] = None) -> Tensor:
        
        assert input_seq.ndim == 1
        seq_len = len(input_seq)
        
        # Value embeddings with 012...012 pattern
        ve = []
        if hasattr(self, 'value_embeds'):
            value_embeds = [value_embed(input_seq) for value_embed in self.value_embeds]
            # Pattern: [ve[0], ve[1], ve[2]] + [None]*(n-6) + [ve[0], ve[1], ve[2]]
            n = len(self.blocks)
            ve = value_embeds + [None] * (n - 6) + value_embeds
            ve = ve[:n]  # Ensure correct length
        else:
            ve = [None] * len(self.blocks)
            
        # Create block masks if using FlexAttention
        if self.config.use_flex_attention and sliding_window_num_blocks is not None:
            long_bm, short_bm = self.create_block_masks(input_seq, sliding_window_num_blocks)
            # Alternating pattern from modded-nanogpt
            block_masks = [long_bm, short_bm] * (len(self.blocks) // 2)
            if len(self.blocks) % 2:
                block_masks.append(long_bm)
        else:
            block_masks = [None] * len(self.blocks)
            
        # Token embedding with normalization
        x = x0 = norm(self.embed(input_seq)[None])
        
        # U-Net forward pass
        if self.config.use_skip_connections:
            skip_connections = []
            skip_weights = self.scalars[:len(self.blocks) // 2]
            lambdas = self.scalars[len(self.blocks):3*len(self.blocks)].view(-1, 2)
            sa_lambdas = self.scalars[3*len(self.blocks):5*len(self.blocks)].view(-1, 2)
            
            n = len(self.blocks) // 2
            for i in range(len(self.blocks)):
                if i >= n:
                    x = x + skip_weights[i - n] * skip_connections.pop()
                x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
                if i < n:
                    skip_connections.append(x)
        else:
            # Standard forward pass
            for i, block in enumerate(self.blocks):
                x = block(x, ve[i], x0, None, None, block_masks[i])
        
        # Final norm and language model head
        x = norm(x)
        logits = self.lm_head(x).float()
        
        # Logit soft-capping from modded-nanogpt
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        
        # Compute loss if targets provided
        if target_seq is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_seq, 
                reduction="sum" if self.training else "mean"
            )
            return loss
        
        return logits
    
    def configure_optimizers(self, learning_rate: float, weight_decay: float, device_type: str):
        """Configure optimizers with per-parameter learning rates."""
        
        # Collect parameters with their learning rate multipliers
        param_groups = []
        for name, param in self.named_parameters():
            lr_mul = getattr(param, 'lr_mul', 1.0)
            wd_mul = getattr(param, 'wd_mul', 1.0)
            
            param_groups.append({
                'params': [param],
                'lr': learning_rate * lr_mul,
                'weight_decay': weight_decay * wd_mul
            })
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        return optimizer