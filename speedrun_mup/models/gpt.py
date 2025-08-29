"""
MuP-aware GPT model with modded-nanogpt optimizations.

Combines the competitive training optimizations from modded-nanogpt
with principled MuP scaling for width-invariant hyperparameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .layers import MuPBlock, MuPReadout, RoPE
from ..utils import init as mup_init


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    # Model architecture
    vocab_size: int = 50304  # Padded to multiple of 128 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
    # Sequence length
    block_size: int = 1024
    
    # Architecture options
    bias: bool = False  # Use bias in linear layers
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    
    # MuP-specific options
    use_mup: bool = True
    
    # Modded-nanogpt features
    use_flex_attention: bool = True
    qk_norm: bool = True
    use_skip_connections: bool = True
    use_rope: bool = True
    rope_base: float = 10000.0
    
    # Value embeddings (modded-nanogpt feature)  
    n_value_embd: int = 3  # Number of value embedding layers
    value_embd_mix: bool = True  # Mix value embeddings into attention
    
    # Output scaling
    logit_soft_capping: Optional[float] = 30.0  # Soft cap for logits
    
    def __post_init__(self):
        # Validate configuration
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.vocab_size % 128 == 0, "vocab_size should be multiple of 128 for efficiency"


class GPTMuP(nn.Module):
    """
    MuP-aware GPT model with modded-nanogpt performance optimizations.
    
    Features:
    - MuP parameterization for width-invariant hyperparameters
    - FlexAttention for efficient attention computation
    - U-Net skip connections for better gradient flow
    - RoPE position encodings
    - Value embeddings for richer representations
    - Logit soft-capping for training stability
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                MuPBlock(
                    d_model=config.n_embd,
                    n_heads=config.n_head,
                    d_ff=4 * config.n_embd,
                    bias=config.bias,
                    dropout=config.dropout,
                    layer_norm_eps=config.layer_norm_eps,
                    use_skip_connections=config.use_skip_connections,
                    use_flex_attention=config.use_flex_attention,
                    qk_norm=config.qk_norm,
                ) for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps, bias=config.bias),
        ))
        
        # Value embeddings (modded-nanogpt feature)
        if config.n_value_embd > 0:
            self.value_embeddings = nn.ModuleList([
                nn.Embedding(config.vocab_size, config.n_embd)
                for _ in range(config.n_value_embd)
            ])
            if config.value_embd_mix:
                self.value_mix_weights = nn.Parameter(torch.ones(config.n_value_embd + 1))
        else:
            self.value_embeddings = None
            self.value_mix_weights = None
        
        # RoPE position embeddings
        if config.use_rope:
            d_head = config.n_embd // config.n_head
            self.rope = RoPE(d_head, config.block_size, config.rope_base)
        else:
            self.rope = None
            # Fallback to learned position embeddings
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Language model head with MuP scaling
        if config.use_mup:
            self.lm_head = MuPReadout(config.n_embd, config.vocab_size, bias=config.bias)
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        
        # Tie weights between embedding and output layer (standard practice)
        if hasattr(self.lm_head, 'linear'):
            self.lm_head.linear.weight = self.transformer.wte.weight
        else:
            self.lm_head.weight = self.transformer.wte.weight
        
        # Initialize parameters with MuP-aware initialization
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPTMuP model initialized with {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights with MuP-aware patterns."""
        if isinstance(module, nn.Linear):
            if any(name in str(module) for name in ['c_proj', 'lm_head']):
                # Output projections: zero initialization
                mup_init.zero_(module.weight)
            else:
                # Input projections: Kaiming initialization  
                mup_init.kaiming_normal_(module.weight)
            if module.bias is not None:
                mup_init.zero_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings: normal initialization
            mup_init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            mup_init.zero_(module.bias) if module.bias is not None else None
            mup_init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            position_ids: Position indices [batch_size, seq_len]
            labels: Target labels for language modeling loss
            use_cache: Whether to return attention cache (not implemented)
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.transformer.wte(input_ids)  # [B, T, C]
        
        # Mix in value embeddings if available
        if self.value_embeddings is not None:
            value_embs = []
            for value_emb_layer in self.value_embeddings:
                value_embs.append(value_emb_layer(input_ids))
            
            if self.config.value_embd_mix and self.value_mix_weights is not None:
                # Weighted combination of token and value embeddings
                weights = F.softmax(self.value_mix_weights, dim=0)
                all_embs = [token_emb] + value_embs
                token_emb = sum(w * emb for w, emb in zip(weights, all_embs))
        
        # Position embeddings
        if self.rope is None:
            # Use learned position embeddings
            if position_ids is None:
                position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(position_ids)
            x = token_emb + pos_emb
        else:
            x = token_emb
        
        x = self.transformer.drop(x)
        
        # Transformer blocks with skip connections
        skip_connections = []
        for i, block in enumerate(self.transformer.h):
            # Get skip connection input (U-Net style)
            skip_input = None
            if self.config.use_skip_connections and len(skip_connections) > 0:
                # Connect to corresponding layer from bottom
                skip_idx = len(skip_connections) - 1 - (i - len(skip_connections) // 2)
                if 0 <= skip_idx < len(skip_connections):
                    skip_input = skip_connections[skip_idx]
            
            x, skip_output = block(x, mask=attention_mask, skip_input=skip_input)
            
            if skip_output is not None:
                skip_connections.append(skip_output)
        
        # Apply RoPE to attention layers if using RoPE
        if self.rope is not None:
            # This would typically be applied inside attention layers
            # For now, we assume attention layers handle RoPE internally
            pass
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Apply logit soft-capping if configured
        if self.config.logit_soft_capping is not None:
            logits = self.config.logit_soft_capping * torch.tanh(
                logits / self.config.logit_soft_capping
            )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            'logits': logits,
            'loss': loss,
        }
    
    def configure_optimizers(self, train_config):
        """
        Configure optimizers with MuP-appropriate learning rates.
        
        This method would typically be called by the training loop.
        """
        from ..models.mup_integration import create_mup_param_groups
        
        # Create parameter groups with MuP learning rates
        if self.config.use_mup:
            param_groups = create_mup_param_groups(self, train_config.learning_rate)
        else:
            param_groups = [{'params': self.parameters(), 'lr': train_config.learning_rate}]
        
        # Create optimizer (would be imported from training module)
        # This is a placeholder - actual optimizer creation would be handled by trainer
        return param_groups
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from checkpoint."""
        # Placeholder for loading pretrained models
        raise NotImplementedError("Pretrained model loading not yet implemented")
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters in model."""
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.transformer.wte.weight.numel()
            if hasattr(self.transformer, 'wpe'):
                n_params -= self.transformer.wpe.weight.numel()
            if self.value_embeddings is not None:
                for emb in self.value_embeddings:
                    n_params -= emb.weight.numel()
        
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization."""
        # Rough approximation of FLOPs for transformer
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        
        # FLOPs per token per parameter (approximate)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Express our throughput in tokens per second
        fwdbwd_per_sec = 1.0 / dt
        flops_achieved = flops_per_iter * fwdbwd_per_sec
        
        # A100 peak flops: 312 TFLOPS for bfloat16
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        
        return mfu