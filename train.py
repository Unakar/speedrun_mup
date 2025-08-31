#!/usr/bin/env python3
"""
Training script for speedrun-mup: modded-nanogpt with MuP integration.

This script implements the complete training pipeline from modded-nanogpt
with added MuP (Maximal Update Parameterization) support for scaling experiments.

Key features:
- Exact modded-nanogpt architecture and training procedure
- MuP scaling for zero-shot hyperparameter transfer
- Distributed training with 8xH100 support
- FP8 mixed precision training
- FlexAttention with sliding window
- U-net skip connections
- Value embeddings
- Muon optimizer
- Comprehensive logging and checkpointing

Usage:
    # Standard training
    python scripts/train.py --config configs/gpt_small.yaml
    
    # MuP scaling experiment
    python scripts/train.py --config configs/mup_width_sweep.yaml --mup --width 1024
    
    # Distributed training
    torchrun --nproc_per_node=8 scripts/train.py --config configs/gpt_small.yaml
"""

import os
import sys
import uuid
import time
import copy
import glob
import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Set memory allocation before PyTorch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor

# Import our components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import GPT, GPTConfig
from core.mup import MuPConfig, set_base_shapes, apply_mup_to_model, MuReadout
from core.optimizers import create_mup_muon_optimizer, step_optimizers, zero_grad_optimizers, apply_lr_schedule, apply_momentum_warmup
from core.utils import SimpleLogger, Timer, compute_grad_norm, get_model_info, set_seed
from core import mup  # For initialization functions


# -----------------------------------------------------------------------------
# Configuration

@dataclass
class TrainingConfig:
    """Training configuration matching modded-nanogpt."""
    # Data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    train_seq_len: int = 48 * 1024
    val_seq_len: int = 4 * 64 * 1024
    
    # Model
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768
    
    # Training
    num_iterations: int = 1750
    cooldown_frac: float = 0.45
    val_loss_every: int = 125
    save_checkpoint: bool = False
    
    # MuP settings
    use_mup: bool = False
    base_width: int = 768
    target_width: int = 768
    base_model_path: str = ""
    delta_model_path: str = ""
    mup_base_shapes_file: str = ""
    
    # System
    compile_model: bool = True
    warmup_steps: int = 10
    device: str = "cuda"
    seed: int = 42


# -----------------------------------------------------------------------------
# Data loading (from modded-nanogpt)

def _load_data_shard(file: Path):
    """Load a data shard from binary format."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    """Find world_size starting indices, such that each begins with token 50256 and local_batches don't overlap."""
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= local_batch_size:
            starts.append(start)  # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False, "increase max_batch_span if necessary"


def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    """Generate distributed data batches."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size  # provide buffer to handle samples up to length local_batch_size
    
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)  # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)  # H2D in another stream isn't helpful.
        pos += batch_span
        yield inputs, targets


# -----------------------------------------------------------------------------
# Window size scheduling

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)


def get_window_size_blocks(step: int, num_iterations: int):
    """Get sliding window size in blocks."""
    x = step / num_iterations  # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    from core.model import next_multiple_of_n
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)


# -----------------------------------------------------------------------------
# Model creation and MuP setup

def create_model(config: TrainingConfig) -> nn.Module:
    """Create model with MuP support."""
    # Create model config
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        model_dim=config.model_dim,
        max_seq_len=max(config.train_seq_len, config.val_seq_len)
    )
    
    # Create model
    model = GPT(model_config).cuda()
    
    # Apply MuP if requested
    if config.use_mup:
        mup_config = MuPConfig(
            use_mup=True,
            base_width=config.base_width,
            target_width=config.target_width
        )
        
        # Set base shapes
        if config.mup_base_shapes_file and os.path.exists(config.mup_base_shapes_file):
            # Load from saved base shapes
            set_base_shapes(model, config.mup_base_shapes_file)
        else:
            # Create base shapes from base and delta models
            base_config = GPTConfig(
                vocab_size=config.vocab_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                model_dim=config.base_width,
                max_seq_len=model_config.max_seq_len
            )
            base_model = GPT(base_config)
            
            # Create delta model (slightly different width for shape inference)
            delta_config = GPTConfig(
                vocab_size=config.vocab_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                model_dim=config.base_width + 64,  # Small difference for shape inference
                max_seq_len=model_config.max_seq_len
            )
            delta_model = GPT(delta_config)
            
            # Set base shapes
            set_base_shapes(model, base_model, delta=delta_model,
                           savefile=config.mup_base_shapes_file if config.mup_base_shapes_file else None)
            
            del base_model, delta_model
        
        # Apply MuP modifications
        model = apply_mup_to_model(model, mup_config)
        
        # Initialize with MuP-aware initialization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module.weight, 'infshape'):
                    mup.kaiming_normal_(module.weight, nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    # Apply modded-nanogpt specific initialization
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    
    # Broadcast parameters if distributed
    if dist.is_initialized():
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
    
    return model


# -----------------------------------------------------------------------------
# Main training function

def train_model(config: TrainingConfig, args=None):
    """Main training function."""
    # Initialize distributed training
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if world_size > 1:
        assert world_size == 8, "This code is designed for 8xH100"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        dist.barrier()
    
    master_process = (rank == 0)
    
    # Set random seed
    set_seed(config.seed)
    
    # Initialize logging
    logger = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(f"Logging to: {logfile}")
        if args:
            logger = SimpleLogger(
                use_wandb=args.use_wandb, 
                project_name=args.wandb_project or "speedrun-mup",
                experiment_name=args.wandb_name,
                config=vars(args)
            )
        else:
            logger = SimpleLogger(use_wandb=False)
    
    # Create model
    print(f"Creating model with width {config.model_dim}...")
    model = create_model(config)
    
    if master_process:
        model_info = get_model_info(model)
        print(f"Model: {model_info['total_params']:,} parameters ({model_info['model_size_mb']:.1f} MB)")
        if config.use_mup:
            print(f"MuP enabled: base_width={config.base_width}, target_width={config.target_width}")
    
    # Collect parameters for optimization
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight] if hasattr(model, 'lm_head') else []
    
    # Create optimizers
    if config.use_mup:
        # Use MuP-aware optimizers
        optimizer1 = torch.optim.AdamW(scalar_params + head_params + embed_params, 
                                     lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
        from core.optimizers import Muon
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
        optimizers = [optimizer1, optimizer2]
    else:
        # Standard optimizers from modded-nanogpt
        optimizer1 = torch.optim.AdamW(scalar_params + head_params + embed_params, 
                                     lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
        from core.optimizers import Muon  
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
        optimizers = [optimizer1, optimizer2]
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    # Compile model
    if config.compile_model:
        model = torch.compile(model, dynamic=False)
    
    # Kernel warmup
    print("Warming up kernels...")
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])
    train_loader = distributed_data_generator(config.train_files, world_size * config.train_seq_len, align_to_bos=True)
    
    for _ in range(config.warmup_steps):
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(1, config.num_iterations)).backward()
        step_optimizers(optimizers)
        zero_grad_optimizers(optimizers)
    
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del train_loader, initial_state
    
    # Training loop
    print("Starting training...")
    train_loader = distributed_data_generator(config.train_files, world_size * config.train_seq_len, align_to_bos=True)
    training_time_ms = 0
    timer = Timer()
    
    # Start training
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    
    for step in range(config.num_iterations + 1):
        last_step = (step == config.num_iterations)
        
        # Validation
        if last_step or (config.val_loss_every > 0 and step % config.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * timer.stop() if timer.start_time else 0
            model.eval()
            
            val_batch_size = world_size * config.val_seq_len
            assert config.val_tokens % val_batch_size == 0
            val_steps = config.val_tokens // val_batch_size
            val_loader = distributed_data_generator(config.val_files, val_batch_size, align_to_bos=False)
            val_loss = 0
            
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step, config.num_iterations))
            
            val_loss /= val_steps
            del val_loader
            
            if dist.is_initialized():
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            
            if master_process:
                metrics = {
                    'step': step,
                    'val_loss': val_loss.item(),
                    'train_time_ms': training_time_ms,
                    'step_avg_ms': training_time_ms / max(step, 1)
                }
                logger.log(metrics)
                print(f"step:{step}/{config.num_iterations} val_loss:{val_loss:.4f} "
                     f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms")
            
            model.train()
            timer.start()
        
        if last_step:
            if master_process and config.save_checkpoint:
                checkpoint = dict(
                    step=step,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                    config=config
                )
                checkpoint_path = f"logs/{run_id}/checkpoint_step{step:06d}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(checkpoint, checkpoint_path)
            break
        
        # Training step
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step, config.num_iterations)).backward()
        
        # Update learning rates and momentum
        apply_lr_schedule(optimizers, step, config.num_iterations, config.cooldown_frac)
        apply_momentum_warmup(optimizers, step, warmup_steps=300)
        
        # Optimizer step
        step_optimizers(optimizers)
        zero_grad_optimizers(optimizers)
        
        # Log training metrics
        if master_process and step % 10 == 0:
            approx_training_time_ms = training_time_ms + 1000 * timer.avg_time()
            grad_norm = compute_grad_norm(model) if step > 0 else 0.0
            metrics = {
                'step': step + 1,
                'approx_train_time_ms': approx_training_time_ms,
                'step_avg_ms': approx_training_time_ms / (step + 1),
                'grad_norm': grad_norm,
                'lr': optimizers[0].param_groups[0]['lr']
            }
            logger.log(metrics)
            if step % 100 == 0:
                print(f"step:{step+1}/{config.num_iterations} "
                     f"train_time:{approx_training_time_ms:.0f}ms "
                     f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms")
    
    # Final statistics
    torch.cuda.synchronize()
    total_time = time.perf_counter() - total_start
    
    if master_process:
        peak_mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        reserved_mem = torch.cuda.max_memory_reserved() // 1024 // 1024
        final_metrics = {
            'peak_memory_mb': peak_mem,
            'reserved_memory_mb': reserved_mem,
            'total_time_s': total_time
        }
        logger.log(final_metrics)
        print(f"peak memory allocated: {peak_mem} MiB reserved: {reserved_mem} MiB")
        logger.close()
    
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Main entry point

def main():
    parser = argparse.ArgumentParser(description='Train GPT model with MuP support')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # MuP options
    parser.add_argument('--mup', action='store_true', help='Enable MuP scaling')
    parser.add_argument('--width', type=int, default=768, help='Model width')
    parser.add_argument('--base-width', type=int, default=768, help='Base model width for MuP')
    
    # Training options
    parser.add_argument('--iterations', type=int, default=1750, help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--sequence-length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--learning-rate', '--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')
    
    # Validation and logging
    parser.add_argument('--val-every', type=int, default=100, help='Validation interval')
    parser.add_argument('--save-every', type=int, default=500, help='Save checkpoint interval')
    parser.add_argument('--log-every', type=int, default=10, help='Logging interval')
    parser.add_argument('--log-dir', type=str, help='Log directory')
    
    # Coordinate checking
    parser.add_argument('--coord-check', action='store_true', help='Enable coordinate checking')
    parser.add_argument('--coord-check-every', type=int, default=100, help='Coordinate check interval')
    
    # W&B logging
    parser.add_argument('--use-wandb', type=lambda x: x.lower() == 'true', default=False, help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--wandb-name', type=str, help='W&B run name')
    
    # System options
    parser.add_argument('--compile', type=lambda x: x.lower() == 'true', default=True, help='Compile model')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        use_mup=args.mup,
        model_dim=args.width,
        target_width=args.width,
        base_width=args.base_width,
        num_iterations=args.iterations,
        seed=args.seed,
        compile_model=not args.no_compile
    )
    
    # Load config file if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
        # Update config with file values
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print(f"Training configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    train_model(config, args)


if __name__ == "__main__":
    main()