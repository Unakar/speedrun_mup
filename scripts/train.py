#!/usr/bin/env python3
"""
Speedrun-MuP training script.

Simple, clean implementation based on modded-nanogpt with MuP integration.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

import torch
import torch.nn.functional as F

# Core imports
from model import GPT, GPTConfig
from mup import MuPConfig, apply_mup, create_mup_optimizer
from utils import (
    setup_distributed, cleanup_distributed, get_rank, is_main_process,
    MetricsLogger, Timer, compute_grad_norm, get_model_info, 
    save_checkpoint, set_seed, estimate_mfu
)


def create_dummy_data_loader(batch_size: int, seq_len: int, vocab_size: int = 50304):
    """Create dummy data loader for testing."""
    def data_gen():
        while True:
            # Generate random sequences
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            yield input_ids, targets
    
    return data_gen()


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT with MuP')
    
    # Model config
    parser.add_argument('--model-dim', type=int, default=768)
    parser.add_argument('--num-heads', type=int, default=12) 
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--vocab-size', type=int, default=50304)
    parser.add_argument('--max-seq-len', type=int, default=1024)
    
    # MuP config
    parser.add_argument('--use-mup', action='store_true', default=True)
    parser.add_argument('--base-width', type=int, default=768)
    parser.add_argument('--coord-check', action='store_true')
    
    # Training config
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--warmup-iters', type=int, default=100)
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)
    
    return parser.parse_args()


def get_lr(iter_num: int, warmup_iters: int, max_iters: int, 
           learning_rate: float, min_lr: float = 0.0):
    """Get learning rate with warmup and cosine decay."""
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    
    if iter_num > max_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def run_coordinate_check(args):
    """Run coordinate checking across different widths."""
    print("Running coordinate check...")
    
    from mup import coord_check, plot_coord_check
    
    widths = [256, 512, 768, 1024]
    
    def model_factory(width):
        config = GPTConfig(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            model_dim=width,
            num_heads=max(1, width // 64),
            num_layers=args.num_layers
        )
        return GPT(config)
    
    results = coord_check(model_factory, widths, device=args.device)
    plot_coord_check(results, save_path=f"{args.log_dir}/coord_check.png")
    
    return results


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_distributed()
    
    if is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        print(f"Training GPT with MuP: {args}")
    
    # Run coordinate check if requested
    if args.coord_check:
        run_coordinate_check(args)
        if is_main_process():
            print("Coordinate check complete. Exiting.")
        cleanup_distributed()
        return
    
    # Initialize logging
    logger = MetricsLogger(
        project_name="speedrun-mup",
        use_wandb=args.wandb and is_main_process(),
        log_dir=args.log_dir if is_main_process() else None
    )
    timer = Timer()
    
    # Create model
    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    
    model = GPT(config).to(args.device)
    
    if is_main_process():
        model_info = get_model_info(model)
        print(f"Model: {model_info}")
        logger.log(model_info)
    
    # Apply MuP
    mup_config = MuPConfig(
        base_width=args.base_width,
        target_width=args.model_dim,
        use_mup=args.use_mup
    )
    
    if args.use_mup:
        apply_mup(model, mup_config)
        if is_main_process():
            print(f"Applied MuP scaling: {args.base_width} -> {args.model_dim}")
    
    # Create optimizer
    if args.use_mup:
        optimizer = create_mup_optimizer(model, args.learning_rate, args.weight_decay, mup_config)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
    
    # Compile model
    if args.compile:
        model = torch.compile(model)
        if is_main_process():
            print("Model compiled")
    
    # Create data loader
    data_loader = create_dummy_data_loader(args.batch_size, args.seq_len, args.vocab_size)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
    if is_main_process():
        print("Starting training...")
    
    for iter_num in range(args.max_iters):
        timer.start()
        
        # Get learning rate
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * getattr(param_group, 'lr_mul', 1.0)
        
        # Get batch
        input_ids, targets = next(data_loader)
        input_ids = input_ids.to(args.device)
        targets = targets.to(args.device)
        
        # Forward pass - reshape for model input format
        if input_ids.dim() == 2:  # (batch, seq) -> (seq,) for each sample
            losses = []
            for i in range(input_ids.size(0)):
                seq_input = input_ids[i]  # (seq,)
                seq_target = targets[i]   # (seq,)
                loss = model(seq_input, seq_target)
                losses.append(loss)
            loss = torch.stack(losses).mean()
        else:
            loss = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = compute_grad_norm(model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Timing
        step_time = timer.stop()
        tokens_per_sec = timer.throughput(args.batch_size, args.seq_len)
        
        # Logging
        if iter_num % args.log_interval == 0:
            # Estimate MFU
            mfu = estimate_mfu(
                model_info.get('total_params', 0) if 'model_info' in locals() else 0,
                args.batch_size, args.seq_len, step_time
            )
            
            metrics = {
                'loss': loss.item(),
                'lr': lr,
                'grad_norm': grad_norm,
                'step_time': step_time,
                'tokens_per_sec': tokens_per_sec,
                'mfu': mfu,
                'iter': iter_num
            }
            
            logger.log(metrics, step=iter_num)
            
            if is_main_process():
                logger.print_metrics()
        
        # Save checkpoint
        if iter_num > 0 and iter_num % args.save_interval == 0 and is_main_process():
            checkpoint_path = f"{args.log_dir}/checkpoint_{iter_num}.pt"
            save_checkpoint(
                model, optimizer, iter_num, loss.item(),
                vars(args), checkpoint_path
            )
        
        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # Final checkpoint
    if is_main_process():
        final_path = f"{args.log_dir}/final_model.pt"
        save_checkpoint(model, optimizer, args.max_iters, best_loss, vars(args), final_path)
        print(f"Training complete. Best loss: {best_loss:.4f}")
    
    # Cleanup
    logger.close()
    cleanup_distributed()


if __name__ == '__main__':
    import math  # For LR scheduling
    main()