#!/usr/bin/env python3
"""
Main training script for speedrun-mup experiments.

This script demonstrates the complete MuP workflow:
1. Load configuration
2. Create MuP-aware models
3. Set up logging and validation
4. Train with comprehensive metrics
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from speedrun_mup.config import ExperimentConfig, MuPConfig, ScalingConfig
from speedrun_mup.models import GPTMuP, apply_mup, make_mup_model
from speedrun_mup.models.gpt import GPTConfig
from speedrun_mup.logging import WandBLogger, MetricsCollector, SpectralMonitor
from speedrun_mup.validation import coord_check, plot_coord_data
from speedrun_mup.utils import set_base_shapes, setup_distributed, cleanup_distributed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MuP-aware GPT model')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration YAML')
    parser.add_argument('--mup-config', type=str, default=None,
                       help='Path to MuP-specific configuration YAML')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Override run name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--coord-check-only', action='store_true',
                       help='Only run coordinate checking, skip training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    return parser.parse_args()


def load_config(config_path: str, mup_config_path: str = None) -> tuple:
    """Load experiment and MuP configurations."""
    # Load main experiment config
    exp_config = ExperimentConfig.from_yaml(config_path)
    
    # Load MuP config if provided
    if mup_config_path:
        with open(mup_config_path, 'r') as f:
            mup_dict = yaml.safe_load(f)
        mup_config = MuPConfig(**mup_dict.get('mup', {}))
        scaling_config = ScalingConfig(**mup_dict.get('scaling', {}))
    else:
        mup_config = MuPConfig()
        scaling_config = ScalingConfig()
    
    return exp_config, mup_config, scaling_config


def create_model(
    scaling_config: ScalingConfig,
    mup_config: MuPConfig,
    use_mup: bool = True
) -> tuple:
    """Create model with MuP parameterization."""
    
    def model_factory(**config_kwargs):
        """Factory function to create models with different configurations."""
        gpt_config = GPTConfig(**config_kwargs)
        return GPTMuP(gpt_config)
    
    if use_mup:
        # Get model configurations
        base_config = scaling_config.get_base_config()
        target_config = scaling_config.get_target_config()
        delta_config = scaling_config.get_delta_config()
        
        # Create models
        base_model = model_factory(**base_config.__dict__)
        target_model = model_factory(**target_config.__dict__)
        
        # Apply MuP parameterization
        if mup_config.base_shapes_file and os.path.exists(mup_config.base_shapes_file):
            # Load pre-computed base shapes
            base_shapes = torch.load(mup_config.base_shapes_file)
            apply_mup(target_model, base_model, base_shapes=base_shapes)
        else:
            # Compute base shapes
            delta_model = model_factory(**delta_config.__dict__)
            apply_mup(target_model, base_model, delta_model)
            
            # Save base shapes if requested
            if mup_config.save_base_shapes and mup_config.base_shapes_file:
                from speedrun_mup.utils.shapes import make_base_shapes, save_base_shapes
                base_shapes = make_base_shapes(base_model, delta_model)
                save_base_shapes(base_shapes, mup_config.base_shapes_file)
        
        return target_model, base_model
    else:
        # Standard model without MuP
        target_config = scaling_config.get_target_config()
        model = model_factory(**target_config.__dict__)
        return model, None


def run_coordinate_check(
    base_model: nn.Module,
    scaling_config: ScalingConfig,
    mup_config: MuPConfig,
    device: str,
    save_dir: str
) -> bool:
    """Run coordinate checking to validate MuP implementation."""
    print("Running coordinate checking...")
    
    def model_factory(width: int):
        """Create model with specified width."""
        config = scaling_config.get_base_config()
        config.n_embd = width
        config.n_head = max(1, width // 64)  # Adjust heads proportionally
        while config.n_embd % config.n_head != 0:
            config.n_head -= 1
        
        model = GPTMuP(config)
        
        # Apply MuP if base model provided
        if base_model is not None:
            # Create a delta model for base shapes
            delta_config = config
            delta_config.n_embd = width * 2  # Different width for delta
            delta_model = GPTMuP(delta_config)
            
            apply_mup(model, base_model, delta_model)
        
        return model
    
    # Test different widths
    widths = [256, 512, 1024]
    
    # Run coordinate check
    coord_results = coord_check(
        model_factory=model_factory,
        widths=widths,
        n_steps=mup_config.coord_check_nsteps,
        n_seeds=mup_config.coord_check_nseeds,
        device=device
    )
    
    # Plot results
    save_path = os.path.join(save_dir, 'coordinate_check.png')
    plot_coord_data(
        coord_results,
        save_path=save_path,
        title="MuP Coordinate Check"
    )
    
    # Validate results
    from speedrun_mup.validation.coord_check import validate_mup_coordinates
    validation_results = validate_mup_coordinates(
        coord_results,
        tolerance=mup_config.coord_check_tolerance
    )
    
    # Print validation summary
    print("\nCoordinate Check Results:")
    print("=" * 40)
    all_passed = True
    for layer_name, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{layer_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All coordinate checks passed!")
    else:
        print("\n❌ Some coordinate checks failed!")
        print("Consider adjusting MuP parameterization.")
    
    return all_passed


def create_dummy_data_loader(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """Create a dummy data loader for testing."""
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size
        
        def __iter__(self):
            for _ in range(self.size):
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                yield {'input_ids': input_ids, 'labels': input_ids}
    
    return DummyDataset()


def main():
    """Main training loop."""
    args = parse_args()
    
    # Load configuration
    exp_config, mup_config, scaling_config = load_config(args.config, args.mup_config)
    
    # Override run name if provided
    if args.run_name:
        exp_config.logging.wandb_run_name = args.run_name
    
    # Disable wandb if requested
    if args.no_wandb:
        exp_config.logging.use_wandb = False
    
    # Setup distributed training if needed
    setup_distributed()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(exp_config.seed)
    
    # Create output directories
    os.makedirs(exp_config.out_dir, exist_ok=True)
    os.makedirs(exp_config.logging.log_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model, base_model = create_model(scaling_config, mup_config, use_mup=mup_config.use_mup)
    model = model.to(device)
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Run coordinate checking
    if mup_config.coord_check_enabled:
        coord_check_passed = run_coordinate_check(
            base_model, scaling_config, mup_config, args.device, exp_config.out_dir
        )
        
        if args.coord_check_only:
            print("Coordinate checking complete. Exiting.")
            return
        
        if not coord_check_passed:
            print("Warning: Coordinate checks failed. Training may not follow MuP assumptions.")
    
    # Initialize logging
    wandb_logger = None
    if exp_config.logging.use_wandb:
        wandb_logger = WandBLogger(
            exp_config.logging,
            project_config=exp_config.to_dict(),
            model_config=model.config.__dict__
        )
        wandb_logger.log_model_info(model, model.config.__dict__)
        wandb_logger.log_system_info()
    
    # Initialize metrics collection
    metrics_collector = MetricsCollector(
        model,
        collect_activations=exp_config.logging.log_activations,
        activation_frequency=exp_config.logging.activation_log_interval
    )
    
    # Initialize spectral monitoring
    spectral_monitor = None
    if exp_config.logging.log_spectral:
        spectral_monitor = SpectralMonitor(model)
    
    # Create optimizer with MuP-appropriate learning rates
    if mup_config.use_mup:
        from speedrun_mup.models.mup_integration import create_mup_param_groups
        param_groups = create_mup_param_groups(model, exp_config.training.learning_rate)
    else:
        param_groups = [{'params': model.parameters(), 'lr': exp_config.training.learning_rate}]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(exp_config.training.beta1, exp_config.training.beta2),
        weight_decay=exp_config.training.weight_decay
    )
    
    # Create dummy data loader (replace with real data loading)
    data_loader = create_dummy_data_loader(
        exp_config.training.micro_batch_size,
        exp_config.training.block_size,
        model.config.vocab_size,
        device
    )
    
    # Training loop
    print("Starting training...")
    model.train()
    
    data_iter = iter(data_loader)
    for step in range(exp_config.training.max_iters):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        
        # Start metrics collection
        metrics_collector.step_start()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], labels=batch['labels'])
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if exp_config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_config.training.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # End metrics collection
        metrics_collector.step_end()
        
        # Collect metrics
        if step % exp_config.logging.log_interval == 0:
            # Collect comprehensive metrics
            mup_metrics = metrics_collector.collect_metrics(loss, optimizer)
            summary_metrics = metrics_collector.get_summary_metrics()
            
            # Log to console
            print(f"Step {step}: loss={loss.item():.4f}, "
                  f"grad_norm={summary_metrics.get('grad_norm_smooth', 0):.4f}")
            
            # Log to wandb
            if wandb_logger:
                wandb_logger.log_training_metrics(
                    loss=loss.item(),
                    learning_rate=optimizer.param_groups[0]['lr'],
                    grad_norm=summary_metrics.get('grad_norm_smooth'),
                    step_time=summary_metrics.get('step_time'),
                    step=step
                )
                wandb_logger.log_mup_metrics(mup_metrics, step=step)
        
        # Spectral monitoring
        if (spectral_monitor and step > 0 and 
            step % exp_config.logging.spectral_log_interval == 0):
            spectral_stats = spectral_monitor.compute_spectral_stats()
            spectral_summary = spectral_monitor.get_summary_metrics(spectral_stats)
            
            print(f"Spectral norms - mean: {spectral_summary.get('spectral_norm_mean', 0):.4f}")
            
            if wandb_logger:
                wandb_logger.log_spectral_stats(spectral_stats, step=step)
                wandb_logger.log_metrics(spectral_summary, step=step, prefix="spectral_summary")
        
        # Save checkpoint
        if step > 0 and step % exp_config.training.save_interval == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': exp_config.to_dict(),
                'mup_config': mup_config.__dict__,
                'scaling_config': scaling_config.__dict__,
            }
            
            save_path = os.path.join(exp_config.out_dir, f'checkpoint_{step}.pt')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint: {save_path}")
    
    # Final checkpoint
    final_checkpoint = {
        'step': exp_config.training.max_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': exp_config.to_dict(),
        'mup_config': mup_config.__dict__,
        'scaling_config': scaling_config.__dict__,
    }
    
    final_save_path = os.path.join(exp_config.out_dir, 'final_model.pt')
    torch.save(final_checkpoint, final_save_path)
    print(f"Saved final model: {final_save_path}")
    
    # Cleanup
    metrics_collector.cleanup()
    if wandb_logger:
        wandb_logger.finish()
    
    cleanup_distributed()
    
    print("Training complete!")


if __name__ == '__main__':
    main()