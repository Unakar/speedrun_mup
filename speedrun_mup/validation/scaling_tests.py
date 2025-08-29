"""
Scaling tests for MuP validation and hyperparameter transfer experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from ..models.mup_integration import apply_mup


@dataclass
class ScalingResult:
    """Results from a scaling test."""
    width: int
    final_loss: float
    min_loss: float
    loss_curve: List[float]
    grad_norms: List[float]
    learning_rate: float
    train_time: float
    converged: bool


def scaling_test(
    model_factory: Callable[[int], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    data_loader: Any,  # DataLoader
    widths: List[int],
    n_steps: int = 1000,
    device: str = 'cuda',
    eval_interval: int = 100,
    convergence_threshold: float = 0.01,
    timeout_minutes: float = 30.0
) -> Dict[int, ScalingResult]:
    """
    Test training dynamics across different model widths.
    
    Args:
        model_factory: Function that creates model given width
        optimizer_factory: Function that creates optimizer given model
        data_loader: Data loader for training
        widths: List of model widths to test
        n_steps: Number of training steps
        device: Device to run on
        eval_interval: Steps between loss evaluations
        convergence_threshold: Loss change threshold for convergence detection
        timeout_minutes: Maximum time per width test
        
    Returns:
        Dictionary mapping widths to scaling results
    """
    results = {}
    
    for width in widths:
        print(f"Testing width {width}...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        # Create model and optimizer
        model = model_factory(width).to(device)
        optimizer = optimizer_factory(model)
        model.train()
        
        # Training loop
        loss_curve = []
        grad_norms = []
        data_iter = iter(data_loader)
        
        min_loss = float('inf')
        last_loss = float('inf')
        converged = False
        
        for step in range(n_steps):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"  Timeout reached for width {width}")
                break
            
            try:
                # Get next batch
                batch = next(data_iter)
            except StopIteration:
                # Reset data iterator
                data_iter = iter(data_loader)
                batch = next(data_iter)
            
            # Move batch to device
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
            else:
                input_ids = batch[0].to(device) if len(batch) > 0 else batch.to(device)
                labels = batch[1].to(device) if len(batch) > 1 else input_ids
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss'] if 'loss' in outputs else outputs['logits'].mean()
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Optimizer step
            optimizer.step()
            
            # Record metrics
            current_loss = loss.item()
            loss_curve.append(current_loss)
            grad_norms.append(total_norm)
            
            min_loss = min(min_loss, current_loss)
            
            # Check convergence
            if step > 0 and step % eval_interval == 0:
                recent_losses = loss_curve[-eval_interval:]
                loss_change = abs(np.mean(recent_losses) - last_loss)
                if loss_change < convergence_threshold:
                    converged = True
                    print(f"  Converged at step {step}")
                    break
                last_loss = np.mean(recent_losses)
            
            # Progress reporting
            if step % (n_steps // 10) == 0:
                print(f"  Step {step}/{n_steps}, Loss: {current_loss:.4f}, Grad norm: {total_norm:.4f}")
        
        # Record results
        train_time = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']
        
        results[width] = ScalingResult(
            width=width,
            final_loss=loss_curve[-1] if loss_curve else float('inf'),
            min_loss=min_loss,
            loss_curve=loss_curve,
            grad_norms=grad_norms,
            learning_rate=lr,
            train_time=train_time,
            converged=converged
        )
        
        print(f"  Finished width {width}: final_loss={results[width].final_loss:.4f}, "
              f"min_loss={min_loss:.4f}, time={train_time:.1f}s")
        
        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
    
    return results


def hyperparameter_transfer_test(
    model_factory: Callable[[int], nn.Module],
    base_width: int,
    target_widths: List[int],
    data_loader: Any,
    base_lr: float,
    n_steps: int = 5000,
    device: str = 'cuda'
) -> Dict[str, Dict[int, ScalingResult]]:
    """
    Test hyperparameter transfer from base width to target widths.
    
    Args:
        model_factory: Function that creates model given width
        base_width: Reference width for hyperparameter tuning
        target_widths: Widths to transfer hyperparameters to
        data_loader: Data loader for training
        base_lr: Base learning rate tuned on base_width
        n_steps: Number of training steps
        device: Device to run on
        
    Returns:
        Dictionary with 'base' and 'transfer' results
    """
    results = {'base': {}, 'transfer': {}}
    
    # First, train base model with base hyperparameters
    print(f"Training base model (width {base_width})...")
    
    def base_optimizer_factory(model):
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.1)
    
    base_results = scaling_test(
        model_factory=model_factory,
        optimizer_factory=base_optimizer_factory,
        data_loader=data_loader,
        widths=[base_width],
        n_steps=n_steps,
        device=device
    )
    results['base'] = base_results
    
    # Now test transfer to target widths with MuP-scaled hyperparameters
    print("Testing hyperparameter transfer...")
    
    def transfer_optimizer_factory(model):
        # This would need to use MuP-appropriate learning rates
        # based on the width multiplier
        width = model.config.n_embd if hasattr(model.config, 'n_embd') else base_width
        width_mult = width / base_width
        
        # MuP scaling: LR scales as 1/width for matrix parameters
        scaled_lr = base_lr / width_mult
        
        return torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=0.1 * width_mult)
    
    transfer_results = scaling_test(
        model_factory=model_factory,
        optimizer_factory=transfer_optimizer_factory,
        data_loader=data_loader,
        widths=target_widths,
        n_steps=n_steps,
        device=device
    )
    results['transfer'] = transfer_results
    
    return results


def plot_scaling_results(
    results: Dict[int, ScalingResult],
    save_path: Optional[str] = None,
    title: str = "Scaling Test Results"
) -> None:
    """
    Plot results from scaling tests.
    
    Args:
        results: Results from scaling_test
        save_path: Path to save plot
        title: Plot title
    """
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    widths = sorted(results.keys())
    
    # Loss curves
    ax = axes[0, 0]
    for width in widths:
        result = results[width]
        steps = range(len(result.loss_curve))
        ax.plot(steps, result.loss_curve, label=f'Width {width}', alpha=0.7)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curves')
    
    # Final loss vs width
    ax = axes[0, 1]
    final_losses = [results[w].final_loss for w in widths]
    min_losses = [results[w].min_loss for w in widths]
    
    ax.loglog(widths, final_losses, 'o-', label='Final Loss', alpha=0.7)
    ax.loglog(widths, min_losses, 's-', label='Min Loss', alpha=0.7)
    ax.set_xlabel('Model Width')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss vs Width')
    
    # Gradient norms
    ax = axes[1, 0]
    for width in widths:
        result = results[width]
        steps = range(len(result.grad_norms))
        # Plot moving average to reduce noise
        window = max(1, len(result.grad_norms) // 50)
        if len(result.grad_norms) > window:
            grad_norms_ma = np.convolve(result.grad_norms, np.ones(window)/window, mode='valid')
            steps_ma = range(window//2, len(result.grad_norms) - window//2)
            ax.plot(steps_ma, grad_norms_ma, label=f'Width {width}', alpha=0.7)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Gradient Norms')
    
    # Training efficiency
    ax = axes[1, 1]
    train_times = [results[w].train_time for w in widths]
    params_per_width = [w**2 for w in widths]  # Rough estimate
    
    ax.loglog(params_per_width, train_times, 'o-', alpha=0.7)
    ax.set_xlabel('Approximate Parameters')
    ax.set_ylabel('Training Time (seconds)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Training Time vs Model Size')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scaling results plot saved to {save_path}")
    
    plt.show()


def analyze_transfer_success(
    transfer_results: Dict[str, Dict[int, ScalingResult]],
    tolerance: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze success of hyperparameter transfer.
    
    Args:
        transfer_results: Results from hyperparameter_transfer_test
        tolerance: Acceptable relative difference in final loss
        
    Returns:
        Analysis summary
    """
    base_results = transfer_results['base']
    transfer_results_data = transfer_results['transfer']
    
    if not base_results:
        return {'error': 'No base results available'}
    
    base_width = list(base_results.keys())[0]
    base_loss = base_results[base_width].final_loss
    
    analysis = {
        'base_width': base_width,
        'base_loss': base_loss,
        'transfer_success': {},
        'summary': {}
    }
    
    successful_transfers = 0
    total_transfers = len(transfer_results_data)
    
    for width, result in transfer_results_data.items():
        relative_diff = abs(result.final_loss - base_loss) / base_loss
        success = relative_diff <= tolerance
        
        analysis['transfer_success'][width] = {
            'final_loss': result.final_loss,
            'relative_diff': relative_diff,
            'success': success
        }
        
        if success:
            successful_transfers += 1
    
    analysis['summary'] = {
        'success_rate': successful_transfers / total_transfers if total_transfers > 0 else 0.0,
        'successful_transfers': successful_transfers,
        'total_transfers': total_transfers,
        'tolerance': tolerance
    }
    
    return analysis