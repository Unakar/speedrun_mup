"""
Weights & Biases logging integration for MuP experiments.
"""

import os
from typing import Dict, Any, Optional, List
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb")

from .metrics import MuPMetrics, MetricsCollector
from .spectral import SpectralStats, SpectralMonitor
from ..config.base import LoggingConfig


class WandBLogger:
    """
    Weights & Biases logger for MuP experiments.
    
    Handles initialization, metric logging, and experiment tracking
    with MuP-specific enhancements.
    """
    
    def __init__(
        self,
        config: LoggingConfig,
        project_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.enabled = config.use_wandb and WANDB_AVAILABLE
        self.run = None
        
        if self.enabled:
            self._initialize_wandb(project_config, model_config)
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Skipping W&B logging.")
    
    def _initialize_wandb(
        self,
        project_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize wandb run."""
        # Prepare config for logging
        wandb_config = {}
        if project_config:
            wandb_config.update(project_config)
        if model_config:
            wandb_config['model'] = model_config
        
        # Initialize run
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=wandb_config,
            reinit=True
        )
        
        print(f"Initialized W&B run: {self.run.name}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """Log metrics to wandb."""
        if not self.enabled or self.run is None:
            return
        
        # Add prefix to metric names if provided
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics
        
        self.run.log(prefixed_metrics, step=step)
    
    def log_mup_metrics(
        self,
        mup_metrics: MuPMetrics,
        step: Optional[int] = None
    ):
        """Log MuP-specific metrics."""
        if not self.enabled:
            return
        
        metrics_dict = mup_metrics.to_dict()
        self.log_metrics(metrics_dict, step=step, prefix="mup")
    
    def log_spectral_stats(
        self,
        spectral_stats: Dict[str, SpectralStats],
        step: Optional[int] = None
    ):
        """Log spectral monitoring statistics."""
        if not self.enabled:
            return
        
        # Log per-layer spectral statistics
        for layer_name, stats in spectral_stats.items():
            layer_metrics = stats.to_dict()
            self.log_metrics(
                layer_metrics,
                step=step,
                prefix=f"spectral/{layer_name}"
            )
    
    def log_training_metrics(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        step_time: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        step: Optional[int] = None
    ):
        """Log standard training metrics."""
        if not self.enabled:
            return
        
        metrics = {
            'train/loss': loss,
            'train/lr': learning_rate,
        }
        
        if grad_norm is not None:
            metrics['train/grad_norm'] = grad_norm
        
        if step_time is not None:
            metrics['train/step_time'] = step_time
            metrics['train/steps_per_sec'] = 1.0 / step_time
        
        if tokens_per_sec is not None:
            metrics['train/tokens_per_sec'] = tokens_per_sec
        
        self.log_metrics(metrics, step=step)
    
    def log_validation_metrics(
        self,
        val_loss: float,
        val_perplexity: Optional[float] = None,
        step: Optional[int] = None
    ):
        """Log validation metrics."""
        if not self.enabled:
            return
        
        metrics = {'val/loss': val_loss}
        if val_perplexity is not None:
            metrics['val/perplexity'] = val_perplexity
        
        self.log_metrics(metrics, step=step)
    
    def log_coordinate_check_results(
        self,
        coord_results: Dict[str, Any],
        step: Optional[int] = None
    ):
        """Log coordinate checking results."""
        if not self.enabled:
            return
        
        # Flatten coordinate check results
        flattened = {}
        for layer_name, stats_list in coord_results.items():
            if isinstance(stats_list, list) and stats_list:
                # Log latest statistics
                latest_stats = stats_list[-1]
                flattened[f"coord_check/{layer_name}/mean"] = latest_stats.mean
                flattened[f"coord_check/{layer_name}/std"] = latest_stats.std
                flattened[f"coord_check/{layer_name}/max"] = latest_stats.max_abs
                flattened[f"coord_check/{layer_name}/width_mult"] = latest_stats.width_mult
        
        self.log_metrics(flattened, step=step)
    
    def log_scaling_test_results(
        self,
        scaling_results: Dict[int, Any],
        test_name: str = "scaling_test"
    ):
        """Log results from scaling tests."""
        if not self.enabled:
            return
        
        # Create summary table
        columns = ["width", "final_loss", "min_loss", "converged", "train_time"]
        data = []
        
        for width, result in scaling_results.items():
            data.append([
                width,
                result.final_loss,
                result.min_loss,
                result.converged,
                result.train_time
            ])
        
        table = wandb.Table(columns=columns, data=data)
        self.run.log({f"{test_name}/results": table})
        
        # Log individual metrics
        for width, result in scaling_results.items():
            metrics = {
                f"{test_name}/final_loss": result.final_loss,
                f"{test_name}/min_loss": result.min_loss,
                f"{test_name}/train_time": result.train_time,
            }
            
            # Use width as "step" for x-axis
            self.run.log(metrics, step=width)
    
    def log_hyperparameter_transfer_results(
        self,
        transfer_results: Dict[str, Dict[int, Any]]
    ):
        """Log hyperparameter transfer test results."""
        if not self.enabled:
            return
        
        base_results = transfer_results.get('base', {})
        transfer_results_data = transfer_results.get('transfer', {})
        
        # Log base model performance
        for width, result in base_results.items():
            metrics = {
                'hp_transfer/base_loss': result.final_loss,
                'hp_transfer/base_width': width,
            }
            self.run.log(metrics, step=width)
        
        # Log transfer results
        for width, result in transfer_results_data.items():
            metrics = {
                'hp_transfer/transfer_loss': result.final_loss,
                'hp_transfer/transfer_width': width,
            }
            self.run.log(metrics, step=width)
    
    def log_model_info(
        self,
        model,
        model_config: Dict[str, Any]
    ):
        """Log model architecture information."""
        if not self.enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model/total_params': total_params,
            'model/trainable_params': trainable_params,
            'model/config': model_config,
        }
        
        # Log model architecture details
        if hasattr(model, 'config'):
            config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else {}
            model_info.update({f'model/config/{k}': v for k, v in config_dict.items()})
        
        self.run.log(model_info)
    
    def watch_model(
        self,
        model,
        log_freq: int = 1000,
        log_graph: bool = False
    ):
        """Watch model for gradient and parameter tracking."""
        if not self.enabled or self.run is None:
            return
        
        wandb.watch(
            model,
            log_freq=log_freq,
            log_graph=log_graph,
            log="all"  # Log both gradients and parameters
        )
    
    def log_system_info(self):
        """Log system and environment information."""
        if not self.enabled:
            return
        
        import torch
        import platform
        
        system_info = {
            'system/python_version': platform.python_version(),
            'system/pytorch_version': torch.__version__,
            'system/cuda_available': torch.cuda.is_available(),
            'system/platform': platform.platform(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'system/cuda_version': torch.version.cuda,
                'system/gpu_count': torch.cuda.device_count(),
                'system/gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown',
            })
        
        self.run.log(system_info)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            self.run.finish()
            self.run = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()