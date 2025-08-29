"""
Spectral monitoring for MuP experiments.

Implements spectral norm tracking and higher-order MuP validation
based on the spectral conditions from advanced MuP research.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import time

from ..utils.shapes import get_infshape


@dataclass
class SpectralStats:
    """Statistics for spectral monitoring."""
    
    layer_name: str
    matrix_shape: Tuple[int, ...]
    
    # Singular values
    top_singular_values: List[float] = field(default_factory=list)
    condition_number: float = 0.0
    spectral_norm: float = 0.0
    
    # Normalized by MuP target scaling
    normalized_spectral_norm: float = 0.0
    expected_scale: float = 1.0
    
    # Update statistics (if available)
    update_spectral_norm: Optional[float] = None
    update_to_param_ratio: Optional[float] = None
    
    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = {
            f"{prefix}spectral_norm": self.spectral_norm,
            f"{prefix}condition_number": self.condition_number,
            f"{prefix}normalized_spectral_norm": self.normalized_spectral_norm,
            f"{prefix}expected_scale": self.expected_scale,
        }
        
        # Add top singular values
        for i, sv in enumerate(self.top_singular_values[:5]):  # Log top 5
            result[f"{prefix}sv_{i+1}"] = sv
        
        # Add update statistics if available
        if self.update_spectral_norm is not None:
            result[f"{prefix}update_spectral_norm"] = self.update_spectral_norm
        if self.update_to_param_ratio is not None:
            result[f"{prefix}update_ratio"] = self.update_to_param_ratio
        
        return result


class SpectralMonitor:
    """
    Monitor spectral properties of model parameters during training.
    
    Tracks spectral norms and validates they follow MuP scaling rules:
    ||W_k||_2 = O(√(d_k/d_{k-1})) and ||ΔW_k||_2 = O(√(d_k/d_{k-1}))
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        top_k: int = 5,
        power_iter_steps: int = 10,
        track_updates: bool = True
    ):
        self.model = model
        self.top_k = top_k
        self.power_iter_steps = power_iter_steps
        self.track_updates = track_updates
        
        # Storage for previous parameters (to compute updates)
        self.prev_params = {}
        
        # Determine which layers to monitor
        self.monitored_layers = self._select_layers(layer_names)
        
        # Initialize previous parameter storage
        if self.track_updates:
            self._store_current_params()
    
    def _select_layers(self, layer_names: Optional[List[str]]) -> List[str]:
        """Select layers to monitor for spectral properties."""
        if layer_names is not None:
            return layer_names
        
        # Auto-select matrix-like parameters
        selected = []
        for name, param in self.model.named_parameters():
            if param.dim() >= 2 and 'weight' in name:  # Matrix-like parameters
                # Skip very small matrices
                if min(param.shape) >= 8:
                    selected.append(name)
        
        return selected
    
    def _store_current_params(self):
        """Store current parameters for update tracking."""
        for name in self.monitored_layers:
            param = self._get_param_by_name(name)
            if param is not None:
                self.prev_params[name] = param.data.clone().detach()
    
    def _get_param_by_name(self, name: str) -> Optional[torch.Tensor]:
        """Get parameter by name."""
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                return param
        return None
    
    def compute_spectral_stats(self) -> Dict[str, SpectralStats]:
        """
        Compute spectral statistics for monitored layers.
        
        Returns:
            Dictionary mapping layer names to spectral statistics
        """
        stats = {}
        
        for layer_name in self.monitored_layers:
            param = self._get_param_by_name(layer_name)
            if param is None:
                continue
            
            # Compute spectral statistics for parameter
            layer_stats = self._compute_layer_spectral_stats(layer_name, param)
            
            # Compute update statistics if tracking updates
            if self.track_updates and layer_name in self.prev_params:
                update_tensor = param.data - self.prev_params[layer_name]
                self._add_update_stats(layer_stats, update_tensor)
            
            stats[layer_name] = layer_stats
        
        # Update stored parameters for next iteration
        if self.track_updates:
            self._store_current_params()
        
        return stats
    
    def _compute_layer_spectral_stats(
        self, 
        layer_name: str, 
        param: torch.Tensor
    ) -> SpectralStats:
        """Compute spectral statistics for a single layer."""
        # Ensure we have a 2D matrix
        if param.dim() > 2:
            # For convolutions, reshape to 2D
            param_2d = param.view(param.size(0), -1)
        else:
            param_2d = param
        
        # Compute top singular values using power iteration
        top_svs = self._power_iteration_svd(param_2d, self.top_k)
        
        # Spectral norm (top singular value)
        spectral_norm = top_svs[0] if top_svs else 0.0
        
        # Condition number (ratio of largest to smallest computed SV)
        condition_number = top_svs[0] / top_svs[-1] if len(top_svs) >= 2 else 1.0
        
        # Get expected scaling from MuP
        expected_scale = self._get_mup_target_scale(param)
        normalized_spectral_norm = spectral_norm / expected_scale if expected_scale > 0 else spectral_norm
        
        return SpectralStats(
            layer_name=layer_name,
            matrix_shape=param.shape,
            top_singular_values=top_svs,
            condition_number=condition_number,
            spectral_norm=spectral_norm,
            normalized_spectral_norm=normalized_spectral_norm,
            expected_scale=expected_scale
        )
    
    def _add_update_stats(self, stats: SpectralStats, update_tensor: torch.Tensor):
        """Add update-related statistics to existing stats."""
        # Reshape update tensor if needed
        if update_tensor.dim() > 2:
            update_2d = update_tensor.view(update_tensor.size(0), -1)
        else:
            update_2d = update_tensor
        
        # Compute spectral norm of update
        update_svs = self._power_iteration_svd(update_2d, 1)
        update_spectral_norm = update_svs[0] if update_svs else 0.0
        
        # Ratio of update to parameter magnitude
        update_to_param_ratio = update_spectral_norm / stats.spectral_norm if stats.spectral_norm > 0 else 0.0
        
        stats.update_spectral_norm = update_spectral_norm
        stats.update_to_param_ratio = update_to_param_ratio
    
    def _power_iteration_svd(self, matrix: torch.Tensor, k: int) -> List[float]:
        """
        Compute top-k singular values using power iteration.
        
        More efficient than full SVD for large matrices when only
        top singular values are needed.
        """
        if matrix.numel() == 0:
            return []
        
        m, n = matrix.shape
        singular_values = []
        
        # Work with a copy to avoid modifying original
        A = matrix.clone().float()
        
        for i in range(min(k, min(m, n))):
            # Power iteration to find top singular vector
            if A.numel() == 0:
                break
            
            # Initialize random vector
            if n > 0:
                v = torch.randn(n, device=A.device, dtype=A.dtype)
                v = v / (torch.norm(v) + 1e-12)
                
                # Power iteration
                for _ in range(self.power_iter_steps):
                    # v -> A^T A v (find right singular vector)
                    Av = A @ v
                    u = Av / (torch.norm(Av) + 1e-12)
                    
                    ATu = A.T @ u
                    v = ATu / (torch.norm(ATu) + 1e-12)
                
                # Compute singular value
                Av = A @ v
                singular_value = torch.norm(Av).item()
                singular_values.append(singular_value)
                
                # Deflation: remove the found singular vector
                if singular_value > 1e-12:
                    u = Av / singular_value
                    A = A - singular_value * torch.outer(u, v)
            else:
                break
        
        return singular_values
    
    def _get_mup_target_scale(self, param: torch.Tensor) -> float:
        """Get target spectral scale according to MuP rules."""
        infshape = get_infshape(param)
        if infshape is None:
            return 1.0
        
        # MuP spectral target: ||W_k||_2 = O(√(d_k/d_{k-1}))
        if infshape.ndim >= 2:
            d_out = infshape.target_shape[-1].size  # Output dimension
            d_in = infshape.target_shape[-2].size   # Input dimension
            
            target_scale = (d_out / d_in) ** 0.5
            return target_scale
        
        return 1.0
    
    def validate_spectral_scaling(
        self, 
        stats: Dict[str, SpectralStats],
        tolerance_range: Tuple[float, float] = (0.1, 10.0)
    ) -> Dict[str, bool]:
        """
        Validate that spectral norms follow MuP scaling rules.
        
        Args:
            stats: Spectral statistics from compute_spectral_stats
            tolerance_range: (min, max) acceptable range for normalized spectral norms
            
        Returns:
            Dictionary mapping layer names to validation status
        """
        results = {}
        min_tol, max_tol = tolerance_range
        
        for layer_name, layer_stats in stats.items():
            normalized_norm = layer_stats.normalized_spectral_norm
            valid = min_tol <= normalized_norm <= max_tol
            results[layer_name] = valid
        
        return results
    
    def get_summary_metrics(self, stats: Dict[str, SpectralStats]) -> Dict[str, float]:
        """Get summary metrics for logging."""
        if not stats:
            return {}
        
        # Aggregate statistics
        all_spectral_norms = [s.spectral_norm for s in stats.values()]
        all_condition_numbers = [s.condition_number for s in stats.values()]
        all_normalized_norms = [s.normalized_spectral_norm for s in stats.values()]
        
        summary = {
            'spectral_norm_mean': np.mean(all_spectral_norms),
            'spectral_norm_max': np.max(all_spectral_norms),
            'condition_number_mean': np.mean(all_condition_numbers),
            'condition_number_max': np.max(all_condition_numbers),
            'normalized_norm_mean': np.mean(all_normalized_norms),
            'normalized_norm_std': np.std(all_normalized_norms),
        }
        
        # Update statistics if available
        update_norms = [s.update_spectral_norm for s in stats.values() 
                       if s.update_spectral_norm is not None]
        if update_norms:
            summary['update_spectral_norm_mean'] = np.mean(update_norms)
            summary['update_spectral_norm_max'] = np.max(update_norms)
        
        return summary


def efficient_spectral_norm(matrix: torch.Tensor, num_iters: int = 10) -> float:
    """
    Efficiently compute spectral norm using power iteration.
    
    Standalone function for quick spectral norm computation.
    """
    if matrix.dim() > 2:
        matrix = matrix.view(matrix.size(0), -1)
    
    if matrix.numel() == 0:
        return 0.0
    
    m, n = matrix.shape
    if n == 0:
        return 0.0
    
    # Initialize random vector
    v = torch.randn(n, device=matrix.device, dtype=matrix.dtype)
    v = v / torch.norm(v)
    
    # Power iteration
    for _ in range(num_iters):
        # v -> A^T A v
        Av = matrix @ v
        u = Av / (torch.norm(Av) + 1e-12)
        
        ATu = matrix.T @ u
        v = ATu / (torch.norm(ATu) + 1e-12)
    
    # Final singular value
    Av = matrix @ v
    return torch.norm(Av).item()