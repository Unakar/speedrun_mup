"""
Speedrun-MuP: Clean MuP implementation based on modded-nanogpt architecture.
"""

from .model import GPT, GPTConfig
from .mup import apply_mup_to_model as apply_mup, MuPConfig
from .utils import SimpleLogger, TrainingMonitor, compute_grad_norm, estimate_mfu
from .optimizers import create_muon_optimizer, create_mup_muon_optimizer

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "GPTConfig", 
    "apply_mup",
    "MuPConfig",
    "SimpleLogger",
    "TrainingMonitor",
    "compute_grad_norm",
    "estimate_mfu",
    "create_muon_optimizer",
    "create_mup_muon_optimizer",
]