"""
Speedrun-MuP: Clean MuP implementation based on modded-nanogpt architecture.
"""

from .model import GPT, GPTConfig
from .utils import SimpleLogger, TrainingMonitor, compute_grad_norm
from .optimizers import create_muon_optimizer

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "GPTConfig", 
    "SimpleLogger",
    "TrainingMonitor",
    "compute_grad_norm",
    "create_muon_optimizer",
]