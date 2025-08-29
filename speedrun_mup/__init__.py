"""
Speedrun-MuP: A clean, engineering-grade experimental stack for MuP scaling experiments
on top of the modded-nanogpt "speedrun" architecture.

This library provides:
- MuP-aware model implementations with competitive training speed
- Comprehensive logging and validation for scaling experiments
- Spectral monitoring and coordinate checking
- Configuration-driven experiment management
"""

from . import models, training, config, data, logging, validation, utils

__version__ = "0.1.0"
__author__ = "Speedrun-MuP Team"

# Core exports for convenience
from .models import GPTMuP
from .training import TrainerMuP
from .config import MuPConfig
from .validation import coord_check, scaling_test

__all__ = [
    "models",
    "training", 
    "config",
    "data",
    "logging",
    "validation",
    "utils",
    "GPTMuP",
    "TrainerMuP", 
    "MuPConfig",
    "coord_check",
    "scaling_test",
]