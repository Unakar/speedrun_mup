"""
Speedrun-MuP: Clean MuP implementation based on modded-nanogpt architecture.
"""

from .model import GPT, GPTConfig
from .mup import apply_mup, MuPConfig

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "GPTConfig", 
    "apply_mup",
    "MuPConfig",
]