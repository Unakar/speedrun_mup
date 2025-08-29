#!/usr/bin/env python3
"""
Simple coordinate checking script for MuP validation.
"""

import sys
from pathlib import Path

# Add core to path  
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from model import GPT, GPTConfig
from mup import MuPConfig, apply_mup, coord_check, plot_coord_check


def main():
    print("Running MuP coordinate check...")
    
    # Test different widths
    widths = [256, 512, 768, 1024]
    
    def model_factory(width):
        """Create model with given width."""
        config = GPTConfig(
            vocab_size=50257,
            max_seq_len=1024,
            model_dim=width,
            num_heads=max(1, width // 128),  # Adjust heads proportionally
            num_layers=12,
            use_fp8=False,  # Disable for simplicity in coord check
            use_flex_attention=False
        )
        return GPT(config)
    
    # Run coordinate check
    results = coord_check(
        model_factory=model_factory,
        widths=widths,
        input_shape=(1024,),
        n_steps=3,
        device='cuda' if __name__ == '__main__' else 'cpu'
    )
    
    # Plot results
    plot_coord_check(results, save_path="coord_check_results.png")
    
    print("Coordinate check complete!")
    print("Check 'coord_check_results.png' for validation plots.")


if __name__ == '__main__':
    main()