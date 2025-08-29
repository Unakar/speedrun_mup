#!/usr/bin/env python3
"""
Coordinate checking script for MuP validation.

Run coordinate checks on different model widths to validate MuP implementation.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from speedrun_mup.config import ExperimentConfig, MuPConfig, ScalingConfig
from speedrun_mup.models import GPTMuP, apply_mup
from speedrun_mup.models.gpt import GPTConfig
from speedrun_mup.validation import coord_check, plot_coord_data, validate_mup_coordinates


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MuP coordinate checking')
    
    parser.add_argument('--mup-config', type=str, required=True,
                       help='Path to MuP configuration YAML')
    parser.add_argument('--widths', nargs='+', type=int, default=[256, 512, 1024],
                       help='Model widths to test')
    parser.add_argument('--n-steps', type=int, default=3,
                       help='Number of training steps for coordinate check')
    parser.add_argument('--n-seeds', type=int, default=1,
                       help='Number of random seeds to average over')
    parser.add_argument('--output-dir', type=str, default='./coord_check_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    
    return parser.parse_args()


def main():
    """Run coordinate checking experiment."""
    args = parse_args()
    
    # Load MuP configuration
    import yaml
    with open(args.mup_config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    mup_config = MuPConfig(**config_dict.get('mup', {}))
    scaling_config = ScalingConfig(**config_dict.get('scaling', {}))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1337)
    
    print("Running MuP coordinate checking...")
    print(f"Testing widths: {args.widths}")
    print(f"Steps: {args.n_steps}, Seeds: {args.n_seeds}")
    print(f"Device: {device}")
    
    # Create base model for MuP setup
    base_config = scaling_config.get_base_config()
    base_model = GPTMuP(base_config)
    
    def model_factory(width: int):
        """Create model with specified width."""
        # Create config for this width
        config = GPTConfig(
            vocab_size=scaling_config.vocab_size,
            n_layer=scaling_config.base_n_layer,
            n_head=max(1, width // 64),  # Scale heads appropriately
            n_embd=width,
            block_size=scaling_config.block_size,
            use_mup=mup_config.use_mup
        )
        
        # Ensure n_embd is divisible by n_head
        while config.n_embd % config.n_head != 0:
            config.n_head -= 1
        
        model = GPTMuP(config)
        
        # Apply MuP parameterization
        if mup_config.use_mup:
            delta_config = scaling_config.get_delta_config()
            delta_model = GPTMuP(delta_config)
            apply_mup(model, base_model, delta_model)
        
        return model
    
    # Run coordinate check
    print("Performing forward/backward passes...")
    coord_results = coord_check(
        model_factory=model_factory,
        widths=args.widths,
        input_shape=(4, 512),  # Small batch for testing
        n_steps=args.n_steps,
        n_seeds=args.n_seeds,
        device=device
    )
    
    # Save and plot results
    plot_path = os.path.join(args.output_dir, 'coordinate_check.png')
    plot_coord_data(
        coord_results,
        save_path=plot_path,
        title="MuP Coordinate Check Results"
    )
    
    # Validate results
    validation_results = validate_mup_coordinates(
        coord_results,
        tolerance=mup_config.coord_check_tolerance
    )
    
    # Save validation results
    results_path = os.path.join(args.output_dir, 'validation_results.txt')
    with open(results_path, 'w') as f:
        f.write("MuP Coordinate Check Validation Results\n")
        f.write("=" * 45 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Widths tested: {args.widths}\n")
        f.write(f"  Steps: {args.n_steps}\n")
        f.write(f"  Seeds: {args.n_seeds}\n")
        f.write(f"  Tolerance: {mup_config.coord_check_tolerance}\n\n")
        
        f.write("Per-layer results:\n")
        all_passed = True
        for layer_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            f.write(f"  {layer_name}: {status}\n")
            if not passed:
                all_passed = False
        
        f.write(f"\nOverall result: {'PASS' if all_passed else 'FAIL'}\n")
        
        if not all_passed:
            f.write("\nNotes:\n")
            f.write("- Failed coordinate checks suggest MuP parameterization issues\n")
            f.write("- Check model initialization and learning rate scaling\n")
            f.write("- Consider adjusting tolerance or model architecture\n")
    
    # Print summary
    print(f"\nCoordinate Check Results:")
    print("=" * 40)
    all_passed = True
    for layer_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{layer_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    
    if all_passed:
        print("\n🎉 MuP coordinate checking successful!")
        print("Your model should exhibit width-invariant training dynamics.")
    else:
        print("\n⚠️  Some coordinate checks failed.")
        print("Consider reviewing MuP parameterization settings.")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Plot: {plot_path}")
    print(f"  - Validation: {results_path}")


if __name__ == '__main__':
    main()