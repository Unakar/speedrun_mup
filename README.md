# Speedrun-MuP 🚀

A clean experimental stack for **Maximal Update Parameterization (μP)** scaling experiments built on top of the modded-nanogpt "speedrun" architecture.

## Overview

Speedrun-MuP combines the world-record training speed of modded-nanogpt with the principled scaling laws of μP to enable:

- **Width-invariant hyperparameters** that transfer across model scales
- **Competitive training speed** with modern optimizations (FlexAttention, FP8, Muon optimizer)
- **Comprehensive validation** through coordinate checking and spectral monitoring
- **Research-ready metrics** with automatic plotting and W&B integration

  speedrun_mup/
  ├── core/                    # Clean, concise implementation
  │   ├── __init__.py         # Simple package initialization
  │   ├── model.py            # GPT implementation following modded-nanogpt
  │   ├── mup.py              # MuP scaling and coordinate checking
  │   └── utils.py            # Consolidated logging and utilities
  ├── scripts/                # Executable training scripts
  │   ├── train.py           # Main training script with MuP
  │   └── coord_check.py     # Coordinate validation
  ├── configs/                # Simplified YAML configurations
  │   ├── gpt_small_mup.yaml # Basic GPT-small config
  │   └── width_sweep.yaml   # Width experiments config
  ├── claude_instructions/    # Preserved unchanged
  └── origin_repos/          # Preserved unchanged
      ├── modded-nanogpt/
      └── mup/

## Quick Start

### 1. Coordinate Checking

First, validate your MuP implementation:

```bash
python scripts/coord_check.py \
    --mup-config configs/mup/width_sweep.yaml \
    --widths 256 512 1024 \
    --output-dir ./coord_check_results
```

This will:
- Test models at different widths
- Generate coordinate check plots
- Validate that activations remain O(1) across widths

### 2. Basic Training

Run a basic MuP training experiment:

```bash
python scripts/train.py \
    --config configs/base/gpt_small.yaml \
    --mup-config configs/mup/width_sweep.yaml \
    --run-name my_mup_experiment
```

### 3. Width Scaling Experiment

Test hyperparameter transfer across multiple widths:

```bash
# Train base model
python scripts/train.py \
    --config configs/base/gpt_small.yaml \
    --mup-config configs/mup/width_sweep.yaml \
    --run-name base_width_256

# Transfer to larger widths (automatically uses MuP scaling)
python scripts/train.py \
    --config configs/base/gpt_medium.yaml \
    --mup-config configs/mup/width_sweep.yaml \
    --run-name transferred_width_768
```

## Project Structure

```
speedrun_mup/
├── speedrun_mup/           # Core package
│   ├── models/            # MuP-aware model implementations
│   ├── training/          # Training loops and optimizers
│   ├── config/            # Configuration management
│   ├── logging/           # Metrics and W&B integration
│   ├── validation/        # Coordinate checking and validation
│   └── utils/             # MuP utilities and shape management
├── configs/               # Experiment configurations
│   ├── base/             # Base model configurations
│   ├── mup/              # MuP-specific configurations
│   └── experiments/      # Full experiment suites
├── scripts/              # Entry point scripts
│   ├── train.py         # Main training script
│   ├── coord_check.py   # Coordinate checking
│   └── analyze_run.py   # Post-training analysis
└── analysis/             # Notebooks and plotting utilities
```


## Troubleshooting

### Coordinate Checks Fail
```bash
# Common issues:
# 1. Incorrect base shapes - regenerate with fresh models
# 2. Architecture incompatibility - check skip connections
# 3. Initialization problems - verify MuP init is applied

# Debug with:
python scripts/coord_check.py --widths 256 512 --n-steps 5
```

## Recommand links (kexue.fm is all you need)

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [μP Practitioner's Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
- [Modded-NanoGPT Repository](https://github.com/KellerJordan/modded-nanogpt)
- [Higher-order μP Spectral Conditions](https://kexue.fm/archives/10795)

---
