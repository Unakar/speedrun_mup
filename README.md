# Speedrun-MuP 🚀

A clean, engineering-grade experimental stack for **Maximal Update Parameterization (μP)** scaling experiments built on top of the modded-nanogpt "speedrun" architecture.

## Overview

Speedrun-MuP combines the world-record training speed of modded-nanogpt with the principled scaling laws of μP to enable:

- **Width-invariant hyperparameters** that transfer across model scales
- **Competitive training speed** with modern optimizations (FlexAttention, FP8, Muon optimizer)
- **Comprehensive validation** through coordinate checking and spectral monitoring
- **Research-ready metrics** with automatic plotting and W&B integration

## Key Features

### 🔬 MuP Integration
- **Complete μP implementation** with proper initialization and learning rate scaling
- **Coordinate checking** to validate width-invariant activations
- **Spectral monitoring** for higher-order μP validation
- **Hyperparameter transfer** testing across model scales

### ⚡ Performance Optimizations
- **FlexAttention** with sliding window patterns
- **U-Net skip connections** for better gradient flow
- **RoPE position encodings** with half-truncated frequencies
- **Mixed precision training** (FP8/BF16/FP32)
- **Value embeddings** for richer representations

### 📊 Comprehensive Logging
- **Weights & Biases** integration with MuP-specific metrics
- **Activation statistics** tracking across layers
- **Gradient and parameter norms** monitoring
- **Spectral properties** analysis (optional)
- **Automated plotting** for validation curves

### 🧪 Experimental Framework
- **Configuration-driven** experiments with YAML configs
- **Reproducible runs** with deterministic seeding
- **Distributed training** support for multi-GPU setups
- **Checkpoint management** with resumable training

## Installation

```bash
git clone <repository-url>
cd speedrun_mup
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- wandb (optional, for logging)
- matplotlib (for plotting)
- PyYAML (for configuration)

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

## Understanding MuP

### What is μP?

Maximal Update Parameterization (μP) is a neural network parameterization that keeps activations, gradients, and updates at the same scale as you increase model width. This means:

1. **Optimal hyperparameters remain stable** across different model sizes
2. **No expensive retuning** when scaling up models
3. **Predictable training dynamics** regardless of width

### Key MuP Rules

1. **Initialization**: Matrix weights scale as `σ² ∝ 1/width`
2. **Learning rates**: Hidden layer LR scales as `η ∝ 1/width`
3. **Output scaling**: Language model head uses `1/width` scaling
4. **Attention**: Use `1/d_head` instead of `1/√d_head` scaling

### Validation

MuP correctness is validated through **coordinate checking**:
- Train models at different widths for a few steps
- Plot activation statistics vs width
- Verify activations remain O(1) (don't grow/shrink with width)

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

## References

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [μP Practitioner's Guide](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization)
- [Modded-NanoGPT Repository](https://github.com/KellerJordan/modded-nanogpt)
- [Higher-order μP Spectral Conditions](https://kexue.fm/archives/10795)

---

Built with ❤️ for scalable, principled neural network training.
