# Speedrun-MuP

A research implementation of Maximal Update Parameterization (MuP) integrated with the modded-nanogpt speedrun architecture for efficient transformer scaling experiments.

## Overview

This project combines the state-of-the-art training optimizations from modded-nanogpt with principled scaling laws from MuP theory. It enables zero-shot hyperparameter transfer across model widths while maintaining competitive training performance.

**Key Features:**
- Full modded-nanogpt architecture implementation (FlexAttention, FP8, U-net skip connections, value embeddings, Muon optimizer)
- Standard MuP implementation with InfShape dimension tracking and coordinate checking
- Distributed training support for 8xH100 systems
- Comprehensive validation tools and experiment tracking

## Project Structure

```
speedrun_mup/
├── core/                   
│   ├── model.py            # modded-nanogpt arch
│   ├── mup.py              # MuP scaling and dimension tracking
│   ├── optimizers.py       # Muon and DistAdam optimizers
│   └── utils.py            # Logging, metrics, and utilities
├── scripts/                
│   └── data                # data download & tokenize
└── train.py                # Start training!
    
```

## Usage

### Basic Training

Train a GPT model with standard hyperparameters:

```bash
python train.py --width 768 --iterations 1750
```

### MuP Scaling Experiments

Train with MuP for zero-shot hyperparameter transfer:

```bash
# Base model (width 768)
python train.py --mup --width 768 --base-width 768 --iterations 1750

# Larger model with transferred hyperparameters
python train.py --mup --width 1024 --base-width 768 --iterations 1750
```

### Distributed Training

For multi-GPU systems:

```bash
torchrun --nproc_per_node=8 train.py --mup --width 1024 --base-width 768
```

### Configuration Options

The training script supports extensive configuration:

```bash
python scripts/train.py \
    --mup                        # Enable MuP scaling
    --width 1024                 # Target model width
    --base-width 768             # Base width for MuP reference
    --iterations 1750            # Training steps
    --seed 42                    # Random seed
    --no-compile                 # Disable torch.compile
```

## Implementation Details

### Architecture Fidelity

The model implementation maintains exact compatibility with modded-nanogpt:
- Custom FP8 operators (`nanogpt::mm`)
- FlexAttention with sliding window block masks
- U-net skip connections with learned scalar weights
- Value embeddings with 012...012 pattern
- Half-truncated RoPE with base frequency tuning
- ReLU² activation and logit soft-capping

### MuP Integration

The MuP implementation follows standard practices:
- InfShape dimension tracking for all parameters
- MuP-aware initialization and optimizers
- Output layer scaling and coordinate checking
- Support for width scaling experiments

### Performance Optimizations

Training includes all modded-nanogpt optimizations:
- Kernel warmup and torch.compile
- Distributed data loading with document alignment  
- Learning rate scheduling with momentum warmup
- Memory-efficient gradient accumulation

## Validation

The implementation includes coordinate checking functionality to validate MuP correctness. Proper MuP implementations should show stable activation magnitudes across different model widths.


## References

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Modded-NanoGPT Repository](https://github.com/KellerJordan/modded-nanogpt)
- [MuP Repository](https://github.com/microsoft/mup)
- [μP Theory and Practice (kexue.fm)](https://kexue.fm/archives/10795)