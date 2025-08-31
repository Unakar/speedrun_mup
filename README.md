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

## Monitoring and Logging System

### Intelligent Monitoring

**SimpleLogger**: Clean logging system aligned with modded-nanogpt style
- Automatic experiment naming: `speedrun_20250831_143022_mup_w1024_base768`
- Structured metric grouping: Time/, Loss/, Optimization/, Model/, Hardware/, Activations/
- Structured logging: console + file + W&B
- Hardware detection: H100(989T), B200(2500T), A100(312T) FLOPS

**Advanced Spectral Norm Monitoring**: Efficient Newton-Schulz implementation
- **ns5-fast mode**: Muon-style quintic Newton-Schulz iteration, speed-prioritized
- **ns3-accurate mode**: Classic convergent cubic iteration, accuracy-prioritized  
- **BF16 friendly**: Fully compatible with mixed-precision training, no type conversion
- **Smart fallback**: Automatic fallback to traditional power iteration for stability

### Complete Monitoring Metrics

**Basic metrics** (lightweight per-step computation):
- `train_loss`: Training loss
- `grad_norm`: Global gradient norm
- `param_norm`: Global parameter norm
- `lr`: Learning rate
- `training_time_ms`: Training time

**Advanced metrics** (configurable intervals):
- `spectral_norm_max/mean/std`: Weight matrix spectral norm statistics (Newton-Schulz)
- `activation_mean/std/max/min/l2_norm`: Activation layer statistics
- `peak_memory_mb`: Peak memory usage
- `reserved_memory_mb`: Reserved memory

### Configurable Monitoring

```bash
# Enable spectral norm monitoring (every 100 steps)
MONITOR_SPECTRAL_EVERY=100 bash scripts/run_basic_speedrun.sh

# Enable activation statistics monitoring
MONITOR_ACTIVATIONS=true bash scripts/run_basic_speedrun.sh

# Combined usage
MONITOR_SPECTRAL_EVERY=50 MONITOR_ACTIVATIONS=true \
bash scripts/run_basic_speedrun.sh 768 1750 1337
```

### Performance Overhead

- **Lightweight monitoring** (<1ms/step): Basic metrics, param/grad norms
- **Newton-Schulz spectral norm** (5-15ms/step): Efficient polar decomposition + power iteration
- **Activation statistics** (10-30ms/step): Depends on number of hooks
- **Traditional spectral norm** (50-200ms/step): SVD decomposition, deprecated
- **Recommended configuration**:
  - Daily training: `MONITOR_SPECTRAL_EVERY=100`
  - Research debugging: `MONITOR_SPECTRAL_EVERY=10 MONITOR_ACTIVATIONS=true`

### W&B Integration

- **Project grouping**: `speedrun-basic`, `speedrun-mup`, `speedrun-mup-group`
- **Automatic naming**: Descriptive names based on timestamp and configuration
- **Configuration tracking**: Complete hyperparameter recording
- **Visualization**: Training curves, activation statistics, coordinate checking plots

## Validation

The implementation includes coordinate checking functionality to validate MuP correctness. Proper MuP implementations should show stable activation magnitudes across different model widths.

```bash
# Run width sweep validation
bash scripts/run_mup_width_group_scaling.sh 256 "512 768 1024" 1000

# Check generated coordinate checking plots
# Correct implementation: activation statistics remain stable across widths
# Incorrect implementation: activation magnitudes vary significantly with width
```


## References

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Modded-NanoGPT Repository](https://github.com/KellerJordan/modded-nanogpt)
- [MuP Repository](https://github.com/microsoft/mup)
- [μP Theory and Practice (kexue.fm)](https://kexue.fm/archives/10795)