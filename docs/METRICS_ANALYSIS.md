# W&B Metrics and Performance Impact Analysis

## Overview

This document analyzes the W&B integration and advanced monitoring features in speedrun-mup, focusing on performance impact and recommended usage patterns, with particular emphasis on the efficient Newton-Schulz based spectral norm computation.

## W&B Metrics Categories

### Basic Training Metrics (Always Logged)
- `step`: Training step number
- `training_time_ms`: Wall-clock training time in milliseconds
- `step_avg_ms`: Average time per step
- `train_loss`: Training loss
- `val_loss`: Validation loss (when available)

### Advanced Monitoring Metrics (Configurable)
- `grad_norm`: Global gradient norm (lightweight, always computed)
- `param_norm`: Global parameter norm (lightweight, always computed)
- `spectral_norm_max/mean/std`: Weight matrix spectral norms (Newton-Schulz, configurable)
- `activation_mean/std/max/min/l2_norm`: Activation statistics (expensive, configurable)

## Performance Impact Analysis

### Lightweight Metrics (Negligible Impact)
- **Gradient norm computation**: ~0.1ms overhead per step
- **Parameter norm computation**: ~0.1ms overhead per step
- **Basic logging to W&B**: ~1-2ms overhead per step
- **Local file logging**: ~0.05ms overhead per step

### Newton-Schulz Spectral Norm (Medium Impact, Efficient Implementation)
- **NS5 fast mode**: 5-10ms overhead per monitored parameter
- **NS3 accurate mode**: 8-15ms overhead per monitored parameter
- **BF16 friendly**: Fully compatible with mixed-precision training
- **Smart optimization**: Only monitors Muon optimizer's hidden matrix parameters

### Traditional Expensive Metrics (Significant Impact, Optimized)
- **SVD spectral norm computation**: 20-50ms overhead per monitored parameter (deprecated)
- **Activation statistics**: 2-10ms overhead per hooked layer
- **Full monitoring on large models**: Can add 50-200ms per step

### Recommended Monitoring Intervals

#### Production Training Configuration
```bash
# High-performance daily training
MONITOR_SPECTRAL_EVERY=100  # Newton-Schulz spectral norm
MONITOR_ACTIVATIONS=false   # Disable activation statistics
bash scripts/run_basic_speedrun.sh
```

#### Research/Debugging Configuration
```bash
# Research debugging mode
MONITOR_SPECTRAL_EVERY=10   # More frequent spectral norm
MONITOR_ACTIVATIONS=true    # Enable activation statistics
bash scripts/run_basic_speedrun.sh
```

#### Final Validation Configuration
```bash
# Final validation runs
MONITOR_SPECTRAL_EVERY=1    # Every step spectral norm
MONITOR_ACTIVATIONS=true    # Full activation statistics
bash scripts/run_basic_speedrun.sh
```

## Newton-Schulz Spectral Norm Technical Details

### Algorithm Advantages

**Polar Decomposition Method**: Uses Newton-Schulz iterations to compute the orthogonal polar factor Q ≈ polar(W)
- **NS5 fast mode**: Muon-style quintic polynomial with coefficients optimized for high slope (3.4445, -4.7750, 2.0315)
- **NS3 accurate mode**: Classic convergent cubic iteration with higher precision

**Power Iteration Optimization**: Performs power iteration on H = Q^T W ≈ (W^T W)^{1/2}
- **Memory efficient**: Never explicitly constructs the H matrix, uses linear operator v → Q^T(Wv)
- **BF16 compatible**: Matrix multiplications in BF16, norm computations in FP32
- **Smart fallback**: Automatic fallback to traditional W^T W power iteration on failure

### Precision vs Performance Trade-offs

**NS5 Fast Mode**:
- ✅ Consistent with Muon optimizer style
- ✅ Speed-prioritized, 5-10ms/parameter
- ⚠️ Slight bias possible (tens of percentage points)
- 🎯 **Recommended for**: Daily training monitoring, trend analysis

**NS3 Accurate Mode**:
- ✅ Closer to strict polar decomposition
- ✅ Higher precision spectral norm estimation
- ⚠️ 8-15ms/parameter, slightly slower
- 🎯 **Recommended for**: Research analysis, precise measurements

### Usage Examples

```python
# Configure Newton-Schulz spectral norm monitoring
from core.utils import compute_weight_spectral_norms

# Fast mode - daily training
spectral_norms = compute_weight_spectral_norms(
    model, 
    target_params=hidden_matrix_params,
    mode="ns5-fast", 
    power_iters=7
)

# Accurate mode - research analysis
spectral_norms = compute_weight_spectral_norms(
    model,
    target_params=hidden_matrix_params, 
    mode="ns3-accurate",
    power_iters=10
)
```

## W&B Integration Benefits

### Experiment Organization
- **Automatic naming**: `speedrun_20250831_143022_mup_w1024_base768`
- **Project grouping**: Separates different experiment types
- **Config logging**: Full hyperparameter tracking
- **Structured logging**: Consistent metric naming and units

### Performance Monitoring
- **Hardware-aware MFU**: Automatically detects H100/B200/A100 for accurate utilization
- **Memory tracking**: Peak and reserved memory usage
- **Training efficiency**: Tokens per second, step timing analysis

### MuP-Specific Metrics
- **Coordinate checking**: Activation magnitude stability across widths
- **Scaling validation**: Hyperparameter transfer verification
- **Width comparison**: Side-by-side comparison of different model widths

### W&B Metric Grouping

Metrics are organized by category for clear visualization:

**Time/** (time-related):
- `Time/training_time_ms`
- `Time/step_avg_ms`
- `Time/total_time_s`

**Loss/** (loss functions):
- `Loss/train_loss`
- `Loss/val_loss`

**Optimization/** (optimization-related):
- `Optimization/lr`
- `Optimization/grad_norm`
- `Optimization/momentum`

**Model/** (model parameters):
- `Model/param_norm`
- `Model/spectral_norm_max`
- `Model/spectral_norm_mean`
- `Model/spectral_norm_std`

**Hardware/** (hardware resources):
- `Hardware/peak_memory_mb`
- `Hardware/reserved_memory_mb`

**Activations/** (activation statistics):
- `Activations/layer_0_mean`
- `Activations/layer_0_std`
- `Activations/attention_l2_norm`

## Recommended Usage Patterns

### For Daily Training
```bash
# Basic monitoring with minimal performance impact
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup"
)
# Use script configuration
MONITOR_SPECTRAL_EVERY=100 bash scripts/run_basic_speedrun.sh
```

### For Research/Debugging
```bash
# Detailed monitoring, research-friendly
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-research"
)
# Use script configuration
MONITOR_SPECTRAL_EVERY=10 MONITOR_ACTIVATIONS=true \
bash scripts/run_basic_speedrun.sh
```

### For MuP Validation
```bash
# MuP coordinate checking specialized
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-coord-check"
)
# Enable coordinate checking functionality
python train.py --mup --coord-check --coord-check-every 50
```

## Performance Recommendations

### High-Frequency Training (>1000 steps/experiment)
- Monitor expensive metrics every 100+ steps
- Use basic metrics for step-by-step tracking
- Enable W&B for experiment organization
- Disable activation hooks during training

### Research Experiments (<500 steps)
- Monitor expensive metrics every 10-50 steps
- Full activation statistics for layer analysis
- Spectral norms for weight analysis
- Detailed coordinate checking for MuP validation

### Distributed Training (8xH100)
- Only log from rank 0 process
- Aggregate metrics across processes before logging
- Use efficient reduction operations
- Consider logging frequency impact on synchronization

## Memory Impact

### W&B Overhead
- Client memory: ~50-100MB baseline
- Metric buffers: ~1-5MB per 1000 steps
- Image logging: Variable (not used in basic setup)

### Monitoring Overhead
- Activation hooks: ~10-50MB per hooked layer
- Spectral norm computation: Minimal memory impact
- Gradient/param norms: Negligible memory impact

## Best Practices

1. **Start Simple**: Use basic metrics first, add advanced monitoring as needed
2. **Monitor Selectively**: Enable expensive metrics only during research phases
3. **Use Intervals**: Don't monitor expensive metrics every step
4. **Clean Up**: Remove activation hooks when not needed
5. **Organize Experiments**: Use descriptive project names and experiment names
6. **Validate MuP**: Use coordinate checking to ensure proper implementation

## Integration with modded-nanogpt Style

The logging system maintains compatibility with modded-nanogpt's simple console output:
```
step:1000/1750 val_loss:3.2847 train_time:180420.0ms step_avg:180.42ms
```

While adding structured logging to files and W&B for advanced analysis and experiment tracking.