# W&B Metrics and Performance Impact Analysis

## Overview

This document analyzes the W&B integration and advanced monitoring features in speedrun-mup, focusing on performance impact and recommended usage patterns.

## W&B Metrics Categories

### Basic Training Metrics (Always Logged)
- `step`: Training step number
- `training_time_ms`: Wall-clock training time in milliseconds
- `step_avg_ms`: Average time per step
- `val_loss`: Validation loss (when available)

### Advanced Monitoring Metrics (Configurable)
- `grad_norm`: Global gradient norm (lightweight, always computed)
- `param_norm`: Global parameter norm (lightweight, always computed)
- `spectral_norm_max/mean/std`: Weight matrix spectral norms (expensive)
- `{layer}_mean/std/max/min/l2_norm`: Activation statistics (expensive)

## Performance Impact Analysis

### Lightweight Metrics (Negligible Impact)
- **Gradient norm computation**: ~0.1ms overhead per step
- **Parameter norm computation**: ~0.1ms overhead per step
- **Basic logging to W&B**: ~1-2ms overhead per step
- **Local file logging**: ~0.05ms overhead per step

### Expensive Metrics (Significant Impact)
- **Spectral norm computation**: 5-20ms overhead per monitored parameter
- **Activation statistics**: 2-10ms overhead per hooked layer
- **Full monitoring on large models**: Can add 50-200ms per step

### Recommended Monitoring Intervals
```python
# For production training
monitor_interval = 100  # Check expensive metrics every 100 steps

# For debugging/research
monitor_interval = 10   # More frequent monitoring

# For final validation runs
monitor_interval = 1    # Monitor every step (expensive but thorough)
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

## Recommended Usage Patterns

### For Daily Training
```python
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup"
)
monitor = TrainingMonitor(
    model, 
    monitor_interval=100,  # Every 100 steps
    enable_spectral_norms=False,  # Disable expensive metrics
    enable_activation_stats=False
)
```

### For Research/Debugging
```python
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-research"
)
monitor = TrainingMonitor(
    model, 
    monitor_interval=10,   # Every 10 steps
    enable_spectral_norms=True,   # Enable for analysis
    enable_activation_stats=True  # Enable for debugging
)
```

### For MuP Validation
```python
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-coord-check"
)
# Use coordinate checking functionality from core/mup.py
# Monitor activation magnitudes across different widths
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