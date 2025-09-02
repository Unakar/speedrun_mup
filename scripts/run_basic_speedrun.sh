#!/bin/bash
# Basic speedrun training - single-node 8-GPU H100 setup
# Comprehensive training script with full configuration support

set -e

# Default configuration (8xH100 optimized)
ITERATIONS=${ITERATIONS:-1750}
SEED=${SEED:-1337}
TRAIN_DATA=${TRAIN_DATA:-"data/finewebedu10B/finewebedu_train_*.bin"}
VAL_DATA=${VAL_DATA:-"data/finewebedu10B/finewebedu_val_*.bin"}

# Advanced training parameters
BATCH_SIZE=${BATCH_SIZE:-8}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-1024}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GRAD_CLIP=${GRAD_CLIP:-1.0}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# Monitoring and logging
VAL_EVERY=${VAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-10}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"speedrun-basic"}
WANDB_NAME=${WANDB_NAME:-"exp1"}

# Advanced monitoring (expensive metrics)
MONITOR_SPECTRAL_EVERY=${MONITOR_SPECTRAL_EVERY:-100}  # 0=disabled, 100=every 100 steps
MONITOR_ACTIVATIONS=${MONITOR_ACTIVATIONS:-false}
MONITOR_ACTIVATIONS_EVERY=${MONITOR_ACTIVATIONS_EVERY:-100}

# System options
COMPILE=${COMPILE:-true}
MUP=${MUP:-false}

echo "=========================================="
echo "Basic Speedrun Training"
echo "Iterations: $ITERATIONS | Seed: $SEED"
echo "Training Data: $TRAIN_DATA"
echo "Validation Data: $VAL_DATA"
echo "Batch Size: $BATCH_SIZE | Seq Length: $SEQUENCE_LENGTH"
echo "Learning Rate: $LEARNING_RATE | Weight Decay: $WEIGHT_DECAY"
echo "W&B Project: $WANDB_PROJECT | Use W&B: $USE_WANDB"
echo "MuP Enabled: $MUP | Base Width: $BASE_WIDTH"
echo "=========================================="

# Check GPU count
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Error: CUDA not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "Warning: Expected 8 GPUs, found $GPU_COUNT"
    echo "This script is optimized for 8xH100"
fi

# Check data
if [ ! -d "data" ] || [ ! -f "data/finewebedu10B/finewebedu_train_000001.bin" ]; then
    echo "Error: Training data not found in data/"
    echo "Run: bash /root/xietian/dev/speedrun_mup/scripts/data_process/download_hf_data.sh"
    exit 1
fi

# Build the training command
TRAIN_CMD=(
    torchrun
    --nproc_per_node=8
    --nnodes=1
    --node_rank=0
    --master_addr=localhost
    --master_port=12345
    train.py
    --train-data "$TRAIN_DATA"
    --val-data "$VAL_DATA"
    --iterations $ITERATIONS
    --seed $SEED
    --batch-size-per-gpu $BATCH_SIZE
    --sequence-length $SEQUENCE_LENGTH
    --learning-rate $LEARNING_RATE
    --weight-decay $WEIGHT_DECAY
    --grad-clip $GRAD_CLIP
    --warmup-steps $WARMUP_STEPS
    --val-every $VAL_EVERY
    --save-every $SAVE_EVERY
    --log-every $LOG_EVERY
    --use-wandb $USE_WANDB
    --wandb-project "$WANDB_PROJECT"
    --monitor-spectral-every $MONITOR_SPECTRAL_EVERY
    --monitor-activations-every $MONITOR_ACTIVATIONS_EVERY
    --compile $COMPILE
)

# Add activation monitoring if enabled
if [ "$MONITOR_ACTIVATIONS" = "true" ]; then
    TRAIN_CMD+=(--monitor-activations)
fi

# Add wandb run name if specified
if [ -n "$WANDB_NAME" ]; then
    TRAIN_CMD+=(--wandb-name "$WANDB_NAME")
fi

# Run distributed training
echo "Executing: ${TRAIN_CMD[*]}"
echo ""
"${TRAIN_CMD[@]}"

echo "Basic speedrun training completed!"
echo "Model width: $WIDTH, Total iterations: $ITERATIONS"