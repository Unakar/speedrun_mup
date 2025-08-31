#!/bin/bash
# Basic speedrun training - single-node 8-GPU H100 setup
# Comprehensive training script with full configuration support
#
# Usage:
#   bash run_basic_speedrun.sh [WIDTH] [ITERATIONS] [SEED] [TRAIN_DATA] [VAL_DATA]
#
# Environment Variables:
#   BATCH_SIZE=8            # Batch size per GPU  
#   SEQUENCE_LENGTH=1024    # Sequence length
#   LEARNING_RATE=3e-4      # Learning rate
#   WEIGHT_DECAY=0.1        # Weight decay
#   GRAD_CLIP=1.0           # Gradient clipping
#   WARMUP_STEPS=100        # Warmup steps
#   VAL_EVERY=100           # Validation interval
#   SAVE_EVERY=500          # Save checkpoint interval
#   LOG_EVERY=10            # Logging interval
#   USE_WANDB=true          # Enable W&B logging
#   WANDB_PROJECT="speedrun-basic"  # W&B project name
#   WANDB_NAME=""           # W&B run name (optional)
#   COMPILE=true            # Enable model compilation
#   MUP=false               # Enable MuP scaling
#   BASE_WIDTH=768          # MuP base width
#
# Examples:
#   bash run_basic_speedrun.sh 768 1750 1337
#   LEARNING_RATE=1e-4 bash run_basic_speedrun.sh 1024 2000 42
#   MUP=true BASE_WIDTH=512 bash run_basic_speedrun.sh 1024 1750 1337

set -e

# Default configuration (8xH100 optimized)
WIDTH=${1:-768}
ITERATIONS=${2:-1750}
SEED=${3:-1337}
TRAIN_DATA=${4:-"data/finewebedu10B/finewebedu_train_*.bin"}
VAL_DATA=${5:-"data/finewebedu10B/finewebedu_val_*.bin"}

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
WANDB_NAME=${WANDB_NAME:-""}

# System options
COMPILE=${COMPILE:-true}
MUP=${MUP:-false}
BASE_WIDTH=${BASE_WIDTH:-768}

echo "=========================================="
echo "Basic Speedrun Training"
echo "Width: $WIDTH | Iterations: $ITERATIONS | Seed: $SEED"
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
    --width $WIDTH
    --iterations $ITERATIONS
    --seed $SEED
    --batch-size $BATCH_SIZE
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
    --compile $COMPILE
)

# Add MuP parameters if enabled
if [ "$MUP" = "true" ]; then
    TRAIN_CMD+=(--mup --base-width $BASE_WIDTH)
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