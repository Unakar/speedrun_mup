#!/bin/bash
# Single MuP width scaling experiment - single-node 8-GPU H100 setup
# Tests hyperparameter transfer from base width to target width

set -e

# Configuration (required arguments)
TARGET_WIDTH=$1
BASE_WIDTH=$2
ITERATIONS=${3:-1000}
SEED=${4:-1337}

if [ -z "$TARGET_WIDTH" ] || [ -z "$BASE_WIDTH" ]; then
    echo "Usage: $0 <target_width> <base_width> [iterations] [seed]"
    echo "Example: $0 1024 768 1000 1337"
    exit 1
fi

# Create experiment name for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="mup_w${TARGET_WIDTH}_base${BASE_WIDTH}_${TIMESTAMP}"

echo "=========================================="
echo "MuP Width Scaling (8xH100)"
echo "Target Width: $TARGET_WIDTH"
echo "Base Width: $BASE_WIDTH"
echo "Iterations: $ITERATIONS"
echo "Experiment: $EXP_NAME"
echo "=========================================="

# Validation
if [ "$TARGET_WIDTH" -le "$BASE_WIDTH" ]; then
    echo "Warning: Target width ($TARGET_WIDTH) <= Base width ($BASE_WIDTH)"
    echo "MuP is typically used for scaling UP from base to larger target"
fi

# Check GPU and data
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "Warning: Expected 8 GPUs, found $GPU_COUNT"
fi

if [ ! -d "data" ] || [ ! -f "data/finewebedu_train_000001.bin" ]; then
    echo "Error: Training data not found"
    exit 1
fi

# Create logs directory for this experiment
mkdir -p "logs/$EXP_NAME"

# Run MuP training with coordinate checking
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    train.py \
    --mup \
    --width $TARGET_WIDTH \
    --base-width $BASE_WIDTH \
    --iterations $ITERATIONS \
    --seed $SEED \
    --batch-size 8 \
    --sequence-length 1024 \
    --learning-rate 3e-4 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --warmup-steps 100 \
    --val-every 100 \
    --save-every 500 \
    --log-every 10 \
    --coord-check \
    --coord-check-every 100 \
    --use-wandb true \
    --wandb-project "speedrun-mup" \
    --wandb-name "$EXP_NAME" \
    --compile true \
    --log-dir "logs/$EXP_NAME" \
    2>&1 | tee "logs/$EXP_NAME/console.log"

echo ""
echo "=========================================="
echo "MuP width scaling completed!"
echo "Target: $TARGET_WIDTH | Base: $BASE_WIDTH"
echo "Logs: logs/$EXP_NAME/"
echo "W&B: speedrun-mup/$EXP_NAME"
echo "Check coordinate plots for MuP validation"
echo "=========================================="