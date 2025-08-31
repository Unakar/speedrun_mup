#!/bin/bash
# Basic speedrun training - single-node 8-GPU H100 setup
# Replicates modded-nanogpt speedrun without MuP

set -e

# Default configuration (8xH100 optimized)
WIDTH=${1:-768}
ITERATIONS=${2:-1750}
SEED=${3:-1337}

echo "=========================================="
echo "Basic Speedrun Training (8xH100)"
echo "Width: $WIDTH | Iterations: $ITERATIONS | Seed: $SEED"
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
if [ ! -d "data" ] || [ ! -f "data/finewebedu_train_000001.bin" ]; then
    echo "Error: Training data not found in data/"
    echo "Run: python scripts/data_process/download_cached_finewebedu_10B.py"
    exit 1
fi

# Run distributed training
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    train.py \
    --width $WIDTH \
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
    --use-wandb true \
    --wandb-project "speedrun-basic" \
    --compile true

echo "Basic speedrun training completed!"
echo "Model width: $WIDTH, Total iterations: $ITERATIONS"