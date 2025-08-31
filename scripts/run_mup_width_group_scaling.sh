#!/bin/bash
# Multi-width MuP scaling validation - single-node 8-GPU H100 setup
# Tests hyperparameter transfer across multiple widths systematically

set -e

# Configuration
BASE_WIDTH=${1:-256}
WIDTHS=${2:-"512 768 1024 1536"}
ITERATIONS=${3:-1000}
SEED=${4:-1337}

echo "=========================================="
echo "MuP Multi-Width Group Scaling (8xH100)"
echo "Base Width: $BASE_WIDTH"
echo "Target Widths: $WIDTHS"
echo "Iterations per width: $ITERATIONS"
echo "Seed: $SEED"
echo "=========================================="

# Convert widths string to array
WIDTH_ARRAY=($WIDTHS)
TOTAL_WIDTHS=${#WIDTH_ARRAY[@]}

echo "Will train $TOTAL_WIDTHS models with widths: ${WIDTH_ARRAY[*]}"
echo "Base width for all experiments: $BASE_WIDTH"
echo ""

# Check prerequisites
if [ ! -d "data" ] || [ ! -f "data/finewebedu10B/finewebedu_train_000001.bin" ]; then
    echo "Error: Training data not found in data/"
    echo "Run: bash /root/xietian/dev/speedrun_mup/scripts/data_process/download_hf_data.sh"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "Warning: Expected 8 GPUs, found $GPU_COUNT"
fi

# Create group experiment timestamp
GROUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GROUP_NAME="mup_group_base${BASE_WIDTH}_${GROUP_TIMESTAMP}"
GROUP_LOG_DIR="logs/$GROUP_NAME"

echo "Group experiment: $GROUP_NAME"
echo "Group logs: $GROUP_LOG_DIR/"
echo ""

mkdir -p "$GROUP_LOG_DIR"

# Save group experiment configuration
cat > "$GROUP_LOG_DIR/experiment_config.json" << EOF
{
  "experiment_type": "mup_width_group_scaling",
  "base_width": $BASE_WIDTH,
  "target_widths": [$(IFS=,; echo "${WIDTH_ARRAY[*]}")],
  "iterations_per_width": $ITERATIONS,
  "seed": $SEED,
  "timestamp": "$GROUP_TIMESTAMP",
  "total_experiments": $TOTAL_WIDTHS,
  "hardware": "8xH100"
}
EOF

# Initialize group summary log
echo "Starting MuP group scaling experiment: $GROUP_NAME" > "$GROUP_LOG_DIR/group_summary.log"
echo "Base width: $BASE_WIDTH" >> "$GROUP_LOG_DIR/group_summary.log"
echo "Target widths: ${WIDTH_ARRAY[*]}" >> "$GROUP_LOG_DIR/group_summary.log"
echo "Started at: $(date)" >> "$GROUP_LOG_DIR/group_summary.log"
echo "" >> "$GROUP_LOG_DIR/group_summary.log"

# Track timing
GROUP_START_TIME=$(date +%s)

# Run each width experiment
for i in "${!WIDTH_ARRAY[@]}"; do
    WIDTH=${WIDTH_ARRAY[$i]}
    EXP_NUM=$((i + 1))
    
    echo "=========================================="
    echo "Experiment $EXP_NUM/$TOTAL_WIDTHS: Width $WIDTH"
    echo "=========================================="
    
    # Individual experiment name
    EXP_NAME="mup_w${WIDTH}_base${BASE_WIDTH}_${GROUP_TIMESTAMP}"
    EXP_LOG_DIR="$GROUP_LOG_DIR/width_$WIDTH"
    
    mkdir -p "$EXP_LOG_DIR"
    
    # Log to group summary
    echo "[$EXP_NUM/$TOTAL_WIDTHS] Starting width $WIDTH at $(date)" >> "$GROUP_LOG_DIR/group_summary.log"
    
    # Track individual timing
    EXP_START_TIME=$(date +%s)
    
    # Run training
    if torchrun \
        --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12345 \
        train.py \
        --mup \
        --width $WIDTH \
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
        --wandb-project "speedrun-mup-group" \
        --wandb-name "$EXP_NAME" \
        --compile true \
        --log-dir "$EXP_LOG_DIR" \
        > "$EXP_LOG_DIR/console.log" 2>&1; then
        
        # Success
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo "[$EXP_NUM/$TOTAL_WIDTHS] Completed width $WIDTH in ${EXP_DURATION}s at $(date)" >> "$GROUP_LOG_DIR/group_summary.log"
        echo "✓ Width $WIDTH completed successfully (${EXP_DURATION}s)"
        
    else
        # Failure
        echo "[$EXP_NUM/$TOTAL_WIDTHS] FAILED width $WIDTH at $(date)" >> "$GROUP_LOG_DIR/group_summary.log"
        echo "✗ Width $WIDTH failed! Check logs: $EXP_LOG_DIR/console.log"
        
        # Continue with remaining widths rather than exit
        echo "Continuing with remaining widths..."
    fi
    
    echo ""
done

# Finalize group experiment
GROUP_END_TIME=$(date +%s)
GROUP_DURATION=$((GROUP_END_TIME - GROUP_START_TIME))

echo "" >> "$GROUP_LOG_DIR/group_summary.log"
echo "Group experiment completed at: $(date)" >> "$GROUP_LOG_DIR/group_summary.log"
echo "Total duration: ${GROUP_DURATION}s" >> "$GROUP_LOG_DIR/group_summary.log"

echo "=========================================="
echo "MuP Group Scaling Completed!"
echo "Base width: $BASE_WIDTH"
echo "Tested widths: ${WIDTH_ARRAY[*]}"
echo "Total duration: ${GROUP_DURATION}s"
echo ""
echo "Results:"
echo "- Group logs: $GROUP_LOG_DIR/"
echo "- Individual experiments: $GROUP_LOG_DIR/width_*/"
echo "- W&B project: speedrun-mup-group"
echo "- Summary: $GROUP_LOG_DIR/group_summary.log"
echo ""
echo "Use coordinate check plots to validate MuP implementation"
echo "=========================================="