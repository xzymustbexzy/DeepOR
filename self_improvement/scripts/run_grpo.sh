#!/usr/bin/env bash
# Run GRPO self-improvement training for DeepOR

set -e

# Adjust paths as needed
CONFIG="configs/example_config.json"

echo "Starting DeepOR GRPO self-improvement training..."
echo "Config: $CONFIG"

# Option 1: Launch with accelerate + DeepSpeed
accelerate launch \
  --config_file configs/accelerate_config.yaml \
  train.py \
  --config "$CONFIG"

# Option 2: Direct launch (single GPU, for testing)
# python train.py --config "$CONFIG"

echo "GRPO training complete."
