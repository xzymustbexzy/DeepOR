#!/usr/bin/env bash
# Run SFT training for DeepOR using LLaMA-Factory

set -e

# Adjust paths as needed
BASE_MODEL="Qwen/Qwen3-8B"         # or your local base model path
DATA_PATH="data/deepor_sft.json"   # path to your SFT dataset
OUTPUT_DIR="saves/deepor/full/sft"

echo "Starting DeepOR SFT training..."
echo "Base model: $BASE_MODEL"
echo "Output dir: $OUTPUT_DIR"

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "$BASE_MODEL" \
  --dataset deepor_sft \
  --template qwen \
  --finetuning_type full \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --deepspeed examples/deepspeed/ds_z3_config.json \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss \
  --overwrite_output_dir \
  --preprocessing_num_workers 16

echo "SFT training complete. Model saved to $OUTPUT_DIR"
