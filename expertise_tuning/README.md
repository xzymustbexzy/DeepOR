# Stage 2: Expertise Tuning (SFT)

This directory contains the configuration and scripts for the **Expertise Tuning** stage of DeepOR, which performs supervised fine-tuning (SFT) on the base model using synthesized Chain-of-Thought (CoT) data.

## Prerequisites

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the SFT framework. Install it first:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Data Format

The training data follows the standard **Alpaca** format:

```json
{
  "instruction": "Below is an operations research question. Build a mathematical model and corresponding python code using `pyomo` that appropriately addresses the question.",
  "input": "... problem description ...",
  "output": "... model code and explanation ...",
  "answer": "... ground-truth objective value ..."
}
```

## Registering the Dataset in LLaMA-Factory

Add the following entry to `LLaMA-Factory/data/dataset_info.json`:

```json
"deepor_sft": {
  "file_name": "deepor_sft.json",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

## Training

We provide an example training config (`configs/deepor_qwen3_full_sft.yaml`) for **full fine-tuning** a Qwen3-8B model. Adjust the paths and hyperparameters as needed.

```bash
# Single node, multi-GPU with DeepSpeed ZeRO-3
llamafactory-cli train configs/deepor_qwen3_full_sft.yaml

# Or with accelerate
accelerate launch --config_file configs/accelerate_config.yaml \
  src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path <BASE_MODEL> \
  --dataset deepor_sft \
  --template qwen \
  --finetuning_type full \
  --output_dir saves/deepor/sft \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --deepspeed configs/ds_z3_config.json
```

## Data Source

The full SFT dataset is synthesized using the [Chain-of-Experts (CoE)](../data_synthesis/) pipeline. A sample is provided in `data/sft_data_sample.jsonl`.
