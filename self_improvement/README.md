# Stage 3: Self-Improvement Learning (RL / GRPO)

This directory implements the **Self-Improvement Learning** stage of DeepOR, which uses **Group Relative Policy Optimization (GRPO)** with a **modeling checklist reward-shaping mechanism** to further enhance the model's reasoning capabilities for optimization modeling.

## Overview

The training loop consists of:

1. **Generate**: The policy model generates multiple candidate answers (a "group") for each optimization problem.
2. **Extract & Execute**: Python code is extracted from each candidate and executed with an optimization solver (e.g., HiGHS via Pyomo).
3. **Reward Shaping**: A **modeling checklist** evaluates each candidate across three dimensions:
   - **Feasibility**: Can the code compile? Is the model solvable?
   - **Correctness**: Does the model produce the correct optimal value?
   - **Robustness**: Are constraints properly tight? Does it handle edge cases?
4. **GRPO Update**: The policy is updated using group-relative advantages, with an additional SFT loss on hard examples.

## File Structure

```
self_improvement/
├── train.py                         # CLI entry point
├── grpo_trainer.py                  # GRPO trainer wrapper (around TRL)
├── self_improvement_trainer.py      # High-level training orchestrator
├── modeling_checklist.py            # Reward-shaping checklist evaluator
├── requirements.txt                 # Python dependencies
├── configs/
│   ├── accelerate_config.yaml       # Accelerate + DeepSpeed config
│   ├── deepspeed_config.json        # DeepSpeed ZeRO-2 config
│   └── example_config.json          # Example training configuration
└── scripts/
    └── run_grpo.sh                  # Launch script
```

## Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.43.0`
- `trl==0.26.2` (with GRPO support)
- `vllm==0.11.0`
- `accelerate==1.10.1`
- `deepspeed==0.18.3`
- `pyomo` + `highs` (for solver execution)
- `openai>=1.0.0` (for LLM-as-a-judge in checklist)

## Quick Start

### 1. Prepare Data

Create a JSONL file where each line is:

```json
{"problem": "... optimization problem description ...", "answer": 123.0}
```

A sample is provided at `../data/rl_data_sample.jsonl`.

### 2. Configure Training

Edit `configs/example_config.json`:

```json
{
  "model_name": "/path/to/your/sft_model",
  "train_data_path": "/path/to/rl_train.jsonl",
  "grpo_config": {
    "reward_model_api_key": "sk-...",
    "reward_model_base_url": "https://your-api-endpoint/v1",
    ...
  }
}
```

### 3. Launch Training

```bash
# Multi-GPU with DeepSpeed
bash scripts/run_grpo.sh

# Or manually
accelerate launch --config_file configs/accelerate_config.yaml \
  train.py --config configs/example_config.json
```

## Modeling Checklist

The reward signal is computed by `modeling_checklist.py`, which evaluates each generated answer along three dimensions:

| Dimension | Weight | Key Checks |
|---|---|---|
| Feasibility | 0.3 | Code execution, solver feasibility, variable definitions |
| Correctness | 0.5 | Objective value accuracy, correct target function, complete constraints |
| Robustness | 0.2 | Constraint tightness, integrality handling, edge-case robustness |

The final reward is a weighted sum. For LLM-based checklist items (e.g., "Is the target function correctly modeled?"), an external LLM judge is queried via the OpenAI API.

## Training Tips

- **GPU memory**: GRPO with `vllm` is memory-intensive. Start with `batch_size=1`, `group_size=4`, and `vllm_gpu_memory_utilization=0.35`.
- **Reward API rate limits**: If you hit rate limits, lower `MAX_CONCURRENT_REQUESTS` in `grpo_trainer.py`.
- **Hard examples**: The trainer automatically re-samples hard examples (groups with zero accuracy) for additional SFT loss, which stabilizes training.

## Reference

This implementation is based on the GRPO algorithm described in:

> Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., & Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.
