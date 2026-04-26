# DeepOR: A Deep Reasoning Foundation Model for Optimization Modeling

[![AAAI 2025](https://img.shields.io/badge/AAAI-2025-blue)](https://aaai.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **DeepOR**, a deep reasoning foundation model specifically designed for optimization modeling, accepted at **AAAI 2025**.

> **DeepOR: A Deep Reasoning Foundation Model for Optimization Modeling**  
> Ziyang Xiao, Yuan Jessica Wang, Xiongwei Han, Shisi Guan, Jingyan Zhu, Jingrong Xie, Lilin Xu, Han Wu, Wing Yin Yu, Zehua Liu, Xiaojin Fu, Gang Chen, Dongxiang Zhang  
> *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2025*

## Overview

DeepOR addresses the challenge of automating operations research (OR) optimization modeling — transforming natural language problem descriptions into formal mathematical models and executable solver code. Unlike prior approaches that directly generate solutions, DeepOR explicitly performs multi-step intermediate reasoning, mimicking how human experts solve complex OR problems.

## Repository Structure

```
DeepOR_repo/
├── data/                         # Sample data
│   ├── seed_problems.json        # 9,132 seed OR problems
│   ├── sft_data_sample.jsonl     # SFT data sample (Alpaca format)
│   ├── rl_data_sample.jsonl      # RL data sample
│   └── synthesis_examples/       # Example CoT outputs (4 problems)
│
├── data_synthesis/               # Stage 1: Cold-start data synthesis
│   ├── README.md
│   ├── requirements.txt
│   ├── run_data_synthesis.py     # CoE + reviser pipeline
│   ├── reviser.py
│   ├── cot_revised.txt           # In-context revision example
│   └── coe/                      # Chain-of-Experts framework
│       ├── llm.py
│       ├── main.py
│       ├── conductor.py
│       ├── reducer.py
│       └── experts/              # Individual expert agents
│
├── expertise_tuning/             # Stage 2: SFT via LLaMA-Factory
│   ├── README.md
│   ├── configs/
│   │   └── deepor_qwen3_full_sft.yaml
│   ├── data/
│   │   └── dataset_info_snippet.json
│   └── scripts/
│       └── run_sft.sh
│
└── self_improvement/             # Stage 3: RL/GRPO post-training
    ├── README.md
    ├── requirements.txt
    ├── train.py                  # CLI entry point
    ├── grpo_trainer.py           # GRPO trainer (wraps TRL)
    ├── self_improvement_trainer.py
    ├── modeling_checklist.py     # Reward-shaping checklist
    ├── configs/
    │   ├── accelerate_config.yaml
    │   ├── deepspeed_config.json
    │   └── example_config.json
    └── scripts/
        └── run_grpo.sh
```

## Installation

### Prerequisites

- Python >= 3.9
- CUDA >= 11.8 (for GPU training)
- 4+ GPUs recommended for distributed training

### Stage 1: Data Synthesis

```bash
cd data_synthesis
pip install -r requirements.txt

# Configure your LLM API key
export LLM_API_KEY="sk-..."
export LLM_MODEL="deepseek-chat"
```

### Stage 2: Expertise Tuning (SFT)

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the SFT framework.

```bash
# Install LLaMA-Factory (do this outside this repo)
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Then copy our config and dataset snippet into LLaMA-Factory (see `expertise_tuning/README.md`).

### Stage 3: Self-Improvement (GRPO)

```bash
cd self_improvement
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.43.0`
- `trl==0.26.2`
- `vllm==0.11.0`
- `deepspeed==0.18.3`
- `pyomo` + `highs` (for solver execution)

## Quick Start

### 1. Data Synthesis

```bash
cd data_synthesis
python run_data_synthesis.py
```

This reads `data/seed_problems.json` and generates CoT reasoning traces using the Chain-of-Experts framework, then revises them into a single fluent trace.

### 2. Expertise Tuning

```bash
cd expertise_tuning
# Register dataset in LLaMA-Factory/data/dataset_info.json
# Then launch training
bash scripts/run_sft.sh
```

### 3. Self-Improvement Learning

```bash
cd self_improvement
# Edit configs/example_config.json with your model path and data path
# Then launch training
bash scripts/run_grpo.sh
```

## Modeling Checklist

The reward-shaping mechanism in the RL stage evaluates each generated answer with a fine-grained checklist across three dimensions:

| Dimension | Weight | Key Checks |
|---|---|---|
| **Feasibility** | 0.3 | Code compilation, solver feasibility, variable definitions |
| **Correctness** | 0.5 | Objective value accuracy, correct target function, complete constraints |
| **Robustness** | 0.2 | Constraint tightness, integrality handling, edge cases |

This checklist-based reward provides more stable and informative training signals than sparse objective-value rewards.

## Data

We provide sample data in the `data/` directory:

- **seed_problems.json** (9,132 instances): Seed OR problems with ground-truth answers.
- **sft_data_sample.jsonl** (20 instances): Sample SFT data in Alpaca format.
- **rl_data_sample.jsonl** (20 instances): Sample RL data (problem + answer).
- **synthesis_examples/**: 4 example outputs showing raw CoT, revised CoT, and answers.

Full datasets can be synthesized/generated using the provided code.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{xiao2025deepor,
  title={DeepOR: A Deep Reasoning Foundation Model for Optimization Modeling},
  author={Xiao, Ziyang and Wang, Yuan Jessica and Han, Xiongwei and Guan, Shisi and Zhu, Jingyan and Xie, Jingrong and Xu, Lilin and Wu, Han and Yu, Wing Yin and Liu, Zehua and Fu, Xiaojin and Chen, Gang and Zhang, Dongxiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025},
  organization={AAAI}
}
```

## License

This project is released under the [MIT License](LICENSE).
