# Stage 1: Cold-Start Data Synthesis

This directory implements the **cold-start data synthesis** pipeline for DeepOR, which uses a **Chain-of-Experts (CoE)** multi-agent framework to generate high-quality Chain-of-Thought (CoT) reasoning traces for operations research (OR) optimization modeling problems.

## Overview

The pipeline consists of two main steps:

1. **Chain-of-Experts Synthesis** (`run_data_synthesis.py`)
   - For each problem, invokes a multi-expert collaboration system (Terminology Interpreter, Parameter Extractor, Modeling Expert, Programming Expert, Code Reviewer, etc.).
   - Each expert contributes a partial reasoning step toward solving the OR problem.
   - The Conductor dynamically schedules the order of expert collaboration.

2. **CoT Revision** (`reviser.py`)
   - Takes the multi-expert output and rewrites it into a single, fluent, human-like reasoning trace.
   - Ensures smooth transitions between reasoning steps while preserving correctness.

## File Structure

```
data_synthesis/
├── run_data_synthesis.py      # Main pipeline entry point
├── reviser.py                  # CoT revision module
├── cot_revised.txt             # Example of a well-revised CoT (used as in-context example)
├── requirements.txt            # Python dependencies
└── coe/                        # Chain-of-Experts framework
    ├── llm.py                  # LLM client wrapper (OpenAI / Azure / DeepSeek)
    ├── main.py                 # CoE orchestration logic
    ├── conductor.py            # Expert scheduling
    ├── reducer.py              # Answer aggregation
    ├── evaluator.py            # Reflection-based evaluator
    ├── comment.py              # Expert comment data structure
    ├── comment_pool.py         # Comment management
    ├── utils.py                # Utility functions
    ├── extract_program.py      # Code extraction helpers
    └── experts/                # Individual expert implementations
        ├── base_expert.py
        ├── terminology_interpreter.py
        ├── parameter_extractor.py
        ├── modeling_expert.py
        ├── programming_expert.py
        ├── code_reviewer.py
        └── programming_example_provider.py
```

## Quick Start

### 1. Install Dependencies

```bash
cd data_synthesis
pip install -r requirements.txt
```

### 2. Configure LLM Access

Set environment variables for your LLM provider:

```bash
# Option A: OpenAI-compatible API (e.g., DeepSeek)
export LLM_PROVIDER="openai"
export LLM_API_KEY="sk-..."
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"

# Option B: Azure OpenAI
export LLM_PROVIDER="azure"
export LLM_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
export LLM_MODEL="gpt-5"
```

### 3. Run Synthesis

```bash
python run_data_synthesis.py
```

The script reads `../data/seed_problems.json` and writes synthesized outputs to `../data/synthesis_examples/` (one folder per problem).

## Parameters

| Environment Variable | Description | Default |
|---|---|---|
| `SYNTHESIS_MODEL` | LLM for CoE synthesis | Value of `LLM_MODEL` |
| `REVISER_MODEL` | LLM for CoT revision | Same as `SYNTHESIS_MODEL` |

## Output Format

Each problem produces a folder `synthesis_examples/prob_{i}/` containing:

- `description.txt`: Original problem description
- `cot.txt`: Raw multi-expert CoT output
- `revised_cot.txt`: Single-expert-style revised CoT
- `answer.txt`: Ground-truth optimal objective value

## Citation

The Chain-of-Experts framework is adapted from:

> Xiao, Z., Zhang, D., Wu, Y., Xu, L., Wang, Y. J., Han, X., Fu, X., Zhong, T., Zeng, J., Song, M., & Chen, G. (2024). Chain-of-Experts: When LLMs Meet Complex Operations Research Problems. *ICLR 2024*.
