#!/usr/bin/env python3
"""
DeepOR Self-Improvement Learning Training Script

Usage:
    accelerate launch --use_deepspeed --zero_stage 3 train.py --config configs/example_config.json
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_improvement_trainer import SelfImprovementTrainer, TrainingConfig
from grpo_trainer import GRPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="DeepOR Self-Improvement Learning Training")

    parser.add_argument("--config", type=str, help="Training config file path (JSON format)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="SFT model path")
    parser.add_argument("--output_dir", type=str, default="./self_improvement_output",
                        help="Output directory")
    parser.add_argument("--train_data", type=str, help="Training data path")
    parser.add_argument("--max_train_samples", type=int, help="Max training samples")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--group_size", type=int, default=4, help="Group size")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps interval")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps interval")
    parser.add_argument("--max_completion_length", type=int, default=8192, help="Max completion length")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.97, help="Top p")
    parser.add_argument("--top_k", type=int, default=20, help="Top k")
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--reward_api_key", type=str, default="", help="Reward model API key")
    parser.add_argument("--reward_base_url", type=str, default="", help="Reward model API base URL")
    parser.add_argument("--reward_model", type=str, default="gpt-5-mini", help="Reward model name")
    parser.add_argument("--vllm_mode", type=str, default="colocate", help="VLLM mode")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35, help="VLLM GPU memory utilization")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4, help="VLLM tensor parallel size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")

    return parser.parse_args()


def load_config_from_file(config_path: str) -> TrainingConfig:
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    if 'grpo_config' in config_dict:
        grpo_dict = config_dict.pop('grpo_config')
        grpo_config = GRPOConfig(**grpo_dict)
        config_dict['grpo_config'] = grpo_config

    return TrainingConfig(**config_dict)


def create_config_from_args(args) -> TrainingConfig:
    grpo_config = GRPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        group_size=args.group_size,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        max_grad_norm=args.max_grad_norm,
        reward_model_api_key=args.reward_api_key,
        reward_model_base_url=args.reward_base_url,
        reward_model=args.reward_model,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        output_dir=args.output_dir
    )

    return TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_data_path=args.train_data or "",
        max_train_samples=args.max_train_samples,
        grpo_config=grpo_config,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
    )


def save_config(config: TrainingConfig, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False, default=str)

    print(f"Training config saved to: {config_path}")


def main():
    args = parse_args()

    if args.config:
        print(f"Loading from config file: {args.config}")
        config = load_config_from_file(args.config)
    else:
        print("Creating config from command-line arguments")
        config = create_config_from_args(args)

    save_config(config, config.output_dir)

    print("Initializing trainer...")
    trainer = SelfImprovementTrainer(config)

    if getattr(args, 'resume_from_checkpoint', None):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)

    if config.train_data_path and os.path.exists(config.train_data_path):
        print("Starting training...")
        trainer.train()
        print("Training complete!")
    else:
        print("Error: No valid training data path provided")
        print(f"Current path: {config.train_data_path}")
        return

    print(f"Training results saved in: {config.output_dir}")


if __name__ == "__main__":
    main()
