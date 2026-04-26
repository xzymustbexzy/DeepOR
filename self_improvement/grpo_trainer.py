"""
Group Relative Policy Optimization (GRPO) Trainer
Based on TRL library, for optimization modeling reinforcement learning training.
"""
import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import time
import random
import torch
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

from trl import GRPOTrainer as TRLGRPOTrainer
from trl import GRPOConfig as TRLGRPOConfig
from modeling_checklist import ModelingChecklist

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Local GRPO training config (passed to the wrapper)"""
    model_name: str = "<YOUR_SFT_MODEL_PATH>"
    learning_rate: float = 1.4e-5

    # Training batch related
    batch_size: int = 4              # per_device_train_batch_size
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    save_steps: int = 200

    # GRPO-specific parameters
    group_size: int = 4              # Corresponds to TRL num_generations
    max_completion_length: int = 4096
    max_prompt_length: int = 2048
    temperature: float = 1.0
    top_p: float = 0.97
    top_k: int = 20

    # Optimization parameters
    beta: float = 0.1                # KL penalty coefficient
    max_grad_norm: float = 1.0

    # Reward model related
    reward_model_api_key: str = ""
    reward_model_base_url: str = ""
    reward_model: str = "gpt-5-mini"

    # VLLM
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.35
    vllm_tensor_parallel_size: int = 4

    # Output directory (required by HF Trainer)
    output_dir: str = "./grpo_output"


class GRPOTrainer:
    """GRPO trainer wrapper"""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Current model_name is: '{config.model_name}'")

        # 1. Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )

        if hasattr(self.model, "hf_device_map"):
            del self.model.hf_device_map
        if hasattr(self.model, "pretrained_model") and hasattr(self.model.pretrained_model, "hf_device_map"):
            del self.model.pretrained_model.hf_device_map

        # 3. Initialize evaluation tool (Checklist)
        self.checklist = ModelingChecklist(
            api_key=config.reward_model_api_key,
            base_url=config.reward_model_base_url,
            model=config.reward_model
        )

        # Logging initialization
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_file = os.path.join(config.output_dir, "generation_logs.jsonl")
        self.write_lock = threading.Lock()
        logger.info(f"Generation logs will be saved to: {self.log_file}")

    def _save_log(self, data: Dict[str, Any]):
        """Thread-safe log saving to JSONL"""
        json_str = json.dumps(data, ensure_ascii=False)
        with self.write_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json_str + "\n")

    def _process_single_sample(self, prompt: str, completion: str, ground_truth) -> float:
        """Process a single sample: parse -> call API (with retry) -> return score"""
        try:
            if "\n\nAnswer:" in prompt:
                problem_desc = prompt.split("\n\nAnswer:")[0].split(":\n\n")[-1]
            else:
                problem_desc = prompt
        except Exception:
            problem_desc = prompt

        max_retries = 5
        base_delay = 1.0
        final_reward = 0.0
        eval_details = {}

        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.1, 0.5))

                eval_result = self.checklist.evaluate_model(
                    problem=problem_desc,
                    modeling_res=completion,
                    ground_truth=ground_truth,
                    solver_log=""
                )

                final_reward = float(eval_result.get("total_reward", 0.0))
                eval_details = eval_result
                break

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "too many requests" in error_msg:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                elif "401" in error_msg:
                    logger.error("API authentication failed")
                    break
                else:
                    if attempt == max_retries - 1:
                        logger.error(f"Evaluation failed: {e}")
                        break

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": problem_desc,
            "modeling_result": completion,
            "reward": final_reward,
            "details": eval_details.get("detailed_scores", {}),
            "feasibility": eval_details.get("feasibility", {}).get("score", 0),
            "correctness": eval_details.get("correctness", {}).get("score", 0)
        }
        self._save_log(log_entry)
        return final_reward

    def _reward_func(self, prompts: List[str], completions: List[str], ground_truth: List[Any], **kwargs) -> List[float]:
        """Parallel reward computation (with max concurrency limit)"""
        MAX_CONCURRENT_REQUESTS = 10
        rewards = []

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            results = executor.map(self._process_single_sample, prompts, completions, ground_truth)
            rewards = list(results)

        return rewards

    def train(self, problems: List[Dict[str, Any]]):
        """Execute GRPO training"""
        data_list = [
            {"prompt": self._format_problem(p), "ground_truth": p.get("answer", p.get("ground_truth", 0.0))}
            for p in problems
        ]
        train_dataset = Dataset.from_list(data_list)

        trl_args = TRLGRPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            num_generations=self.config.group_size,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            fp16=False,
            bf16=True,
            max_grad_norm=self.config.max_grad_norm,
            beta=self.config.beta,
            use_vllm=True,
            gradient_checkpointing=True,
            vllm_mode=self.config.vllm_mode,
            vllm_gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            vllm_tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            vllm_enable_sleep_mode=True,
            vllm_max_model_length=self.config.max_completion_length + self.config.max_prompt_length,
            logging_steps=1,
            save_steps=self.config.save_steps,
            report_to="none",
        )

        trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self._reward_func,
            args=trl_args,
            train_dataset=train_dataset,
        )

        logger.info("Starting GRPO training...")
        train_result = trainer.train()
        self.save_model(os.path.join(self.config.output_dir, "final_model"))
        return train_result

    def _format_problem(self, problem: Dict[str, Any]) -> str:
        """Format problem as model input"""
        return f"<|im_start|>user\nPlease build a mathematical model and write the corresponding Python code for the following optimization problem:\n\n{problem['problem']}\n\nAnswer:\n<|im_end|>"

    def save_model(self, save_path: str):
        """Save model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to: {save_path}")


if __name__ == "__main__":
    config = GRPOConfig(
        model_name="<YOUR_SFT_MODEL_PATH>",
        batch_size=1,
        group_size=4,
        output_dir="./test_grpo_output"
    )

    trainer_wrapper = GRPOTrainer(config)

    test_problems = [
        {
            "description": "A factory needs to decide production quantities for two products to maximize profit. Product A yields 3 yuan/unit, Product B yields 2 yuan/unit. Labor constraint: total hours <= 100, Product A needs 2 hours/unit, Product B needs 1 hour/unit.",
            "ground_truth": "Optimal: A=0, B=100, max profit=200"
        }
    ] * 2

    trainer_wrapper.train(test_problems)
