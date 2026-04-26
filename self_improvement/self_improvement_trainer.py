"""
Self-Improvement Learning Trainer
Integrates the modeling checklist evaluator and GRPO training.
Compatible with the new TRL GRPOTrainer interface.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch

from grpo_trainer import GRPOTrainer, GRPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    model_name: str = "<YOUR_SFT_MODEL_PATH>"
    output_dir: str = "./self_improvement_output"
    train_data_path: str = ""
    max_train_samples: Optional[int] = 3000
    num_train_epochs: int = 1
    grpo_config: GRPOConfig = None
    seed: int = 42


class DataUtils:
    """Data loading utilities"""

    @staticmethod
    def load_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load data and return a list of dicts"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading data: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                data = json.load(f)
            elif data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Only JSON and JSONL formats are supported")

        if max_samples is not None:
            data = data[:max_samples]

        logger.info(f"Successfully loaded {len(data)} samples")
        return data


class SelfImprovementTrainer:
    """
    Self-improvement learning main trainer (wrapper)
    Responsible for config assembly and launching the TRL Trainer
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        torch.manual_seed(config.seed)
        os.makedirs(config.output_dir, exist_ok=True)

        # Assemble GRPOConfig
        self.grpo_config = config.grpo_config

        # Initialize underlying GRPOTrainer wrapper
        self.grpo_trainer = GRPOTrainer(self.grpo_config)

        # Load data
        self.train_data = []
        if config.train_data_path:
            self.train_data = DataUtils.load_data(
                config.train_data_path,
                config.max_train_samples
            )

        logger.info("Self-improvement trainer initialized")

    def train(self):
        """Execute the full training pipeline"""
        if not self.train_data:
            raise ValueError("No training data provided")

        logger.info(f"Starting GRPO training with {len(self.train_data)} samples")

        # This internally:
        # 1. Converts List[Dict] to Dataset
        # 2. Initializes TRL GRPOTrainer
        # 3. Calls trainer.train() to start the loop
        train_result = self.grpo_trainer.train(self.train_data)

        logger.info("Training complete")
        return train_result

    def load_checkpoint(self, checkpoint_path: str):
        """Load from a checkpoint (placeholder for future use)"""
        logger.info(f"Checkpoint loading not yet implemented: {checkpoint_path}")


if __name__ == "__main__":
    # Example config
    config = TrainingConfig(
        model_name="<YOUR_SFT_MODEL_PATH>",
        output_dir="./self_improvement_output_test",
        train_data_path="<YOUR_RL_DATA_PATH>",
        num_train_epochs=1,
    )

    trainer = SelfImprovementTrainer(config)

    if os.path.exists(config.train_data_path):
        trainer.train()
    else:
        logger.warning(f"Test data not found: {config.train_data_path}")
        print("Please configure the correct data path in the config.")
