#!/usr/bin/env python3
"""Standalone script to train a PPO agent for Lunar Lander RL Lab."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lunar_rl.training.trainer import Trainer
from src.lunar_rl.utils.config_loader import load_config
from src.lunar_rl.utils.logger import get_logger

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on LunarLander-v3 with customizable reward engineering."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "reward_hacking", "corrected_reward"],
        help="Experiment configuration to run (default: baseline)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps (e.g., 300000 or short test like 5000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional custom YAML config file to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Loading configuration for experiment: {args.experiment}")

    cfg = load_config(experiment_name=args.experiment, config_file=args.config)
    if args.seed is not None:
        cfg["environment"]["seed"] = args.seed

    trainer = Trainer(config=cfg)
    result = trainer.train(override_timesteps=args.timesteps)

    logger.info("Training pipeline complete!")
    logger.info(f"Final Model:   {result['model_path']}")
    logger.info(f"Metrics CSV:   {result['metrics_path']}")
    logger.info(f"Metadata JSON: {result['metadata_path']}")


if __name__ == "__main__":
    main()
