#!/usr/bin/env python3
"""Standalone script to evaluate a trained Lunar Lander agent."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lunar_rl.evaluation.evaluator import Evaluator
from src.lunar_rl.utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardized evaluation of trained Lunar Lander agents."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "reward_hacking", "corrected_reward", "all"],
        help="Experiment to evaluate (default: baseline, or 'all')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Evaluation random seed (default: 12345)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = Evaluator()

    experiments = (
        ["baseline", "reward_hacking", "corrected_reward"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        logger.info(f"--- Commencing Evaluation for: {exp} ---")
        try:
            summary = evaluator.evaluate_experiment(
                experiment_name=exp,
                num_episodes=args.episodes,
                deterministic=not args.stochastic,
                seed=args.seed,
            )
            nat = summary["native_evaluation"]
            logger.info("Evaluation Results (Unmodified Native Environment):")
            logger.info(f"  • Mean Reward:  {nat['mean_reward']:.2f} ± {nat['std_reward']:.2f}")
            logger.info(f"  • Success Rate: {nat['success_rate'] * 100:.1f}%")
            logger.info(f"  • Crash Rate:   {nat['crash_rate'] * 100:.1f}%")
            logger.info(f"  • Timeout Rate: {nat['timeout_rate'] * 100:.1f}%")
            logger.info(f"  • Mean Length:  {nat['mean_episode_length']:.1f} steps")
            logger.info(f"  • Stability:    {nat['landing_stability_score']:.3f}")
        except FileNotFoundError as e:
            logger.error(f"Cannot evaluate '{exp}': {e}")


if __name__ == "__main__":
    main()
