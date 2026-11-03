#!/usr/bin/env python3
"""Unified Command-Line Interface for Lunar Lander RL Lab."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lunar_rl.environment.random_agent import run_random_agent
from src.lunar_rl.evaluation.comparison import (
    generate_comparison_report,
    generate_comparison_table,
)
from src.lunar_rl.evaluation.evaluator import Evaluator
from src.lunar_rl.training.trainer import Trainer
from src.lunar_rl.utils.config_loader import load_config
from src.lunar_rl.utils.logger import get_logger
from src.lunar_rl.visualization.plots import generate_all_plots
from src.lunar_rl.visualization.video_recorder import VideoRecorder

logger = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="lunar-rl",
        description="Lunar Lander RL Lab — Reinforcement Learning & Reward Engineering Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  random     Run random agent to establish baseline performance
  train      Train PPO agent on baseline, reward_hacking, or corrected_reward
  evaluate   Evaluate trained models over multiple episodes in pure native env
  compare    Generate cross-regime comparison table and report
  record     Record gameplay video (MP4) using RGB array rendering
  plot       Generate all visualization charts in results/plots/

Examples:
  python main.py random
  python main.py train --experiment baseline --timesteps 300000
  python main.py train --experiment reward_hacking
  python main.py train --experiment corrected_reward
  python main.py evaluate --experiment baseline --episodes 100
  python main.py compare
  python main.py record --experiment corrected_reward
  python main.py plot
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Command: random
    sub_random = subparsers.add_parser(
        "random", help="Run random agent to benchmark untrained behavior"
    )
    sub_random.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes (default: 5)"
    )
    sub_random.add_argument(
        "--render", action="store_true", help="Render human visualization window"
    )
    sub_random.add_argument(
        "--record", action="store_true", help="Record MP4 gameplay video of random agent"
    )
    sub_random.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility (default: 42)"
    )

    # Command: train
    sub_train = subparsers.add_parser(
        "train", help="Train a PPO agent under a selected reward configuration"
    )
    sub_train.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "reward_hacking", "corrected_reward"],
        help="Experiment regime to train (default: baseline)",
    )
    sub_train.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps (e.g., 300000)",
    )
    sub_train.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment and neural network",
    )
    sub_train.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML file",
    )

    # Command: evaluate
    sub_eval = subparsers.add_parser(
        "evaluate", help="Evaluate trained agent in pure native LunarLander-v3"
    )
    sub_eval.add_argument(
        "--experiment",
        type=str,
        default="baseline",
        choices=["baseline", "reward_hacking", "corrected_reward", "all"],
        help="Experiment to evaluate (default: baseline, or 'all')",
    )
    sub_eval.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    sub_eval.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for evaluation (default: 12345)",
    )
    sub_eval.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of deterministic",
    )

    # Command: compare
    subparsers.add_parser(
        "compare", help="Compare all evaluated experiments, generate table and report"
    )

    # Command: record
    sub_record = subparsers.add_parser(
        "record", help="Record gameplay video for an agent or random baseline"
    )
    sub_record.add_argument(
        "--experiment",
        type=str,
        default="corrected_reward",
        choices=["baseline", "reward_hacking", "corrected_reward", "random"],
        help="Experiment model to record (default: corrected_reward)",
    )
    sub_record.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    sub_record.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recording seed (default: 42)",
    )
    sub_record.add_argument(
        "--journey",
        action="store_true",
        help="Capture failure attempt, perfect safe landing attempt, and merged continuous video",
    )

    # Command: plot
    subparsers.add_parser(
        "plot", help="Generate all analysis charts and save to results/plots/"
    )

    return parser


def handle_random(args: argparse.Namespace) -> None:
    render_mode = "human" if args.render else None
    run_random_agent(
        episodes=args.episodes,
        render_mode=render_mode,
        seed=args.seed,
        log_observations=True,
    )
    if args.record:
        print("\nRecording video for Random Agent...")
        recorder = VideoRecorder()
        v_path = recorder.record_experiment("random", episodes=args.episodes, seed=args.seed)
        print(f"Random agent video recorded: {v_path}")


def handle_train(args: argparse.Namespace) -> None:
    cfg = load_config(experiment_name=args.experiment, config_file=args.config)
    if args.seed is not None:
        cfg["environment"]["seed"] = args.seed

    trainer = Trainer(config=cfg)
    result = trainer.train(override_timesteps=args.timesteps)
    print("\n" + "=" * 70)
    print(" TRAINING SUMMARY")
    print("=" * 70)
    print(f"Experiment:    {result['experiment_name']}")
    print(f"Model Path:    {result['model_path']}")
    print(f"Metrics CSV:   {result['metrics_path']}")
    print(f"Metadata JSON: {result['metadata_path']}")
    print(f"Episodes:      {result['episodes']}")
    print(f"Duration:      {result['duration_seconds']:.2f} s")
    print("=" * 70 + "\n")


def handle_evaluate(args: argparse.Namespace) -> None:
    evaluator = Evaluator()
    experiments = (
        ["baseline", "reward_hacking", "corrected_reward"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        print("\n" + "=" * 70)
        print(f" EVALUATING EXPERIMENT: {exp.upper()}")
        print("=" * 70)
        try:
            summary = evaluator.evaluate_experiment(
                experiment_name=exp,
                num_episodes=args.episodes,
                deterministic=not args.stochastic,
                seed=args.seed,
            )
            nat = summary["native_evaluation"]
            tra = summary["training_env_evaluation"]
            print("\nResults in Pure Native LunarLander-v3 (Fair Comparison):")
            print(f"  • Mean Return:    {nat['mean_reward']:.2f} ± {nat['std_reward']:.2f}")
            print(f"  • Median Return:  {nat['median_reward']:.2f}")
            print(f"  • Success Rate:   {nat['success_rate'] * 100:.1f}%")
            print(f"  • Crash Rate:     {nat['crash_rate'] * 100:.1f}%")
            print(f"  • Timeout Rate:   {nat['timeout_rate'] * 100:.1f}%")
            print(f"  • Mean Length:    {nat['mean_episode_length']:.1f} steps")
            print(f"  • Stability:      {nat['landing_stability_score']:.3f}")
            print("\nResults in Training Environment (Agent's Internal Reward):")
            print(f"  • Mean Return:    {tra['mean_reward']:.2f} ± {tra['std_reward']:.2f}")
            print(f"  • Reward Gap:     {tra['mean_reward'] - nat['mean_reward']:+.2f}")
            print("=" * 70)
        except FileNotFoundError as e:
            logger.error(f"Cannot evaluate '{exp}': {e}")


def handle_compare(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print(" CROSS-REGIME COMPARISON")
    print("=" * 70)
    df = generate_comparison_table()
    if not df.empty:
        print(df.to_string(index=False))
        generate_comparison_report(df)
        plots = generate_all_plots()
        print(f"\nSaved comparison table to results/metrics/comparison_table.csv")
        print(f"Saved report to results/metrics/comparison_report.md")
        print(f"Generated {len(plots)} charts in results/plots/")
    else:
        print("No evaluation data found. Run evaluations first via: python main.py evaluate --experiment all")
    print("=" * 70 + "\n")


def handle_record(args: argparse.Namespace) -> None:
    recorder = VideoRecorder()
    if args.journey:
        print(f"\nRecording Failure + Safe Landing Journey for '{args.experiment}'...")
        from src.lunar_rl.agents.model_manager import ModelManager
        from src.lunar_rl.agents.ppo_agent import PPOAgent

        model = ModelManager().load_model(args.experiment)
        cfg = load_config(experiment_name=args.experiment)
        agent = PPOAgent(config=cfg, model=model)

        paths = recorder.record_multi_attempt_journey(
            agent=agent,
            fail_seed_1=26,
            fail_seed_2=27,
            success_seed=24,
            combined_filename="continuous_journey_landing.mp4",
        )
        print("\n" + "=" * 70)
        print(" FLIGHT RECORDING JOURNEY COMPLETE")
        print("=" * 70)
        for key, p in paths.items():
            print(f"  • {key.upper():8s} -> {p}")
        print("=" * 70)
    else:
        print(f"\nRecording video for '{args.experiment}' ({args.episodes} episode)...")
        try:
            video_path = recorder.record_experiment(
                experiment_name=args.experiment,
                episodes=args.episodes,
                deterministic=True,
                seed=args.seed,
            )
            print(f"Video recorded successfully: {video_path}")
        except Exception as e:
            logger.error(f"Recording failed: {e}")


def handle_plot(args: argparse.Namespace) -> None:
    plots = generate_all_plots()
    print(f"\nGenerated {len(plots)} visualization plots in results/plots/:")
    for p in plots:
        print(f"  • {p.name}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "random": handle_random,
        "train": handle_train,
        "evaluate": handle_evaluate,
        "compare": handle_compare,
        "record": handle_record,
        "plot": handle_plot,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
