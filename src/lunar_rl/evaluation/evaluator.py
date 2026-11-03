"""Evaluation engine for assessing trained Lunar Lander agents in standard and shaped environments."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.lunar_rl.agents.model_manager import ModelManager
from src.lunar_rl.agents.ppo_agent import PPOAgent
from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.evaluation.metrics import (
    EvaluationMetrics,
    compute_evaluation_summary,
)
from src.lunar_rl.rewards.reward_wrapper import wrap_environment
from src.lunar_rl.utils.config_loader import load_config
from src.lunar_rl.utils.logger import get_logger
from src.lunar_rl.utils.reproducibility import set_seed

logger = get_logger(__name__)


class Evaluator:
    """Evaluator providing standardized, fair evaluation across all experiments."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        results_dir: Path = Path("results/metrics"),
    ) -> None:
        self.config = config or load_config()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = ModelManager()

    def evaluate_policy(
        self,
        agent: Union[PPOAgent, Any],
        experiment_name: str,
        num_episodes: int = 100,
        deterministic: bool = True,
        seed: int = 12345,
        use_reward_wrapper: bool = False,
        reward_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[EvaluationMetrics, pd.DataFrame]:
        """Runs evaluation rollouts in either pure native environment or shaped environment.

        Args:
            agent: Policy agent implementing .predict(obs, deterministic).
            experiment_name: Identifier for tracking.
            num_episodes: Number of evaluation episodes (default 100).
            deterministic: Whether to use deterministic action prediction.
            seed: Base random seed for evaluation reproducibility.
            use_reward_wrapper: True to apply reward shaping wrapper, False for pure native env.
            reward_config: Reward config if wrapper is applied.

        Returns:
            Tuple of (EvaluationMetrics summary, episode-level DataFrame).
        """
        set_seed(seed)
        env = make_lunar_environment(
            env_id="LunarLander-v3",
            render_mode=None,
            seed=seed,
        )

        if use_reward_wrapper and reward_config:
            env = wrap_environment(env, reward_config)

        episode_records: List[Dict[str, Any]] = []

        try:
            for ep in range(1, num_episodes + 1):
                obs, info = env.reset(seed=seed + ep)
                total_reward = 0.0
                total_native_reward = 0.0
                total_custom_reward = 0.0
                steps = 0
                done = False

                last_reward = 0.0
                last_obs = obs

                while not done:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)

                    native_r = info.get("original_reward", reward)
                    custom_r = info.get("custom_reward", 0.0)

                    total_reward += reward
                    total_native_reward += native_r
                    total_custom_reward += custom_r
                    steps += 1
                    done = terminated or truncated

                    last_reward = native_r
                    last_obs = obs

                # Criteria:
                # Native LunarLander delivers +100 on safe rest, -100 on crash / out-of-bounds
                is_success = bool(terminated and last_reward >= 100.0)
                is_crash = bool(terminated and last_reward <= -100.0)
                is_timeout = bool(truncated and not is_success and not is_crash)

                # Final state variables
                final_vx = float(last_obs[2])
                final_vy = float(last_obs[3])
                final_speed = float(np.sqrt(final_vx * final_vx + final_vy * final_vy))
                final_angle = float(last_obs[4])
                final_ang_vel = float(last_obs[5])

                episode_records.append({
                    "episode": ep,
                    "total_reward": total_reward,
                    "native_reward": total_native_reward,
                    "custom_reward": total_custom_reward,
                    "length": steps,
                    "is_success": is_success,
                    "is_crash": is_crash,
                    "is_timeout": is_timeout,
                    "final_angle": final_angle,
                    "final_linear_velocity": final_speed,
                    "final_angular_velocity": final_ang_vel,
                })
        finally:
            env.close()

        metrics = compute_evaluation_summary(experiment_name, episode_records)
        df = pd.DataFrame(episode_records)
        return metrics, df

    def evaluate_experiment(
        self,
        experiment_name: str,
        num_episodes: int = 100,
        model_filename: Optional[str] = None,
        deterministic: bool = True,
        seed: int = 12345,
    ) -> Dict[str, Any]:
        """Performs full evaluation of a saved experiment model.

        Evaluates the model in BOTH:
        1. Pure native LunarLander-v3 (unmodified reference for fair comparison).
        2. Shaped training environment (to compare training vs evaluation reward).

        Args:
            experiment_name: Name of experiment.
            num_episodes: Number of episodes to evaluate (default 100).
            model_filename: Specific model zip filename or latest.
            deterministic: Deterministic action selection.
            seed: Evaluation seed.

        Returns:
            Dictionary with native and shaped evaluation summaries.
        """
        logger.info(f"Loading trained agent for experiment '{experiment_name}'...")
        exp_config = load_config(experiment_name=experiment_name)
        model = self.model_manager.load_model(experiment_name, filename=model_filename)
        agent = PPOAgent(config=exp_config, model=model)

        logger.info(
            f"Evaluating '{experiment_name}' on {num_episodes} episodes in pure native LunarLander-v3..."
        )
        # 1. Benchmark in pure native environment (MANDATORY FAIR COMPARISON)
        native_metrics, native_df = self.evaluate_policy(
            agent=agent,
            experiment_name=experiment_name,
            num_episodes=num_episodes,
            deterministic=deterministic,
            seed=seed,
            use_reward_wrapper=False,
        )

        # Save native evaluation CSV
        native_csv_path = self.results_dir / f"{experiment_name}_native_eval.csv"
        native_df.to_csv(native_csv_path, index=False)

        # 2. Benchmark in training environment (with reward wrapper)
        reward_cfg = exp_config.get("reward", {})
        logger.info(
            f"Evaluating '{experiment_name}' in training-environment with reward type '{reward_cfg.get('type')}'..."
        )
        shaped_metrics, shaped_df = self.evaluate_policy(
            agent=agent,
            experiment_name=f"{experiment_name}_shaped",
            num_episodes=num_episodes,
            deterministic=deterministic,
            seed=seed,
            use_reward_wrapper=True,
            reward_config=reward_cfg,
        )

        shaped_csv_path = self.results_dir / f"{experiment_name}_training_env_eval.csv"
        shaped_df.to_csv(shaped_csv_path, index=False)

        # Save summary JSON
        summary = {
            "experiment_name": experiment_name,
            "episodes": num_episodes,
            "deterministic": deterministic,
            "seed": seed,
            "native_evaluation": native_metrics.to_dict(),
            "training_env_evaluation": shaped_metrics.to_dict(),
        }

        summary_json_path = self.results_dir / f"{experiment_name}_eval_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Evaluation finished for '{experiment_name}'. Native Mean Reward: "
            f"{native_metrics.mean_reward:.2f} | Native Success Rate: {native_metrics.success_rate * 100:.1f}%"
        )
        return summary
