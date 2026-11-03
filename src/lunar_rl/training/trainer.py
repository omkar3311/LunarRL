"""Reusable training system for Lunar Lander RL Lab experiments."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.lunar_rl.agents.model_manager import ModelManager
from src.lunar_rl.agents.ppo_agent import PPOAgent
from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.rewards.reward_wrapper import wrap_environment
from src.lunar_rl.training.callbacks import CheckpointCallback, RewardDecompositionCallback
from src.lunar_rl.utils.config_loader import load_config
from src.lunar_rl.utils.logger import get_logger
from src.lunar_rl.utils.reproducibility import set_seed

logger = get_logger(__name__)


class Trainer:
    """End-to-end trainer orchestrating environment setup, reward wrapping, PPO training, and logging."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Initializes Trainer with configuration.

        Args:
            config: Full configuration dictionary. If None, loaded via experiment_name.
            experiment_name: Name of experiment to load config for ('baseline', 'reward_hacking', 'corrected_reward').
        """
        if config is None:
            if experiment_name is None:
                experiment_name = "baseline"
            self.config = load_config(experiment_name=experiment_name)
        else:
            self.config = config

        self.experiment_name = self.config.get("experiment_name", "baseline")
        self.model_manager = ModelManager(Path("models"))

    def train(self, override_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Runs the complete training lifecycle.

        Args:
            override_timesteps: Optional override for total training timesteps.

        Returns:
            Dictionary containing paths to artifacts and training metadata.
        """
        env_cfg = self.config.get("environment", {})
        training_cfg = self.config.get("training", {})
        reward_cfg = self.config.get("reward", {})

        seed = env_cfg.get("seed", 42)
        total_timesteps = override_timesteps or training_cfg.get("total_timesteps", 300000)
        save_freq = training_cfg.get("save_frequency", 25000)

        # 1. Ensure deterministic seed
        set_seed(seed)

        # 2. Setup directories
        exp_dir = Path("experiments") / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = self.model_manager.get_checkpoint_dir(self.experiment_name)
        tb_dir = Path("logs") / "tensorboard" / self.experiment_name
        tb_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info(f"STARTING EXPERIMENT TRAINING: {self.experiment_name.upper()}")
        logger.info(f"Target Timesteps: {total_timesteps:,} | Seed: {seed}")
        logger.info(f"Reward Type: {reward_cfg.get('type')}")
        logger.info("=" * 70)

        # 3. Create and wrap environment
        env = make_lunar_environment(
            env_id=env_cfg.get("id", "LunarLander-v3"),
            render_mode=env_cfg.get("render_mode", None),
            seed=seed,
            continuous=env_cfg.get("continuous", False),
            gravity=env_cfg.get("gravity", -10.0),
            enable_wind=env_cfg.get("enable_wind", False),
        )

        env = wrap_environment(env, reward_cfg)

        # 4. Initialize PPO Agent
        # Update config with tensorboard directory
        self.config["training"]["tensorboard_log"] = str(tb_dir)
        agent = PPOAgent(env=env, config=self.config)

        # 5. Setup callbacks
        metrics_csv_path = exp_dir / "training_metrics.csv"
        decomp_cb = RewardDecompositionCallback(metrics_save_path=metrics_csv_path)
        ckpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=ckpt_dir,
            name_prefix=self.experiment_name,
        )

        # 6. Execute training
        start_time = time.time()
        try:
            agent.train(
                total_timesteps=total_timesteps,
                callback=[decomp_cb, ckpt_cb],
                tb_log_name=self.experiment_name,
                progress_bar=False,
            )
        finally:
            env.close()

        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds.")

        # 7. Save final model and metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "timesteps": total_timesteps,
            "training_duration_seconds": round(duration, 2),
            "seed": seed,
            "environment": env_cfg,
            "agent": self.config.get("agent", {}),
            "reward": reward_cfg,
            "episodes_trained": decomp_cb.episode_count,
        }

        model_path = self.model_manager.save_model(
            model=agent.model,
            experiment_name=self.experiment_name,
            metadata=metadata,
        )

        # Also write metadata into experiments/<experiment_name>/metadata.json
        meta_json_path = exp_dir / "metadata.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "experiment_name": self.experiment_name,
            "model_path": str(model_path),
            "metrics_path": str(metrics_csv_path),
            "metadata_path": str(meta_json_path),
            "episodes": decomp_cb.episode_count,
            "duration_seconds": duration,
        }


def train_experiment(
    experiment_name: str,
    override_timesteps: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience entry point to run an experiment training pipeline."""
    trainer = Trainer(experiment_name=experiment_name)
    return trainer.train(override_timesteps=override_timesteps)
