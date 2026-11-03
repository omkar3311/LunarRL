"""PPO Reinforcement Learning Agent wrapper built on Stable-Baselines3."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


class PPOAgent:
    """Configurable PPO agent for LunarLander-v3."""

    def __init__(
        self,
        env: Optional[Env] = None,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[PPO] = None,
    ) -> None:
        """Initializes the PPO agent.

        Args:
            env: Optional Gymnasium environment. Required if creating a new model.
            config: Full configuration dictionary containing 'agent' parameters.
            model: Optional pre-instantiated or loaded PPO model.
        """
        self.config = config or {}
        agent_cfg = self.config.get("agent", {})
        self.policy = agent_cfg.get("policy", "MlpPolicy")
        self.learning_rate = agent_cfg.get("learning_rate", 3e-4)
        self.n_steps = agent_cfg.get("n_steps", 1024)
        self.batch_size = agent_cfg.get("batch_size", 64)
        self.n_epochs = agent_cfg.get("n_epochs", 10)
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.gae_lambda = agent_cfg.get("gae_lambda", 0.95)
        self.clip_range = agent_cfg.get("clip_range", 0.2)
        self.ent_coef = agent_cfg.get("ent_coef", 0.01)
        self.vf_coef = agent_cfg.get("vf_coef", 0.5)
        self.max_grad_norm = agent_cfg.get("max_grad_norm", 0.5)
        self.policy_kwargs = agent_cfg.get("policy_kwargs", {"net_arch": [64, 64]})

        tb_log = self.config.get("training", {}).get("tensorboard_log", None)

        if model is not None:
            self.model = model
            if env is not None:
                self.model.set_env(env)
        elif env is not None:
            logger.info("Initializing Stable-Baselines3 PPO model...")
            self.model = PPO(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=self.policy_kwargs,
                tensorboard_log=str(tb_log) if tb_log else None,
                verbose=1,
            )
        else:
            self.model = None

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, list]] = None,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "PPOAgent":
        """Executes explicit training of the PPO model.

        Args:
            total_timesteps: Total environment timesteps to collect.
            callback: SB3 callback or list of callbacks.
            tb_log_name: Experiment name for TensorBoard run.
            reset_num_timesteps: Whether to reset internal timestep counter.
            progress_bar: Whether to display SB3 tqdm progress bar.

        Returns:
            Self instance.
        """
        if self.model is None:
            raise RuntimeError("Cannot train: model is not initialized with an environment.")

        logger.info(f"Starting PPO training for {total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
        logger.info("Training cycle complete.")
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[Tuple[np.ndarray, ...]]]:
        """Predicts the next action for an observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic mode (greedy mode).

        Returns:
            Tuple of (discrete action, hidden states).
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: Union[str, Path]) -> None:
        """Saves the underlying model weights and parameters."""
        if self.model is None:
            raise RuntimeError("Cannot save: model is not initialized.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[Env] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "PPOAgent":
        """Loads a saved PPO model from disk.

        Args:
            path: Path to the .zip model file.
            env: Optional environment to bind.
            config: Optional config dictionary.

        Returns:
            Initialized PPOAgent wrapping the loaded model.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading PPOAgent from {path}")
        loaded_model = PPO.load(str(path), env=env)
        return cls(env=env, config=config, model=loaded_model)
