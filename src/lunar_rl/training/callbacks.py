"""Custom Stable-Baselines3 callbacks for reward decomposition and metric logging."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


class RewardDecompositionCallback(BaseCallback):
    """Callback tracking and logging native vs shaped reward contributions.

    Records detailed per-episode metrics into TensorBoard and maintains an
    in-memory history exported to CSV for visualization and report generation.
    """

    def __init__(
        self,
        metrics_save_path: Optional[Path] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.metrics_save_path = Path(metrics_save_path) if metrics_save_path else None
        self.episode_records: List[Dict[str, Any]] = []

        # Current episode accumulators (per env if vectorized)
        self._cur_native_reward: float = 0.0
        self._cur_shaped_reward: float = 0.0
        self._cur_custom_reward: float = 0.0
        self._cur_length: int = 0
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        # In single env, locals['rewards'] is array of length 1
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])

        for info in infos:
            native_r = info.get("original_reward", 0.0)
            shaped_r = info.get("shaped_reward", native_r)
            custom_r = info.get("custom_reward", 0.0)

            self._cur_native_reward += native_r
            self._cur_shaped_reward += shaped_r
            self._cur_custom_reward += custom_r
            self._cur_length += 1

            if dones[0]:
                self.episode_count += 1
                is_success = bool(native_r >= 100.0)
                is_crash = bool(native_r <= -100.0)

                record = {
                    "timestep": int(self.num_timesteps),
                    "episode": self.episode_count,
                    "length": self._cur_length,
                    "native_reward": self._cur_native_reward,
                    "shaped_reward": self._cur_shaped_reward,
                    "custom_reward": self._cur_custom_reward,
                    "is_success": is_success,
                    "is_crash": is_crash,
                }
                self.episode_records.append(record)

                # Log to TensorBoard through SB3 Logger
                self.logger.record("rollout/native_reward", self._cur_native_reward)
                self.logger.record("rollout/shaped_reward", self._cur_shaped_reward)
                self.logger.record("rollout/custom_reward", self._cur_custom_reward)
                self.logger.record("rollout/episode_length", self._cur_length)
                self.logger.record("rollout/is_success", 1.0 if is_success else 0.0)
                self.logger.record("rollout/is_crash", 1.0 if is_crash else 0.0)

                # Reset accumulator for next episode
                self._cur_native_reward = 0.0
                self._cur_shaped_reward = 0.0
                self._cur_custom_reward = 0.0
                self._cur_length = 0

        return True

    def _on_training_end(self) -> None:
        """Saves accumulated episode history to CSV."""
        if self.metrics_save_path and self.episode_records:
            self.metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(self.episode_records)
            df.to_csv(self.metrics_save_path, index=False)
            logger.info(
                f"Saved {len(df)} episode training metrics to {self.metrics_save_path}"
            )


class CheckpointCallback(BaseCallback):
    """Callback for periodically saving model checkpoints during training."""

    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            ckpt_file = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            logger.info(f"Saving training checkpoint to {ckpt_file}")
            self.model.save(str(ckpt_file))
        return True
