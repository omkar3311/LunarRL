"""Baseline reward shaper: leaves native LunarLander-v3 reward unchanged."""

from typing import Any, Dict, Tuple
import numpy as np
from src.lunar_rl.rewards.base_reward import BaseRewardShaper


class BaselineRewardShaper(BaseRewardShaper):
    """Pass-through shaper that preserves LunarLander-v3 native reward."""

    @property
    def name(self) -> str:
        return "baseline"

    def reset(self, initial_obs: np.ndarray) -> None:
        pass

    def compute_reward(
        self,
        obs: np.ndarray,
        action: int,
        native_reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Returns unmodified native reward."""
        components = {
            "native_reward": float(native_reward),
            "custom_bonus": 0.0,
        }
        return float(native_reward), components
