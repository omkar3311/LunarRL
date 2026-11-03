"""Abstract base class for reward shapers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseRewardShaper(ABC):
    """Abstract interface defining custom reward shaping logic."""

    @abstractmethod
    def reset(self, initial_obs: np.ndarray) -> None:
        """Resets shaper internal state at episode start.

        Args:
            initial_obs: Initial 8D observation from env.reset().
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        obs: np.ndarray,
        action: int,
        native_reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Computes shaped reward and individual component contributions.

        Args:
            obs: Current 8D observation vector.
            action: Action executed.
            native_reward: Raw reward emitted by LunarLander-v3.
            terminated: Whether episode ended due to goal/crash.
            truncated: Whether episode reached time limit.
            info: Environment info dictionary.

        Returns:
            Tuple of:
                - shaped_reward: Total reward returned to the RL agent.
                - components: Dictionary of named reward components for diagnostics.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of this reward shaper."""
        pass
