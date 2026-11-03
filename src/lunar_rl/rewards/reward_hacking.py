"""Intentionally flawed reward function demonstrating reward hacking."""

from typing import Any, Dict, Tuple
import numpy as np
from src.lunar_rl.rewards.base_reward import BaseRewardShaper


class RewardHackingShaper(BaseRewardShaper):
    """Reward shaper with intentional design flaws that incentivize hovering/stalling.

    Flaw explanation:
        Rewards the agent for existing in a desirable state (upright orientation and low
        velocity) on a per-step basis, rather than rewarding progress toward task completion.
        An agent can exploit this by gently firing thrusters to remain airborne and balanced,
        accumulating hundreds of points across 1,000 steps without ever attempting to land.
    """

    def __init__(
        self,
        upright_bonus: float = 0.5,
        velocity_bonus: float = 0.5,
        angle_threshold: float = 0.20,
        velocity_threshold: float = 0.25,
    ) -> None:
        """Initializes the flawed reward shaper.

        Args:
            upright_bonus: Bonus reward given per step when upright.
            velocity_bonus: Bonus reward given per step when moving slowly.
            angle_threshold: Maximum angle magnitude (radians) to qualify as upright.
            velocity_threshold: Maximum speed magnitude to qualify as low velocity.
        """
        self.upright_bonus = upright_bonus
        self.velocity_bonus = velocity_bonus
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold

    @property
    def name(self) -> str:
        return "reward_hacking"

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
        """Calculates native reward plus flawed frame-by-frame state bonuses."""
        # Observation indices:
        # 2: vx, 3: vy, 4: angle
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        speed = float(np.sqrt(vx * vx + vy * vy))

        # 1. Bonus for remaining upright
        is_upright = abs(angle) < self.angle_threshold
        step_upright_bonus = self.upright_bonus if is_upright else 0.0

        # 2. Bonus for maintaining low velocity
        is_low_velocity = speed < self.velocity_threshold
        step_velocity_bonus = self.velocity_bonus if is_low_velocity else 0.0

        custom_total = step_upright_bonus + step_velocity_bonus
        shaped_reward = native_reward + custom_total

        components = {
            "native_reward": float(native_reward),
            "upright_bonus": float(step_upright_bonus),
            "velocity_bonus": float(step_velocity_bonus),
            "custom_bonus": float(custom_total),
        }
        return float(shaped_reward), components
