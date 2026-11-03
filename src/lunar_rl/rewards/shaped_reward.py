"""Corrected reward shaper: progress-based potential shaping with time penalty."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.lunar_rl.rewards.base_reward import BaseRewardShaper


class CorrectedRewardShaper(BaseRewardShaper):
    """Corrected reward shaper utilizing progress differentials and time penalties.

    Key Engineering Principles:
        1. Progress over State: Rewards rate of approach (Δdistance) toward pad (0, 0),
           rather than statically rewarding altitude or speed per frame.
        2. Anti-Hovering Time Penalty: Deducts a small penalty per step to eliminate
           stalling incentives.
        3. Touchdown Softness: Damps descent velocity only when close to the ground.
        4. Goal Alignment: Reinforces native landing bonus (+50) and penalizes crashes (-50).
    """

    def __init__(
        self,
        time_penalty: float = 0.04,
        progress_scale: float = 1.5,
        velocity_alignment_weight: float = 0.5,
        safe_landing_bonus: float = 50.0,
        crash_penalty: float = 50.0,
    ) -> None:
        """Initializes the corrected reward shaper.

        Args:
            time_penalty: Penalty deducted each step to discourage stalling.
            progress_scale: Multiplier for pad approach progress (prev_dist - curr_dist).
            velocity_alignment_weight: Weight for soft landing dampening near ground.
            safe_landing_bonus: Additional incentive upon confirmed safe landing.
            crash_penalty: Additional penalty upon crash.
        """
        self.time_penalty = time_penalty
        self.progress_scale = progress_scale
        self.velocity_alignment_weight = velocity_alignment_weight
        self.safe_landing_bonus = safe_landing_bonus
        self.crash_penalty = crash_penalty

        self._prev_distance: Optional[float] = None

    @property
    def name(self) -> str:
        return "corrected_reward"

    def reset(self, initial_obs: np.ndarray) -> None:
        """Resets distance tracking using initial lander position."""
        x0, y0 = float(initial_obs[0]), float(initial_obs[1])
        self._prev_distance = float(np.sqrt(x0 * x0 + y0 * y0))

    def compute_reward(
        self,
        obs: np.ndarray,
        action: int,
        native_reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Computes well-aligned shaped reward."""
        x = float(obs[0])
        y = float(obs[1])
        vy = float(obs[3])
        curr_distance = float(np.sqrt(x * x + y * y))

        # 1. Progress toward landing pad
        if self._prev_distance is not None:
            progress = (self._prev_distance - curr_distance) * self.progress_scale
        else:
            progress = 0.0
        self._prev_distance = curr_distance

        # 2. Time penalty
        step_time_penalty = -abs(self.time_penalty)

        # 3. Soft touchdown damping when near ground (y < 0.25)
        touchdown_damping = 0.0
        if y < 0.25 and vy < -0.3:
            # Penalize dangerous descent speed near ground
            touchdown_damping = -self.velocity_alignment_weight * abs(vy)

        # 4. Terminal outcomes
        terminal_bonus = 0.0
        if terminated:
            if native_reward >= 100.0:
                terminal_bonus = self.safe_landing_bonus
            elif native_reward <= -100.0:
                terminal_bonus = -self.crash_penalty

        custom_total = progress + step_time_penalty + touchdown_damping + terminal_bonus
        shaped_reward = native_reward + custom_total

        components = {
            "native_reward": float(native_reward),
            "progress_bonus": float(progress),
            "time_penalty": float(step_time_penalty),
            "touchdown_damping": float(touchdown_damping),
            "terminal_bonus": float(terminal_bonus),
            "custom_bonus": float(custom_total),
        }
        return float(shaped_reward), components
