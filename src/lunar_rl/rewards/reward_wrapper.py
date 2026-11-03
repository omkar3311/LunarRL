"""Gymnasium wrapper integrating reward shapers into LunarLander environments."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np

from src.lunar_rl.rewards.base_reward import BaseRewardShaper
from src.lunar_rl.rewards.baseline_reward import BaselineRewardShaper
from src.lunar_rl.rewards.reward_hacking import RewardHackingShaper
from src.lunar_rl.rewards.shaped_reward import CorrectedRewardShaper
from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


class RewardWrapper(gym.Wrapper):
    """Gymnasium Wrapper that applies a BaseRewardShaper to environment rewards.

    Transparently intercepts environment steps, applies the specified reward shaping,
    and enriches the step info dictionary with original and decomposed reward components.
    """

    def __init__(self, env: gym.Env, shaper: BaseRewardShaper) -> None:
        super().__init__(env)
        self.shaper = shaper
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_custom_reward = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.shaper.reset(obs)
        self.episode_original_reward = 0.0
        self.episode_shaped_reward = 0.0
        self.episode_custom_reward = 0.0
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, native_reward, terminated, truncated, info = self.env.step(action)

        shaped_reward, components = self.shaper.compute_reward(
            obs=obs,
            action=action,
            native_reward=float(native_reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        custom_bonus = shaped_reward - native_reward

        self.episode_original_reward += native_reward
        self.episode_shaped_reward += shaped_reward
        self.episode_custom_reward += custom_bonus

        # Enrich info dictionary
        info["original_reward"] = float(native_reward)
        info["shaped_reward"] = float(shaped_reward)
        info["custom_reward"] = float(custom_bonus)
        info["reward_components"] = components
        info["episode_original_reward_accum"] = self.episode_original_reward
        info["episode_shaped_reward_accum"] = self.episode_shaped_reward
        info["episode_custom_reward_accum"] = self.episode_custom_reward

        return obs, float(shaped_reward), terminated, truncated, info


def get_reward_shaper(reward_config: Dict[str, Any]) -> BaseRewardShaper:
    """Factory creating the appropriate reward shaper from configuration.

    Args:
        reward_config: Reward configuration sub-dictionary.

    Returns:
        Instance of BaseRewardShaper subclass.
    """
    reward_type = reward_config.get("type", "baseline")

    if reward_type == "baseline":
        return BaselineRewardShaper()
    elif reward_type == "reward_hacking":
        return RewardHackingShaper(
            upright_bonus=reward_config.get("upright_bonus", 0.5),
            velocity_bonus=reward_config.get("velocity_bonus", 0.5),
            angle_threshold=reward_config.get("angle_threshold", 0.20),
            velocity_threshold=reward_config.get("velocity_threshold", 0.25),
        )
    elif reward_type == "corrected_reward":
        return CorrectedRewardShaper(
            time_penalty=reward_config.get("time_penalty", 0.04),
            progress_scale=reward_config.get("progress_scale", 1.5),
            velocity_alignment_weight=reward_config.get("velocity_alignment_weight", 0.5),
            safe_landing_bonus=reward_config.get("safe_landing_bonus", 50.0),
            crash_penalty=reward_config.get("crash_penalty", 50.0),
        )
    else:
        raise ValueError(f"Unknown reward type: '{reward_type}'")


def wrap_environment(env: gym.Env, reward_config: Dict[str, Any]) -> gym.Env:
    """Wraps environment with RewardWrapper according to config.

    If reward_type is 'baseline', wrapping is optional but can still be applied
    for uniform info logging.
    """
    shaper = get_reward_shaper(reward_config)
    return RewardWrapper(env, shaper)
