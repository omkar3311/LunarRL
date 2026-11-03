"""Reward engineering module for Lunar Lander RL Lab."""

from src.lunar_rl.rewards.base_reward import BaseRewardShaper
from src.lunar_rl.rewards.baseline_reward import BaselineRewardShaper
from src.lunar_rl.rewards.reward_hacking import RewardHackingShaper
from src.lunar_rl.rewards.shaped_reward import CorrectedRewardShaper
from src.lunar_rl.rewards.reward_wrapper import (
    RewardWrapper,
    get_reward_shaper,
    wrap_environment,
)

__all__ = [
    "BaseRewardShaper",
    "BaselineRewardShaper",
    "RewardHackingShaper",
    "CorrectedRewardShaper",
    "RewardWrapper",
    "get_reward_shaper",
    "wrap_environment",
]
