"""Unit tests for RewardWrapper integrating with Gymnasium."""

import gymnasium as gym
import numpy as np
import pytest

from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.rewards.reward_wrapper import RewardWrapper, get_reward_shaper, wrap_environment


def test_reward_wrapper_enrichment():
    env = make_lunar_environment(env_id="LunarLander-v3", seed=42)
    reward_cfg = {
        "type": "reward_hacking",
        "upright_bonus": 0.5,
        "velocity_bonus": 0.5,
    }
    wrapped_env = wrap_environment(env, reward_cfg)

    obs, info = wrapped_env.reset(seed=42)
    obs, reward, terminated, truncated, step_info = wrapped_env.step(0)

    assert "original_reward" in step_info
    assert "shaped_reward" in step_info
    assert "custom_reward" in step_info
    assert "reward_components" in step_info
    assert "episode_original_reward_accum" in step_info

    # Difference between shaped and original should match custom_reward
    assert step_info["shaped_reward"] - step_info["original_reward"] == pytest.approx(
        step_info["custom_reward"]
    )

    wrapped_env.close()


def test_get_reward_shaper_factory():
    shaper_base = get_reward_shaper({"type": "baseline"})
    assert shaper_base.name == "baseline"

    shaper_hack = get_reward_shaper({"type": "reward_hacking", "upright_bonus": 1.0})
    assert shaper_hack.name == "reward_hacking"

    shaper_corr = get_reward_shaper({"type": "corrected_reward", "time_penalty": 0.05})
    assert shaper_corr.name == "corrected_reward"

    with pytest.raises(ValueError):
        get_reward_shaper({"type": "invalid_reward_type"})
