"""Unit tests for LunarLander environment creation and observation parsing."""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.environment.observation import (
    ACTION_DESCRIPTIONS,
    OBSERVATION_DESCRIPTIONS,
    LunarObservation,
)


def test_environment_initialization():
    env = make_lunar_environment(env_id="LunarLander-v3")
    assert env is not None
    assert isinstance(env.observation_space, Box)
    assert env.observation_space.shape == (8,)
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 4
    env.close()


def test_environment_reset_and_step():
    env = make_lunar_environment(env_id="LunarLander-v3", seed=42)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    assert isinstance(info, dict)

    # Execute a step
    action = 0  # Do nothing
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (8,)
    assert isinstance(reward, (float, int, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(next_info, dict)

    env.close()


def test_lunar_observation_parsing():
    raw_obs = np.array([0.1, 0.8, -0.05, -0.2, 0.04, 0.01, 1.0, 0.0], dtype=np.float32)
    parsed = LunarObservation.from_array(raw_obs)

    assert parsed.x == pytest.approx(0.1)
    assert parsed.y == pytest.approx(0.8)
    assert parsed.vx == pytest.approx(-0.05)
    assert parsed.vy == pytest.approx(-0.2)
    assert parsed.angle == pytest.approx(0.04)
    assert parsed.angular_velocity == pytest.approx(0.01)
    assert parsed.left_leg_contact is True
    assert parsed.right_leg_contact is False

    obs_dict = parsed.to_dict()
    assert "x_position" in obs_dict
    assert "angle_deg" in obs_dict

    explanation = parsed.explain()
    assert "Horizontal Pos" in explanation
    assert "Left Leg Ground Contact" in explanation


def test_observation_invalid_length():
    with pytest.raises(ValueError):
        LunarObservation.from_array(np.array([1.0, 2.0]))
