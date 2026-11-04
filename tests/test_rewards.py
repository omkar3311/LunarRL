"""Unit tests for reward shapers and calculations."""

import numpy as np
import pytest

from src.lunar_rl.rewards.baseline_reward import BaselineRewardShaper
from src.lunar_rl.rewards.reward_hacking import RewardHackingShaper
from src.lunar_rl.rewards.shaped_reward import CorrectedRewardShaper


def test_baseline_reward_shaper():
    shaper = BaselineRewardShaper()
    assert shaper.name == "baseline"

    obs = np.zeros(8, dtype=np.float32)
    shaped_r, comps = shaper.compute_reward(
        obs=obs,
        action=0,
        native_reward=15.5,
        terminated=False,
        truncated=False,
        info={},
    )

    assert shaped_r == 15.5
    assert comps["native_reward"] == 15.5
    assert comps["custom_bonus"] == 0.0


def test_reward_hacking_shaper_bonuses():
    shaper = RewardHackingShaper(
        upright_bonus=0.5,
        velocity_bonus=0.5,
        angle_threshold=0.2,
        velocity_threshold=0.3,
    )
    assert shaper.name == "reward_hacking"

    # Case 1: Both conditions satisfied (upright and slow)
    obs_ideal = np.array([0.0, 1.0, 0.05, 0.05, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
    r1, c1 = shaper.compute_reward(
        obs=obs_ideal,
        action=0,
        native_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert c1["upright_bonus"] == 0.5
    assert c1["velocity_bonus"] == 0.5
    assert r1 == pytest.approx(1.0)

    # Case 2: Tilted (angle > threshold) and fast (velocity > threshold)
    obs_bad = np.array([0.0, 1.0, 0.5, 0.5, 0.45, 0.0, 0.0, 0.0], dtype=np.float32)
    r2, c2 = shaper.compute_reward(
        obs=obs_bad,
        action=0,
        native_reward=-2.0,
        terminated=False,
        truncated=False,
        info={},
    )
    assert c2["upright_bonus"] == 0.0
    assert c2["velocity_bonus"] == 0.0
    assert r2 == pytest.approx(-2.0)


def test_corrected_reward_shaper_progress():
    shaper = CorrectedRewardShaper(
        time_penalty=0.04,
        progress_scale=2.0,
        safe_landing_bonus=50.0,
        crash_penalty=50.0,
    )
    assert shaper.name == "corrected_reward"

    # Reset with initial distance: (x=0, y=1.0) -> distance = 1.0
    obs_init = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    shaper.reset(obs_init)

    # Step closer: (x=0, y=0.8) -> distance = 0.8 -> delta_d = 0.2
    obs_closer = np.array([0.0, 0.8, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r_progress, c_progress = shaper.compute_reward(
        obs=obs_closer,
        action=0,
        native_reward=0.0,
        terminated=False,
        truncated=False,
        info={},
    )

    # Expected: progress = 0.2 * 2.0 = +0.40, time_penalty = -0.04
    assert c_progress["progress_bonus"] == pytest.approx(0.40)
    assert c_progress["time_penalty"] == pytest.approx(-0.04)
    assert r_progress == pytest.approx(0.40 - 0.04)

    # Terminal safe landing check (+100 native -> +50 terminal bonus)
    obs_landed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    r_land, c_land = shaper.compute_reward(
        obs=obs_landed,
        action=0,
        native_reward=100.0,
        terminated=True,
        truncated=False,
        info={},
    )
    assert c_land["terminal_bonus"] == pytest.approx(50.0)
    assert r_land > 100.0
