"""Unit tests for evaluation metrics and evaluator engine."""

import pytest
from src.lunar_rl.environment.random_agent import RandomAgent
from src.lunar_rl.evaluation.evaluator import Evaluator
from src.lunar_rl.evaluation.metrics import (
    calculate_landing_stability,
    compute_evaluation_summary,
)


def test_landing_stability_calculation():
    # Zero velocity and upright tilt should yield zero score (most stable)
    score_perfect = calculate_landing_stability([0.0], [0.0], [0.0])
    assert score_perfect == 0.0

    # Tilted and moving fast yields higher score (less stable)
    score_turbulent = calculate_landing_stability([0.5], [1.2], [0.8])
    assert score_turbulent > score_perfect


def test_compute_evaluation_summary():
    dummy_records = [
        {
            "total_reward": 150.0,
            "native_reward": 150.0,
            "custom_reward": 0.0,
            "length": 200,
            "is_success": True,
            "is_crash": False,
            "is_timeout": False,
            "final_angle": 0.02,
            "final_linear_velocity": 0.1,
            "final_angular_velocity": 0.05,
        },
        {
            "total_reward": -100.0,
            "native_reward": -100.0,
            "custom_reward": 0.0,
            "length": 80,
            "is_success": False,
            "is_crash": True,
            "is_timeout": False,
            "final_angle": 0.45,
            "final_linear_velocity": 1.5,
            "final_angular_velocity": 0.8,
        },
    ]

    metrics = compute_evaluation_summary("test_exp", dummy_records)

    assert metrics.num_episodes == 2
    assert metrics.mean_reward == pytest.approx(25.0)
    assert metrics.success_rate == 0.5
    assert metrics.crash_rate == 0.5
    assert metrics.timeout_rate == 0.0
    assert metrics.mean_episode_length == 140.0
    assert metrics.landing_stability_score > 0.0


def test_evaluator_evaluate_policy(tmp_path):
    evaluator = Evaluator(results_dir=tmp_path)

    # Use a dummy agent that always outputs action 0
    class DummyAgent:
        def predict(self, obs, deterministic=True):
            return 0, None

    metrics, df = evaluator.evaluate_policy(
        agent=DummyAgent(),
        experiment_name="test_dummy",
        num_episodes=2,
        seed=42,
    )

    assert metrics.num_episodes == 2
    assert len(df) == 2
    assert "total_reward" in df.columns
    assert "is_success" in df.columns
