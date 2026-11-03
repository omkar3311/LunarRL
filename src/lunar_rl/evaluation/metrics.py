"""Metrics definitions and statistical calculations for Lunar Lander evaluation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for an agent across multiple episodes."""

    experiment_name: str
    num_episodes: int
    mean_reward: float
    median_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    success_rate: float
    crash_rate: float
    timeout_rate: float
    mean_episode_length: float
    std_episode_length: float
    landing_stability_score: float
    mean_native_reward: float
    mean_custom_reward: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics dataclass to dictionary."""
        return asdict(self)


def calculate_landing_stability(
    final_angles: List[float],
    final_linear_velocities: List[float],
    final_angular_velocities: List[float],
) -> float:
    """Calculates landing stability score based on touchdown kinetic energy and tilt.

    Lower scores indicate more calm, stable, upright landings.
    Stability = mean(|angle|) + 0.5 * mean(linear_speed) + 0.2 * mean(|angular_velocity|).
    """
    if not final_angles:
        return 0.0

    mean_angle = float(np.mean(np.abs(final_angles)))
    mean_lin_vel = float(np.mean(final_linear_velocities))
    mean_ang_vel = float(np.mean(np.abs(final_angular_velocities)))

    score = mean_angle + 0.5 * mean_lin_vel + 0.2 * mean_ang_vel
    return float(round(score, 4))


def compute_evaluation_summary(
    experiment_name: str,
    episode_records: List[Dict[str, Any]],
) -> EvaluationMetrics:
    """Computes aggregate statistical metrics from a list of episode records.

    Criteria:
        - Success: Terminated with positive terminal landing bonus (reward >= 100).
        - Crash: Terminated with terminal crash penalty (reward <= -100).
        - Timeout: Truncated without successful landing or crash.

    Args:
        experiment_name: Name of the evaluated experiment.
        episode_records: List of dictionaries, each containing single-episode stats:
            - total_reward (float)
            - native_reward (float)
            - custom_reward (float)
            - length (int)
            - is_success (bool)
            - is_crash (bool)
            - is_timeout (bool)
            - final_angle (float)
            - final_linear_velocity (float)
            - final_angular_velocity (float)

    Returns:
        Populated EvaluationMetrics dataclass.
    """
    n_episodes = len(episode_records)
    if n_episodes == 0:
        raise ValueError("Cannot calculate metrics on an empty list of episode records.")

    rewards = [rec["total_reward"] for rec in episode_records]
    native_rewards = [rec.get("native_reward", rec["total_reward"]) for rec in episode_records]
    custom_rewards = [rec.get("custom_reward", 0.0) for rec in episode_records]
    lengths = [rec["length"] for rec in episode_records]
    successes = sum(1 for rec in episode_records if rec.get("is_success", False))
    crashes = sum(1 for rec in episode_records if rec.get("is_crash", False))
    timeouts = sum(1 for rec in episode_records if rec.get("is_timeout", False))

    final_angles = [rec.get("final_angle", 0.0) for rec in episode_records]
    final_lin_vels = [rec.get("final_linear_velocity", 0.0) for rec in episode_records]
    final_ang_vels = [rec.get("final_angular_velocity", 0.0) for rec in episode_records]

    stability = calculate_landing_stability(final_angles, final_lin_vels, final_ang_vels)

    return EvaluationMetrics(
        experiment_name=experiment_name,
        num_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        median_reward=float(np.median(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        success_rate=float(successes / n_episodes),
        crash_rate=float(crashes / n_episodes),
        timeout_rate=float(timeouts / n_episodes),
        mean_episode_length=float(np.mean(lengths)),
        std_episode_length=float(np.std(lengths)),
        landing_stability_score=stability,
        mean_native_reward=float(np.mean(native_rewards)),
        mean_custom_reward=float(np.mean(custom_rewards)),
    )
