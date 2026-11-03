"""Random Agent baseline runner and observation demonstrator."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from gymnasium import Env

from src.lunar_rl.environment.factory import make_lunar_environment
from src.lunar_rl.environment.observation import LunarObservation, ACTION_DESCRIPTIONS
from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


class RandomAgent:
    """Agent that samples actions uniformly at random to establish baseline performance."""

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """Selects random action from action space."""
        return int(self.action_space.sample())


def run_random_agent(
    episodes: int = 5,
    render_mode: Optional[str] = None,
    seed: int = 42,
    log_observations: bool = True,
    env: Optional[Env] = None,
) -> Dict[str, Any]:
    """Runs a random policy on LunarLander-v3 to benchmark untrained behavior.

    Args:
        episodes: Number of episodes to run.
        render_mode: Render mode ('human', 'rgb_array', or None).
        seed: Random seed.
        log_observations: Whether to print detailed observation breakdowns.
        env: Optional existing environment to use.

    Returns:
        Dictionary containing run statistics.
    """
    close_env = False
    if env is None:
        env = make_lunar_environment(render_mode=render_mode, seed=seed)
        close_env = True

    agent = RandomAgent(env.action_space)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    crashes = 0
    successes = 0

    print("=" * 70)
    print(" LUNAR LANDER RL LAB — RANDOM AGENT BASELINE DEMONSTRATION")
    print("=" * 70)
    print("Action space available to agent:")
    for act_idx, act_desc in ACTION_DESCRIPTIONS.items():
        print(f"  [{act_idx}] {act_desc}")
    print("-" * 70)

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=seed + ep if seed is not None else None)
            lunar_obs = LunarObservation.from_array(obs)
            total_reward = 0.0
            steps = 0
            done = False

            if log_observations and ep == 1:
                print(f"\n--- Episode {ep} Initial State Observation ---")
                print(lunar_obs.explain())

            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                # Log periodic state updates during first episode
                if log_observations and ep == 1 and (steps % 50 == 0 or done):
                    curr_obs = LunarObservation.from_array(obs)
                    print(f"\n[Step {steps:3d}] Action taken: [{action}] {ACTION_DESCRIPTIONS[action]}")
                    print(f"Step Reward: {reward:+.3f} | Cumulative: {total_reward:+.3f}")
                    print(curr_obs.explain())

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            outcome = "Timeout"
            if terminated:
                if reward <= -100:
                    outcome = "Crash (Terminal Penalty)"
                    crashes += 1
                elif reward >= 100:
                    outcome = "Safe Landing!"
                    successes += 1
                else:
                    outcome = "Terminated"

            print(
                f"Episode {ep:02d}/{episodes:02d} | Total Reward: {total_reward:8.2f} "
                f"| Steps: {steps:4d} | Outcome: {outcome}"
            )

    finally:
        if close_env:
            env.close()

    summary = {
        "episodes": episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "median_reward": float(np.median(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(successes / episodes),
        "crash_rate": float(crashes / episodes),
    }

    print("-" * 70)
    print("Random Agent Summary:")
    print(f"  Mean Reward:   {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Mean Steps:    {summary['mean_length']:.1f}")
    print(f"  Success Rate:  {summary['success_rate'] * 100:.1f}%")
    print(f"  Crash Rate:    {summary['crash_rate'] * 100:.1f}%")
    print("=" * 70)

    return summary
