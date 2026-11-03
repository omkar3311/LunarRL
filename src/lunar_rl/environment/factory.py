"""Environment factory for LunarLander-v3."""

from typing import Any, Callable, Dict, Optional
import gymnasium as gym
from gymnasium import Env

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


def make_lunar_environment(
    env_id: str = "LunarLander-v3",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    wrapper_fn: Optional[Callable[[Env], Env]] = None,
    **env_kwargs: Any,
) -> Env:
    """Creates and configures a Gymnasium LunarLander environment.

    Args:
        env_id: Gymnasium environment identifier (default: 'LunarLander-v3').
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        seed: Random seed for environment reproducibility.
        wrapper_fn: Optional callable to wrap the created environment.
        **env_kwargs: Additional arguments passed to gym.make (e.g. continuous, gravity).

    Returns:
        Configured Gymnasium environment.
    """
    logger.info(
        f"Creating environment '{env_id}' (render_mode={render_mode}, seed={seed})"
    )

    env = gym.make(
        env_id,
        render_mode=render_mode,
        **env_kwargs,
    )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    if wrapper_fn is not None:
        logger.info(f"Applying custom wrapper: {wrapper_fn}")
        env = wrapper_fn(env)

    return env


def make_vec_env_fn(
    env_id: str = "LunarLander-v3",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    wrapper_fn: Optional[Callable[[Env], Env]] = None,
    **env_kwargs: Any,
) -> Callable[[], Env]:
    """Helper to return an environment instantiation callable for vectorized environments."""
    def _thunk() -> Env:
        return make_lunar_environment(
            env_id=env_id,
            render_mode=render_mode,
            seed=seed,
            wrapper_fn=wrapper_fn,
            **env_kwargs,
        )
    return _thunk
