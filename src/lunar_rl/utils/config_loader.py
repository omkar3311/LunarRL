"""Configuration loader and validator for Lunar Lander RL Lab."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively updates a nested dictionary.

    Args:
        base: Base dictionary.
        override: Dictionary with overriding keys.

    Returns:
        Updated merged dictionary.
    """
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = _deep_update(base[key].copy(), value)
        else:
            base[key] = value
    return base


def validate_config(config: Dict[str, Any]) -> None:
    """Validates configuration parameters.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        ValueError: If required fields are missing or values are invalid.
    """
    required_sections = ["environment", "agent", "training", "evaluation", "reward"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: '{section}'")

    # Validate environment
    env_cfg = config["environment"]
    if "id" not in env_cfg or env_cfg["id"] != "LunarLander-v3":
        raise ValueError(f"Environment ID must be 'LunarLander-v3', got: {env_cfg.get('id')}")

    # Validate agent
    agent_cfg = config["agent"]
    if agent_cfg.get("algorithm") != "PPO":
        raise ValueError(f"Unsupported algorithm: {agent_cfg.get('algorithm')}. Only 'PPO' is supported.")
    if agent_cfg.get("learning_rate", 0) <= 0:
        raise ValueError("Agent learning_rate must be positive.")
    if agent_cfg.get("n_steps", 0) <= 0:
        raise ValueError("Agent n_steps must be positive.")
    if agent_cfg.get("batch_size", 0) <= 0:
        raise ValueError("Agent batch_size must be positive.")

    # Validate reward
    reward_cfg = config["reward"]
    valid_reward_types = {"baseline", "reward_hacking", "corrected_reward"}
    if reward_cfg.get("type") not in valid_reward_types:
        raise ValueError(
            f"Invalid reward type '{reward_cfg.get('type')}'. Must be one of {valid_reward_types}."
        )

    # Validate training
    training_cfg = config["training"]
    if training_cfg.get("total_timesteps", 0) <= 0:
        raise ValueError("training total_timesteps must be a positive integer.")


def load_config(
    experiment_name: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Loads base config and optionally merges experiment override.

    Args:
        experiment_name: Name of experiment ('baseline', 'reward_hacking', 'corrected_reward').
        config_file: Direct path to a config file. If provided, overrides experiment_name lookup.
        project_root: Root directory of project. Defaults to current working directory.

    Returns:
        Consolidated and validated configuration dictionary.
    """
    root = Path(project_root) if project_root else Path.cwd()
    base_config_path = root / "config.yaml"

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config file not found at {base_config_path}")

    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_file:
        custom_path = Path(config_file)
        if not custom_path.is_absolute():
            custom_path = root / custom_path
        if custom_path.exists():
            with open(custom_path, "r", encoding="utf-8") as f:
                custom_cfg = yaml.safe_load(f) or {}
                config = _deep_update(config, custom_cfg)
    elif experiment_name:
        exp_path = root / "configs" / f"{experiment_name}.yaml"
        if exp_path.exists():
            with open(exp_path, "r", encoding="utf-8") as f:
                exp_cfg = yaml.safe_load(f) or {}
                config = _deep_update(config, exp_cfg)
        else:
            # If experiment is explicitly named, update experiment_name in config
            config["experiment_name"] = experiment_name
            config["reward"]["type"] = experiment_name

    validate_config(config)
    return config
