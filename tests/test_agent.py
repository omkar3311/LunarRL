"""Unit tests for PPOAgent and ModelManager."""

from pathlib import Path
import numpy as np
import pytest

from src.lunar_rl.agents.model_manager import ModelManager
from src.lunar_rl.agents.ppo_agent import PPOAgent
from src.lunar_rl.environment.factory import make_lunar_environment


def test_ppo_agent_initialization():
    env = make_lunar_environment()
    config = {
        "agent": {
            "policy": "MlpPolicy",
            "learning_rate": 0.0003,
            "n_steps": 128,
            "batch_size": 64,
            "policy_kwargs": {"net_arch": [64, 64]},
        }
    }
    agent = PPOAgent(env=env, config=config)

    assert agent.model is not None
    assert agent.policy_kwargs == {"net_arch": [64, 64]}

    # Test predict
    dummy_obs = np.zeros(8, dtype=np.float32)
    action, _ = agent.predict(dummy_obs, deterministic=True)
    assert action in [0, 1, 2, 3]

    env.close()


def test_ppo_model_save_and_load(tmp_path: Path):
    env = make_lunar_environment()
    config = {
        "agent": {
            "n_steps": 128,
            "batch_size": 64,
            "policy_kwargs": {"net_arch": [64, 64]},
        }
    }
    agent = PPOAgent(env=env, config=config)

    # Train for a minimal 128 steps
    agent.train(total_timesteps=128)

    manager = ModelManager(base_models_dir=tmp_path)
    save_path = manager.save_model(
        model=agent.model,
        experiment_name="test_exp",
        metadata={"test_key": "test_val"},
    )
    assert save_path.exists()
    assert save_path.with_suffix(".json").exists()

    # Load model back
    loaded_model = manager.load_model("test_exp")
    assert loaded_model is not None

    meta = manager.load_metadata("test_exp")
    assert meta.get("test_key") == "test_val"

    env.close()
