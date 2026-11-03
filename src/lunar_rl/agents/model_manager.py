"""Model storage, metadata management, and checkpoint discovery."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from stable_baselines3 import PPO

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manages saving, loading, metadata tracking, and path resolution for RL models."""

    def __init__(self, base_models_dir: Path = Path("models")) -> None:
        self.base_models_dir = Path(base_models_dir)

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Returns the model directory for a specific experiment."""
        exp_dir = self.base_models_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def get_checkpoint_dir(self, experiment_name: str) -> Path:
        """Returns the checkpoints directory for a specific experiment."""
        ckpt_dir = self.get_experiment_dir(experiment_name) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir

    def get_final_model_path(self, experiment_name: str) -> Path:
        """Returns the standard path for the final trained model."""
        return self.get_experiment_dir(experiment_name) / f"{experiment_name}_final.zip"

    def save_model(
        self,
        model: PPO,
        experiment_name: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves a model and its associated metadata.

        Args:
            model: Trained PPO model.
            experiment_name: Experiment identifier.
            filename: Name of the zip file (default: '{experiment_name}_final.zip').
            metadata: Optional dictionary with training metadata to serialize.

        Returns:
            Path to the saved model file.
        """
        exp_dir = self.get_experiment_dir(experiment_name)
        file_name = filename or f"{experiment_name}_final.zip"
        if not file_name.endswith(".zip"):
            file_name = f"{file_name}.zip"

        save_path = exp_dir / file_name
        logger.info(f"Saving model to {save_path}")
        model.save(str(save_path))

        # Save metadata JSON alongside the model
        meta = metadata or {}
        meta.update({
            "experiment_name": experiment_name,
            "saved_at": datetime.now().isoformat(),
            "model_path": str(save_path),
            "algorithm": "PPO",
        })

        meta_path = save_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved experiment metadata to {meta_path}")

        return save_path

    def load_model(
        self,
        experiment_name: str,
        filename: Optional[str] = None,
        env: Optional[Any] = None,
    ) -> PPO:
        """Loads a saved PPO model.

        Args:
            experiment_name: Experiment identifier.
            filename: Specific model filename (if None, looks for final model, then latest checkpoint).
            env: Optional environment to bind to the loaded model.

        Returns:
            Loaded PPO model instance.

        Raises:
            FileNotFoundError: If no model matching the criteria is found.
        """
        exp_dir = self.get_experiment_dir(experiment_name)

        if filename:
            target_path = exp_dir / filename if not Path(filename).is_absolute() else Path(filename)
        else:
            final_path = self.get_final_model_path(experiment_name)
            if final_path.exists():
                target_path = final_path
            else:
                # Fallback to latest checkpoint
                checkpoints = self.list_checkpoints(experiment_name)
                if not checkpoints:
                    raise FileNotFoundError(
                        f"No model or checkpoints found for experiment '{experiment_name}' in {exp_dir}"
                    )
                target_path = checkpoints[-1]

        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path}")

        logger.info(f"Loading PPO model from {target_path}")
        return PPO.load(str(target_path), env=env)

    def list_checkpoints(self, experiment_name: str) -> List[Path]:
        """Lists all checkpoint files sorted by step count."""
        ckpt_dir = self.get_checkpoint_dir(experiment_name)
        ckpts = list(ckpt_dir.glob("*.zip"))
        # Sort by modification time or step number if formatted
        ckpts.sort(key=lambda p: p.stat().st_mtime)
        return ckpts

    def load_metadata(self, experiment_name: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Loads metadata dictionary for an experiment model."""
        exp_dir = self.get_experiment_dir(experiment_name)
        if filename:
            meta_path = (exp_dir / filename).with_suffix(".json")
        else:
            meta_path = self.get_final_model_path(experiment_name).with_suffix(".json")

        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
