"""Checkpoint discovery and management utilities."""

import re
from pathlib import Path
from typing import List, Optional

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)


def get_checkpoint_step(path: Path) -> int:
    """Extracts step number from checkpoint filename (e.g. checkpoint_50000_steps.zip)."""
    match = re.search(r"(\d+)_steps", path.stem)
    if match:
        return int(match.group(1))
    return int(path.stat().st_mtime)


def list_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """Returns sorted list of checkpoint paths in directory."""
    dir_path = Path(checkpoint_dir)
    if not dir_path.exists():
        return []
    ckpts = list(dir_path.glob("*.zip"))
    ckpts.sort(key=get_checkpoint_step)
    return ckpts


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Finds the most recent checkpoint in directory by step count."""
    ckpts = list_checkpoints(checkpoint_dir)
    return ckpts[-1] if ckpts else None


def clean_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """Removes older checkpoints, preserving only the most recent keep_last_n files."""
    ckpts = list_checkpoints(checkpoint_dir)
    if len(ckpts) > keep_last_n:
        for old_ckpt in ckpts[:-keep_last_n]:
            logger.info(f"Removing old checkpoint: {old_ckpt}")
            try:
                old_ckpt.unlink()
            except OSError as e:
                logger.warning(f"Could not delete {old_ckpt}: {e}")
