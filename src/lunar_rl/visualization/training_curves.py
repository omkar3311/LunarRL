"""Training curves visualization utilities using Matplotlib and Seaborn."""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.lunar_rl.utils.logger import get_logger

logger = get_logger(__name__)

# Apply publication-quality style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})


def load_training_metrics(experiment_name: str) -> Optional[pd.DataFrame]:
    """Loads training metrics CSV for an experiment."""
    csv_path = Path("experiments") / experiment_name / "training_metrics.csv"
    if not csv_path.exists():
        logger.warning(f"Training metrics CSV not found for '{experiment_name}' at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def plot_training_reward_curve(
    metrics_df: pd.DataFrame,
    experiment_name: str,
    save_path: Path,
    window: int = 25,
) -> Path:
    """Plots raw and moving average reward curves across training episodes."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    episodes = metrics_df["episode"].values
    rewards = metrics_df["shaped_reward"].values

    # Raw rewards with transparency
    ax.plot(episodes, rewards, color="#1f77b4", alpha=0.25, label="Episode Shaped Reward")

    # Rolling moving average
    if len(rewards) >= window:
        rolling_mean = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        rolling_std = pd.Series(rewards).rolling(window=window, min_periods=1).std()
        ax.plot(
            episodes,
            rolling_mean,
            color="#1f77b4",
            linewidth=2.2,
            label=f"Rolling Mean (window={window})",
        )
        ax.fill_between(
            episodes,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            color="#1f77b4",
            alpha=0.15,
            label="±1 Std Dev",
        )

    # Reference threshold for solved Lunar Lander (+200)
    ax.axhline(200, color="green", linestyle="--", linewidth=1.2, label="Solved Threshold (+200)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)

    ax.set_title(f"Training Reward Progression — {experiment_name.replace('_', ' ').title()}")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved training reward curve to {save_path}")
    return save_path


def plot_native_vs_shaped_training(
    metrics_df: pd.DataFrame,
    experiment_name: str,
    save_path: Path,
    window: int = 25,
) -> Path:
    """Plots native vs shaped reward trajectories to visualize reward shaping divergence."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    episodes = metrics_df["episode"].values
    native = metrics_df["native_reward"].values
    shaped = metrics_df["shaped_reward"].values

    # Moving averages
    win = min(window, max(2, len(episodes) // 5))
    native_ma = pd.Series(native).rolling(window=win, min_periods=1).mean()
    shaped_ma = pd.Series(shaped).rolling(window=win, min_periods=1).mean()

    ax.plot(episodes, shaped_ma, color="#d62728", linewidth=2.0, label="Agent Shaped Return (MA)")
    ax.plot(episodes, native_ma, color="#2ca02c", linewidth=2.0, label="True Native Return (MA)")

    ax.set_title(
        f"Native vs Shaped Reward Trajectory — {experiment_name.replace('_', ' ').title()}"
    )
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Return (Moving Average)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved native vs shaped comparison to {save_path}")
    return save_path
