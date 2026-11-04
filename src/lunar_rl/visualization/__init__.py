"""Visualization and recording module for Lunar Lander RL Lab."""

from src.lunar_rl.visualization.plots import (
    generate_all_plots,
    plot_success_rate_comparison,
    plot_episode_length_comparison,
    plot_tri_experiment_comparison,
    plot_native_vs_shaped_gap,
    plot_reward_component_breakdown,
)
from src.lunar_rl.visualization.training_curves import (
    load_training_metrics,
    plot_training_reward_curve,
    plot_native_vs_shaped_training,
)
from src.lunar_rl.visualization.video_recorder import VideoRecorder

__all__ = [
    "generate_all_plots",
    "plot_success_rate_comparison",
    "plot_episode_length_comparison",
    "plot_tri_experiment_comparison",
    "plot_native_vs_shaped_gap",
    "plot_reward_component_breakdown",
    "load_training_metrics",
    "plot_training_reward_curve",
    "plot_native_vs_shaped_training",
    "VideoRecorder",
]
