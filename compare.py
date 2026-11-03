#!/usr/bin/env python3
"""Standalone script to run comparison analysis and generate plots and reports."""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lunar_rl.evaluation.comparison import (
    generate_comparison_report,
    generate_comparison_table,
)
from src.lunar_rl.visualization.plots import generate_all_plots
from src.lunar_rl.utils.logger import get_logger

logger = get_logger("compare")


def main() -> None:
    logger.info("Generating experiment comparison table...")
    df = generate_comparison_table()

    if df.empty:
        logger.warning(
            "No evaluation data found yet. Run 'python main.py evaluate --experiment all' first."
        )
    else:
        logger.info("\n" + df.to_string(index=False))

    logger.info("Generating analytical comparison report...")
    report_path = Path("results/metrics/comparison_report.md")
    generate_comparison_report(df, output_path=report_path)

    logger.info("Generating publication-quality comparison charts...")
    plots = generate_all_plots()
    logger.info(f"Generated {len(plots)} charts in 'results/plots/':")
    for p in plots:
        logger.info(f"  • {p.name}")


if __name__ == "__main__":
    main()
