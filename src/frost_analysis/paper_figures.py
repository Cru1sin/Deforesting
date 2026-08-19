"""Small source-table transforms for the cost-to-RGB paper figures."""

from __future__ import annotations

import pandas as pd

from .rgb_evaluation import high_confidence_coverage


def regret_threshold_summary(
    bands: pd.DataFrame, label_balance: pd.DataFrame
) -> pd.DataFrame:
    """Summarize timing ambiguity and retained image coverage by regret threshold."""
    summary = (
        bands.groupby("relative_regret_threshold", as_index=False)["band_width_minutes"]
        .median()
        .rename(
            columns={
                "relative_regret_threshold": "regret_threshold",
                "band_width_minutes": "median_width_minutes",
            }
        )
    )
    summary["eligible_image_coverage"] = [
        high_confidence_coverage(label_balance, "all", threshold)
        for threshold in summary["regret_threshold"]
    ]
    return summary
