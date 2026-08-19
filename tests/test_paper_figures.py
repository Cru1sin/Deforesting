from __future__ import annotations

import pandas as pd

from frost_analysis.paper_figures import regret_threshold_summary


def test_regret_threshold_summary_joins_width_and_absolute_coverage() -> None:
    bands = pd.DataFrame(
        {
            "relative_regret_threshold": [0.01, 0.01, 0.02, 0.02],
            "band_width_minutes": [10.0, 30.0, 20.0, 40.0],
        }
    )
    balance = pd.DataFrame(
        {
            "regret_threshold": [0.01] * 3 + [0.02] * 3,
            "camera_group": ["all"] * 6,
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"] * 2,
            "image_count": [30, 40, 30, 20, 60, 20],
        }
    )

    summary = regret_threshold_summary(bands, balance)

    assert summary["median_width_minutes"].tolist() == [20.0, 30.0]
    assert summary["eligible_image_coverage"].tolist() == [0.6, 0.4]
