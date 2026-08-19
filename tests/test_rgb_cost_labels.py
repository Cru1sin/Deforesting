from __future__ import annotations

import pandas as pd

from frost_analysis.rgb_cost_labels import assign_image_cost_states


def test_image_states_use_pointwise_regret_and_preserve_candidate_domain() -> None:
    curve = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                [
                    "2026-01-01 00:10",
                    "2026-01-01 00:20",
                    "2026-01-01 00:30",
                    "2026-01-01 00:40",
                    "2026-01-01 00:50",
                ]
            ),
            "relative_regret": [0.2, 0.0, 0.2, 0.0, 0.2],
        }
    )
    images = pd.to_datetime(
        [
            "2026-01-01 00:05",
            "2026-01-01 00:15",
            "2026-01-01 00:20",
            "2026-01-01 00:30",
            "2026-01-01 00:40",
            "2026-01-01 00:55",
        ]
    )

    labels = assign_image_cost_states(images, curve, regret_threshold=0.05)

    assert labels["cost_state"].tolist() == [
        "outside_candidate_domain",
        "pre_optimal",
        "near_optimal",
        "post_optimal",
        "near_optimal",
        "outside_candidate_domain",
    ]
