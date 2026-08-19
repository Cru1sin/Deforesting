from __future__ import annotations

import pandas as pd

from frost_analysis.rgb_evaluation import leave_one_experiment_out_predictions


def test_leave_one_experiment_out_predicts_every_row_once() -> None:
    rows = []
    for experiment, offset in (("a", 0.0), ("b", 0.2), ("c", -0.2)):
        for target, value in ((0, -1.0 + offset), (1, 1.0 + offset)):
            for repeat in range(3):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "cycle_name": f"{experiment}_{repeat}",
                        "camera_role": "top",
                        "target": target,
                        "feature_000": value + repeat * 0.01,
                    }
                )

    predictions = leave_one_experiment_out_predictions(pd.DataFrame(rows))

    assert len(predictions) == len(rows)
    assert predictions.groupby("experiment_id")["held_out_experiment"].nunique().eq(1).all()
    assert predictions["experiment_id"].eq(predictions["held_out_experiment"]).all()
    assert set(predictions["predicted_target"]) == {0, 1}
