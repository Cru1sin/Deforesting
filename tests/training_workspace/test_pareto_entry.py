from pathlib import Path

import pandas as pd
import pytest


def test_fold_exclusions_and_changed_run_settings(tmp_path: Path) -> None:
    from train_pareto_boundary import fold_exclusions, save_settings

    folds = fold_exclusions(["d", "b", "a", "c"])
    assert set(folds) == {"a", "b", "c", "d"}
    for test, inner in folds.items():
        assert test != inner
        assert inner in folds
    path = tmp_path / "settings.json"
    save_settings(path, {"seed": 0})
    save_settings(path, {"seed": 0})
    with pytest.raises(ValueError, match="new output"):
        save_settings(path, {"seed": 1})


def test_native_frame_cohort_keeps_only_observed_preparation_prefixes() -> None:
    from train_pareto_boundary import native_frame_cycles

    catalog = pd.DataFrame(
        {
            "cycle_name": ["short", "normal"],
            "heating_start": ["2026-08-07 10:02:59"] * 2,
            "defrost_preparation_start": ["2026-08-07 10:03:08", "2026-08-07 11:03:08"],
        }
    )
    images = pd.DataFrame(
        {
            "cycle_name": ["short", "normal"],
            "camera_role": ["front"] * 2,
            "image_time": ["2026-08-07 10:04:00"] * 2,
        }
    )
    assert native_frame_cycles(catalog, images) == {"normal"}
