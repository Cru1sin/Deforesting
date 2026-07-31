from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.config import Config
from frost_analysis.process import process


def _config(root: Path, *, continuous_gap: float = 60.0) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=root / "data",
        channels_path=root / "channels.yaml",
        camera_mapping_path=root / "camera.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        cycles={},
        process={
            "resample_interval_seconds": 10,
            "continuous_max_gap_seconds": continuous_gap,
            "control_max_gap_seconds": 30,
            "baseline": {
                "stage": "frost_development",
                "search_start_minutes": 0,
                "search_end_minutes": 5,
                "window_minutes": 1,
                "window_step_minutes": 1,
                "minimum_observed_coverage": 0.8,
                "maximum_imputed_fraction": 0.0,
                "required_anchor_channels": ["anchor"],
                "anchor_maximum_std": {"anchor": 1.0},
            },
            "features": {"windows_minutes": [1]},
        },
        analysis={},
    )


def _channels() -> dict[str, dict[str, object]]:
    return {
        "temperature": {
            "kind": "continuous",
            "role": "sensor",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
        },
        "step_signal": {
            "kind": "step",
            "role": "control",
            "resample": "last",
            "missing": "forward_fill",
            "analysis_candidate": False,
        },
        "event_signal": {
            "kind": "event",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
        },
        "categorical_signal": {
            "kind": "categorical",
            "role": "event",
            "resample": "last",
            "missing": "none",
            "analysis_candidate": False,
        },
        "anchor": {
            "kind": "continuous",
            "role": "context",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
        },
        "heating_capacity": {
            "kind": "continuous",
            "role": "performance",
            "resample": "mean",
            "missing": "interpolate",
            "analysis_candidate": False,
        },
        "power_total": {
            "kind": "continuous",
            "role": "performance",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
        },
        "cop": {
            "kind": "derived",
            "role": "performance",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
            "formula": "cop",
            "dependencies": ["heating_capacity", "power_total"],
        },
    }


def _summary(
    *,
    status: str = "valid",
    heating: str | None = None,
    stable: str = "2026-07-15 00:00:00",
    defrost: str = "2026-07-15 00:05:00",
) -> pd.DataFrame:
    heating_start = pd.Timestamp(heating or stable)
    return pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": [status],
            "cycle_status_reason": [""],
            "heating_start": [heating_start],
            "stable_heating_start": [pd.Timestamp(stable)],
            "defrost_start": [pd.Timestamp(defrost)],
            "defrost_end": [pd.Timestamp(defrost) + pd.Timedelta(minutes=1)],
        }
    )


def _frame(
    timestamps: pd.DatetimeIndex,
    *,
    temperature: list[float],
    stage: str = "frost_development",
    include_images: bool = False,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "timestamp": timestamps,
            "cycle_id": "cycle_001",
            "cycle_stage": stage,
            "cycle_status": "valid",
            "cycle_status_reason": "",
            "cycle_elapsed_seconds": np.nan,
            "cycle_progress": np.nan,
            "temperature": temperature,
            "step_signal": [1.0] * len(timestamps),
            "event_signal": pd.Series([True] * len(timestamps), dtype="boolean"),
            "categorical_signal": ["heat"] * len(timestamps),
            "anchor": [20.0] * len(timestamps),
            "heating_capacity": [10.0] * len(timestamps),
            "power_total": [2.0] * len(timestamps),
        }
    )
    if include_images:
        frame["image_front_center_path"] = pd.NA
        frame["image_front_center_time"] = pd.NaT
        frame["image_front_center_offset_seconds"] = np.nan
    return frame


def test_process_excludes_partial_and_recomputes_coordinates(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=4, freq="10s")
    frame = _frame(timestamps, temperature=[10.0, 9.0, 8.0, 7.0])
    frame.loc[3, "cycle_stage"] = "partial"
    summary = _summary(defrost="2026-07-15 00:01:00")

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert not processed["cycle_stage"].eq("partial").any()
    assert processed["cycle_progress"].dropna().tolist() == [0.0, 1 / 6, 1 / 3]
    assert final_summary.loc[0, "baseline_status"] == "available"


def test_stage_boundary_bucket_uses_overlap_winner_and_drops_ties(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:02",
            "2026-07-15 00:00:03",
            "2026-07-15 00:00:09",
            "2026-07-15 00:00:11",
        ]
    )
    frame = _frame(timestamps, temperature=[100.0, 100.0, 1.0, 9.0, 2.0])
    frame.loc[:1, "cycle_stage"] = "recovery"
    summary = _summary(
        heating="2026-07-15 00:00:00",
        stable="2026-07-15 00:00:03",
        defrost="2026-07-15 00:01:00",
    )

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert not processed.duplicated(["experiment_id", "timestamp"]).any()
    transition = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:00"))
    assert processed.loc[transition, "cycle_stage"].eq("frost_development").all()
    assert processed.loc[transition, "temperature"].iloc[0] == 5.0

    tie_timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:04",
            "2026-07-15 00:00:05",
            "2026-07-15 00:00:09",
        ]
    )
    tie_frame = _frame(tie_timestamps, temperature=[100.0, 100.0, 1.0, 9.0])
    tie_frame.loc[:1, "cycle_stage"] = "recovery"
    tie_summary = _summary(
        heating="2026-07-15 00:00:00",
        stable="2026-07-15 00:00:05",
        defrost="2026-07-15 00:01:00",
    )

    tie_processed, tie_final_summary = process(
        tie_frame, tie_summary, _config(tmp_path), _channels()
    )

    assert tie_processed.empty
    assert tie_final_summary.loc[0, "excluded_transition_bucket_count"] == 1


def test_cycle_boundary_bucket_is_excluded_instead_of_duplicated(tmp_path: Path) -> None:
    first_timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:29",
            "2026-07-15 00:00:31",
        ]
    )
    first = _frame(first_timestamps, temperature=[1.0, 2.0, 3.0])
    first["cycle_id"] = "cycle_001"
    first.loc[:1, "cycle_stage"] = "frost_development"
    first.loc[2, "cycle_stage"] = "defrost"

    second_timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:32",
            "2026-07-15 00:00:34",
            "2026-07-15 00:00:41",
        ]
    )
    second = _frame(second_timestamps, temperature=[10.0, 11.0, 12.0])
    second["cycle_id"] = "cycle_002"
    second.loc[:1, "cycle_stage"] = "recovery"

    first_summary = _summary(
        heating="2026-07-15 00:00:00",
        stable="2026-07-15 00:00:03",
        defrost="2026-07-15 00:00:30",
    )
    first_summary["cycle_id"] = "cycle_001"
    first_summary["defrost_end"] = pd.Timestamp("2026-07-15 00:00:32")
    second_summary = _summary(
        heating="2026-07-15 00:00:32",
        stable="2026-07-15 00:00:35",
        defrost="2026-07-15 00:01:00",
    )
    second_summary["cycle_id"] = "cycle_002"
    summary = pd.concat([first_summary, second_summary], ignore_index=True)

    processed, final_summary = process(
        pd.concat([first, second], ignore_index=True),
        summary,
        _config(tmp_path),
        _channels(),
    )

    assert not processed.duplicated(["experiment_id", "timestamp"]).any()
    assert not processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:30")).any()
    assert final_summary["excluded_transition_bucket_count"].sum() >= 1


def test_long_continuous_gap_is_kept_entirely_nan(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(["2026-07-15 00:00:00", "2026-07-15 00:01:10"])
    frame = _frame(timestamps, temperature=[1.0, 3.0])
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    gap = processed["timestamp"].between(
        pd.Timestamp("2026-07-15 00:00:10"), pd.Timestamp("2026-07-15 00:01:00")
    )
    assert processed.loc[gap, "temperature"].isna().all()
    assert not processed.loc[gap, "temperature__imputed"].any()


def test_gap_threshold_below_grid_interval_does_not_impute(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(["2026-07-15 00:00:00", "2026-07-15 00:00:20"])
    frame = _frame(timestamps, temperature=[1.0, 3.0])
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(
        frame, summary, _config(tmp_path, continuous_gap=5.0), _channels()
    )

    middle = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:10"))
    assert processed.loc[middle, "temperature"].isna().all()


def test_step_fill_is_bounded_and_other_kinds_are_not_filled(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(["2026-07-15 00:00:00", "2026-07-15 00:00:40"])
    frame = _frame(timestamps, temperature=[1.0, 3.0])
    frame.loc[1, "step_signal"] = np.nan
    frame.loc[1, "event_signal"] = pd.NA
    frame.loc[1, "categorical_signal"] = pd.NA
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    late = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:40"))
    assert processed.loc[late, "step_signal"].isna().all()
    assert processed.loc[late, "event_signal"].isna().all()
    assert processed.loc[late, "categorical_signal"].isna().all()


def test_derived_imputation_is_boolean_or_of_dependencies(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0])
    frame.loc[1, "heating_capacity"] = np.nan
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed["cop"].notna().iloc[1]
    assert processed["cop__imputed"].dtype == bool
    assert bool(processed.loc[1, "cop__imputed"])


def test_invalid_cycle_has_no_baseline_and_does_not_change_status(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0])
    summary = _summary(status="invalid", defrost="2026-07-15 00:05:00")

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert final_summary.loc[0, "cycle_status"] == "invalid"
    assert final_summary.loc[0, "baseline_status"] == "not_applicable"
    assert processed["temperature__baseline"].isna().all()
    assert processed["temperature__baseline_residual"].isna().all()


def test_baseline_uses_one_common_non_imputed_anchor_window(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=37, freq="10s")
    frame = _frame(timestamps, temperature=list(np.linspace(10, 8, len(timestamps))))
    frame.loc[2, "anchor"] = np.nan
    frame["anchor__imputed"] = False
    frame["temperature__imputed"] = False
    frame["heating_capacity__imputed"] = False
    frame["power_total__imputed"] = False
    summary = _summary(defrost="2026-07-15 00:06:00")

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert final_summary.loc[0, "baseline_status"] == "available"
    assert final_summary.loc[0, "baseline_reference_type"] == "cycle_local_early_stable_proxy"
    assert pd.Timestamp(final_summary.loc[0, "baseline_start"]) == timestamps[0]
    assert pd.Timestamp(final_summary.loc[0, "baseline_end"]) == timestamps[0] + pd.Timedelta(
        minutes=1
    )
    assert processed["temperature__baseline"].notna().any()


def test_duplicate_source_value_is_not_observed_by_resampling(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=2, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 99.0])
    frame["temperature__duplicate"] = [False, True]
    frame["temperature__conflict"] = [False, True]
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    duplicate_bucket = processed["timestamp"].eq(timestamps[1])
    assert processed.loc[duplicate_bucket, "temperature"].isna().all()


def test_images_are_bucketed_without_forward_fill_and_offsets_are_recomputed(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0], include_images=True)
    frame.loc[0, "image_front_center_path"] = "front-0.jpg"
    frame.loc[0, "image_front_center_time"] = timestamps[0] + pd.Timedelta(seconds=1)
    frame.loc[2, "image_front_center_path"] = "front-2.jpg"
    frame.loc[2, "image_front_center_time"] = timestamps[2] - pd.Timedelta(seconds=2)
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed.loc[0, "image_front_center_path"] == "front-0.jpg"
    assert pd.isna(processed.loc[1, "image_front_center_path"])
    assert processed.loc[2, "image_front_center_path"] == "front-2.jpg"
    assert processed.loc[0, "image_front_center_offset_seconds"] == 1.0
    assert processed.loc[2, "image_front_center_offset_seconds"] == -2.0


def test_dynamic_features_require_a_full_past_window(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=8, freq="10s")
    frame = _frame(timestamps, temperature=list(range(8)))
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed["temperature__rolling_mean_1min"].iloc[5] != processed[
        "temperature__rolling_mean_1min"
    ].iloc[5]
    assert processed.loc[6, "temperature__rolling_mean_1min"] == 2.5
    assert processed.loc[6, "temperature__lag_1min"] == 0.0
    assert processed.loc[6, "temperature__delta_1min"] == 6.0
    assert not any("__slope" in column for column in processed.columns)
