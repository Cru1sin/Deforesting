from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.config import Config
from frost_analysis.process import process


def _config(
    root: Path,
    *,
    continuous_gap: float = 60.0,
    interval: int = 10,
    expected_interval: int = 1,
) -> Config:
    return Config(
        project_root=root,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=root / "data",
        channels_path=root / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="时间",
        expected_sensor_interval_seconds=expected_interval,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
        cycles={},
        process={
            "resample_interval_seconds": interval,
            "continuous_max_gap_seconds": continuous_gap,
            "control_max_gap_seconds": 30,
            "baseline": {
                "stage": "frost_development",
                "search_start_minutes": 0,
                "search_end_minutes": 5,
                "window_minutes": 1,
                "window_step_minutes": 1,
                "minimum_observed_coverage": 0.8,
                "required_anchor_channels": ["anchor"],
                "anchor_maximum_std": {"anchor": 1.0},
            },
            "features": {"windows_minutes": [1]},
        },
        analysis={},
        camera_roles={},
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
        "compressor_power": {
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
        "evaporator_capacity": {
            "kind": "derived",
            "role": "performance",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
            "formula": "evaporator_capacity",
            "dependencies": ["heating_capacity", "compressor_power"],
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
    assert processed["cycle_progress"].dropna().tolist() == [index / 6 for index in range(6)]
    assert final_summary.loc[0, "baseline_status"] == "unavailable"
    assert final_summary.loc[0, "baseline_failure_reason"] == "insufficient_observed_coverage"


def test_process_aggregates_environment_channels_and_keeps_rh_over_100(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=6, freq="2s")
    frame = _frame(timestamps, temperature=[10.0] * len(timestamps))
    frame["environment_temperature"] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    frame["environment_relative_humidity"] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    channels = _channels()
    channels.update(
        {
            "environment_temperature": {
                "kind": "continuous",
                "role": "context",
                "resample": "mean",
                "missing": "interpolate",
                "analysis_candidate": False,
            },
            "environment_relative_humidity": {
                "kind": "continuous",
                "role": "context",
                "resample": "mean",
                "missing": "interpolate",
                "analysis_candidate": False,
            },
        }
    )

    processed, _ = process(
        frame,
        _summary(defrost="2026-07-15 00:01:00"),
        _config(tmp_path),
        channels,
    )

    first_bucket = processed.loc[processed["timestamp"].eq(pd.Timestamp("2026-07-15"))].iloc[0]
    assert first_bucket["environment_temperature"] == pytest.approx(12.0)
    assert first_bucket["environment_relative_humidity"] == pytest.approx(103.0)


def test_incomplete_cycle_with_observed_defrost_onset_keeps_observed_processed_grid(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=30, freq="10s")
    frame = _frame(timestamps, temperature=[10.0] * len(timestamps))
    frame["cycle_status"] = "incomplete"
    frame["cycle_status_reason"] = "defrost_end_not_observed"
    frame.loc[:4, "cycle_stage"] = "recovery"
    frame.loc[5:19, "cycle_stage"] = "frost_development"
    frame.loc[20:, "cycle_stage"] = "defrost"
    summary = _summary(status="incomplete", defrost="2026-07-15 00:03:20")
    summary["cycle_status_reason"] = "defrost_end_not_observed"
    summary["defrost_end"] = pd.NaT

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert not processed.empty
    assert processed["temperature"].notna().any()
    assert processed["cop"].notna().all()
    assert processed["cop"].tolist() == [5.0] * len(processed)


def test_partial_cycle_uses_observed_fallback(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0], stage="partial")
    frame["cycle_id"] = "partial_001"
    frame["cycle_status"] = "incomplete"
    summary = _summary(status="incomplete")
    summary["cycle_id"] = "partial_001"
    summary[["heating_start", "stable_heating_start", "defrost_start", "defrost_end"]] = pd.NaT

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert processed["timestamp"].tolist() == timestamps.tolist()
    assert processed["cycle_stage"].eq("partial").all()
    assert processed["cycle_status"].eq("incomplete").all()
    assert processed["temperature"].tolist() == [1.0, 2.0, 3.0]
    assert final_summary.loc[0, "processed_row_count"] == 3


def test_explicit_partial_status_uses_observed_fallback(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=2, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0], stage="partial")
    frame["cycle_status"] = "partial"
    frame["cycle_id"] = "partial_001"
    summary = _summary(status="partial")
    summary["cycle_id"] = "partial_001"
    summary[["heating_start", "stable_heating_start", "defrost_start", "defrost_end"]] = pd.NaT

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed["cycle_status"].eq("partial").all()
    assert processed["cycle_stage"].eq("partial").all()
    assert processed["temperature"].tolist() == [1.0, 2.0]


def test_fallback_cycles_anchor_buckets_to_each_cycle_start(tmp_path: Path) -> None:
    first_timestamps = pd.to_datetime(
        ["2026-07-15 00:00:11", "2026-07-15 00:00:15"]
    )
    first = _frame(first_timestamps, temperature=[1.0, 2.0], stage="partial")
    first["cycle_id"] = "partial_001"
    first["cycle_status"] = "incomplete"

    second_timestamps = pd.to_datetime(
        ["2026-07-15 00:00:16", "2026-07-15 00:00:19"]
    )
    second = _frame(second_timestamps, temperature=[3.0, 4.0], stage="partial")
    second["cycle_id"] = "partial_002"
    second["cycle_status"] = "incomplete"

    first_summary = _summary(status="incomplete")
    first_summary["cycle_id"] = "partial_001"
    second_summary = _summary(status="incomplete")
    second_summary["cycle_id"] = "partial_002"
    summary = pd.concat([first_summary, second_summary], ignore_index=True)
    summary[["heating_start", "stable_heating_start", "defrost_start", "defrost_end"]] = pd.NaT

    processed, _ = process(
        pd.concat([first, second], ignore_index=True),
        summary,
        _config(tmp_path),
        _channels(),
    )

    assert not processed.duplicated(["experiment_id", "timestamp"]).any()
    assert processed["timestamp"].tolist() == [first_timestamps[0], second_timestamps[0]]
    assert processed["cycle_id"].tolist() == ["partial_001", "partial_002"]


def test_cop_preserves_dependency_gaps_instead_of_using_filled_values(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=6, freq="10s")
    frame = _frame(timestamps, temperature=[10.0] * len(timestamps))
    frame.loc[2, "heating_capacity"] = np.nan
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    by_time = processed.set_index("timestamp")
    assert by_time.loc[timestamps[1], "cop"] == 5.0
    assert pd.isna(by_time.loc[timestamps[2], "cop"])
    assert by_time.loc[timestamps[3], "cop"] == 5.0


def test_evaporator_capacity_subtracts_compressor_power(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=4, freq="10s")
    frame = _frame(
        timestamps,
        temperature=[10.0] * len(timestamps),
        stage="partial",
    )
    frame["cycle_status"] = "incomplete"
    frame["heating_capacity"] = 10.0
    frame["compressor_power"] = [2.428, np.nan, 2.0, 2.0]

    processed, _ = process(
        frame,
        _summary(status="incomplete"),
        _config(tmp_path),
        _channels(),
    )

    assert processed.loc[0, "evaporator_capacity"] == pytest.approx(7.572)
    assert pd.isna(processed.loc[1, "evaporator_capacity"])


def test_stage_boundary_buckets_are_excluded_only_when_boundary_is_inside_bucket(
    tmp_path: Path,
) -> None:
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
    assert not processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:00")).any()
    assert processed.loc[
        processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:10")), "cycle_stage"
    ].iloc[0] == "frost_development"

    edge_frame = _frame(
        pd.date_range("2026-07-15", periods=3, freq="10s"), temperature=[1.0, 2.0, 3.0]
    )
    edge_frame.loc[0, "cycle_stage"] = "frost_development"
    edge_summary = _summary(
        heating="2026-07-15 00:00:00",
        stable="2026-07-15 00:00:00",
        defrost="2026-07-15 00:01:00",
    )
    edge_processed, edge_final_summary = process(
        edge_frame, edge_summary, _config(tmp_path), _channels()
    )

    assert edge_processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:00")).any()
    assert edge_final_summary.loc[0, "excluded_transition_bucket_count"] == 0


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


def test_step_fill_is_bounded_per_timestamp(tmp_path: Path) -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:10",
            "2026-07-15 00:00:40",
            "2026-07-15 00:00:50",
        ]
    )
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0, 4.0])
    frame.loc[1:2, "step_signal"] = np.nan
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed.loc[processed["timestamp"].eq(timestamps[1]), "step_signal"].iloc[0] == 1
    assert processed.loc[processed["timestamp"].eq(timestamps[2]), "step_signal"].isna().all()


def test_continuous_coverage_counts_before_fill_and_excludes_unavailable_channels(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=30, freq="s")
    frame = _frame(timestamps, temperature=[1.0] * 30)
    frame["temperature__missing"] = False
    frame["invalid_sensor"] = np.nan
    frame["invalid_sensor__missing"] = False
    frame["absent_sensor"] = np.nan
    frame["absent_sensor__missing"] = True
    channels = _channels()
    channels.update(
        {
            "invalid_sensor": {
                "kind": "continuous",
                "role": "sensor",
                "resample": "mean",
                "missing": "none",
                "analysis_candidate": False,
            },
            "absent_sensor": {
                "kind": "continuous",
                "role": "sensor",
                "resample": "mean",
                "missing": "none",
                "analysis_candidate": False,
            },
        }
    )
    frame.loc[10:12, "temperature"] = np.nan
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, final_summary = process(frame, summary, _config(tmp_path), channels)

    assert final_summary.loc[0, "eligible_continuous_channel_bucket_count"] == 72
    assert final_summary.loc[0, "low_coverage_channel_bucket_count"] == 70
    low_bucket = processed["timestamp"].eq(pd.Timestamp("2026-07-15 00:00:10"))
    assert processed.loc[low_bucket, "temperature__imputed"].all()
    assert processed.loc[low_bucket, "temperature"].notna().all()


def test_complete_cycle_grid_uses_summary_boundaries_for_unobserved_edges(
    tmp_path: Path,
) -> None:
    timestamps = pd.to_datetime(
        ["2026-07-15 00:00:05", "2026-07-15 00:00:45"]
    )
    frame = _frame(timestamps, temperature=[1.0, 2.0])
    frame["temperature__missing"] = False
    frame.loc[0, "cycle_stage"] = "recovery"
    frame.loc[1, "cycle_stage"] = "frost_development"
    summary = _summary(
        heating="2026-07-15 00:00:00",
        stable="2026-07-15 00:10:00",
        defrost="2026-07-15 00:50:00",
    )
    summary["defrost_end"] = pd.Timestamp("2026-07-15 00:01:00")
    summary["processed_available_fraction"] = 1.0
    summary["imputed_fraction"] = 0.0

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    expected_timestamps = pd.date_range("2026-07-15 00:00:00", periods=6, freq="10s")
    assert processed["timestamp"].tolist() == list(expected_timestamps)
    first_bucket = processed["timestamp"].eq(expected_timestamps[0])
    last_bucket = processed["timestamp"].eq(expected_timestamps[-1])
    assert processed.loc[first_bucket, "temperature"].isna().all()
    assert processed.loc[last_bucket, "temperature"].isna().all()
    assert final_summary.loc[0, "eligible_continuous_channel_bucket_count"] == 6
    assert "processed_available_fraction" not in final_summary
    assert "imputed_fraction" not in final_summary


def test_process_rejects_non_divisible_coverage_grid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divisible"):
        _config(tmp_path, expected_interval=3)


def test_incomplete_cycle_without_boundaries_uses_observed_ten_second_fallback(
    tmp_path: Path,
) -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 00:00:00",
            "2026-07-15 00:00:01",
            "2026-07-15 00:00:10",
            "2026-07-15 00:00:11",
        ]
    )
    frame = _frame(timestamps, temperature=[1.0, 3.0, 5.0, 7.0])
    frame["cycle_status"] = "incomplete"
    for suffix in ("__missing", "__invalid", "__duplicate", "__conflict"):
        frame[f"temperature{suffix}"] = False
    summary = _summary(status="incomplete", defrost="2026-07-15 00:05:00")
    summary["defrost_end"] = pd.NaT

    processed, final_summary = process(
        frame,
        summary,
        _config(tmp_path, expected_interval=10),
        _channels(),
    )

    expected_timestamps = list(pd.date_range(timestamps[0], periods=2, freq="10s"))
    assert processed["timestamp"].tolist() == expected_timestamps
    assert processed["temperature"].tolist() == [2.0, 6.0]
    assert processed["cycle_stage"].eq("frost_development").all()
    assert processed["cycle_status"].eq("incomplete").all()
    assert processed["cycle_progress"].isna().all()
    assert processed["cycle_elapsed_seconds"].isna().all()
    assert processed["cop"].notna().all()
    assert processed["cop"].tolist() == [5.0, 5.0]
    assert processed["cop__imputed"].eq(False).all()
    assert processed["temperature__baseline"].isna().all()
    assert processed["temperature__baseline_residual"].isna().all()
    assert processed["temperature__lag_1min"].isna().all()
    assert processed["temperature__delta_1min"].isna().all()
    assert processed["temperature__rolling_mean_1min"].isna().all()
    imputed_columns = [column for column in processed if column.endswith("__imputed")]
    assert imputed_columns
    assert not processed[imputed_columns].any(axis=None)
    assert not any(
        column.endswith(("__missing", "__invalid", "__duplicate", "__conflict"))
        for column in processed
    )
    assert final_summary.loc[0, "processed_row_count"] == 2


def test_valid_cycle_without_boundaries_uses_observed_fallback(tmp_path: Path) -> None:
    frame = _frame(pd.date_range("2026-07-15", periods=2, freq="10s"), temperature=[1.0, 2.0])
    summary = _summary()
    summary["defrost_end"] = pd.NaT

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert processed["timestamp"].tolist() == frame["timestamp"].tolist()
    assert processed["temperature"].tolist() == frame["temperature"].tolist()
    assert processed["cop"].tolist() == [5.0, 5.0]
    assert processed["cycle_progress"].isna().all()
    assert final_summary.loc[0, "processed_row_count"] == 2


def test_nonpartial_cycle_without_summary_is_a_process_contract_error(tmp_path: Path) -> None:
    frame = _frame(pd.date_range("2026-07-15", periods=2, freq="10s"), temperature=[1.0, 2.0])

    with pytest.raises(ValueError, match="missing cycle summary"):
        process(frame, _summary().iloc[0:0], _config(tmp_path), _channels())


def test_summary_cycle_without_prepared_rows_is_a_process_contract_error(tmp_path: Path) -> None:
    frame = _frame(pd.date_range("2026-07-15", periods=2, freq="10s"), temperature=[1.0, 2.0])
    summary_only = _summary(status="incomplete")
    summary_only["cycle_id"] = "cycle_002"
    summary = pd.concat([_summary(), summary_only], ignore_index=True)

    with pytest.raises(ValueError, match="without Prepared rows"):
        process(frame, summary, _config(tmp_path), _channels())


def test_cop_does_not_mark_an_unfilled_dependency_as_imputed(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0])
    frame.loc[1, "heating_capacity"] = np.nan
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed["cop"].isna().iloc[1]
    assert processed["cop__imputed"].dtype == bool
    assert not processed["cop__imputed"].any()


def test_invalid_cycle_has_no_baseline_and_does_not_change_status(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0])
    frame["cycle_status"] = "invalid"
    summary = _summary(status="invalid", defrost="2026-07-15 00:05:00")

    processed, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert final_summary.loc[0, "cycle_status"] == "invalid"
    assert final_summary.loc[0, "baseline_status"] == "not_applicable"
    assert processed["temperature"].tolist() == [1.0, 2.0, 3.0]
    assert processed["cop"].notna().any()
    assert processed["cycle_progress"].isna().all()
    assert processed["temperature__baseline"].isna().all()
    assert processed["temperature__baseline_residual"].isna().all()


def test_invalid_cycle_fallback_recomputes_cop_from_aggregated_dependencies(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=3, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0, 3.0])
    frame["cycle_status"] = "invalid"
    summary = _summary(status="invalid", defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed["heating_capacity"].notna().all()
    assert processed["power_total"].notna().all()
    assert processed["cop"].notna().all()
    assert processed["cop"].tolist() == [5.0, 5.0, 5.0]


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


def test_baseline_does_not_shift_past_fixed_recovery_window(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-07-15", periods=37, freq="10s")
    frame = _frame(timestamps, temperature=list(np.linspace(10, 8, len(timestamps))))
    frame.loc[:1, "anchor"] = np.nan
    frame["anchor__imputed"] = False
    frame["temperature__imputed"] = False
    frame["heating_capacity__imputed"] = False
    frame["power_total__imputed"] = False
    summary = _summary(defrost="2026-07-15 00:06:00")

    _, final_summary = process(frame, summary, _config(tmp_path), _channels())

    assert final_summary.loc[0, "baseline_status"] == "unavailable"
    assert final_summary.loc[0, "baseline_failure_reason"] == "insufficient_observed_coverage"
    assert pd.isna(final_summary.loc[0, "baseline_start"])
    assert pd.isna(final_summary.loc[0, "baseline_end"])


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
    assert processed.loc[1, "image_front_center_path"] == "front-2.jpg"
    assert pd.isna(processed.loc[2, "image_front_center_path"])
    assert processed.loc[0, "image_front_center_offset_seconds"] == 1.0
    assert processed.loc[1, "image_front_center_offset_seconds"] == 8.0


def test_image_bucket_membership_uses_image_time_not_sensor_attachment_time(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-07-15", periods=2, freq="10s")
    frame = _frame(timestamps, temperature=[1.0, 2.0], include_images=True)
    frame.loc[1, "image_front_center_path"] = "front-near-boundary.jpg"
    frame.loc[1, "image_front_center_time"] = timestamps[0] + pd.Timedelta(seconds=9.9)
    summary = _summary(defrost="2026-07-15 00:05:00")

    processed, _ = process(frame, summary, _config(tmp_path), _channels())

    assert processed.loc[0, "image_front_center_path"] == "front-near-boundary.jpg"
    assert pd.isna(processed.loc[1, "image_front_center_path"])
    assert processed.loc[0, "image_front_center_offset_seconds"] == 9.9


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
