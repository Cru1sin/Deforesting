from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from frost_analysis import cli
from frost_analysis.report import (
    _AXIS_LABELS,
    _COP_INSET_BBOX,
    _PUBLICATION_AXIS_LABELS,
    _PUBLICATION_PANEL_CHANNELS,
    _PUBLICATION_RIBBON_ALPHA,
    _PUBLICATION_X_GRID_ALPHA,
    _STAGE_BACKGROUND_ALPHA,
    _add_cycle_title,
    _add_freezing_reference,
    _add_qa_summary_line,
    _add_startup_annotation,
    _cop_inset_required,
    _cycle_image_counts,
    _finish_cycle_axes,
    _observed_segments,
    _plot_defrost_state_strip,
    _plot_prepared_line,
    _plot_publication_stage_ribbon,
    _plot_stage_and_defrost,
    _prepared_observed_series,
    _processed_observed_series,
    _publication_stage_labels,
    generate_report,
)

_SOURCE_CHANNELS = [
    "ambient_temperature",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "heating_capacity",
    "evaporating_pressure",
    "evaporating_temperature",
    "coil_temperature",
]


def _write_run(run_dir: Path, *, manifest: str | None = None) -> None:
    run_dir.mkdir(parents=True)
    timestamps = pd.to_datetime(
        ["2026-07-15 08:00:00", "2026-07-15 08:00:10", "2026-07-15 08:00:20"]
    )
    prepared = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "timestamp": timestamps,
            "cycle_id": "cycle_001",
            "cycle_stage": ["recovery", "frost_development", "defrost"],
            "cycle_status": "valid",
            "cycle_status_reason": "",
            "cycle_progress": [np.nan, 0.5, np.nan],
            "cycle_elapsed_seconds": [np.nan, 10.0, np.nan],
            "defrost_active": [False, False, True],
            "image_top_center_path": ["top/a.jpg", np.nan, np.nan],
            "image_top_center_time": [timestamps[0], pd.NaT, pd.NaT],
            "image_top_center_offset_seconds": [0.0, np.nan, np.nan],
        }
    )
    for index, channel in enumerate(_SOURCE_CHANNELS, start=1):
        prepared[channel] = float(index)
        prepared[f"{channel}__missing"] = False
        prepared[f"{channel}__invalid"] = False
        prepared[f"{channel}__duplicate"] = False
        prepared[f"{channel}__conflict"] = False
    prepared["defrost_active__missing"] = False
    prepared["defrost_active__invalid"] = False
    prepared["defrost_active__duplicate"] = False
    prepared["defrost_active__conflict"] = False
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)

    processed = prepared.drop(
        columns=[
            column
            for column in prepared.columns
            if str(column).endswith(("__missing", "__invalid", "__duplicate", "__conflict"))
        ]
    ).copy()
    processed["ambient_temperature__imputed"] = False
    processed["water_in_temperature__imputed"] = False
    processed["water_out_temperature__imputed"] = False
    processed["compressor_frequency__imputed"] = False
    processed["heating_capacity__imputed"] = False
    processed["evaporating_pressure__imputed"] = False
    processed["evaporating_temperature__imputed"] = False
    processed["coil_temperature__imputed"] = False
    processed["cop"] = [2.0, 2.1, np.nan]
    processed["cop__imputed"] = False
    processed["coil_temperature__baseline"] = 1.0
    processed["coil_temperature__baseline_residual"] = [0.0, 0.1, 0.2]
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)

    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "heating_start": [timestamps[0]],
            "stable_heating_start": [timestamps[0] + pd.Timedelta(seconds=5)],
            "defrost_start": [timestamps[2]],
            "defrost_end": [timestamps[2] + pd.Timedelta(seconds=10)],
            "baseline_status": ["available"],
            "baseline_failure_reason": [""],
            "baseline_start": [timestamps[0]],
            "baseline_end": [timestamps[1]],
        }
    )
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)

    evidence = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "channel": ["coil_temperature"],
            "trend_cycle_count": [1],
            "reset_pair_count": [0],
            "future_cycle_count": [0],
            "context_cycle_count": [0],
            "trend_effect": [0.5],
            "direction_consistency": [1.0],
            "reset_effect": [np.nan],
            "reset_evidence_status": ["not_evaluated"],
            "reset_evidence_reason": ["independent_reference_unavailable"],
            "future_performance_association": [np.nan],
            "median_max_abs_context_spearman": [np.nan],
            "decision": ["insufficient_coverage"],
            "reason": ["trend_cycles_below_minimum"],
        }
    )
    evidence.to_csv(run_dir / "candidate_channel_evidence.csv", index=False)
    if manifest is not None:
        (run_dir / "manifest.json").write_text(manifest, encoding="utf-8")


def test_generate_report_writes_four_figure_contract_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "qa"
    _write_run(
        run_dir,
        manifest=json.dumps(
            {
                "experiment_id": "exp_test",
                "git_commit": "abc",
                "config_provenance": {
                    "schema_version": 2,
                    "defaults_path": "configs/defaults.yaml",
                    "resolved_config_sha256": "resolved",
                },
                "resolved_config": {"experiment_id": "exp_test"},
            }
        ),
    )

    result = generate_report(run_dir, output_dir)

    assert result == output_dir
    assert (output_dir / "cycles" / "cycle_001_overview.png").is_file()
    assert (output_dir / "publication" / "cycle_001_publication.png").is_file()
    assert (output_dir / "coverage.png").is_file()
    assert (output_dir / "baseline.png").is_file()
    assert (output_dir / "candidate.png").is_file()
    summary = json.loads((output_dir / "report_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["manifest_present"] is True
    assert summary["input_files"]["prepared_data.parquet"]["sha256"]
    assert summary["provenance"]["config_provenance"]["schema_version"] == 2
    assert "publication/cycle_001_publication.png" in summary["figures"]

    with Image.open(output_dir / "publication" / "cycle_001_publication.png") as image:
        assert image.info["dpi"][0] >= 299
        assert image.info["dpi"][1] >= 299


def test_report_masks_prepared_quality_flags_and_processed_imputation() -> None:
    prepared = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0],
            "signal__missing": [False, False, False],
            "signal__invalid": [False, True, False],
            "signal__duplicate": [False, False, False],
            "signal__conflict": [False, False, False],
        }
    )
    processed = pd.DataFrame(
        {"signal": [1.0, 2.0, 3.0], "signal__imputed": [False, True, False]}
    )

    prepared_observed = _prepared_observed_series(prepared, "signal")
    processed_observed = _processed_observed_series(processed, "signal")
    assert prepared_observed.iloc[0] == 1.0
    assert pd.isna(prepared_observed.iloc[1])
    assert prepared_observed.iloc[2] == 3.0
    assert processed_observed.iloc[0] == 1.0
    assert pd.isna(processed_observed.iloc[1])
    assert processed_observed.iloc[2] == 3.0


def test_cycle_image_counts_only_include_the_requested_cycle() -> None:
    prepared = pd.DataFrame(
        {
            "cycle_id": ["cycle_001", "cycle_002", "cycle_002"],
            "image_top_center_path": ["top/001.jpg", "top/002.jpg", "top/002.jpg"],
        }
    )

    assert _cycle_image_counts(prepared, "cycle_001") == {"top_center": 1}
    assert _cycle_image_counts(prepared, "cycle_002") == {"top_center": 1}


def test_report_warns_when_a_visual_channel_has_no_observed_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    prepared = pd.read_parquet(run_dir / "prepared_data.parquet")
    prepared["ambient_temperature"] = np.nan
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert any(
        warning["code"] == "empty_visual_channel"
        and warning["field"] == "ambient_temperature"
        for warning in report_summary["warnings"]
    )


def test_report_warns_when_valid_cycle_has_no_processed_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    processed = pd.read_parquet(run_dir / "processed_data.parquet").iloc[0:0]
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert any(
        warning["code"] == "cycle_without_processed_rows"
        and warning["cycle_id"] == "cycle_001"
        for warning in report_summary["warnings"]
    )


def test_report_skips_partial_and_summary_only_cycle_overviews(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    summary = pd.read_csv(run_dir / "cycle_summary.csv")
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                {
                    "experiment_id": ["exp_test"],
                    "experiment_date": ["2026-07-15"],
                    "cycle_id": ["cycle_002"],
                    "cycle_status": ["incomplete"],
                    "cycle_status_reason": ["defrost_state_gap"],
                    "heating_start": [pd.NaT],
                    "stable_heating_start": [pd.NaT],
                    "defrost_start": [pd.NaT],
                    "defrost_end": [pd.NaT],
                    "baseline_status": ["not_applicable"],
                    "baseline_failure_reason": ["cycle_not_valid"],
                    "baseline_start": [pd.NaT],
                    "baseline_end": [pd.NaT],
                }
            ),
            pd.DataFrame(
                {
                    "experiment_id": ["exp_test"],
                    "experiment_date": ["2026-07-15"],
                    "cycle_id": ["partial_001"],
                    "cycle_status": ["incomplete"],
                    "cycle_status_reason": ["outside_complete_cycle"],
                    "heating_start": [pd.NaT],
                    "stable_heating_start": [pd.NaT],
                    "defrost_start": [pd.NaT],
                    "defrost_end": [pd.NaT],
                    "baseline_status": ["not_applicable"],
                    "baseline_failure_reason": ["cycle_not_valid"],
                    "baseline_start": [pd.NaT],
                    "baseline_end": [pd.NaT],
                }
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert not any(
        warning.get("cycle_id") == "cycle_002"
        and warning["code"] == "empty_visual_channel"
        for warning in report_summary["warnings"]
    )
    assert any(
        warning.get("cycle_id") == "cycle_002"
        and warning["code"] == "skipped_cycle_without_prepared_rows"
        for warning in report_summary["warnings"]
    )
    assert not any(
        warning.get("cycle_id") == "partial_001"
        for warning in report_summary["warnings"]
    )
    assert "cycles/cycle_002_overview.png" not in report_summary["figures"]
    assert "publication/cycle_002_publication.png" not in report_summary["figures"]
    assert not (tmp_path / "qa" / "cycles" / "cycle_002_overview.png").exists()
    assert not (tmp_path / "qa" / "publication" / "cycle_002_publication.png").exists()
    assert not (tmp_path / "qa" / "publication" / "partial_001_publication.png").exists()


@pytest.mark.parametrize("status", ["incomplete", "invalid"])
def test_report_draws_formal_cycle_with_data_even_when_not_valid(
    tmp_path: Path, status: str
) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    summary = pd.read_csv(run_dir / "cycle_summary.csv")
    summary.loc[0, "cycle_status"] = status
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert "cycles/cycle_001_overview.png" in report_summary["figures"]
    assert "publication/cycle_001_publication.png" in report_summary["figures"]


def test_observed_segments_break_on_nan() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 08:00:00",
            "2026-07-15 08:00:10",
            "2026-07-15 08:00:20",
            "2026-07-15 08:00:30",
        ]
    )
    segments = _observed_segments(timestamps, pd.Series([1.0, 2.0, np.nan, 4.0]))

    assert len(segments) == 2
    assert segments[0][1].tolist() == [1.0, 2.0]
    assert segments[1][1].tolist() == [4.0]


def test_observed_segments_break_on_timestamp_gap_and_keep_values_aligned() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 08:00:10",
            "2026-07-15 08:00:00",
            "2026-07-15 08:00:01",
            "2026-07-15 08:00:02",
            "2026-07-15 08:00:03",
            "2026-07-15 08:00:11",
        ]
    )
    segments = _observed_segments(timestamps, pd.Series([10.0, 0.0, 1.0, 2.0, 3.0, 11.0]))

    assert len(segments) == 2
    assert segments[0][1].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert segments[1][1].tolist() == [10.0, 11.0]


def test_prepared_segments_keep_the_same_variable_color_after_a_gap() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 08:00:00",
            "2026-07-15 08:00:10",
            "2026-07-15 08:00:20",
            "2026-07-15 08:00:30",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "compressor_frequency": [30.0, 31.0, np.nan, 33.0],
            "compressor_frequency__missing": [False] * 4,
            "compressor_frequency__invalid": [False] * 4,
            "compressor_frequency__duplicate": [False] * 4,
            "compressor_frequency__conflict": [False] * 4,
        }
    )
    figure, axis = plt.subplots()

    _plot_prepared_line(
        axis,
        frame,
        "compressor_frequency",
        "Compressor frequency",
        "cycle_001",
        [],
    )

    assert len(axis.lines) == 2
    assert {line.get_color() for line in axis.lines} == {"#0072B2"}
    assert axis.lines[0].get_label() == "Compressor frequency"
    assert axis.lines[1].get_label() == "_child1"
    plt.close(figure)


def test_defrost_step_plot_breaks_at_unknown_state() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 08:00:00",
            "2026-07-15 08:00:10",
            "2026-07-15 08:00:20",
            "2026-07-15 08:00:30",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["recovery", "defrost", "partial", "defrost"],
            "defrost_active": [False, True, np.nan, True],
            "defrost_active__missing": [False] * 4,
            "defrost_active__invalid": [False] * 4,
            "defrost_active__duplicate": [False] * 4,
            "defrost_active__conflict": [False] * 4,
        }
    )
    figure, axis = plt.subplots()

    _plot_stage_and_defrost(axis, frame, "cycle_001", [])

    assert len(axis.lines) == 2
    plt.close(figure)


def test_defrost_state_strip_uses_neutral_off_and_red_on() -> None:
    timestamps = pd.to_datetime(
        [
            "2026-07-15 08:00:00",
            "2026-07-15 08:00:10",
            "2026-07-15 08:00:20",
            "2026-07-15 08:00:30",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "defrost_active": [False, True, np.nan, True],
            "defrost_active__missing": [False] * 4,
            "defrost_active__invalid": [False] * 4,
            "defrost_active__duplicate": [False] * 4,
            "defrost_active__conflict": [False] * 4,
        }
    )
    figure, axis = plt.subplots()

    _plot_defrost_state_strip(axis, frame, timestamps[0])

    colors = {line.get_color() for line in axis.lines}
    assert "#6B7280" in colors
    assert "#C1121F" in colors
    assert len(axis.lines) == 3
    plt.close(figure)


def test_cop_inset_rule_requires_startup_peak_and_narrow_stable_range() -> None:
    timestamps = pd.date_range("2026-07-15 08:00:00", periods=4, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cop": [14.0, 2.2, 2.4, 2.6],
            "cop__imputed": [False] * 4,
        }
    )
    cycle = pd.Series(
        {
            "heating_start": timestamps[0],
            "stable_heating_start": timestamps[1],
        }
    )

    assert _cop_inset_required(frame, cycle) is True

    cycle["stable_heating_start"] = pd.NaT
    assert _cop_inset_required(frame, cycle) is False


def test_qa_summary_line_uses_compact_minute_units() -> None:
    start = pd.Timestamp("2026-07-15 08:00:00")
    cycle = pd.Series(
        {
            "cycle_id": "cycle_001",
            "cycle_duration_seconds": 4218.0,
            "baseline_start": start + pd.Timedelta(minutes=5),
            "baseline_end": start + pd.Timedelta(minutes=10),
        }
    )

    line = _add_qa_summary_line(
        cycle,
        [(start + pd.Timedelta(minutes=20), start + pd.Timedelta(minutes=26))],
        start,
    )

    assert "Duration: 70.3 min" in line
    assert "Defrost-state gap: 6.0 min" in line
    assert "Baseline: 5–10 min" in line


def test_cycle_axes_label_time_from_heating_start() -> None:
    figure, axis = plt.subplots()

    _finish_cycle_axes([axis], None)

    assert axis.get_xlabel() == "Time from heating start [min]"
    plt.close(figure)


def test_cycle_panel_contract_matches_final_five_panel_layout() -> None:
    from frost_analysis.report import _CYCLE_PANEL_CHANNELS

    assert _CYCLE_PANEL_CHANNELS == (
        ("compressor_frequency", "compressor_frequency_setpoint"),
        ("heating_capacity",),
        ("cop",),
        ("water_in_temperature", "water_out_temperature", "water_temperature_setpoint"),
        ("evaporating_temperature", "coil_temperature", "ambient_temperature"),
    )


def test_publication_panel_contract_uses_formal_labels_and_physical_order() -> None:
    assert _PUBLICATION_PANEL_CHANNELS[-1] == (
        "ambient_temperature",
        "coil_temperature",
        "evaporating_temperature",
    )
    assert _PUBLICATION_AXIS_LABELS == (
        "Compressor frequency [Hz]",
        "Heating capacity [kW]",
        "COP",
        "Water temperature [°C]",
        "Temperature [°C]",
    )
    assert _AXIS_LABELS["cop"] == "COP"


def test_publication_inset_is_small_and_upper_right() -> None:
    left, bottom, width, height = _COP_INSET_BBOX

    assert 0.60 <= left < 0.80
    assert 0.50 <= bottom < 0.75
    assert width <= 0.24
    assert height <= 0.28


def test_publication_phase_layers_are_visible_but_subordinate() -> None:
    assert 0.07 <= _STAGE_BACKGROUND_ALPHA <= 0.12
    assert 0.70 <= _PUBLICATION_RIBBON_ALPHA <= 0.85


def test_publication_stage_labels_only_include_observed_boundaries() -> None:
    start = pd.Timestamp("2026-07-15 08:00:00")
    cycle = pd.Series(
        {
            "heating_start": start,
            "stable_heating_start": start + pd.Timedelta(minutes=3),
            "defrost_start": pd.NaT,
            "defrost_end": pd.NaT,
        }
    )

    assert _publication_stage_labels(cycle) == ("Recovery",)


def test_publication_ribbon_explains_state_gap_texture_even_for_short_gaps() -> None:
    start = pd.Timestamp("2026-07-15 08:00:00")
    cycle = pd.Series(
        {
            "heating_start": start,
            "stable_heating_start": start + pd.Timedelta(minutes=3),
            "defrost_start": start + pd.Timedelta(minutes=20),
            "defrost_end": start + pd.Timedelta(minutes=22),
        }
    )
    figure, axis = plt.subplots()

    _plot_publication_stage_ribbon(
        axis,
        cycle,
        [(start + pd.Timedelta(minutes=8), start + pd.Timedelta(minutes=9))],
        start,
    )

    assert any(
        "Defrost state unavailable" in text.get_text() for text in axis.texts
    )
    plt.close(figure)


def test_publication_title_omits_nan_reason() -> None:
    figure = plt.figure()
    cycle = pd.Series(
        {
            "cycle_id": "cycle_004",
            "cycle_status": "valid",
            "cycle_status_reason": np.nan,
        }
    )

    _add_cycle_title(figure, cycle, publication=True)

    assert all("nan" not in text.get_text().lower() for text in figure.texts)
    assert len(figure.texts) == 1
    plt.close(figure)


def test_publication_title_omits_nan_status() -> None:
    figure = plt.figure()
    cycle = pd.Series(
        {
            "cycle_id": "cycle_004",
            "cycle_status": np.nan,
            "cycle_status_reason": "",
        }
    )

    _add_cycle_title(figure, cycle, publication=True)

    assert all("nan" not in text.get_text().lower() for text in figure.texts)
    plt.close(figure)


def test_freezing_reference_adds_zero_degree_line_and_label() -> None:
    figure, axis = plt.subplots()

    _add_freezing_reference(axis)

    assert any(float(line.get_ydata()[0]) == 0.0 for line in axis.lines)
    reference_text = next(
        text for text in axis.texts if "0°C freezing threshold" in text.get_text()
    )
    assert reference_text.get_position()[1] > 0.70
    assert _PUBLICATION_X_GRID_ALPHA <= 0.15
    plt.close(figure)


def test_publication_startup_annotation_is_in_the_top_margin() -> None:
    figure, axis = plt.subplots()
    start = pd.Timestamp("2026-07-15 08:00:00")
    cycle = pd.Series(
        {
            "heating_start": start,
            "stable_heating_start": start + pd.Timedelta(minutes=3),
        }
    )

    _add_startup_annotation(axis, cycle, start)

    annotation = axis.texts[0]
    assert annotation.get_position()[0] >= 0.10
    assert annotation.get_position()[1] > 1.0
    plt.close(figure)


def test_report_records_baseline_unavailable_even_without_processed_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    processed = pd.read_parquet(run_dir / "processed_data.parquet").iloc[0:0]
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)
    summary = pd.read_csv(run_dir / "cycle_summary.csv")
    summary["baseline_status"] = "unavailable"
    summary["baseline_failure_reason"] = "no_candidate_window"
    summary["baseline_start"] = pd.NaT
    summary["baseline_end"] = pd.NaT
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert any(
        warning["code"] == "baseline_unavailable"
        and warning["cycle_id"] == "cycle_001"
        for warning in report_summary["warnings"]
    )


def test_report_rejects_incomplete_image_role_columns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    prepared = pd.read_parquet(run_dir / "prepared_data.parquet")
    prepared = prepared.drop(columns=["image_top_center_offset_seconds"])
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)

    with pytest.raises(ValueError, match="image role columns"):
        generate_report(run_dir, tmp_path / "qa")


def test_report_rejects_corrupt_manifest_without_publishing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "qa"
    _write_run(run_dir, manifest="not-json")

    with pytest.raises(ValueError, match="manifest.json"):
        generate_report(run_dir, output_dir)

    assert not output_dir.exists()


def test_report_rejects_manifest_for_a_different_experiment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, manifest=json.dumps({"experiment_id": "other_experiment"}))

    with pytest.raises(ValueError, match="experiment identity"):
        generate_report(run_dir, tmp_path / "qa")


def test_report_rejects_null_identity_in_nonempty_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    evidence = pd.read_csv(run_dir / "candidate_channel_evidence.csv")
    evidence.loc[0, "experiment_id"] = pd.NA
    evidence.to_csv(run_dir / "candidate_channel_evidence.csv", index=False)

    with pytest.raises(ValueError, match="experiment identity"):
        generate_report(run_dir, tmp_path / "qa")


def test_report_rejects_missing_boundary_for_valid_cycle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    summary = pd.read_csv(run_dir / "cycle_summary.csv")
    summary.loc[0, "defrost_end"] = pd.NaT
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)

    with pytest.raises(ValueError, match="cycle boundaries"):
        generate_report(run_dir, tmp_path / "qa")


def test_report_rejects_row_with_partial_image_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    prepared = pd.read_parquet(run_dir / "prepared_data.parquet")
    prepared.loc[0, "image_top_center_time"] = pd.NaT
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)

    with pytest.raises(ValueError, match="image role row"):
        generate_report(run_dir, tmp_path / "qa")


def test_report_records_no_image_roles_without_blocking_sensor_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    image_columns = [
        "image_top_center_path",
        "image_top_center_time",
        "image_top_center_offset_seconds",
    ]
    for name in ("prepared_data.parquet", "processed_data.parquet"):
        frame = pd.read_parquet(run_dir / name).drop(columns=image_columns)
        frame.to_parquet(run_dir / name, index=False)

    generate_report(run_dir, tmp_path / "qa")

    report_summary = json.loads(
        (tmp_path / "qa" / "report_summary.json").read_text(encoding="utf-8")
    )
    assert report_summary["status"] == "success_with_warnings"
    assert any(
        warning["code"] == "no_image_roles_exported"
        for warning in report_summary["warnings"]
    )


def test_report_cli_reads_explicit_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    exit_code = cli.main(
        ["report", "--input", str(run_dir), "--output", str(tmp_path / "qa")]
    )

    assert exit_code == 0
    assert str(tmp_path / "qa") in capsys.readouterr().out


def test_report_cli_returns_nonzero_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_report(*args: object, **kwargs: object) -> Path:
        raise ValueError("bad report input")

    monkeypatch.setattr(cli, "generate_report", fail_report)

    exit_code = cli.main(
        ["report", "--input", str(tmp_path / "run"), "--output", str(tmp_path / "qa")]
    )

    assert exit_code == 1
    assert "QA report failed" in capsys.readouterr().err


def test_report_preserves_old_output_when_publish_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "qa"
    _write_run(run_dir)
    generate_report(run_dir, output_dir)
    old_summary = (output_dir / "report_summary.json").read_text(encoding="utf-8")
    original_rename = Path.rename

    def fail_old_output_rename(self: Path, target: Path) -> Path:
        if self == output_dir:
            raise OSError("rename failed")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_old_output_rename)
    with pytest.raises(OSError, match="rename failed"):
        generate_report(run_dir, output_dir, overwrite=True)

    assert (output_dir / "report_summary.json").read_text(encoding="utf-8") == old_summary


def test_run_report_failure_returns_nonzero_after_scientific_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "run_pipeline", lambda *args, **kwargs: run_dir)

    def fail_report(*args: object, **kwargs: object) -> Path:
        raise ValueError("render failed")

    monkeypatch.setattr(cli, "generate_report", fail_report)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output",
            str(run_dir),
            "--report",
        ]
    )

    assert exit_code == 1
    assert "scientific run succeeded, QA report failed" in capsys.readouterr().err
    assert run_dir.is_dir()
