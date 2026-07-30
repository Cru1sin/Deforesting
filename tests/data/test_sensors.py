from __future__ import annotations

from pathlib import Path

import pandas as pd

from frost_analysis.data.sensors import merge_parameter_fragments, preprocess_directory


def _write(path: Path, rows: list[tuple[str, str, str]]) -> None:
    text = "时间\t温度\tDeforst\n" + "".join("\t".join(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def test_fragments_sort_deduplicate_and_record_conflicts(tmp_path: Path) -> None:
    first = tmp_path / "b参数1.xls"
    second = tmp_path / "a参数1.xls"
    _write(first, [("2026-07-15 00:00:02", "2", "OFF"), ("2026-07-15 00:00:01", "1", "OFF")])
    _write(second, [("2026-07-15 00:00:02", "20", "OFF"), ("2026-07-15 00:00:03", "3", "OFF")])

    result = merge_parameter_fragments("1", [first, second], tmp_path)
    assert result.frame["sensor_time"].is_monotonic_increasing
    assert result.frame["sensor_time"].is_unique
    assert (
        result.frame.loc[
            result.frame["sensor_time"].eq(pd.Timestamp("2026-07-15 00:00:02")), "p1__温度"
        ].item()
        == 20.0
    )
    assert len(result.conflicts) == 2
    assert result.conflicts["duplicate_conflict"].all()


def test_nonnumeric_tokens_have_separate_raw_clean_and_invalid_flags(tmp_path: Path) -> None:
    path = tmp_path / "x参数1.xls"
    _write(
        path,
        [
            ("2026-07-15 00:00:00", "1", "OFF"),
            ("2026-07-15 00:00:01", "ERR", "OFF"),
            ("2026-07-15 00:00:02", "3", "OFF"),
        ],
    )
    result = merge_parameter_fragments("1", [path], tmp_path)
    assert result.frame.loc[1, "p1__温度__raw"] == "ERR"
    assert pd.isna(result.frame.loc[1, "p1__温度"])
    assert bool(result.frame.loc[1, "p1__温度__invalid"])


def test_short_interpolation_is_limited_and_guarded_by_state_transition(tmp_path: Path) -> None:
    path = tmp_path / "x参数1.xls"
    _write(
        path,
        [
            ("2026-07-15 00:00:00", "0", "OFF"),
            ("2026-07-15 00:00:01", "", "OFF"),
            ("2026-07-15 00:00:02", "2", "OFF"),
            ("2026-07-15 00:00:10", "", "OFF"),
            ("2026-07-15 00:00:20", "20", "OFF"),
            ("2026-07-15 00:00:21", "", "ON"),
            ("2026-07-15 00:00:22", "22", "ON"),
        ],
    )
    result = merge_parameter_fragments(
        "1", [path], tmp_path, short_gap_max_seconds=3, transition_guard_seconds=1
    )
    assert result.frame.loc[1, "p1__温度"] == 1.0
    assert bool(result.frame.loc[1, "p1__温度__interpolated"])
    assert pd.isna(result.frame.loc[3, "p1__温度"])
    assert pd.isna(result.frame.loc[5, "p1__温度"])


def test_directory_outer_aligns_groups_and_reports_irregular_sampling(tmp_path: Path) -> None:
    _write(
        tmp_path / "a参数1.xls",
        [
            ("2026-07-15 00:00:00", "1", "OFF"),
            ("2026-07-15 00:00:01", "2", "OFF"),
            ("2026-07-15 00:00:03", "3", "OFF"),
        ],
    )
    _write(tmp_path / "b参数2.xls", [("2026-07-15 00:00:02", "3", "OFF")])
    result = preprocess_directory(tmp_path, short_gap_max_seconds=0)
    assert result.frame["sensor_time"].tolist() == list(
        pd.date_range("2026-07-15", periods=4, freq="s")
    )
    assert {"p1__温度", "p2__温度"} <= set(result.frame.columns)
    assert result.sampling_summary["irregular_interval_count"].sum() >= 1
    assert not result.missing_intervals.empty
