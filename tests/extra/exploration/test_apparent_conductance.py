from __future__ import annotations

import pandas as pd

from frost_analysis.exploration.evaporator_ua import (
    image_evaporator_ua,
    summarize_evaporator_ua,
    write_evaporator_ua_outputs,
)


def test_image_evaporator_ua_uses_te_and_keeps_t3_as_a_diagnostic() -> None:
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=13, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["frost_development"] * len(timestamps),
            "defrost_active": [False] * len(timestamps),
            "ambient_temperature": [5.0] * len(timestamps),
            "evaporating_temperature": [-5.0] * len(timestamps),
            "coil_temperature": [0.0] * len(timestamps),
            "evaporator_capacity": [5.0] * len(timestamps),
            "compressor_power": [1.0] * len(timestamps),
            "power_total": [1.2] * len(timestamps),
            "cop": [5.0] * len(timestamps),
        }
    )
    images = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000020"] * 2,
            "camera_role": ["front", "top"],
            "file_name": ["front.jpg", "top.jpg"],
            "frame_index": [1, 1],
            "image_time": [timestamps[10] + pd.Timedelta(seconds=2)] * 2,
            "cycle_stage": ["frost_development"] * 2,
        }
    )
    record = {
        "cycle_name": "frost_cycle_000020",
        "status": "valid",
        "boundaries": {
            "baseline_start": timestamps[0].isoformat(),
            "baseline_end": timestamps[6].isoformat(),
        },
    }

    result = image_evaporator_ua(frame, images, record)

    assert result["quality_status"].eq("available").all()
    assert result["sensor_time"].eq(timestamps[10]).all()
    assert result["ua_evaporator_kw_per_k"].eq(0.5).all()
    assert result["ua_t3_diagnostic_kw_per_k"].eq(1.0).all()
    assert result["ua_baseline_kw_per_k"].eq(0.5).all()
    assert result["ua_over_baseline"].eq(1.0).all()
    assert result["cop_from_ua"].eq(5.0).all()
    assert "k_rel" not in result


def test_summary_counts_unique_times_and_marks_stable_and_missing_cycles() -> None:
    times = pd.date_range("2026-01-01", periods=12, freq="30s")
    images = pd.DataFrame(
        [
            {
                "cycle_name": "frost_cycle_000020",
                "image_time": time,
                "camera_role": role,
                "quality_status": "available",
                "ua_evaporator_kw_per_k": 0.5 - index * 0.0025,
                "ua_t3_diagnostic_kw_per_k": (
                    float("nan") if index == 5 else 1.0 - index * 0.01
                ),
                "ua_baseline_kw_per_k": 0.5,
                "ua_t3_baseline_kw_per_k": 1.0,
            }
            for index, time in enumerate(times)
            for role in ("front", "top")
        ]
    )
    records = [
        {"cycle_name": "frost_cycle_000020", "status": "valid"},
        {"cycle_name": "frost_cycle_000021", "status": "valid"},
    ]

    summary = summarize_evaporator_ua(images, records).set_index("cycle_name")

    stable = summary.loc["frost_cycle_000020"]
    assert stable["image_count"] == 24
    assert stable["unique_image_times"] == 12
    assert stable["available_unique_times"] == 12
    assert stable["max_time_gap_minutes"] == 0.5
    assert stable["large_step_count"] == 0
    assert stable["continuity_status"] == "stable"
    assert stable["preferred_temperature_basis"] == "Te"
    assert pd.notna(stable["t3_p95_abs_step_fraction"])
    assert summary.loc["frost_cycle_000021", "continuity_status"] == "no_images"


def test_writer_emits_reusable_tables_summary_and_plot(tmp_path) -> None:
    images = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000020"] * 2,
            "image_time": pd.date_range("2026-01-01", periods=2, freq="30s"),
            "camera_role": ["front", "front"],
            "quality_status": ["available", "available"],
            "ua_evaporator_kw_per_k": [0.5, 0.495],
            "ua_t3_diagnostic_kw_per_k": [1.0, 0.98],
            "ua_baseline_kw_per_k": [0.5, 0.5],
            "ua_t3_baseline_kw_per_k": [1.0, 1.0],
        }
    )
    summary = summarize_evaporator_ua(
        images, [{"cycle_name": "frost_cycle_000020", "status": "valid"}]
    )

    result = write_evaporator_ua_outputs(images, summary, tmp_path)

    assert result == tmp_path
    assert (tmp_path / "image_evaporator_ua.csv").is_file()
    assert (tmp_path / "cycle_summary.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (tmp_path / f"ua_timeseries.{suffix}").is_file()
    findings = (tmp_path / "findings.md").read_text(encoding="utf-8")
    assert "## 主标签" in findings
    assert "$$" in findings
