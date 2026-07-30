from __future__ import annotations

import pandas as pd
import pytest

from frost_analysis.core.validation import validate_image_artifacts
from frost_analysis.data.alignment import (
    attach_image_paths,
    attach_cycle_labels,
    build_multiview,
    match_images_to_sensors,
)


def test_nearest_match_prefers_earlier_tie_and_keeps_unmatched_candidate() -> None:
    images = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_time": pd.to_datetime(["2026-07-15 09:00:00.500", "2026-07-15 09:00:03.000"]),
            "camera_id": ["cam_a", "cam_a"],
        }
    )
    sensors = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15 09:00:00", "2026-07-15 09:00:01"]),
            "must_not_be_copied": [10, 20],
        }
    )

    matched = match_images_to_sensors(images, sensors, tolerance_s=0.5)

    assert matched.loc[0, "candidate_timestamp"] == pd.Timestamp("2026-07-15 09:00:00")
    assert matched.loc[0, "time_delta_s"] == -0.5
    assert bool(matched.loc[0, "matched"]) is True
    assert matched.loc[0, "timestamp"] == pd.Timestamp("2026-07-15 09:00:00")
    assert bool(matched.loc[1, "matched"]) is False
    assert matched.loc[1, "candidate_timestamp"] == pd.Timestamp("2026-07-15 09:00:01")
    assert pd.isna(matched.loc[1, "timestamp"])
    assert "must_not_be_copied" not in matched


def test_cycle_labels_are_attached_only_to_matched_sensor_rows() -> None:
    aligned = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "matched": [True, False],
            "timestamp": pd.to_datetime(["2026-07-15 09:00:00", None]),
        }
    )
    sensors = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15 09:00:00"]),
            "cycle_id": ["cycle_001"],
            "cycle_quality": ["complete"],
            "stage": ["frost_development"],
            "cycle_phase": [0.4],
            "qf_gap": [False],
            "irrelevant_sensor": [999],
        }
    )

    labeled = attach_cycle_labels(aligned, sensors)

    assert labeled.loc[0, "cycle_id"] == "cycle_001"
    assert labeled.loc[0, "stage"] == "frost_development"
    assert labeled.loc[0, "cycle_phase"] == 0.4
    assert pd.isna(labeled.loc[1, "cycle_id"])
    assert "irrelevant_sensor" not in labeled


def test_multiview_uses_every_image_once_and_keeps_incomplete_groups() -> None:
    images = pd.DataFrame(
        {
            "sample_id": ["a0", "b0", "a1"],
            "camera_id": ["cam_a", "cam_b", "cam_a"],
            "camera_role": ["A", "B", "A"],
            "image_time": pd.to_datetime(
                [
                    "2026-07-15 09:00:00.000",
                    "2026-07-15 09:00:00.001",
                    "2026-07-15 09:00:30.000",
                ]
            ),
            "image_path": ["a0.jpg", "b0.jpg", "a1.jpg"],
            "image_ok": [True, True, True],
        }
    )

    multiview = build_multiview(images, tolerance_ms=10)

    assert multiview["camera_count"].tolist() == [2, 1]
    assert multiview["all_cameras_present"].tolist() == [True, False]
    used = multiview.filter(like="__sample_id").stack().tolist()
    assert sorted(used) == ["a0", "a1", "b0"]


def test_alignment_validates_tolerances_and_empty_schemas() -> None:
    images = pd.DataFrame({"sample_id": ["bad"], "camera_id": ["cam_a"], "image_time": [pd.NaT]})
    sensors = pd.DataFrame(columns=["timestamp", "value"])

    unmatched = match_images_to_sensors(images, sensors, tolerance_s=0)
    multiview = build_multiview(images, tolerance_ms=0)

    assert bool(unmatched.loc[0, "matched"]) is False
    assert multiview.empty
    assert {"group_id", "group_time", "camera_count", "all_cameras_present"} <= set(multiview)
    with pytest.raises(ValueError):
        match_images_to_sensors(images, sensors, tolerance_s=-0.1)
    with pytest.raises(ValueError):
        build_multiview(images, tolerance_ms=-1)


def test_attach_image_paths_requires_contract_and_uses_camera_roles() -> None:
    """Wide image columns must be keyed by stable roles and nearest deltas."""
    prepared = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-07-15 09:00:00", "2026-07-15 09:00:01"])}
    )
    alignment = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-07-15 09:00:00",
                    "2026-07-15 09:00:00",
                    "2026-07-15 09:00:01",
                ]
            ),
            "matched": [True, True, True],
            "camera_role": ["front", "front", "left"],
            "image_path": ["front-old.jpg", "front-new.jpg", "left.jpg"],
            "time_delta_s": [0.2, 0.1, -0.3],
        }
    )

    result = attach_image_paths(prepared, alignment)

    assert result.loc[0, "image_front_path"] == "front-new.jpg"
    assert result.loc[0, "image_front_offset_seconds"] == 0.1
    assert result.loc[1, "image_left_path"] == "left.jpg"
    assert "front" in result.columns[1]
    with pytest.raises(ValueError, match="camera_role"):
        attach_image_paths(prepared, alignment.drop(columns="camera_role"))


def test_multiview_separates_repeated_images_and_uses_exact_median() -> None:
    images = pd.DataFrame(
        {
            "sample_id": ["a0", "a1", "b0", "b1"],
            "camera_id": ["cam_a", "cam_a", "cam_b", "cam_b"],
            "camera_role": ["A", "A", "B", "B"],
            "image_time": pd.to_datetime(
                [
                    "2026-07-15 09:00:00.000",
                    "2026-07-15 09:00:00.001",
                    "2026-07-15 09:00:00.002",
                    "2026-07-15 09:00:00.003",
                ]
            ),
            "image_path": ["a0.jpg", "a1.jpg", "b0.jpg", "b1.jpg"],
            "image_ok": True,
        }
    )

    multiview = build_multiview(images, tolerance_ms=10)

    assert multiview["camera_count"].tolist() == [2, 2]
    used = multiview.filter(like="__sample_id").stack().tolist()
    assert sorted(used) == ["a0", "a1", "b0", "b1"]
    assert multiview.loc[0, "group_time"] == pd.Timestamp("2026-07-15 09:00:00.001")
    assert multiview.loc[0, "cam_a__delta_s"] == -0.001
    assert multiview.loc[0, "cam_b__delta_s"] == 0.001


def test_image_validator_rejects_duplicate_ids_tolerance_label_and_reuse(tmp_path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "image_time": pd.to_datetime(["2026-07-15 09:00:00"] * 2),
            "camera_id": ["cam_a", "cam_a"],
            "image_path": ["a.jpg", "a-copy.jpg"],
            "timestamp_ok": [True, True],
            "image_ok": [True, True],
        }
    )
    alignment = pd.DataFrame(
        {
            "sample_id": ["a"],
            "candidate_timestamp": pd.to_datetime(["2026-07-15 09:00:01"]),
            "timestamp": pd.to_datetime(["2026-07-15 09:00:01"]),
            "time_delta_s": [1.0],
            "matched": [True],
            "cycle_id": ["wrong_cycle"],
            "cycle_quality": ["complete"],
            "stage": ["frost_development"],
            "cycle_phase": [0.5],
        }
    )
    cycles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-15 09:00:01"]),
            "cycle_id": ["cycle_001"],
            "cycle_quality": ["complete"],
            "stage": ["frost_development"],
            "cycle_phase": [0.5],
        }
    )
    multiview = pd.DataFrame(
        {
            "group_id": ["g1", "g2"],
            "group_time": pd.to_datetime(["2026-07-15 09:00:00"] * 2),
            "camera_count": [1, 1],
            "all_cameras_present": [True, True],
            "cam_a__sample_id": ["a", "a"],
        }
    )
    manifest.to_parquet(processed / "image_manifest.parquet", index=False)
    alignment.to_parquet(processed / "image_sensor_alignment.parquet", index=False)
    multiview.to_parquet(processed / "multiview_index.parquet", index=False)
    cycles.to_parquet(processed / "cycle_labeled_timeseries.parquet", index=False)

    errors = validate_image_artifacts(tmp_path, tolerance_s=0.5)

    assert any("duplicate sample_id" in error for error in errors)
    assert any("exceeds tolerance" in error for error in errors)
    assert any("cycle label mismatch" in error for error in errors)
    assert any("reuses sample_id" in error for error in errors)
