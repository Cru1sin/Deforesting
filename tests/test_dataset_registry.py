from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from frost_analysis.dataset_registry import (
    IMAGE_COLUMNS,
    canonical_frame,
    canonical_registry_hash,
    drop_image_columns,
    merge_registries,
    registry_from_frame,
)
from frost_analysis.dataset_v3 import (
    make_image_id,
    resolve_project_root,
    source_fingerprint,
)


def _registry(*, signal_type: str = "double") -> dict[str, object]:
    return {
        "registry_version": 1,
        "resample_interval_seconds": 10,
        "channels": {
            "signal": {
                "kind": "continuous",
                "unit": "C",
                "dtype": signal_type,
                "resample": "mean",
                "formula": None,
                "dependencies": [],
                "analysis_candidate": False,
                "coverage_required": True,
            }
        },
        "fields": [
            {"name": "timestamp", "logical_type": "timestamp[ns]", "nullable": True},
            {"name": "signal", "logical_type": signal_type, "nullable": True},
        ],
        "analysis_settings": {},
    }


def test_drop_image_columns_removes_all_image_triples() -> None:
    frame = pd.DataFrame(
        columns=[
            "timestamp",
            "image_front_path",
            "image_front_time",
            "image_front_offset_seconds",
            "image_unverified_camera_01_path",
            "image_unverified_camera_01_time",
            "image_unverified_camera_01_offset_seconds",
            "signal",
        ]
    )

    result = drop_image_columns(frame)

    assert result.columns.tolist() == ["timestamp", "signal"]
    assert IMAGE_COLUMNS


def test_registry_merge_allows_new_channel_and_preserves_old_order() -> None:
    merged = merge_registries(_registry(), {**_registry(), "channels": {
        **_registry()["channels"],
        "humidity": {
            "kind": "continuous",
            "unit": "%",
            "dtype": "double",
            "resample": "mean",
            "formula": None,
            "dependencies": [],
            "analysis_candidate": False,
            "coverage_required": False,
        },
    }, "fields": [
        *_registry()["fields"],
        {"name": "humidity", "logical_type": "double", "nullable": True},
    ]})

    assert list(merged["channels"]) == ["signal", "humidity"]
    assert [field["name"] for field in merged["fields"]] == [
        "timestamp",
        "signal",
        "humidity",
    ]
    assert canonical_registry_hash(merged) == canonical_registry_hash(dict(merged))


def test_registry_allows_existing_channel_to_be_absent_in_new_date() -> None:
    existing = _registry()
    candidate = _registry(signal_type="null")
    candidate["fields"] = [candidate["fields"][0]]

    merged = merge_registries(existing, candidate)

    assert merged["channels"]["signal"]["dtype"] == "double"
    assert merged["fields"] == existing["fields"]


def test_canonical_frame_promotes_existing_all_null_field_to_registry_type() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-14 10:00:00"]),
            "signal": pd.Series([pd.NA], dtype="object"),
        }
    )

    canonical = canonical_frame(frame, _registry())

    assert pd.api.types.is_float_dtype(canonical["signal"])


def test_canonical_frame_writes_unknown_all_null_field_as_arrow_null(tmp_path) -> None:
    frame = pd.DataFrame({"signal": pd.Series([float("nan")], dtype="float64")})
    registry = registry_from_frame(
        frame,
        {"signal": {"kind": "continuous", "resample": "mean"}},
    )

    canonical = canonical_frame(frame, registry)
    path = tmp_path / "canonical.parquet"
    canonical.to_parquet(path, index=False)

    assert str(pq.read_schema(path).field("signal").type) == "null"


def test_registry_treats_all_null_candidate_field_as_nullable_unknown() -> None:
    channels = {
        "signal": {
            "kind": "continuous",
            "unit": "C",
            "resample": "mean",
            "missing": "none",
            "analysis_candidate": False,
        }
    }
    existing = registry_from_frame(pd.DataFrame({"signal": [True]}), channels)
    candidate = registry_from_frame(
        pd.DataFrame({"signal": [float("nan")]}), channels
    )

    merged = merge_registries(existing, candidate)

    assert merged["fields"][0]["logical_type"] == "bool"


def test_registry_merge_rejects_same_channel_semantic_change() -> None:
    changed = _registry(signal_type="string")

    try:
        merge_registries(_registry(), changed)
    except ValueError as error:
        assert "channel registry conflict" in str(error)
    else:
        raise AssertionError("semantic channel change must be rejected")


def test_registry_captures_source_channel_semantic_changes() -> None:
    frame = pd.DataFrame({"signal": [1.0]})
    base_channel = {
        "kind": "continuous",
        "unit": "C",
        "resample": "mean",
        "missing": "none",
        "analysis_candidate": False,
        "valid_range": [0.0, 10.0],
        "scale": 1.0,
        "offset": 0.0,
        "allowed_values": None,
    }
    changed_channel = {**base_channel, "valid_range": [0.0, 20.0]}
    existing = registry_from_frame(frame, {"signal": base_channel})
    candidate = registry_from_frame(frame, {"signal": changed_channel})

    with pytest.raises(ValueError, match="channel registry conflict"):
        merge_registries(existing, candidate)


def test_image_id_uses_immutable_source_identity_not_frame_index() -> None:
    first = make_image_id("exp::cycle_1", "camera01", "camera01/frame.jpg")
    second = make_image_id("exp::cycle_1", "camera01", "camera01/other.jpg")

    assert first.startswith("img_")
    assert first != second
    assert make_image_id("exp::cycle_1", "camera01", "camera01/frame.jpg") == first


def test_source_fingerprint_includes_run_inventory_and_registry() -> None:
    first = source_fingerprint("exp", "2026-07-14", "run", "inventory", "registry")
    changed = source_fingerprint("exp", "2026-07-14", "run-2", "inventory", "registry")

    assert len(first) == 64
    assert first != changed


def test_project_root_is_resolved_from_repository_not_cwd(tmp_path):
    root = tmp_path / "project"
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert resolve_project_root(nested) == root
