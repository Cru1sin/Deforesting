from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.dataset_images import (
    collect_matched_images,
    rewrite_processed_image_paths,
)


def _prepared_with_images(input_dir: Path) -> pd.DataFrame:
    image_dir = input_dir / "camera01"
    image_dir.mkdir(parents=True)
    first = image_dir / "20260715100002000.jpg"
    second = image_dir / "20260715100001000.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    timestamps = pd.to_datetime(
        ["2026-07-15 10:00:01", "2026-07-15 10:00:02"]
    )
    return pd.DataFrame(
        {
            "experiment_id": ["frost_0715", "frost_0715"],
            "timestamp": timestamps,
            "cycle_id": ["cycle_1", "cycle_1"],
            "cycle_stage": ["frost_development", "frost_development"],
            "image_front_path": [
                "camera01/20260715100002000.jpg",
                "camera01/20260715100001000.jpg",
            ],
            "image_front_time": pd.to_datetime(
                ["2026-07-15 10:00:02", "2026-07-15 10:00:01"]
            ),
            "image_front_offset_seconds": [1.0, -1.0],
        }
    )


def test_collect_matched_images_sorts_and_builds_stable_records(tmp_path: Path) -> None:
    prepared = _prepared_with_images(tmp_path / "input")

    records, inventory_hash = collect_matched_images(
        prepared,
        input_dir=tmp_path / "input",
        cycle_names={("frost_0715", "cycle_1"): "frost_cycle_000001"},
    )

    assert [record["image_id"] for record in records] == [
        "frost_cycle_000001__front__000001__20260715T100001000",
        "frost_cycle_000001__front__000002__20260715T100002000",
    ]
    assert all(record["image_path"].startswith("images/") for record in records)
    assert len(inventory_hash) == 64


def test_rewrite_processed_paths_uses_cycle_role_source_scope(tmp_path: Path) -> None:
    prepared = _prepared_with_images(tmp_path / "input")
    records, _ = collect_matched_images(
        prepared,
        input_dir=tmp_path / "input",
        cycle_names={("frost_0715", "cycle_1"): "frost_cycle_000001"},
    )
    processed = prepared.copy()

    rewritten = rewrite_processed_image_paths(processed, records)

    assert rewritten["image_front_path"].tolist() == [
        "images/frost_cycle_000001__front__000002__20260715T100002000.jpg",
        "images/frost_cycle_000001__front__000001__20260715T100001000.jpg",
    ]


def test_collect_matched_images_rejects_partial_image_triplet(tmp_path: Path) -> None:
    prepared = _prepared_with_images(tmp_path / "input")
    prepared.loc[0, "image_front_offset_seconds"] = pd.NA

    with pytest.raises(ValueError, match="image_front"):
        collect_matched_images(
            prepared,
            input_dir=tmp_path / "input",
            cycle_names={("frost_0715", "cycle_1"): "frost_cycle_000001"},
        )
