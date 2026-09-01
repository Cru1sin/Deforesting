from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def test_main_data_help_lists_dataset_actions() -> None:
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = "/private/tmp/pinn4soh-uv-cache"
    result = subprocess.run(
        ["uv", "run", "python", "main_data.py", "--help"],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    for action in (
        "validate",
        "add",
        "replace",
        "aggregate-original",
        "remove",
        "refresh",
        "review-cycle",
        "edit",
        "render",
    ):
        assert action in result.stdout


def test_main_data_defaults_to_validate_local_dataset() -> None:
    from main_data import build_parser

    arguments = build_parser().parse_args([])

    assert arguments.action == "validate"
    assert arguments.dataset == Path("dataset")


def test_dataset_loader_reads_catalog_from_absolute_path_without_writing(tmp_path: Path) -> None:
    from dataloader.dataloader import DatasetLoader

    dataset = tmp_path / "external-dataset"
    (dataset / "cycles").mkdir(parents=True)
    (dataset / "cycles_original").mkdir()
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_schema_version": 3,
                "dataset_id": "frost_cycle_dataset",
                "experiments": [],
            }
        ),
        encoding="utf-8",
    )
    (dataset / "cycle_catalog.json").write_text(
        json.dumps({"cycles": [], "catalog_note": "absolute-path"}), encoding="utf-8"
    )
    (dataset / "channel_registry.json").write_text(
        json.dumps({"columns": []}), encoding="utf-8"
    )
    pd.DataFrame(columns=["cycle_name", "camera_role", "file_name"]).to_parquet(
        dataset / "image_metadata.parquet", index=False
    )
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }

    loader = DatasetLoader(dataset.resolve())

    assert loader.dataset_root == dataset.resolve()
    assert loader.catalog["catalog_note"] == "absolute-path"
    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before
