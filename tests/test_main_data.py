from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pandas as pd


def test_validate_dataset_help_exposes_only_read_only_validation(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    result = subprocess.run(
        ["uv", "run", "python", "validate_dataset.py", "--help"],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert "--dataset" in result.stdout
    for maintenance_action in (
        "add",
        "replace",
        "aggregate-original",
        "remove",
        "refresh",
        "review-cycle",
        "edit",
        "render",
    ):
        assert maintenance_action not in result.stdout


def test_validate_dataset_defaults_to_validate_local_dataset() -> None:
    from validate_dataset import build_parser

    arguments = build_parser().parse_args([])

    assert arguments.dataset == Path("dataset")


def test_quality_tools_target_new_workspace_entries() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]

    assert config["pytest"]["ini_options"]["pythonpath"] == ["."]
    assert config["ruff"]["lint"]["select"] == ["E", "F", "I", "B"]
    assert "mypy" not in config
    assert "coverage" not in config


def test_validate_dispatches_to_read_only_domain_functions(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import validate_dataset

    validated: list[Path] = []

    class Loader:
        def __init__(self, dataset: Path) -> None:
            assert dataset == tmp_path

        def list_cycles(self) -> pd.DataFrame:
            return pd.DataFrame(index=range(2))

        def load_image_metadata(self) -> pd.DataFrame:
            return pd.DataFrame(index=range(3))

    monkeypatch.setattr(validate_dataset, "validate_dataset", validated.append)
    monkeypatch.setattr(validate_dataset, "DatasetLoader", Loader)

    assert validate_dataset.main(["--dataset", str(tmp_path)]) == 0
    assert validated == [tmp_path]
    assert capsys.readouterr().out == "dataset valid: 2 cycles, 3 images\n"


def test_advanced_dataset_maintenance_keeps_edit_options(tmp_path: Path, monkeypatch) -> None:
    from dataset_tools import manage_dataset

    calls: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        manage_dataset,
        "edit_dataset",
        lambda dataset, **options: calls.append((dataset, options)) or dataset,
    )

    assert (
        manage_dataset.main(
            [
                "edit",
                "--dataset",
                str(tmp_path),
                "--recovery-seconds",
                "45",
                "--skip-rgb-panels",
            ]
        )
        == 0
    )
    assert calls == [
        (
            tmp_path,
            {
                "baseline_seconds": None,
                "recovery_seconds": 45,
                "recovery_end_by": None,
                "defrost_preparation": False,
                "render_rgb_panels": False,
            },
        )
    ]


def test_advanced_dataset_maintenance_keeps_render_options(tmp_path: Path, monkeypatch) -> None:
    from dataset_tools import manage_dataset

    calls: list[tuple[Path, str, dict[str, object]]] = []
    monkeypatch.setattr(
        manage_dataset,
        "render_dataset",
        lambda dataset, cycle, **options: calls.append((dataset, cycle, options)) or dataset,
    )

    assert manage_dataset.main(["render", "cycle_001", "--dataset", str(tmp_path)]) == 0
    assert calls == [
        (
            tmp_path,
            "cycle_001",
            {"publication": True, "panel": True, "fetch_cloud_images": False},
        )
    ]


def test_uv_edit_and_render_reach_domain_without_import_error(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    missing = tmp_path / "missing"
    render_dataset = tmp_path / "render-dataset"
    render_dataset.mkdir()
    (render_dataset / "cycle_catalog.json").write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "cycle_name": "cycle_001",
                        "experiment_id": "experiment_001",
                        "assets": {"parquet": "cycles/cycle_001.parquet"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    commands = (
        [
            "uv",
            "run",
            "python",
            "-m",
            "dataset_tools.manage_dataset",
            "edit",
            "--dataset",
            str(missing),
            "--baseline-seconds",
            "60",
        ],
        [
            "uv",
            "run",
            "python",
            "-m",
            "dataset_tools.manage_dataset",
            "render",
            "cycle_001",
            "--dataset",
            str(render_dataset),
            "--publication",
        ],
    )

    for command in commands:
        result = subprocess.run(command, capture_output=True, env=environment, text=True)
        assert result.returncode != 0
        assert "ModuleNotFoundError" not in result.stderr
        assert "FileNotFoundError" in result.stderr


def test_dataset_loader_reads_catalog_from_absolute_path_without_writing(tmp_path: Path) -> None:
    from dataset_tools.load_dataset import DatasetLoader

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
    (dataset / "channel_registry.json").write_text(json.dumps({"columns": []}), encoding="utf-8")
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
