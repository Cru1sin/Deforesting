from __future__ import annotations

from pathlib import Path

import pandas as pd

from frost_analysis import cli

from .conftest import frame_for, write_dataset


def test_dataset_validate_command_remains_registered(monkeypatch, tmp_path: Path) -> None:
    class Loader:
        def list_cycles(self) -> pd.DataFrame:
            return pd.DataFrame({"cycle_name": ["c1"]})

        def load_image_metadata(self) -> pd.DataFrame:
            return pd.DataFrame({"image_id": []})

    monkeypatch.setattr(cli, "validate_dataset", lambda _path: None)
    monkeypatch.setattr(cli, "DatasetLoader", lambda _path: Loader())

    assert cli.main(["dataset", "validate", "--dataset", str(tmp_path)]) == 0


def test_evidence_command_uses_dataset_loader_and_new_settings(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    write_dataset(dataset, [("c1", "2026-07-01", "valid", frame_for())])
    config = Path(__file__).parents[2] / "configs" / "evidence.yaml"

    result = cli.main(
        [
            "evidence",
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "evidence"),
        ]
    )

    assert result == 0
    assert (tmp_path / "evidence" / "analysis_manifest.json").is_file()
