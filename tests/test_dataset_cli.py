from __future__ import annotations

import json
from pathlib import Path

import pytest

from frost_analysis import cli


def test_dataset_cli_dispatches_build(monkeypatch: object, tmp_path: Path, capsys: object) -> None:
    calls: dict[str, object] = {}

    def fake_build(run_paths: list[Path], output: Path) -> Path:
        calls["runs"] = run_paths
        calls["output"] = output
        return output

    monkeypatch.setattr(cli, "build_dataset", fake_build)
    output = tmp_path / "frost_cycles_v1"

    assert cli.main(
        [
            "dataset",
            "build",
            "--run",
            "outputs/runs/0715",
            "--run",
            "outputs/runs/0716",
            "--output",
            str(output),
        ]
    ) == 0
    assert calls == {
        "runs": [Path("outputs/runs/0715"), Path("outputs/runs/0716")],
        "output": output,
    }
    assert capsys.readouterr().out.strip() == str(output)


def test_dataset_cli_dispatches_append_and_validate(
    monkeypatch: object, tmp_path: Path, capsys: object
) -> None:
    calls: dict[str, object] = {}
    dataset = tmp_path / "frost_cycles_v1"
    dataset.mkdir()
    (dataset / "dataset_manifest.json").write_text(
        json.dumps({"published_cycle_count": 1, "image_count": 2}),
        encoding="utf-8",
    )

    def fake_append(run: Path, output: Path) -> Path:
        calls["append"] = (run, output)
        return output

    def fake_validate(output: Path) -> None:
        calls["validate"] = output

    monkeypatch.setattr(cli, "append_dataset", fake_append)
    monkeypatch.setattr(cli, "validate_dataset", fake_validate)
    assert cli.main(
        ["dataset", "append", "--run", "outputs/runs/0720", "--dataset", str(dataset)]
    ) == 0
    assert calls["append"] == (Path("outputs/runs/0720"), dataset)
    assert capsys.readouterr().out.strip() == str(dataset)

    assert cli.main(["dataset", "validate", "--input", str(dataset)]) == 0
    assert calls["validate"] == dataset
    assert capsys.readouterr().out.strip() == "dataset valid: 1 published cycles, 2 images"


def test_dataset_cli_dispatches_v2_add_and_review(
    monkeypatch: object, tmp_path: Path, capsys: object
) -> None:
    calls: dict[str, object] = {}
    dataset = tmp_path / "frost_dataset"

    def fake_add(run: Path, output: Path) -> Path:
        calls["add"] = (run, output)
        return output

    def fake_review(output: Path, cycle: str, *, status: str, note: str | None) -> None:
        calls["review"] = (output, cycle, status, note)

    monkeypatch.setattr(cli, "add_dataset", fake_add)
    monkeypatch.setattr(cli, "review_cycle", fake_review)
    assert cli.main(
        ["dataset", "add", "--run", "outputs/runs/0714", "--dataset", str(dataset)]
    ) == 0
    assert calls["add"] == (Path("outputs/runs/0714"), dataset)
    assert capsys.readouterr().out.strip() == str(dataset)

    assert cli.main(
        [
            "dataset",
            "review-cycle",
            "--dataset",
            str(dataset),
            "--cycle",
            "frost_cycle_000001",
            "--status",
            "partial",
            "--note",
            "manual",
        ]
    ) == 0
    assert calls["review"] == (dataset, "frost_cycle_000001", "partial", "manual")


def test_cli_help_does_not_expose_legacy_analyze_entry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])

    assert "analyze" not in capsys.readouterr().out
