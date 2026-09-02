from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main_evaluate

SETTING = {
    "representation": "handcrafted",
    "head": "logistic",
    "camera": "front",
    "modality": "rgb",
}


def _write_run(path: Path, held_out: str = "a") -> None:
    path.mkdir()
    pd.DataFrame(
        [
            {
                "held_out_experiment": held_out,
                **SETTING,
                "train_images": 2,
                "test_images": 1,
                "status": "ok",
                "message": "",
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
            }
        ]
    ).to_csv(path / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": held_out,
                "held_out_experiment": held_out,
                **SETTING,
                "target": 0,
                "prediction": 0,
            }
        ]
    ).to_parquet(path / "predictions.parquet", index=False)


def test_parser_defaults() -> None:
    args = main_evaluate.build_parser().parse_args([])

    assert args.results == [Path("output/models/current")]
    assert args.output == Path("output/models/evaluation")
    assert args.task == "binary"
    assert args.overwrite is False


def test_main_reads_multiple_runs_and_writes_exactly_four_outputs(tmp_path: Path) -> None:
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    _write_run(first, held_out="a")
    _write_run(second, held_out="b")
    output = tmp_path / "evaluation"
    args = main_evaluate.build_parser().parse_args(
        [
            "--results",
            str(first),
            str(second),
            "--output",
            str(output),
        ]
    )

    assert main_evaluate.run(args, command=["main_evaluate.py", "--results", "..."]) == 0

    assert {path.name for path in output.iterdir()} == {
        "command.txt",
        "args.json",
        "experiment_metrics.csv",
        "summary.csv",
    }
    experiment_metrics = pd.read_csv(output / "experiment_metrics.csv")
    summary = pd.read_csv(output / "summary.csv")
    assert len(experiment_metrics) == 2
    assert len(summary) == 2
    assert set(experiment_metrics["source"]) == {str(first), str(second)}
    assert json.loads((output / "args.json").read_text())["task"] == "binary"
    assert (output / "command.txt").read_text().strip() == "main_evaluate.py --results ..."


def test_nonempty_output_requires_overwrite(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "keep.txt").write_text("owned")
    args = main_evaluate.build_parser().parse_args(
        ["--results", str(run), "--output", str(output)]
    )

    with pytest.raises(FileExistsError, match="not empty"):
        main_evaluate.run(args)
