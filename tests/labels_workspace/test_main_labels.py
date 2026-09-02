from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main_labels


def _canonical_cost() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a"],
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="min"),
            "relative_regret": [0.0, 0.1],
            "optimization_eligible": [True, True],
            "is_censored": [False, False],
            "label_eligible": [True, True],
            "variant": [None, None],
        }
    )


def test_parser_exposes_only_the_label_run_arguments() -> None:
    args = main_labels.build_parser().parse_args([])

    assert vars(args) == {
        "dataset": Path("dataset"),
        "cost_csv": Path("output/cost/v1/cost.csv"),
        "output": Path("output/labels/v1"),
        "thresholds": [0.01, 0.02, 0.05, 0.10],
        "overwrite": False,
        "figures": False,
        "figure_output": None,
        "figure_format": ["png"],
    }


def test_main_gates_cost_before_calling_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cost = _canonical_cost()
    cost.loc[0, "label_eligible"] = False
    monkeypatch.setattr(main_labels.pd, "read_csv", lambda _: cost)
    calls = 0

    def forbidden_build(*_: object, **__: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(main_labels, "build_labels", forbidden_build)

    with pytest.raises(ValueError, match="label_eligible must be True for every row"):
        main_labels.main(["--cost-csv", str(tmp_path / "cost.csv")])
    assert calls == 0


def test_main_calls_build_once_and_records_copyable_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cost = _canonical_cost()
    read_csv = pd.read_csv
    monkeypatch.setattr(main_labels.pd, "read_csv", lambda _: cost)
    calls: list[tuple[object, ...]] = []

    def fake_build(
        dataset: Path,
        received_cost: pd.DataFrame,
        thresholds: list[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        calls.append((dataset, received_cost, thresholds))
        return (
            pd.DataFrame({"label": [1]}),
            pd.DataFrame({"balance": [2]}),
            pd.DataFrame({"audit": [3]}),
        )

    monkeypatch.setattr(main_labels, "build_labels", fake_build)
    output = tmp_path / "labels"
    arguments = [
        "--dataset",
        "chosen_dataset",
        "--cost-csv",
        "chosen_cost.csv",
        "--output",
        str(output),
        "--thresholds",
        "0.02",
        "0.05",
        "--overwrite",
    ]

    assert main_labels.main(arguments) == 0

    assert len(calls) == 1
    assert calls[0][0] == Path("chosen_dataset")
    assert calls[0][1] is cost
    assert calls[0][2] == [0.02, 0.05]
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "image_cost_labels.parquet"),
        pd.DataFrame({"label": [1]}),
    )
    pd.testing.assert_frame_equal(
        read_csv(output / "label_balance.csv"), pd.DataFrame({"balance": [2]})
    )
    pd.testing.assert_frame_equal(
        read_csv(output / "cycle_audit.csv"), pd.DataFrame({"audit": [3]})
    )
    assert (output / "command.txt").read_text(encoding="utf-8") == (
        "uv run python main_labels.py " + " ".join(arguments) + "\n"
    )
    assert json.loads((output / "args.json").read_text(encoding="utf-8")) == {
        "cost_csv": "chosen_cost.csv",
        "dataset": "chosen_dataset",
        "output": str(output),
        "overwrite": True,
        "thresholds": [0.02, 0.05],
    }
    printed = capsys.readouterr().out
    assert "chosen_cost.csv" in printed
    assert str(output) in printed


def test_main_rejects_existing_output_before_building_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "labels"
    output.mkdir()
    monkeypatch.setattr(main_labels.pd, "read_csv", lambda _: _canonical_cost())

    def forbidden_build(*_: object, **__: object) -> None:
        raise AssertionError("build_labels must not run for an existing output")

    monkeypatch.setattr(main_labels, "build_labels", forbidden_build)

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        main_labels.main(["--output", str(output)])


def test_main_does_not_create_output_when_label_build_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "labels"
    monkeypatch.setattr(main_labels.pd, "read_csv", lambda _: _canonical_cost())

    def failed_build(*_: object, **__: object) -> None:
        raise RuntimeError("label build failed")

    monkeypatch.setattr(main_labels, "build_labels", failed_build)

    with pytest.raises(RuntimeError, match="label build failed"):
        main_labels.main(["--output", str(output)])
    assert not output.exists()
