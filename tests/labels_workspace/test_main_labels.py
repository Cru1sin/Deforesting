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
    monkeypatch.setattr(main_labels.pd, "read_csv", lambda _: cost)
    calls: list[tuple[object, ...]] = []

    def fake_build(
        dataset: Path,
        received_cost: pd.DataFrame,
        output: Path,
        thresholds: list[float],
        *,
        overwrite: bool,
    ) -> None:
        calls.append((dataset, received_cost, output, thresholds, overwrite))
        output.mkdir(parents=True)

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
    assert calls[0][2:] == (output, [0.02, 0.05], True)
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
