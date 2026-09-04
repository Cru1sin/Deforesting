from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import main_evaluate
import plots.model as model_plots
from plots.model import (
    _robust_trigger_error_limit,
    plot_trigger_error_figures,
    trigger_error_table,
    two_of_three_trigger,
)

SETTING = {
    "representation": "handcrafted",
    "head": "logistic",
    "camera": "front",
    "modality": "rgb",
}


def test_trigger_error_axis_ignores_one_extreme_outlier() -> None:
    errors = pd.Series([*range(1, 61)] * 20 + [122.0])

    assert _robust_trigger_error_limit(errors) < 70


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


def test_two_of_three_trigger_allows_one_negative_between_positive_frames() -> None:
    times = pd.date_range("2026-01-01", periods=5, freq="min")
    trigger, rolling = two_of_three_trigger(times, pd.Series([0.1, 0.8, 0.2, 0.9, 0.1]))

    assert trigger == times[3]
    assert rolling.tolist() == [False, False, False, True, False]


def test_two_of_three_trigger_fires_on_the_second_adjacent_positive() -> None:
    times = pd.date_range("2026-01-01", periods=3, freq="min")
    trigger, rolling = two_of_three_trigger(times, pd.Series([0.8, 0.9, 0.1]))

    assert trigger == times[1]
    assert rolling.tolist() == [False, True, True]


def test_trigger_error_table_keeps_the_sign_for_both_control_rules() -> None:
    times = pd.date_range("2026-01-01", periods=3, freq="min")
    predictions = pd.DataFrame(
        {
            "experiment_id": ["exp_20260101"] * 3,
            "cycle_name": ["frost_cycle_000001"] * 3,
            "camera": ["front"] * 3,
            "modality": ["rgb"] * 3,
            "image_time": times,
            "decision_score": [0.6, 0.2, 0.7],
        }
    )
    policy = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"],
            "selected": [True],
            "selected_time": [times[2]],
        }
    )

    result = trigger_error_table(predictions, policy)

    errors = result.set_index("strategy")["trigger_error_minutes"]
    assert errors["first_positive"] == pytest.approx(-2.0)
    assert errors["two_of_three"] == pytest.approx(0.0)
    assert set(result["experiment_id"]) == {"exp_20260101"}


def test_trigger_error_figures_are_one_scatter_plot_per_camera_and_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras = ["top", "top_close", "left", "left_close", "front", "extreme"]
    modalities = ["rgb", "rgb_sensor", "rgb_sensor_slope"]
    prediction_rows = []
    policy_rows = []
    for cycle_id, experiment_id in ((1, "exp_20260101"), (2, "exp_20260101"), (3, "exp_20260102")):
        cycle_name = f"frost_cycle_{cycle_id:06d}"
        times = pd.date_range(f"2026-01-0{cycle_id} 00:00", periods=3, freq="min")
        policy_rows.append(
            {"cycle_name": cycle_name, "selected": True, "selected_time": times[1]}
        )
        for camera in cameras:
            for modality in modalities:
                for image_time, score in zip(times, (0.2, 0.8, 0.9), strict=True):
                    prediction_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "cycle_name": cycle_name,
                            "camera": camera,
                            "modality": modality,
                            "representation": "dinov2",
                            "head": "logistic",
                            "image_time": image_time,
                            "decision_score": score,
                        }
                    )

    exports: list[Path] = []

    def record_export(figure, stem, formats=("png",)) -> None:
        exports.append(stem)
        axis = figure.axes[0]
        assert len(figure.axes) == 1
        assert axis.collections
        assert len(axis.patches) == 2
        assert all(len(line.get_xdata()) <= 2 for line in axis.lines)
        model_plots.plt.close(figure)

    monkeypatch.setattr(model_plots, "_export", record_export)
    plot_trigger_error_figures(
        predictions=pd.DataFrame(prediction_rows),
        policy=pd.DataFrame(policy_rows),
        output=tmp_path / "figures",
        source_output=tmp_path / "source",
        representation="dinov2",
        head="logistic",
    )

    assert len(exports) == 12
    assert {path.parent.name for path in exports} == {"two_of_three", "first_positive"}
    assert {path.name for path in exports} == set(cameras)


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
    recorded_args = json.loads((output / "args.json").read_text())
    assert set(recorded_args) == {"results", "output", "task", "overwrite"}
    assert recorded_args["task"] == "binary"
    assert (output / "command.txt").read_text().strip() == (
        "uv run python main_evaluate.py --results ..."
    )


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


def test_overwrite_replaces_stale_evaluation_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "stale.txt").write_text("old")
    args = main_evaluate.build_parser().parse_args(
        ["--results", str(run), "--output", str(output), "--overwrite"]
    )

    assert main_evaluate.run(args) == 0

    assert not (output / "stale.txt").exists()
    assert {path.name for path in output.iterdir()} == {
        "command.txt",
        "args.json",
        "experiment_metrics.csv",
        "summary.csv",
    }


def test_figures_route_current_evaluation_and_paired_optional_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    _write_run(run)
    output = tmp_path / "evaluation"
    optima = tmp_path / "optima.csv"
    concentration = tmp_path / "concentration.csv"
    pd.DataFrame({"cycle_name": ["a"]}).to_csv(optima, index=False)
    pd.DataFrame({"camera_group": ["front"]}).to_csv(concentration, index=False)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main_evaluate, "plot_model_figures", lambda **kwargs: calls.append(kwargs))
    args = main_evaluate.build_parser().parse_args(
        [
            "--results",
            str(run),
            "--output",
            str(output),
            "--figures",
            "--figure-output",
            str(tmp_path / "paper"),
            "--figure-format",
            "svg",
            "pdf",
            "--optima",
            str(optima),
            "--concentration",
            str(concentration),
        ]
    )

    assert main_evaluate.run(args) == 0
    assert len(calls) == 1
    assert calls[0]["output"] == tmp_path / "paper"
    assert calls[0]["source_output"] == output / "figure_source_data"
    assert calls[0]["figure_formats"] == ("svg", "pdf")
    assert isinstance(calls[0]["summary"], pd.DataFrame)
    assert isinstance(calls[0]["optima"], pd.DataFrame)
    assert isinstance(calls[0]["concentration"], pd.DataFrame)

    unpaired = main_evaluate.build_parser().parse_args(["--optima", str(optima)])
    with pytest.raises(ValueError, match="provided together"):
        main_evaluate.run(unpaired)
