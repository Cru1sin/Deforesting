from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

import main_cost
from plots import cost, labels, model


class ImageLoader:
    calls: list[str | None] = []

    def __init__(self, dataset: Path) -> None:
        self.dataset = dataset

    def load_image_metadata(self, cycle_name: str | None = None) -> pd.DataFrame:
        type(self).calls.append(cycle_name)
        table = pd.DataFrame(
            {
                "cycle_name": ["cycle_a", "cycle_b"],
                "camera_role": ["front", "front"],
                "file_name": ["nearest.png", "missing.png"],
                "image_time": ["2026-01-01 00:02:30", "2026-01-01 00:02:30"],
            }
        )
        return table if cycle_name is None else table.loc[table["cycle_name"].eq(cycle_name)]


def _cost_rows(cycle: str, offset: int = 0) -> list[dict[str, object]]:
    values = [0.04, 0.009, 0.0, 0.02, 0.009]
    return [
        {
            "cycle_name": cycle,
            "candidate_time": f"2026-01-01 00:0{minute + offset}:00",
            "candidate_elapsed_minutes": minute + offset,
            "optimization_eligible": True,
            "relative_regret": regret,
            "near_optimal_1pct": regret <= 0.01,
            "inverse_cop": 0.4 + regret,
        }
        for minute, regret in enumerate(values)
    ]


def _run(tmp_path: Path, name: str, basis: str, rows: list[dict[str, object]]) -> Path:
    result = tmp_path / name
    result.mkdir()
    base, _, variant = name.partition("__")
    (result / "recipe.json").write_text(
        json.dumps({"base_cost": base, "variant": variant or None, "heat_basis": basis}),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(result / "cost.csv", index=False)
    return result


def test_cost_plots_write_connected_optima_cycles_and_local_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images/cycle_a/front/nearest.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "white").save(image)
    runs = [
        _run(tmp_path, "v1__trial", "unit", _cost_rows("cycle_a")),
        _run(tmp_path, "v2.5", "water", _cost_rows("cycle_a", 1) + _cost_rows("cycle_b")),
        _run(tmp_path, "v2.6.8", "unit", _cost_rows("cycle_b", 2)),
    ]
    monkeypatch.setattr(cost, "DatasetLoader", ImageLoader)

    optima = cost.compare_results(runs, dataset, tmp_path / "plots", camera="front")

    assert isinstance(optima, pd.DataFrame)
    selected = optima.loc[optima["version"].eq("v1 (trial)")].iloc[0]
    assert selected["optimum_elapsed_minutes"] == 2
    assert selected["interval_start_elapsed_minutes"] == 1
    assert selected["interval_end_elapsed_minutes"] == 2
    assert selected["interval_width_minutes"] == 1
    assert selected["image_path"] == "images/cycle_a/front/nearest.png"
    assert selected["image_delta_seconds"] == 30
    assert {path.name for path in (tmp_path / "plots").iterdir()} == {
        "optima.csv",
        "optimum_comparison.svg",
        "optimum_comparison.png",
        "cycles",
        "images",
    }
    assert len(list((tmp_path / "plots/cycles").iterdir())) == 4
    assert {path.name for path in (tmp_path / "plots/images").iterdir()} == {"cycle_a.png"}
    cycle_svg = (tmp_path / "plots/cycles/cycle_a.svg").read_text(encoding="utf-8")
    assert "<text" in cycle_svg
    assert "Absolute inverse COP — unit heat basis" in cycle_svg
    assert "Absolute inverse COP — water heat basis" in cycle_svg


def test_cost_plot_overwrite_replaces_the_whole_plot_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "plots"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    run = _run(tmp_path, "v1", "unit", _cost_rows("cycle_a"))
    monkeypatch.setattr(cost, "DatasetLoader", ImageLoader)

    with pytest.raises(FileExistsError, match="overwrite"):
        cost.compare_results([run], tmp_path / "dataset", output, camera="front")
    cost.compare_results([run], tmp_path / "dataset", output, camera="front", overwrite=True)

    assert not (output / "stale.txt").exists()


def test_cost_loads_image_metadata_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ImageLoader.calls.clear()
    monkeypatch.setattr(cost, "DatasetLoader", ImageLoader)
    run = _run(tmp_path, "v1", "unit", _cost_rows("cycle_a"))

    cost.compare_results([run], tmp_path / "dataset", tmp_path / "plots", camera="front")

    assert ImageLoader.calls == [None]


def test_connected_interval_excludes_ineligible_near_optimal_rows() -> None:
    curve = pd.DataFrame(
        {
            "candidate_elapsed_minutes": [0, 1, 2, 3],
            "optimization_eligible": [True, True, True, False],
            "relative_regret": [0.02, 0.0, 0.009, 0.009],
            "near_optimal_1pct": [False, True, True, True],
        }
    )

    start, end = cost._connected_interval(curve, 1)

    assert start["candidate_elapsed_minutes"] == 1
    assert end["candidate_elapsed_minutes"] == 2


@pytest.mark.parametrize("flag", [None, [False, False, False, False]])
def test_connected_interval_derives_1pct_mask_without_persisted_flag(
    flag: list[bool] | None,
) -> None:
    curve = pd.DataFrame(
        {
            "candidate_elapsed_minutes": [0, 1, 2, 3],
            "optimization_eligible": [True, True, True, False],
            "relative_regret": [0.02, 0.0, 0.009, 0.009],
        }
    )
    if flag is not None:
        curve["near_optimal_1pct"] = flag

    try:
        start, end = cost._connected_interval(curve, 1)
    except KeyError as error:
        pytest.fail(f"interval must not require persisted flags: {error}")

    assert start["candidate_elapsed_minutes"] == 1
    assert end["candidate_elapsed_minutes"] == 2


def test_cycle_figure_sorts_shuffled_candidate_curves() -> None:
    source = Path("/runs/v1")
    table = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"] * 3,
            "candidate_elapsed_minutes": [2, 0, 1],
            "relative_regret": [0.01, 0.02, 0.0],
            "inverse_cop": [0.41, 0.42, 0.40],
        }
    )
    optima = pd.DataFrame(
        {
            "source": [str(source)],
            "cycle": ["cycle_a"],
            "optimum_elapsed_minutes": [1],
            "interval_start_elapsed_minutes": [1],
            "interval_end_elapsed_minutes": [1],
        }
    )

    figure = cost._cycle_figure(
        "cycle_a", [("v1", "unit", table, source)], optima, {str(source): "C0"}, {str(source): "v1"}
    )

    assert figure.axes[0].lines[0].get_xdata().tolist() == [0, 1, 2]
    assert figure.axes[1].lines[0].get_xdata().tolist() == [0, 1, 2]
    plt.close(figure)


def test_cycle_legend_uses_the_source_display_label() -> None:
    source = Path("/runs/first")
    table = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "candidate_elapsed_minutes": [1],
            "relative_regret": [0.0],
            "inverse_cop": [0.4],
        }
    )
    optima = pd.DataFrame(
        {
            "source": [str(source)],
            "cycle": ["cycle_a"],
            "optimum_elapsed_minutes": [1],
            "interval_start_elapsed_minutes": [1],
            "interval_end_elapsed_minutes": [1],
        }
    )

    figure = cost._cycle_figure(
        "cycle_a",
        [("v1", "unit", table, source)],
        optima,
        {str(source): "C0"},
        {str(source): "v1 (first)"},
    )

    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == ["v1 (first)"]
    plt.close(figure)


def test_cost_plots_keep_duplicate_versions_as_distinct_colored_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _run(tmp_path, "first", "unit", _cost_rows("cycle_a"))
    second = _run(tmp_path, "second", "unit", _cost_rows("cycle_a"))
    for run in (first, second):
        (run / "recipe.json").write_text(
            json.dumps({"base_cost": "v1", "variant": None, "heat_basis": "unit"}),
            encoding="utf-8",
        )
    monkeypatch.setattr(cost, "DatasetLoader", ImageLoader)

    optima = cost.compare_results(
        [first, second], tmp_path / "dataset", tmp_path / "plots", camera="front"
    )

    assert isinstance(optima, pd.DataFrame)
    summary_svg = (tmp_path / "plots/optimum_comparison.svg").read_text(encoding="utf-8")
    cycle_svg = (tmp_path / "plots/cycles/cycle_a.svg").read_text(encoding="utf-8")
    assert "v1 (first)" in summary_svg
    assert "v1 (second)" in summary_svg
    assert "v1 (first)" in cycle_svg
    assert "v1 (second)" in cycle_svg


def test_comparison_offsets_coincident_source_runs() -> None:
    optima = pd.DataFrame(
        {
            "source": ["/runs/first", "/runs/second"],
            "version": ["v1", "v1"],
            "cycle": ["cycle_a", "cycle_a"],
            "optimum_elapsed_minutes": [1.0, 1.0],
            "interval_start_elapsed_minutes": [0.0, 0.0],
            "interval_end_elapsed_minutes": [2.0, 2.0],
        }
    )

    figure = cost._comparison_figure(
        optima,
        {"/runs/first": "C0", "/runs/second": "C1"},
        {"/runs/first": "v1 (first)", "/runs/second": "v1 (second)"},
    )

    xdata = [line.get_xdata().tolist() for line in figure.axes[0].lines]
    assert xdata == [[-0.2], [0.2]]
    plt.close(figure)


def test_comparison_figure_caps_69_cycle_width() -> None:
    cycles = [f"cycle_{index:02}" for index in range(69)]
    optima = pd.DataFrame(
        {
            "source": "/runs/v1",
            "version": "v1",
            "cycle": cycles,
            "optimum_elapsed_minutes": 1.0,
            "interval_start_elapsed_minutes": 0.0,
            "interval_end_elapsed_minutes": 2.0,
        }
    )

    figure = cost._comparison_figure(optima, {"/runs/v1": "C0"}, {"/runs/v1": "v1"})

    assert figure.get_size_inches()[0] <= 14
    assert {label.get_rotation() for label in figure.axes[0].get_xticklabels()} == {90.0}
    plt.close(figure)


def test_label_plot_sums_splits_for_requested_cameras(tmp_path: Path) -> None:
    balance = tmp_path / "label_balance.csv"
    pd.DataFrame(
        {
            "regret_threshold": [0.01] * 6,
            "camera_group": ["front"] * 3 + ["all"] * 3,
            "split": ["train", "validation", "test"] * 2,
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"] * 2,
            "image_count": [3, 4, 5, 6, 7, 8],
        }
    ).to_csv(balance, index=False)

    labels.main(["--label-balance", str(balance), "--output", str(tmp_path / "labels")])

    assert {path.name for path in (tmp_path / "labels").iterdir()} == {
        "label_balance.svg",
        "label_balance.png",
    }


def test_model_plot_shows_full_setting_and_each_valid_fold(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    setting = {
        "source": "/runs/first",
        "representation": "handcrafted",
        "head": "logistic",
        "camera": "front",
        "modality": "rgb",
    }
    pd.DataFrame(
        [
            {**setting, "held_out_experiment": "a", "status": "ok", "macro_f1": 0.50},
            {**setting, "held_out_experiment": "b", "status": "ok", "macro_f1": 0.70},
        ]
    ).to_csv(evaluation / "experiment_metrics.csv", index=False)
    pd.DataFrame([{**setting, "macro_f1_mean": 0.60, "macro_f1_std": 0.14}]).to_csv(
        evaluation / "summary.csv", index=False
    )

    result = model.plot(evaluation, tmp_path / "models")

    assert result is None
    assert {path.name for path in (tmp_path / "models").iterdir()} == {
        "model_macro_f1.svg",
        "model_macro_f1.png",
    }
    model_svg = (tmp_path / "models/model_macro_f1.svg").read_text(encoding="utf-8")
    assert "<text" in model_svg
    assert (
        "source=/runs/first; representation=handcrafted; head=logistic; camera=front; modality=rgb"
        in model_svg
    )


def test_cost_compare_cli_keeps_results_and_accepts_an_rgb_camera() -> None:
    args = main_cost.build_parser().parse_args(
        ["--action", "compare", "--results", "v1", "v2.5", "--camera", "front"]
    )

    assert [str(value) for value in args.results] == ["v1", "v2.5"]
    assert args.camera == "front"


@pytest.mark.parametrize("module", ["plots.labels", "plots.model"])
def test_documented_module_entry_points_show_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
