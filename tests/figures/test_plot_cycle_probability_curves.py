from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/figures/plot_cycle_probability_curves.py")
    spec = importlib.util.spec_from_file_location("plot_cycle_probability_curves", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_specific_loader_uses_selected_stage_and_skips_incomplete_runs(
    tmp_path: Path,
) -> None:
    module = _module()
    front = tmp_path / "unit_front_boundary_20260827"
    front.mkdir()
    (front / "selected_stage.json").write_text(
        json.dumps({"stage": "head"}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "cycle": ["cycle_1", "cycle_1", "cycle_1"],
            "camera": ["front", "front", "left"],
            "stage": ["head", "finetune", "head"],
            "time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "p1": [0.1, 0.9, 0.8],
        }
    ).to_parquet(front / "test_predictions.parquet")
    pd.DataFrame(
        {"cycle_name": ["cycle_1"], "stable_heating_start": ["2026-01-01"]}
    ).to_parquet(front / "manifest.parquet")

    incomplete = tmp_path / "unit_left_selected_20260827"
    incomplete.mkdir()
    (incomplete / "selected_stage.json").write_text(
        json.dumps({"stage": "finetune"}), encoding="utf-8"
    )
    pd.DataFrame({"camera": ["left"], "stage": ["finetune"]}).to_parquet(
        incomplete / "test_predictions.parquet"
    )

    predictions, manifest, cameras = module.load_camera_specific_runs(tmp_path)

    assert cameras == ("front",)
    assert predictions[["camera", "selected_stage", "model_run", "p1"]].to_dict("records") == [
        {
            "camera": "front",
            "selected_stage": "head",
            "model_run": "unit_front_boundary_20260827",
            "p1": 0.1,
        }
    ]
    assert manifest.index.tolist() == ["cycle_1"]


def _plot_inputs() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    times = pd.date_range("2026-01-01", periods=3, freq="min")
    probabilities = pd.DataFrame(
        {
            "camera": ["front"] * 3,
            "time": times,
            "p1": [0.4, 0.6, 0.7],
            "stage": ["head"] * 3,
            "selected_stage": ["head"] * 3,
            "model_run": ["unit_front_boundary_20260827"] * 3,
        }
    )
    boundary = pd.Series({"stable_heating_start": times[0]})
    point = pd.Series({"t_star_unit": times[1]})
    near = pd.DataFrame({"candidate_time": [times[0], times[2]]})
    return probabilities, boundary, point, near


def test_default_event_schema_stays_unchanged(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    module = _module()
    monkeypatch.setattr(module, "save", lambda fig, stem: plt.close(fig))
    probabilities, boundary, point, near = _plot_inputs()

    plotted, event = module.plot_cycle(
        "frost_cycle_000001",
        probabilities.drop(columns=["model_run", "selected_stage"]),
        boundary,
        point,
        near,
        cameras=("front",),
        output_dir=tmp_path,
    )

    assert plotted.columns.tolist() == [
        "cycle",
        "camera",
        "time",
        "frost_minutes",
        "p1",
        "rolling_min_3",
        "phase",
    ]
    assert set(event) == {
        "cycle",
        "stable_heating_start",
        "optimal_point_minutes",
        "near_optimum_start_minutes",
        "near_optimum_end_minutes",
        "front_three_frame_trigger_minutes",
        "threshold",
        "rolling_rule",
        "model",
        "heat_basis",
        "split",
    }
    assert event["model"] == "ResNet50 finetune checkpoint"


def test_camera_specific_source_and_event_keep_selected_stage_provenance(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    module = _module()
    monkeypatch.setattr(module, "save", lambda fig, stem: plt.close(fig))
    probabilities, boundary, point, near = _plot_inputs()

    plotted, event = module.plot_cycle(
        "frost_cycle_000001",
        probabilities,
        boundary,
        point,
        near,
        cameras=("front",),
        output_dir=tmp_path,
        camera_specific=True,
    )

    assert plotted[["stage", "selected_stage", "model_run"]].drop_duplicates().to_dict(
        "records"
    ) == [
        {
            "stage": "head",
            "selected_stage": "head",
            "model_run": "unit_front_boundary_20260827",
        }
    ]
    assert event["model"] == "ResNet50 camera-specific checkpoints; selected stages: front=head"
    assert event["camera_count"] == 1
    assert event["cameras"] == "front"
