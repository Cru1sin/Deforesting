from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def _load_plotter():
    path = Path("scripts/figures/plot_camera_specific_model_comparison.py")
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (FileNotFoundError, AttributeError) as error:
        pytest.fail(f"camera comparison implementation is missing: {error}")


def _predictions(paths: list[str], predictions: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": paths,
            "camera": ["left"] * len(paths),
            "relative_regret": [0.2] * len(paths),
            "target": [0, 1][: len(paths)],
            "prediction": predictions,
            "p1": [0.1, 0.9][: len(paths)],
            "cycle": ["cycle_1", "cycle_2"][: len(paths)],
            "experiment": ["experiment_1"] * len(paths),
        }
    )


def test_align_predictions_rejects_mismatched_image_keys() -> None:
    plotter = _load_plotter()
    baseline = _predictions(["a.jpg", "b.jpg"], [0, 1])
    dedicated = _predictions(["a.jpg", "c.jpg"], [0, 1])

    with pytest.raises(ValueError, match="image keys"):
        plotter._align_predictions(baseline, dedicated)


def test_comparison_delta_is_dedicated_minus_baseline() -> None:
    plotter = _load_plotter()
    baseline = _predictions(["a.jpg", "b.jpg"], [0, 0])
    dedicated = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted = _predictions(["a.jpg", "b.jpg"], [1, 1])

    rows = plotter._comparison_rows("left", baseline, dedicated, adapted)
    macro = next(row for row in rows if row["scope"] == "full" and row["metric"] == "macro_f1")

    assert macro["baseline"] == pytest.approx(1 / 3)
    assert macro["dedicated"] == pytest.approx(1.0)
    assert macro["delta"] == pytest.approx(2 / 3)
    assert macro["adapted"] == pytest.approx(1 / 3)
    assert macro["adapted_minus_baseline"] == pytest.approx(0.0)
    assert macro["adapted_minus_dedicated"] == pytest.approx(-2 / 3)
    assert macro["n_images"] == 2
    assert macro["n_cycles"] == 2
    assert macro["n_experiments"] == 1
    assert "n" not in macro


def test_read_selected_predictions_uses_selected_stage_json(tmp_path) -> None:
    plotter = _load_plotter()
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected_stage.json").write_text(json.dumps({"stage": "head"}))
    predictions = pd.DataFrame(
        {
            "image_path": ["head.jpg", "finetune.jpg"],
            "stage": ["head", "finetune"],
            "target": [0, 1],
            "prediction": [0, 1],
            "p1": [0.1, 0.9],
            "camera": ["left", "left"],
            "relative_regret": [0.2, 0.2],
            "cycle": ["cycle_1", "cycle_2"],
            "experiment": ["experiment_1", "experiment_1"],
        }
    )
    predictions.to_parquet(run / "test_predictions.parquet")

    selected, stage = plotter._read_selected_predictions(run)

    assert stage == "head"
    assert selected["image_path"].tolist() == ["head.jpg"]


def test_formal_schedule_rejects_exploratory_one_plus_one_run() -> None:
    plotter = _load_plotter()

    with pytest.raises(ValueError, match="non-formal training schedule"):
        plotter._require_formal_schedule(
            {"head_epochs": 1, "finetune_epochs": 1},
            "unit_front_boundary_lr1e4_20260827",
        )


def test_empty_near_scope_returns_nan_without_warnings(recwarn) -> None:
    plotter = _load_plotter()
    predictions = _predictions(["a.jpg", "b.jpg"], [0, 1])

    rows = plotter._comparison_rows("left", predictions, predictions, predictions)

    near = [row for row in rows if row["scope"] == "near_1pct"]
    assert all(pd.isna(row["baseline"]) for row in near)
    assert not recwarn


def test_comparison_strictly_aligns_all_three_prediction_tables() -> None:
    plotter = _load_plotter()
    baseline = _predictions(["a.jpg", "b.jpg"], [0, 1])
    dedicated = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted = _predictions(["a.jpg", "c.jpg"], [0, 1])

    with pytest.raises(ValueError, match="image keys"):
        plotter._comparison_rows("left", baseline, dedicated, adapted)


def test_comparison_uses_adapted_regret_for_near_scope() -> None:
    plotter = _load_plotter()
    baseline = _predictions(["a.jpg", "b.jpg"], [0, 0])
    dedicated = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted["relative_regret"] = [0.005, 0.2]

    rows = plotter._comparison_rows("left", baseline, dedicated, adapted)

    near = [row for row in rows if row["scope"] == "near_1pct"]
    assert all(row["n_images"] == 1 for row in near)
    assert next(row for row in near if row["metric"] == "macro_f1")["adapted"] == 0.5


def test_comparison_rejects_target_difference_despite_regret_version_change() -> None:
    plotter = _load_plotter()
    baseline = _predictions(["a.jpg", "b.jpg"], [0, 1])
    dedicated = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted = _predictions(["a.jpg", "b.jpg"], [0, 1])
    adapted["relative_regret"] = [0.005, 0.2]
    adapted["target"] = [1, 1]

    with pytest.raises(ValueError, match="target differs"):
        plotter._comparison_rows("left", baseline, dedicated, adapted)


def test_metrics_keep_missing_binary_class_in_macro_f1(recwarn) -> None:
    plotter = _load_plotter()
    predictions = pd.DataFrame(
        {"target": [0, 0], "prediction": [0, 0], "p1": [0.1, 0.2]}
    )

    scores = plotter._metrics(predictions)

    assert scores["class0_f1"] == 1.0
    assert scores["class1_f1"] == 0.0
    assert scores["macro_f1"] == 0.5
    assert not recwarn


def test_pair_breakdown_counts_images_cycles_and_experiments() -> None:
    plotter = _load_plotter()
    predictions = _predictions(["a.jpg", "b.jpg"], [0, 1])

    rows = plotter._pair_rows(
        "left_pair", "left", predictions, predictions, predictions
    )

    macro = next(row for row in rows if row["scope"] == "full" and row["metric"] == "macro_f1")
    assert (macro["n_images"], macro["n_cycles"], macro["n_experiments"]) == (2, 2, 1)


def test_pair_breakdown_still_rejects_regret_difference() -> None:
    plotter = _load_plotter()
    mixed = _predictions(["a.jpg", "b.jpg"], [0, 1])
    pair = mixed.copy()
    pair["relative_regret"] = [0.005, 0.2]

    with pytest.raises(ValueError, match="relative_regret differs"):
        plotter._pair_rows("left_pair", "left", mixed, mixed, pair)


def test_write_outputs_exports_tables_and_three_figure_bundles(tmp_path) -> None:
    plotter = _load_plotter()
    comparison = pd.DataFrame(
        [
            {
                "camera": camera,
                "scope": scope,
                "metric": metric,
                "baseline": 0.80,
                "dedicated": 0.81,
                "delta": 0.01,
                "adapted": 0.82,
                "adapted_minus_baseline": 0.02,
                "adapted_minus_dedicated": 0.01,
                "n_images": 20,
                "n_cycles": 8,
                "n_experiments": 3,
            }
            for camera in plotter.CAMERAS
            for scope in ("full", "near_1pct")
            for metric in plotter.METRICS
        ]
    )
    pair = pd.DataFrame(
        [
            {
                "pair_group": "left_pair" if camera.startswith("left") else "top_pair",
                "camera": camera,
                "scope": scope,
                "metric": metric,
                "mixed": 0.80,
                "single_dedicated": 0.81,
                "pair": 0.805,
                "pair_minus_mixed": 0.005,
                "pair_minus_single": -0.005,
                "n_images": 20,
                "n_cycles": 8,
                "n_experiments": 3,
            }
            for camera in ("left", "left_close", "top", "top_close")
            for scope in ("full", "near_1pct")
            for metric in plotter.METRICS
        ]
    )
    summary = pd.DataFrame(
        {
            "run": [f"run_{i}" for i in range(8)],
            "selection_basis": [plotter.FORMAL_SELECTION_BASIS] * 8,
            "eligibility_rule": [plotter.ELIGIBILITY_RULE] * 8,
        }
    )

    plotter._write_outputs(comparison, pair, summary, tmp_path)

    source = tmp_path / "源数据"
    figures = tmp_path / "图表"
    for name in (
        "camera_specific_comparison.csv",
        "pair_model_breakdown.csv",
        "selected_model_summary.csv",
        "figure_camera_specific_macro_f1_delta.csv",
        "figure_camera_specific_near_1pct_macro_f1_delta.csv",
        "figure_pair_model_macro_f1_comparison.csv",
    ):
        table = pd.read_csv(source / name)
        assert table.shape[0] > 0
        if name != "selected_model_summary.csv":
            assert {"n_images", "n_cycles", "n_experiments"}.issubset(table.columns)
            assert "n" not in table.columns
    selected_summary = pd.read_csv(source / "selected_model_summary.csv")
    assert selected_summary["selection_basis"].eq(
        "validation-selected stage within predeclared formal run"
    ).all()
    assert selected_summary["eligibility_rule"].str.contains("head_epochs=5").all()
    assert selected_summary["eligibility_rule"].str.contains("finetune_epochs=5").all()
    assert selected_summary["eligibility_rule"].str.contains(
        "unit_front_boundary_lr1e4_20260827"
    ).all()
    assert selected_summary["eligibility_rule"].str.contains(r"1\+1", regex=True).all()
    for stem in (
        "figure_camera_specific_macro_f1_delta",
        "figure_camera_specific_near_1pct_macro_f1_delta",
        "figure_pair_model_macro_f1_comparison",
    ):
        for suffix in (".svg", ".pdf", ".png"):
            assert (figures / f"{stem}{suffix}").is_file()
        svg = (figures / f"{stem}.svg").read_text()
        assert "n=image frames" in svg
        assert "8 test cycles" in svg and "3 held-out experiments" in svg
        assert "adjacent frames are not independent replicates" in svg
        assert "no seed/fold CI" in svg
        if "near_1pct" in stem:
            assert "1% near-optimal subset" in svg
            assert "relative_regret" in svg and "0.01" in svg
        if "camera_specific" in stem:
            assert "Dedicated" in svg and "Adapted" in svg
    full_width = Image.open(figures / "figure_camera_specific_macro_f1_delta.png").width
    near_width = Image.open(
        figures / "figure_camera_specific_near_1pct_macro_f1_delta.png"
    ).width
    assert near_width <= 1.1 * full_width
    full_height = Image.open(figures / "figure_camera_specific_macro_f1_delta.png").height
    near_height = Image.open(
        figures / "figure_camera_specific_near_1pct_macro_f1_delta.png"
    ).height
    assert near_height >= 1.1 * full_height


def _write_selected_run(run: Path, *, adapted: bool) -> None:
    run.mkdir(parents=True)
    (run / "selected_stage.json").write_text(
        json.dumps(
            {
                "stage": "adapt" if adapted else "finetune",
                "checkpoint": "best_adapt.pt" if adapted else "best_finetune.pt",
                "validation_macro_f1": 0.8,
                "near_1pct_validation_macro_f1": 0.7,
            }
        )
    )
    stage = "adapt" if adapted else "finetune"
    pd.DataFrame(
        {
            "stage": [stage] * 4,
            "split": ["validation", "near_1pct_validation", "test", "near_1pct_test"],
            "macro_f1": [0.8, 0.7, 0.75, 0.65],
        }
    ).to_csv(run / "stage_metrics.csv", index=False)
    config = {"elapsed_seconds": 3600}
    if adapted:
        config.update(
            init_checkpoint="mixed.pt", adapt_epochs=3, adapt_lr=2e-5
        )
    else:
        config.update(
            head_epochs=5,
            finetune_epochs=5,
            init_checkpoint=None,
            adapt_epochs=3,
            adapt_lr=2e-5,
        )
    (run / "config.json").write_text(json.dumps(config))


def test_selected_summary_appends_adapted_model_without_formal_schedule_rule(
    tmp_path,
) -> None:
    plotter = _load_plotter()
    camera_root, adaptation_root = tmp_path / "camera", tmp_path / "adaptation"
    plotter.DEDICATED_RUNS = {"left": "dedicated"}
    plotter.PAIR_RUNS = {}
    plotter.ADAPTED_RUNS = {"left": "unit_left_adapted_20260827"}
    _write_selected_run(camera_root / "dedicated", adapted=False)
    _write_selected_run(adaptation_root / "unit_left_adapted_20260827", adapted=True)

    summary = plotter._selected_summary(camera_root, adaptation_root)

    assert summary["model_family"].tolist() == [
        "from_scratch_dedicated",
        "mixed_to_camera_adapted",
    ]
    assert summary.iloc[0][["init_checkpoint", "adapt_epochs", "adapt_lr"]].eq("").all()
    adapted = summary.iloc[1]
    assert adapted["init_checkpoint"] == "mixed.pt"
    assert adapted["adapt_epochs"] == 3
    assert adapted["adapt_lr"] == pytest.approx(2e-5)
    assert "head_epochs=5" not in adapted["eligibility_rule"]
