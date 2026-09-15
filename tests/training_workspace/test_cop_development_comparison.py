import json

import pandas as pd
import pytest


RECIPES = (
    "cop_development_static_sensor",
    "cop_development_static_rgb",
    "cop_development_smooth_sensor",
    "cop_development_smooth_rgb",
    "cop_development_history_sensor",
    "cop_development_history_rgb",
    "cop_development_delta_rgb",
)


def test_compare_development_uses_one_common_clock_and_cycle_mean_bootstrap(tmp_path):
    from image_models.cop_development import compare_development

    keys = pd.DataFrame({
        "cycle_name": ["a", "a", "b", "b", "c", "c"],
        "experiment_id": ["e1", "e1", "e2", "e2", "e2", "e2"],
        "candidate_defrost_time": pd.date_range("2026-01-01", periods=6, freq="30s"),
        "development_fold": ["fold_0"] * 6,
        "binary_target": [0., 1.] * 3,
    })
    runs = []
    for recipe in RECIPES:
        run = tmp_path / recipe
        run.mkdir()
        method = "sensor" if recipe.endswith("sensor") else "rgb"
        probability = [.9, .9, .1, .9, .1, .9] if method == "sensor" else [.1, .9, .1, .9, .1, .9]
        if recipe == "cop_development_delta_rgb":
            probability[-1] = float("nan")
        keys.assign(probability=probability, method=method).to_parquet(
            run / "validation_predictions.parquet", index=False
        )
        pd.DataFrame([{"epoch": 3, "threshold": .5, "validation_bce": .4}]).to_csv(
            run / "selected_configuration.csv", index=False
        )
        run_settings = {
            "method": method,
            "mechanism": "delta" if "delta" in recipe else "static",
            "development_loss": "label-smoothing" if any(
                name in recipe for name in ("smooth", "history", "delta")
            ) else "bce",
            "require_visual_history": "history" in recipe or "delta" in recipe,
        }
        if recipe.startswith("cop_development_static_"):
            run_settings.pop("development_loss")
        (run / "settings.json").write_text(json.dumps(run_settings))
        runs.append(run)

    summary, bootstrap = compare_development(
        runs, tmp_path / "comparison", bootstrap_replicates=50, seed=0
    )

    assert set(summary.recipe_id) == set(RECIPES)
    assert summary.common_labeled_rows.eq(5).all()
    assert summary.full_clock_rows.eq(6).all()
    assert summary.full_clock_cycles.eq(3).all()
    assert summary.common_labeled_cycles.eq(3).all()
    assert summary.loc[summary.recipe_id.eq("cop_development_static_rgb"), "selected"].item()
    static = bootstrap.loc[bootstrap.comparison.eq("static_rgb_minus_sensor")].iloc[0]
    assert static.mean_difference == pytest.approx(1 / 6)
    assert static.paired_cycles == 3
    assert static.experiments == 2
    assert "delta_rgb_minus_history_rgb" in set(bootstrap.comparison)
    assert (tmp_path / "comparison/development_common_comparison_source.csv").exists()
    assert (tmp_path / "comparison/development_common_paired_bootstrap.csv").exists()


def test_cli_exposes_development_comparison_and_refit_actions():
    from train_pareto_boundary import build_parser

    parser = build_parser()
    compared = parser.parse_args(["--action", "compare-development", "--runs", *RECIPES])
    assert compared.action == "compare-development"
    refit = parser.parse_args(["--action", "refit-development"])
    assert refit.action == "refit-development"
    frozen = parser.parse_args(["--action", "evaluate-frozen-development"])
    assert frozen.action == "evaluate-frozen-development"
