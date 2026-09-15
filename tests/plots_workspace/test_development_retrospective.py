import numpy as np
import pandas as pd
import pytest


def test_retrospective_renderer_averages_seeds_before_cluster_bootstrap(
    tmp_path, monkeypatch
):
    from plots import pareto_learning

    cycles = []
    for recipe, method in (("history_sensor", "sensor"), ("delta_rgb", "rgb")):
        for seed in (0, 1):
            for index in range(5):
                evaluated = index < 3
                value = .5 if evaluated else np.nan
                if recipe == "delta_rgb" and seed == 0 and index == 0:
                    value = 1.
                cycles.append({
                    "recipe_id": recipe,
                    "method": method,
                    "seed": seed,
                    "outer_experiment": "e1" if index == 0 else "e2",
                    "experiment_id": "e1" if index == 0 else "e2",
                    "cycle_name": f"cycle_{index:03d}",
                    "status": (
                        "evaluated" if evaluated else
                        "no_supported_labels" if index == 3 else "no_prediction"
                    ),
                    "balanced_accuracy": value,
                    "macro_f1": value,
                    "both_classes": evaluated,
                })
    exported = []
    monkeypatch.setattr(
        pareto_learning, "_export",
        lambda figure, path, formats=("png",): exported.append(
            (figure, path, formats)
        ),
    )

    summary, bootstrap = pareto_learning.render_development_retrospective(
        pd.DataFrame(cycles), tmp_path, bootstrap_replicates=50, seed=0
    )

    assert summary.loc[summary.scope.eq("seed"), "cohort_cycles"].eq(5).all()
    assert summary.loc[summary.scope.eq("seed"), "evaluated_cycles"].eq(3).all()
    assert summary.loc[summary.scope.eq("seed"), "no_supported_labels_cycles"].eq(1).all()
    assert summary.loc[summary.scope.eq("seed"), "no_prediction_cycles"].eq(1).all()
    pooled = summary.loc[
        summary.scope.eq("pooled") & summary.recipe_id.eq("delta_rgb")
    ].iloc[0]
    assert pooled.balanced_accuracy == pytest.approx(.5 + .25 / 3)
    assert bootstrap.mean_difference.item() == pytest.approx(.25 / 3)
    assert bootstrap.paired_cycles.item() == 3
    assert bootstrap.experiments.item() == 2
    figure, path, formats = exported[0]
    assert path == tmp_path / "development_frozen_retrospective"
    assert formats == ("png", "pdf", "svg")
    annotations = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert figure.axes[0].get_xticks().tolist() == [0, 1]
    assert "5 cycles" in annotations
    assert "3 evaluated" in annotations
    assert "3 paired cycles" in annotations
    assert "2 seeds/cycle" in annotations


def test_frozen_policy_cop_renderer_averages_seeds_and_shows_unscoreable(tmp_path, monkeypatch):
    from plots import pareto_learning

    policies = (
        ("delta_rgb", "Delta RGB", "current_frozen", (0, 1)),
        ("history_sensor", "History Sensor", "current_frozen", (0, 1)),
        ("legacy_after_optimum_sensor", "Legacy Sensor", "legacy_frozen", (None,)),
        ("legacy_after_optimum_rgb", "Legacy RGB", "legacy_frozen", (None,)),
        ("chen_dinov2_binary", "Chen DINOv2", "legacy_frozen", (None,)),
        ("rb", "Fixed RB", "baseline", (None,)),
    )
    rows = []
    for policy_id, name, family, seeds in policies:
        for seed in seeds:
            for index, (status, scope) in enumerate((
                ("scored", "supported"),
                ("before_reference_accounting_start", "unscored"),
                ("trigger_outside_reference_support", "extrapolated"),
                ("trigger_outside_reference_support", "extrapolated"),
            )):
                gain = np.nan
                if scope == "supported":
                    gain = 10 + (4 if policy_id == "delta_rgb" and seed == 1 else 0)
                elif scope == "extrapolated":
                    gain = -5 + (2 if policy_id == "delta_rgb" and seed == 1 else 0)
                    if index == 3:
                        gain = (
                            100 if policy_id == "delta_rgb" and seed == 0
                            else np.nan if policy_id == "delta_rgb"
                            else -2
                        )
                rows.append({
                    "policy_id": policy_id, "policy_name": name,
                    "policy_family": family, "seed": seed,
                    "cycle_name": f"cycle_{index}", "experiment_id": "experiment",
                    "status": status, "estimate_scope": scope,
                    "outside_reference_support": scope == "extrapolated",
                    "cop_gain_vs_rb_pct": gain if scope == "supported" else np.nan,
                    "formal_reference_headroom_vs_rb_pct": (
                        20 if scope == "supported" else np.nan
                    ),
                    "sensitivity_cop_gain_vs_rb_pct": (
                        gain if scope == "extrapolated" else np.nan
                    ),
                    "sensitivity_reference_headroom_vs_rb_pct": (
                        -2 if scope == "extrapolated" else np.nan
                    ),
                })
    exported = []
    monkeypatch.setattr(
        pareto_learning, "_export",
        lambda figure, path, formats=("png",): exported.append((figure, path, formats)),
    )

    summary = pareto_learning.render_frozen_policy_cop_comparison(
        pd.DataFrame(rows), tmp_path
    )

    formal = summary.loc[
        summary.policy_id.eq("delta_rgb")
        & summary.evaluation_scope.eq("formal_supported_only")
        & summary.aggregation.eq("own_complete_seeds")
    ].iloc[0]
    sensitivity = summary.loc[
        summary.policy_id.eq("delta_rgb")
        & summary.evaluation_scope.eq("reliable_extrapolation_sensitivity")
        & summary.aggregation.eq("own_complete_seeds")
    ].iloc[0]
    assert formal.seed_count == 2
    assert formal.cohort_cycles == 4
    assert formal.before_accounting_cycles_per_seed == pytest.approx(1)
    assert formal.mean_cop_gain_pct == pytest.approx(12)
    assert sensitivity.mean_cop_gain_pct == pytest.approx(-4)
    assert sensitivity.gain_paired_cycles == 1
    assert sensitivity.headroom_paired_cycles == 2
    common = summary.loc[
        summary.policy_id.eq("delta_rgb")
        & summary.evaluation_scope.eq("reliable_extrapolation_sensitivity")
        & summary.aggregation.eq("common_complete_seeds")
    ].iloc[0]
    assert common.gain_paired_cycles == 1
    assert common.mean_cop_gain_pct == pytest.approx(-4)
    assert formal.mean_remaining_gap_on_rb_denominator_pp == pytest.approx(8)
    assert pd.isna(sensitivity.mean_remaining_gap_on_rb_denominator_pp)
    assert formal.point_support_scope == "ridge_domain_supported_points"
    assert sensitivity.point_support_scope == (
        "reliable_points_with_ridge_domain_relaxed"
    )
    figure, path, formats = exported[0]
    assert len(figure.axes) == 3
    assert figure.axes[1].get_title() == "All-policy common complete cycles"
    assert "joint n=1" in " ".join(text.get_text() for text in figure.axes[2].texts)
    assert path == tmp_path / "frozen_policy_cop_comparison"
    assert formats == ("png", "pdf", "svg")
    assert (tmp_path / "frozen_policy_cop_comparison_source.csv").exists()
