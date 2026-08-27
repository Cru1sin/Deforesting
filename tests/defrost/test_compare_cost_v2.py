from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/defrost/compare_cost_v2.py")
    spec = importlib.util.spec_from_file_location("compare_cost_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_name": ["event_a1", "event_a2", "event_b1", "event_b2"],
            "experiment_id": ["a", "a", "b", "b"],
            "ambient_temperature": [0.0, 2.0, 0.0, 2.0],
            "evaporating_temperature": [-8.0, -6.0, -8.0, -6.0],
            "coil_temperature": [-7.0, -5.0, -7.0, -5.0],
            "minutes_from_stable": [40.0, 60.0, 40.0, 60.0],
            "electricity_kwh": [0.2, 0.4, 0.3, 0.5],
            "measured_signed_transient_user_heat_kwh": [-0.2, 0.2, -0.1, 0.3],
        }
    )


def _curves() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a", "cycle_b", "cycle_b"],
            "experiment_id": ["a", "a", "b", "b"],
            "candidate_time": pd.to_datetime(
                [
                    "2026-01-01 00:20",
                    "2026-01-01 00:30",
                    "2026-01-02 00:20",
                    "2026-01-02 00:30",
                ]
            ),
            "ambient_temperature": [0.5, 1.5, 0.5, 1.5],
            "evaporating_temperature": [-7.5, -6.5, -7.5, -6.5],
            "coil_temperature": [-6.5, -5.5, -6.5, -5.5],
            "minutes_from_stable": [45.0, 55.0, 45.0, 55.0],
            "heating_electricity_kwh": [0.4, 0.6, 0.4, 0.6],
            "water_heating_kwh": [1.2, 1.6, 1.2, 1.6],
            "unit_heating_kwh": [1.1, 1.5, 1.1, 1.5],
            "inverse_cop_unit": [0.55, 0.50, 0.55, 0.50],
            "optimization_eligible": [True, True, True, True],
        }
    )


def test_build_v2_curves_uses_ticket_energy_and_signed_heat() -> None:
    module = _module()

    result = module.build_v2_curves(_events(), _curves())
    row = result.loc[result["cycle_name"].eq("cycle_a")].iloc[0]

    assert row["v2_total_electricity_kwh"] == pytest.approx(
        row["heating_electricity_kwh"] + row["predicted_ticket_electricity_kwh"]
    )
    assert row["v2_total_water_heat_kwh"] == pytest.approx(
        row["water_heating_kwh"] + row["predicted_ticket_signed_heat_kwh"]
    )
    assert row["inverse_cop_v2"] == pytest.approx(
        row["v2_total_electricity_kwh"] / row["v2_total_water_heat_kwh"]
    )
    assert bool(row["v2_eligible"])


def test_build_v2_curves_holds_out_candidate_experiment_and_rejects_extrapolation() -> None:
    module = _module()
    baseline = module.build_v2_curves(_events(), _curves())
    changed = _events()
    changed.loc[changed["experiment_id"].eq("a"), "electricity_kwh"] = 99.0
    repeated = module.build_v2_curves(changed, _curves())

    baseline_a = baseline.loc[baseline["experiment_id"].eq("a")]
    repeated_a = repeated.loc[repeated["experiment_id"].eq("a")]
    assert baseline_a["predicted_ticket_electricity_kwh"].tolist() == pytest.approx(
        repeated_a["predicted_ticket_electricity_kwh"].tolist()
    )

    outside = _curves()
    outside.loc[outside["cycle_name"].eq("cycle_a"), "ambient_temperature"] = 10.0
    unsupported = module.build_v2_curves(_events(), outside)
    assert not unsupported.loc[
        unsupported["cycle_name"].eq("cycle_a"), "v2_eligible"
    ].any()


def test_compare_points_reports_v2_unit_and_rule_time_differences() -> None:
    module = _module()
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a"],
            "candidate_time": pd.to_datetime(
                ["2026-01-01 00:20", "2026-01-01 00:30"]
            ),
            "inverse_cop_v2": [0.4, 0.5],
            "inverse_cop_v2_zero_transient_heat": [0.6, 0.5],
            "inverse_cop_v2_unit_ablation": [0.7, 0.6],
            "inverse_cop_unit": [0.8, 0.6],
            "v2_eligible": [True, True],
        }
    )
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "t_star_unit": [pd.Timestamp("2026-01-01 00:30")],
            "t_RB": [pd.Timestamp("2026-01-01 00:40")],
            "rb_status": ["triggered"],
            "trigger_type": ["Case1"],
        }
    )

    result = module.compare_points(curves, points).set_index("cycle_name").loc[
        "cycle_a"
    ]

    assert result["t_star_v2"] == pd.Timestamp("2026-01-01 00:20")
    assert result["t_star_unit_original"] == pd.Timestamp("2026-01-01 00:30")
    assert result["t_rule"] == pd.Timestamp("2026-01-01 00:40")
    assert result["unit_minus_v2_minutes"] == pytest.approx(10.0)
    assert result["rule_minus_v2_minutes"] == pytest.approx(20.0)
    assert result["t_star_unit_common_support"] == pd.Timestamp(
        "2026-01-01 00:30"
    )
    assert result["unit_common_minus_v2_minutes"] == pytest.approx(10.0)
    assert result["v2_minimum_location"] == "left_boundary"
    assert result["t_star_v2_zero_transient_heat"] == pd.Timestamp(
        "2026-01-01 00:30"
    )


def test_compare_points_maps_rule_and_unit_regret_on_v2_curve() -> None:
    module = _module()
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a"],
            "candidate_time": pd.to_datetime(
                ["2026-01-01 00:20", "2026-01-01 00:30"]
            ),
            "inverse_cop_v2": [0.4, 0.5],
            "inverse_cop_v2_zero_transient_heat": [0.6, 0.5],
            "inverse_cop_v2_unit_ablation": [0.7, 0.6],
            "inverse_cop_unit": [0.8, 0.6],
            "v2_eligible": [True, True],
        }
    )
    points = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "t_star_unit": [pd.Timestamp("2026-01-01 00:30")],
            "t_RB": [pd.Timestamp("2026-01-01 00:30:20")],
            "rb_status": ["triggered"],
            "trigger_type": ["Case1"],
        }
    )

    result = module.compare_points(curves, points).iloc[0]

    assert result["v2_regret_at_unit_original"] == pytest.approx(0.25)
    assert result["v2_regret_at_rule"] == pytest.approx(0.25)
    assert result["rule_candidate_time"] == pd.Timestamp("2026-01-01 00:30")


def test_prepare_cycle_overlay_uses_stable_time_and_keeps_v2_support_gaps() -> None:
    module = _module()
    stable = pd.Timestamp("2026-01-01 00:00")
    curves = pd.DataFrame(
        {
            "candidate_time": pd.date_range(
                "2026-01-01 00:10", periods=3, freq="10min"
            ),
            "inverse_cop_unit": [0.50, 0.40, 0.45],
            "inverse_cop_v2": [0.55, 0.50, 0.42],
            "v2_eligible": [True, False, True],
        }
    )
    comparison = pd.Series(
        {
            "t_star_v2": pd.Timestamp("2026-01-01 00:30"),
            "t_star_unit_original": pd.Timestamp("2026-01-01 00:20"),
            "t_rule": pd.Timestamp("2026-01-01 00:25"),
            "v2_minimum_location": "right_boundary",
        }
    )

    prepared = module.prepare_cycle_overlay(curves, comparison, stable)

    assert prepared["minutes_from_stable"].tolist() == [10.0, 20.0, 30.0]
    assert prepared["inverse_cop_v2_plot"].isna().tolist() == [False, True, False]
    assert prepared.attrs["v2_optimum_minutes"] == pytest.approx(30.0)
    assert prepared.attrs["unit_optimum_minutes"] == pytest.approx(20.0)
    assert prepared.attrs["rule_minutes"] == pytest.approx(25.0)
    assert prepared.attrs["v2_minimum_location"] == "right_boundary"


def test_analyze_writes_v2_comparison_artifacts(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source"
    evidence = tmp_path / "evidence"
    output = tmp_path / "output"
    source.mkdir()
    evidence.mkdir()
    _curves().to_parquet(evidence / "conditional_candidate_costs.parquet", index=False)
    _events().to_csv(
        evidence / "ticket_event_features_and_predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_b"],
            "t_heating_stable": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-02 00:00"]
            ),
            "t_star_unit": pd.to_datetime(
                ["2026-01-01 00:30", "2026-01-02 00:30"]
            ),
            "t_RB": pd.to_datetime(
                ["2026-01-01 00:30", "2026-01-02 00:30"]
            ),
            "rb_status": ["triggered", "triggered"],
            "trigger_type": ["Case1", "Case1"],
        }
    ).to_csv(source / "cycle_optimal_points.csv", index=False)

    module.analyze(source, evidence, output)

    expected = {
        "源数据/cost_v2_candidate_curves.parquet",
        "源数据/cost_v2_point_comparison.csv",
        "证据/cost_v2_ticket_model_metrics.csv",
        "summary.csv",
        "比较图.png",
        "图表/逐循环成本曲线/cycle_a.png",
        "图表/逐循环成本曲线/cycle_b.png",
        "图表/逐循环成本曲线图集.pdf",
        "报告.md",
    }
    artifacts = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    assert artifacts == expected
    comparison = pd.read_csv(output / "源数据/cost_v2_point_comparison.csv")
    assert set(comparison["cycle_name"]) == {"cycle_a", "cycle_b"}
    assert pd.read_csv(output / "summary.csv")["metric"].notna().all()
    report = (output / "报告.md").read_text(encoding="utf-8")
    assert "联合支持不是四维联合密度支持" in report
    assert "前瞻分歧集随机试验" in report


def test_comparison_summary_uses_only_valid_v2_cycles_for_interior_fraction() -> None:
    module = _module()
    comparison = pd.DataFrame(
        {
            "t_star_v2": pd.to_datetime(["2026-01-01", "2026-01-02", None]),
            "v2_minimum_location": [
                "interior",
                "left_boundary",
                "no_supported_candidate",
            ],
            **{
                column: [0.0, 1.0, None]
                for column in (
                    "unit_minus_v2_minutes",
                    "unit_common_minus_v2_minutes",
                    "rule_minus_v2_minutes",
                    "zero_transient_heat_minus_v2_minutes",
                    "unit_ablation_minus_v2_minutes",
                    "unit_ablation_minus_unit_original_minutes",
                    "v2_regret_at_unit_original",
                    "v2_regret_at_rule",
                )
            },
        }
    )
    curves = pd.DataFrame({"v2_model_supported": [True, False]})

    summary = module.comparison_summary(comparison, curves).set_index("metric")

    assert summary.loc["v2_interior_minimum_fraction", "mean"] == pytest.approx(0.5)
