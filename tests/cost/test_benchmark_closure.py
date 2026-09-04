from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frost_analysis.cost.benchmark import (
    absolute_rate_metric_tables,
    benchmark_table,
    bootstrap_absolute_rate_trajectories,
    bootstrap_failure_anatomy,
    bootstrap_fixed_support_stability,
    bootstrap_ho_cofailure,
    bootstrap_stability,
    bootstrap_validity_taxonomy,
    ch_high_value_overlap,
    ch_tradeoff_diagnostic,
    cross_objective_regret,
    cycle_trigger_validation,
    experiment_leverage,
    ho_paired_decisions,
    local_ratio_attribution,
    matched_decision_regret,
    outdoor_event_model_ablation,
    pareto_nondominated,
    regret_coverage,
    regret_distribution,
    same_cycle_regret,
    stability_to_basin_ratio,
)
from frost_analysis.cost.outcome import DYNAMIC_8


def test_benchmark_derives_boundary_location_from_native_eligible_domain() -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": pd.date_range(start, periods=3, freq="min"),
            "t_star": start,
            "objective_value": [1.0, 0.9, 0.8],
            "optimization_eligible": [True, True, False],
            "pre_action_window_valid": True,
        }
    )

    result = benchmark_table({"C": curve})

    assert result.loc[0, "extreme_location"] == "left_boundary"


def test_regret_coverage_uses_selector_decisions_as_denominator() -> None:
    regret = pd.DataFrame(
        {
            "cycle_name": ["a", "b", "a"],
            "selector_metric": ["C", "C", "C"],
            "target_metric": ["C", "C", "H"],
            "decision_type": ["point", "point", "point"],
        }
    )

    result = regret_coverage(regret).set_index("target_metric")

    assert result.loc["C", "available_cycles"] == 2
    assert result.loc["H", "available_cycles"] == 1
    assert result.loc["H", "coverage_fraction"] == 0.5


def test_matched_decision_regret_requires_every_selector_target_pair() -> None:
    rows = [
        {
            "cycle_name": cycle,
            "selector_metric": selector,
            "target_metric": target,
            "decision_type": "point",
            "cross_objective_regret": 0.0,
        }
        for cycle in ("complete", "incomplete")
        for selector in ("C", "H", "O")
        for target in ("C", "H", "O")
        if not (cycle == "incomplete" and selector == "O" and target == "H")
    ]

    result = matched_decision_regret(pd.DataFrame(rows), "point", ("C", "H", "O"))

    assert set(result["cycle_name"]) == {"complete"}
    assert len(result) == 9


def test_pareto_nondominated_uses_only_consequence_dimensions() -> None:
    consequences = pd.DataFrame(
        {
            "r_C": [0.0, 0.05, 0.10],
            "r_H": [0.10, 0.0, 0.10],
            "r_O": [0.10, 0.05, 0.10],
        },
        index=["C", "H", "O"],
    )

    result = pareto_nondominated(consequences)

    assert result.to_dict() == {"C": True, "H": True, "O": False}


def test_bootstrap_taxonomy_separates_formula_support_and_location() -> None:
    rows = []
    for cycle, values, eligible in (
        ("formula", [float("nan"), float("nan")], [False, False]),
        ("support", [1.0, 1.1], [False, False]),
        ("endpoint", [2.0, 1.0], [True, True]),
        ("interior", [1.0, 2.0, 1.0], [True, True, True]),
    ):
        for minute, (value, valid) in enumerate(zip(values, eligible, strict=True)):
            rows.append(
                {
                    "replicate_id": 0,
                    "cycle_name": cycle,
                    "candidate_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(minutes=minute),
                    "C": value,
                    "C_eligible": valid,
                }
            )

    result = bootstrap_validity_taxonomy(pd.DataFrame(rows), ("C",))

    assert dict(zip(result["cycle_name"], result["status"], strict=True)) == {
        "formula": "formula_unavailable",
        "support": "support_or_measurement_limited",
        "endpoint": "valid_endpoint",
        "interior": "valid_interior",
    }


def test_bootstrap_anatomy_preserves_shared_and_objective_native_failures() -> None:
    rows = []
    for replicate, q_t, p_comp in ((0, False, True), (1, True, False), (2, True, True)):
        for minute in range(2):
            rows.append(
                {
                    "replicate_id": replicate,
                    "cycle_name": "cycle",
                    "experiment_id": "heldout",
                    "candidate_time": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(minutes=minute),
                    "eta_h_cyc": 1.0 + minute,
                    "eta_e_cyc": 1.0 + minute,
                    "support_Q_T": q_t,
                    "support_Qw0": True,
                    "support_D_T": True,
                    "support_Pcomp0": p_comp,
                    "support_E_comp_T": True,
                    "eta_h_cyc_model_supported": q_t,
                    "eta_e_cyc_model_supported": q_t and p_comp,
                    "eta_h_cyc_measurement_eligible": True,
                    "eta_e_cyc_measurement_eligible": True,
                    "eta_h_cyc_physical_valid": True,
                    "eta_e_cyc_physical_valid": True,
                    "eta_h_cyc_base_eligible": q_t,
                    "eta_e_cyc_base_eligible": q_t and p_comp,
                    "eta_h_cyc_eligible": q_t,
                    "eta_e_cyc_eligible": q_t and p_comp,
                }
            )

    anatomy = bootstrap_failure_anatomy(pd.DataFrame(rows))
    cofailure = bootstrap_ho_cofailure(anatomy).set_index("statistic")["value"]
    draws = pd.DataFrame(
        {
            "replicate_id": [0, 1, 2],
            "heldout_experiment_id": "heldout",
            "source_experiment_id": "source",
            "draw_count": [0, 1, 1],
        }
    )
    leverage = experiment_leverage(anatomy, draws)

    failures = anatomy.loc[~anatomy["valid"], ["replicate_id", "metric_id", "failure_reason"]]
    assert (0, "eta_h_cyc", "Q_T_support") in set(failures.itertuples(index=False, name=None))
    assert (1, "eta_e_cyc", "Pcomp0_support") in set(
        failures.itertuples(index=False, name=None)
    )
    assert cofailure["P(H_invalid|O_invalid)"] == pytest.approx(0.5)
    assert cofailure["P(O_invalid|H_invalid)"] == pytest.approx(1.0)
    h = leverage.loc[leverage.metric_id.eq("eta_h_cyc")].iloc[0]
    assert h["leverage"] == pytest.approx(1.0)


def test_fixed_support_and_same_cycle_comparisons_use_one_frozen_subset() -> None:
    start = pd.Timestamp("2026-01-01")
    trajectory = pd.DataFrame(
        {
            "replicate_id": [0, 0, 1, 1],
            "cycle_name": "cycle",
            "candidate_time": [start, start + pd.Timedelta("1min")] * 2,
            "cop_cyc_evt": [1.0, 2.0, 2.0, 1.0],
        }
    )
    point = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": [start, start + pd.Timedelta("1min")],
            "t_star": start + pd.Timedelta("1min"),
            "optimization_eligible": [True, True],
        }
    )
    stability = bootstrap_fixed_support_stability(
        trajectory, {"cop_cyc_evt": point}, metrics=("cop_cyc_evt",)
    )
    regret = pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "selector_metric": selector,
                "target_metric": target,
                "decision_type": decision,
                "cross_objective_regret": 0.01,
            }
            for cycle in ("complete", "point_only")
            for decision in ("point", "latest_W1", "latest_W2", "latest_W5")
            for selector in ("C", "H", "O")
            for target in ("C", "H", "O")
            if cycle == "complete" or decision == "point"
        ]
    )

    common = same_cycle_regret(regret, metric_order=("C", "H", "O"))

    assert stability.loc[0, "valid_fraction"] == 1
    assert stability.loc[0, "IQR_tau_minutes"] == pytest.approx(0.5)
    assert set(common["cycle_name"]) == {"complete"}


def test_paired_ho_and_regret_tail_statistics_are_cycle_level() -> None:
    regret = pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "selector_metric": selector,
                "target_metric": target,
                "decision_type": "point",
                "decision_time": pd.Timestamp("2026-01-01")
                + pd.Timedelta(minutes={"eta_h_cyc": 10, "eta_e_cyc": 12}.get(selector, 8)),
                "cross_objective_regret": {
                    ("eta_h_cyc", "cop_cyc_evt"): 0.004,
                    ("eta_e_cyc", "cop_cyc_evt"): 0.003,
                    ("eta_h_cyc", "eta_e_cyc"): 0.009,
                    ("eta_e_cyc", "eta_h_cyc"): 0.008,
                }.get((selector, target), 0.0),
            }
            for cycle in ("a", "b")
            for selector in ("cop_cyc_evt", "eta_h_cyc", "eta_e_cyc")
            for target in ("cop_cyc_evt", "eta_h_cyc", "eta_e_cyc")
        ]
    )
    paired = ho_paired_decisions(regret)
    tails = regret_distribution(regret)
    benchmark = pd.DataFrame(
        {"cycle_name": ["a"], "metric_id": ["eta_h_cyc"], "W5_minutes": [20.0]}
    )
    stability = pd.DataFrame(
        {"cycle_name": ["a"], "metric_id": ["eta_h_cyc"], "IQR_tau_minutes": [10.0]}
    )

    rho = stability_to_basin_ratio(benchmark, stability)

    assert paired.loc[0, "delta_t_O_minus_H_minutes"] == 2
    assert paired.loc[0, "delta_C_regret_O_minus_H"] == pytest.approx(-0.001)
    assert paired.loc[0, "H_in_O_W1"]
    assert paired.loc[0, "O_in_H_W1"]
    assert tails.loc[
        tails.selector_metric.eq("eta_h_cyc") & tails.target_metric.eq("cop_cyc_evt"),
        "p90_regret",
    ].iloc[0] == pytest.approx(0.004)
    assert rho.loc[0, "rho_IQR_over_W5"] == pytest.approx(0.5)


def test_ho_family_figure_uses_current_matplotlib_boxplot_api() -> None:
    path = Path(__file__).parents[2] / "scripts/cost/benchmark.py"
    spec = importlib.util.spec_from_file_location("cost_benchmark_figure", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paired = pd.DataFrame(
        {
            "delta_t_O_minus_H_minutes": [1.0, 2.0],
            "C_regret_at_H": [0.01, 0.02],
            "C_regret_at_O": [0.02, 0.01],
            "H_regret_at_O": [0.01, 0.02],
            "O_regret_at_H": [0.02, 0.01],
            **{
                f"{direction}_W{percent}": [True, False]
                for percent in (1, 2, 5)
                for direction in ("H_in_O", "O_in_H")
            },
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = module._ho_family_figure(paired)

    assert len(figure.axes) == 4


def test_absolute_rate_ablation_removes_only_healthy_counterfactual_support() -> None:
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=7, freq="min")
    common = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": times,
            "stable_start_fixed9": start,
            "pre_action_window_valid": True,
            "water_heating_kwh": range(1, 8),
            "evaporator_heating_kwh": [0.8 * value for value in range(1, 8)],
            "Q_T_hat_kwh": 0.1,
            "E_comp_T_hat_kwh": 0.02,
            "D_T_hat_minutes": 1.0,
            "Q_T_supported": True,
            "D_T_supported": True,
            "E_comp_T_supported": True,
            "water_heating_measurement_eligible": True,
            "heating_compressor_measurement_eligible": True,
        }
    )
    unsupported = common.assign(cycle_name="unsupported", Q_T_supported=False)
    point_source = pd.concat([common, unsupported], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        absolute = absolute_rate_metric_tables(
            {
                "eta_h_cyc": point_source.assign(metric_id="eta_h_cyc"),
                "eta_e_cyc": point_source.assign(metric_id="eta_e_cyc"),
            }
        )
    trajectory = common.rename(
        columns={
            "Q_T_hat_kwh": "Q_T",
            "E_comp_T_hat_kwh": "E_comp_T",
            "D_T_hat_minutes": "D_T",
        }
    ).assign(
        replicate_id=0,
        heating_compressor_electricity_kwh=0.2,
        support_Q_T=True,
        support_Qw0=False,
        support_D_T=True,
        support_Pcomp0=False,
        support_E_comp_T=True,
        eta_h_cyc_measurement_eligible=True,
        eta_e_cyc_measurement_eligible=True,
    )
    bootstrap = bootstrap_absolute_rate_trajectories(trajectory)
    stability = bootstrap_stability(
        bootstrap, absolute, metrics=("h_abs_rate", "o_abs_rate")
    )
    ablation_regret = cross_objective_regret(
        {
            "cop_cyc_evt": absolute["h_abs_rate"],
            **absolute,
        },
        metric_order=("cop_cyc_evt", "h_abs_rate", "o_abs_rate"),
    )
    paired = ho_paired_decisions(
        ablation_regret,
        h_metric="h_abs_rate",
        o_metric="o_abs_rate",
        metric_order=("cop_cyc_evt", "h_abs_rate", "o_abs_rate"),
    )

    assert absolute["h_abs_rate"].loc[
        absolute["h_abs_rate"].cycle_name.eq("cycle"), "t_star"
    ].notna().all()
    assert absolute["o_abs_rate"].loc[
        absolute["o_abs_rate"].cycle_name.eq("cycle"), "t_star"
    ].notna().all()
    assert bootstrap["h_abs_rate_eligible"].all()
    assert bootstrap["o_abs_rate_eligible"].all()
    assert set(stability["metric_id"]) == {"h_abs_rate", "o_abs_rate"}
    assert set(ablation_regret["selector_metric"]) == {
        "cop_cyc_evt",
        "h_abs_rate",
        "o_abs_rate",
    }
    assert len(paired) == 1


def test_outdoor_event_ablation_uses_one_dynamic8_row_per_event() -> None:
    rows = []
    for experiment_index in range(4):
        for event_index in range(2):
            value = float(experiment_index * 2 + event_index)
            row = {
                "cycle_name": f"cycle_{experiment_index}_{event_index}",
                "experiment_id": f"exp_{experiment_index}",
                "event_valid": True,
                "model_name": "dynamic_8",
                "Q_T_observed_kwh": 1.0 + value,
                "E_comp_T_observed_kwh": 0.2 + 0.1 * value,
                "Q_T_prediction_kwh": 0.9 + value,
                "E_comp_T_prediction_kwh": 0.1 + 0.1 * value,
                "supported": True,
                "v27_E_comp_T_supported": True,
                **{feature: value + offset for offset, feature in enumerate(DYNAMIC_8)},
            }
            rows.extend([row, {**row, "model_name": "mean_baseline"}])

    result = outdoor_event_model_ablation(pd.DataFrame(rows))

    assert len(result) == 8
    assert result["cycle_name"].is_unique
    assert np.allclose(
        result["Qe_T_observed_kwh"],
        result["Q_T_observed_kwh"] - result["E_comp_T_observed_kwh"],
    )
    assert np.allclose(result["Qe_T_component_prediction_kwh"], 0.8 + 0.9 * np.arange(8))
    assert result["Qe_T_direct_prediction_kwh"].notna().all()


def test_outdoor_event_ablation_figure_is_warning_free() -> None:
    path = Path(__file__).parents[2] / "scripts/cost/benchmark.py"
    spec = importlib.util.spec_from_file_location("cost_benchmark_event_figure", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "Qe_T_observed_kwh": [-0.2, 0.0, 0.2, 0.4],
            "Qe_T_component_prediction_kwh": [-0.1, 0.1, 0.1, 0.5],
            "Qe_T_direct_prediction_kwh": [-0.15, 0.05, 0.15, 0.45],
            "Qe_T_component_supported": [True, True, False, True],
            "Qe_T_direct_supported": [True, True, False, True],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = module._outdoor_event_ablation_figure(values)

    assert len(figure.axes) == 4


def test_ch_diagnostics_compare_native_near_optimal_sets_without_a_joint_selector() -> None:
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=5, freq="min")
    common = {
        "cycle_name": "cycle",
        "candidate_time": times,
        "optimization_eligible": True,
    }
    c = pd.DataFrame(
        {
            **common,
            "objective_value": [10.0, 9.9, 9.8, 9.7, 9.6],
            "relative_optimality_gap": [0.0, 0.01, 0.02, 0.03, 0.04],
            "t_star": start,
        }
    )
    h = pd.DataFrame(
        {
            **common,
            "objective_value": [8.0, 8.4, 8.8, 9.0, 8.9],
            "relative_optimality_gap": [1 - 8 / 9, 1 - 8.4 / 9, 1 - 8.8 / 9, 0, 1 - 8.9 / 9],
            "t_star": start + pd.Timedelta(minutes=3),
        }
    )

    tradeoff = ch_tradeoff_diagnostic(c, h, epsilon_c=(0.0, 0.02))
    c_with_unidentified = pd.concat([c, c.assign(cycle_name="unidentified")])
    h_with_unidentified = pd.concat([h, h.assign(cycle_name="unidentified")])
    c_with_unidentified.loc[c_with_unidentified["cycle_name"].eq("unidentified"), "t_star"] = pd.NaT
    h_with_unidentified.loc[h_with_unidentified["cycle_name"].eq("unidentified"), "t_star"] = pd.NaT
    overlap = ch_high_value_overlap(
        c_with_unidentified,
        h_with_unidentified,
        epsilon_c=(0.02,),
        epsilon_h=(0.03,),
    )

    assert set(tradeoff["epsilon_C"]) == {0.0, 0.02}
    gain = tradeoff.loc[
        tradeoff["epsilon_C"].eq(0.02), "H_gain_upper_bound"
    ].iloc[0]
    assert gain == pytest.approx(0.1)
    assert overlap.loc[0, "longest_overlap_minutes"] == 0
    assert set(overlap["cycle_name"]) == {"cycle"}
    assert "selected_time" not in tradeoff
    assert "selected_time" not in overlap


def test_ch_tradeoff_requires_native_optima_to_be_cross_evaluable() -> None:
    start = pd.Timestamp("2026-01-01")
    common = {
        "cycle_name": "cycle",
        "experiment_id": "experiment",
        "candidate_time": pd.date_range(start, periods=3, freq="min"),
        "relative_optimality_gap": [0.0, 0.01, 0.02],
        "t_star": start,
    }
    c = pd.DataFrame(
        {**common, "objective_value": [3.0, 2.9, 2.8], "optimization_eligible": True}
    )
    h = pd.DataFrame(
        {
            **common,
            "objective_value": [8.0, 8.1, 8.2],
            "optimization_eligible": [False, True, True],
        }
    )

    assert ch_tradeoff_diagnostic(c, h).empty


def test_ch_frontier_figures_are_warning_free() -> None:
    path = Path(__file__).parents[2] / "scripts/cost/benchmark.py"
    spec = importlib.util.spec_from_file_location("cost_benchmark_ch_figures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tradeoff = pd.DataFrame(
        {
            "cycle_name": ["a", "b"] * 2,
            "experiment_id": ["x", "y"] * 2,
            "epsilon_C": [0.005, 0.005, 0.01, 0.01],
            "H_gain_upper_bound": [0.01, 0.02, 0.02, 0.04],
            "H_regret_at_C": [0.02, 0.03] * 2,
            "C_regret_at_H": [0.01, 0.02] * 2,
            "compatible_candidate_count": [2, 3, 4, 5],
        }
    )
    overlap = pd.DataFrame(
        {
            "cycle_name": ["a", "b", "a", "b"],
            "epsilon_C": [0.005, 0.005, 0.01, 0.01],
            "epsilon_H": 0.01,
            "longest_overlap_minutes": [2.0, np.nan, 6.0, 3.0],
            "overlap_candidate_count": [3, 0, 7, 4],
        }
    )
    uncertainty = pd.DataFrame(
        {"cycle_name": ["a", "b"], "C_IQR_minutes": [4.0, 5.0], "H_IQR_minutes": [3.0, 6.0]}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tradeoff_figure = module._ch_tradeoff_figure(tradeoff)
        overlap_figure = module._ch_overlap_figure(overlap, uncertainty)
        gate_figure = module._g1_g2_decision_gate_figure(tradeoff, overlap, uncertainty)

    assert len(tradeoff_figure.axes) == 4
    assert len(overlap_figure.axes) >= 4
    assert len(gate_figure.axes) == 1
    text = " ".join(item.get_text() for item in gate_figure.axes[0].texts)
    assert "valid → guardrail → Pareto → latest" in text


def test_local_ratio_attribution_exactly_reconstructs_cost_change() -> None:
    start = pd.Timestamp("2026-01-01")
    curve = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": [
                start,
                start + pd.Timedelta(minutes=5),
                start + pd.Timedelta(10, "min"),
            ],
            "t_star": start + pd.Timedelta(minutes=5),
            "heating_electricity_kwh": [1.0, 2.0, 4.0],
            "E_T_hat_kwh": [1.0, 1.2, 1.4],
            "water_heating_kwh": [2.0, 3.0, 4.0],
            "Q_T_hat_kwh": [0.5, 0.6, 0.8],
            "optimization_eligible": True,
        }
    )

    result = local_ratio_attribution(curve)

    for _, row in result.iterrows():
        reconstructed = sum(
            row[column]
            for column in (
                "heating_energy_contribution",
                "event_energy_contribution",
                "heating_heat_contribution",
                "event_heat_contribution",
            )
        )
        assert reconstructed == pytest.approx(row["delta_inverse_cop"])


def test_cycle_trigger_uses_the_confirmation_frame_and_scores_each_objective() -> None:
    start = pd.Timestamp("2026-01-01")
    predictions = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "camera_group": "front",
            "modality": "rgb",
            "fold_evaluable": True,
            "image_time": pd.date_range(start, periods=4, freq="min"),
            "decision_score": [0.1, 0.6, 0.7, 0.8],
        }
    )
    curve = pd.DataFrame(
        {
            "cycle_name": "cycle",
            "candidate_time": pd.date_range(start, periods=4, freq="min"),
            "t_star": start + pd.Timedelta(minutes=2),
            "objective_value": [1.0, 2.0, 3.0, 2.7],
            "relative_optimality_gap": [2 / 3, 1 / 3, 0.0, 0.1],
            "optimization_eligible": True,
            "basin_1pct_start": start + pd.Timedelta(minutes=2),
            "basin_1pct_end": start + pd.Timedelta(minutes=2),
            "basin_5pct_start": start + pd.Timedelta(minutes=2),
            "basin_5pct_end": start + pd.Timedelta(minutes=2),
        }
    )

    result = cycle_trigger_validation(
        predictions,
        {metric: curve.assign(metric_id=metric) for metric in ("C", "H", "O")},
        metric_order=("C", "H", "O"),
        consecutive=3,
    )

    assert result.loc[0, "trigger_time"] == start + pd.Timedelta(minutes=3)
    assert result.loc[0, "signed_error_minutes"] == 1
    assert result.loc[0, "regret_C"] == pytest.approx(0.1)
