from __future__ import annotations

import importlib.util
import warnings
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb


def _module():
    path = Path("scripts/cost/plot.py")
    spec = importlib.util.spec_from_file_location("cost_function_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(algorithm: str) -> pd.DataFrame:
    stable = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "experiment_id": ("exp_20260101" if cycle == "cycle_003" else "exp_20260102"),
                "candidate_time": stable + pd.Timedelta(minutes=minute),
                "t_heating_stable": stable,
                "t_star": stable + pd.Timedelta(minutes=optimum),
                "water_reference_t_star": stable + pd.Timedelta(minutes=10),
                "t_RB": stable + pd.Timedelta(minutes=14),
                "rb_status": "triggered",
                "inverse_cop": cost,
                "relative_regret": 0.0 if minute == optimum else 0.1,
                "water_reference_inverse_cop": cost + 0.1,
                "water_reference_relative_regret": 0.0 if minute == 10 else 0.1,
                "optimization_eligible": True,
                "valid": True,
                "is_censored": False,
                "algorithm": algorithm,
            }
            for cycle, optimum, end in (("cycle_003", 10, 16), ("cycle_005", 12, 20))
            for minute, cost in ((optimum, 0.4), (end, 0.5))
        ]
    )


def test_cycle_points_accept_mixed_fractional_timestamp_formats() -> None:
    module = _module()
    table = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000058", "frost_cycle_000071"],
            "candidate_time": [
                "2026-01-01 00:10:00.000000000",
                "2026-01-01 00:20:00",
            ],
            "t_heating_stable": [
                "2026-01-01 00:00:00.000000000",
                "2026-01-01 00:00:00",
            ],
            "cycle_start": [
                "2025-12-31 23:58:00.000000000",
                "2025-12-31 23:58:00",
            ],
            "t_star": [
                "2026-01-01 00:08:00.000000000",
                "2026-01-01 00:18:00",
            ],
            "t_RB": [
                "2026-01-01 00:09:00.000000000",
                "2026-01-01 00:19:00",
            ],
            "rb_status": ["triggered", "triggered"],
        }
    )

    points = module._cycle_points(table)

    assert points["length_minutes"].tolist() == [12.0, 22.0]
    assert points["optimum_minutes"].tolist() == [10.0, 20.0]
    assert points["rb_minutes"].tolist() == [11.0, 21.0]


def test_cycle_points_preserves_unknown_support() -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        t_star_model_supported=np.nan,
    )

    points = module._cycle_points(table)

    assert points["optimum_supported"].isna().all()


def test_read_tables_admits_cycles_by_any_usable_candidate_and_keeps_full_curve(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "v2.6.6.csv"
    pd.DataFrame(
        {
            "cycle_name": ["keep", "keep", "drop", "drop"],
            "algorithm": ["v2.6.6"] * 4,
            "optimization_eligible": [True, False, False, False],
            "valid": [True, False, False, False],
            "is_censored": [False, True, False, True],
        }
    ).to_csv(path, index=False)

    table = module._read_tables({"V2.6.6": path})["v2.6.6"]

    assert table["cycle_name"].tolist() == ["keep", "keep"]
    assert table["is_censored"].tolist() == [False, True]


def test_read_tables_keeps_v268_cycles_without_a_diagnostic_minimum(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "v2.6.8.csv"
    pd.DataFrame(
        {
            "cycle_name": ["supported", "supported", "no_minimum", "no_minimum"],
            "algorithm": ["v2.6.8"] * 4,
            "optimization_eligible": [True, True, False, False],
            "valid": [True, True, False, False],
        }
    ).to_csv(path, index=False)

    table = module._read_tables({"V2.6.8": path})["v2.6.8"]

    assert table["cycle_name"].tolist() == [
        "supported",
        "supported",
        "no_minimum",
        "no_minimum",
    ]


def test_v27_metric_reader_preserves_multiple_native_metrics_from_one_csv(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "cost_function_v2.7.0.csv"
    pd.DataFrame(
        {
            "cycle_name": ["cycle_003"] * 4,
            "experiment_id": ["exp_20260101"] * 4,
            "candidate_time": pd.date_range("2026-01-01 00:10", periods=2, freq="min").tolist() * 2,
            "algorithm": ["v2.7.0"] * 4,
            "metric_id": ["eta_e_cyc"] * 2 + ["cop_e"] * 2,
            "optimization_direction": ["max"] * 4,
            "objective_value": [0.8, 0.9, 1.8, 2.0],
            "optimization_eligible": True,
            "valid": True,
        }
    ).to_csv(path, index=False)

    metrics = module._read_v27_metrics({"V2.7.0": path})

    assert set(metrics) == {"eta_e_cyc", "cop_e"}
    assert metrics["eta_e_cyc"]["objective_value"].tolist() == [0.8, 0.9]
    assert metrics["cop_e"]["optimization_direction"].eq("max").all()


def test_v27_validation_figure_shows_all_loeo_targets_and_derived_dynamic_loss() -> None:
    module = _module()
    rows = []
    for experiment_index, experiment in enumerate(("exp_20260101", "exp_20260102")):
        for cycle_index in range(2):
            observed_e = 0.30 + 0.02 * experiment_index + 0.01 * cycle_index
            observed_q = 0.20 + 0.03 * experiment_index + 0.01 * cycle_index
            observed_j = 0.40 + 0.02 * experiment_index + 0.01 * cycle_index
            for model_index, model_name in enumerate(
                ("mean_baseline", "static_5", "physical_static_6", "dynamic_8")
            ):
                rows.append(
                    {
                        "cycle_name": f"cycle_{experiment_index * 2 + cycle_index + 1:03d}",
                        "experiment_id": experiment,
                        "event_valid": True,
                        "model_name": model_name,
                        "E_T_observed_kwh": observed_e,
                        "E_T_prediction_kwh": observed_e + 0.005 * model_index,
                        "Q_T_observed_kwh": observed_q,
                        "Q_T_prediction_kwh": observed_q - 0.004 * model_index,
                        "J_w_observed": observed_j,
                        "J_w_prediction": observed_j + 0.003 * model_index,
                        "L_T_dynamic_observed_kwh": 1.0 + 0.05 * cycle_index,
                        "L_T_dynamic_prediction_kwh": 1.0
                        + 0.05 * cycle_index
                        + 0.01 * model_index,
                    }
                )

    figure = module._validation_figure(pd.DataFrame(rows))
    legend_labels = [
        label.get_text()
        for axis in figure.axes
        if axis.get_legend() is not None
        for label in axis.get_legend().texts
    ]
    labels = " ".join(
        [
            figure._suptitle.get_text() if figure._suptitle is not None else "",
            *(axis.get_title() for axis in figure.axes),
            *(text.get_text() for axis in figure.axes for text in axis.texts),
            *legend_labels,
        ]
    ).lower()

    assert all(model.replace("_", "-") in labels for model in ("static_5", "dynamic_8"))
    assert "mean" in labels and "physical-static-6" in labels
    assert "e_t" in labels and "q_t" in labels and "j_w" in labels
    assert "calibration" in labels
    assert "cross-fitted healthy-reference-derived target" in labels
    assert "dynamic event heat loss observed" not in labels
    plt.close(figure)


def test_v27_notation_figure_separates_definitions_from_estimators() -> None:
    module = _module()

    figure = module._metric_formula_boundary_figure()
    labels = " ".join(
        [
            figure._suptitle.get_text(),
            *(text.get_text() for axis in figure.axes for text in axis.texts),
        ]
    )

    assert all(
        term in labels
        for term in (
            "$COP_{cyc}=Q/E$",
            "$\\eta_H=Q/Q_0$",
            "$\\eta_{out}=Q_{out}/Q_{out,0}$",
        )
    )
    assert "only future recovery is excluded; leading recovery: included" in labels
    assert "Current V2.7 fields are estimators or sensitivity analyses" in labels
    plt.close(figure)


def test_v27_csv_only_summary_does_not_rebuild_models(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    sources = {}
    metric_specs = {
        "v2.7.0": ("eta_e_cyc", "max", [0.8, 0.9]),
        "v2.7.1": ("epsilon_hl", "min", [0.2, 0.1]),
        "v2.7.2": ("cop_cyc_k", "max", [1.8, 2.0]),
        "v2.7.3": ("epsilon_hl_2a", "min", [0.3, 0.2]),
    }
    for algorithm, (metric_id, direction, objective) in metric_specs.items():
        path = tmp_path / f"cost_function_{algorithm}.csv"
        pd.DataFrame(
            {
                "cycle_name": ["cycle_003"] * 2,
                "experiment_id": ["exp_20260101"] * 2,
                "candidate_time": [
                    start + pd.Timedelta(minutes=10),
                    start + pd.Timedelta(minutes=11),
                ],
                "cycle_start": [start] * 2,
                "algorithm": [algorithm] * 2,
                "metric_id": [metric_id] * 2,
                "optimization_direction": [direction] * 2,
                "objective_value": objective,
                "relative_optimality_gap": [0.1, 0.0],
                "optimization_eligible": True,
                "supported": True,
                "physical_valid": True,
                "identifiable": True,
                "t_star": [start + pd.Timedelta(minutes=11)] * 2,
                "t_RB": [start + pd.Timedelta(minutes=12)] * 2,
                "rb_status": ["triggered"] * 2,
                "basin_1pct_width_minutes": [1.0] * 2,
                "basin_5pct_width_minutes": [2.0] * 2,
                "bootstrap_in_original_5pct_basin_fraction": [0.8] * 2,
            }
        ).to_csv(path, index=False)
        sources[algorithm] = path

    saved: list[str] = []
    monkeypatch.setattr(
        module,
        "_save_svg_png",
        lambda figure, path: (saved.append(path.name), plt.close(figure)),
    )

    metrics = module._read_v27_metrics(sources)
    module._write_v27_summaries(
        metrics, {}, sources, tmp_path / "figures", diagnostics=False
    )

    assert "comparison_v2.7_RB.png" in saved
    assert "comparison_v1_v2.5_v2.6.7_v2.6.8_v2.7_RB.png" not in saved

    saved.clear()
    module._write_v27_summaries(
        metrics, {}, sources, tmp_path / "figures", diagnostics=True
    )
    assert {
        "01_评价指标迁移与文献定位.png",
        "02_完整性极值位置区分度与稳定性.png",
        "03_支持域与可识别覆盖.png",
        "04_方向感知成本形状比较.png",
        "05_新增事件目标LOEO.png",
        "06_全历史成本函数经验链.png",
    } <= set(saved)


def test_v27_csv_only_history_uses_v27_cycle_origin_when_old_csv_lacks_cycle_start(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    v27_path = tmp_path / "cost_function_v2.7.0.csv"
    pd.DataFrame(
        {
            "cycle_name": ["cycle_003"] * 2,
            "experiment_id": ["exp_20260101"] * 2,
            "candidate_time": pd.date_range(
                start + pd.Timedelta(minutes=10), periods=2, freq="min"
            ),
            "cycle_start": [start] * 2,
            "algorithm": ["v2.7.0"] * 2,
            "metric_id": ["eta_e_cyc"] * 2,
            "optimization_direction": ["max"] * 2,
            "objective_value": [0.8, 0.9],
            "optimization_eligible": True,
            "valid": True,
            "t_star": [start + pd.Timedelta(minutes=11)] * 2,
            "t_RB": [start + pd.Timedelta(minutes=12)] * 2,
            "rb_status": ["triggered"] * 2,
        }
    ).to_csv(v27_path, index=False)
    sources = {"V2.7.0": v27_path}
    for algorithm in ("v1", "v2.5", "v2.6.7", "v2.6.8"):
        path = tmp_path / f"cost_function_{algorithm}.csv"
        _table(algorithm).loc[lambda table: table["cycle_name"].eq("cycle_003")].to_csv(
            path, index=False
        )
        sources[algorithm] = path

    saved: list[str] = []
    monkeypatch.setattr(
        module,
        "_save_svg_png",
        lambda figure, path: (saved.append(path.name), plt.close(figure)),
    )

    class Loader:
        @staticmethod
        def get_cycle_record(_cycle_name: str) -> dict[str, object]:
            return {"boundaries": {"start_time": start}}

    module.generate_cost_function_figures(
        sources, Loader(), tmp_path / "figures", comparison_only=True
    )

    assert "comparison_v1_v2.5_v2.6.7_v2.6.8_v2.7_RB.png" in saved


def test_v27_shape_comparison_skips_cycles_without_finite_gap() -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    rows = []
    for cycle_id in range(1, 5):
        for minute in (10, 11):
            rows.append(
                {
                    "cycle_name": f"cycle_{cycle_id:03d}",
                    "cycle_start": start,
                    "candidate_time": start + pd.Timedelta(minutes=minute),
                    "relative_optimality_gap": (
                        np.nan if cycle_id == 1 else 0.01 * abs(minute - 11)
                    ),
                }
            )
    table = pd.DataFrame(rows)

    figure = module._normalized_gap_figure({"eta_e_cyc": table})

    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "cycle_002",
        "cycle_003",
        "cycle_004",
    ]
    plt.close(figure)


def test_v27_boundary_sensitivity_labels_map_all_protocols_without_aliasing() -> None:
    module = _module()
    metrics = {
        metric_id: _table("v2.7.1").assign(
            cycle_start=pd.Timestamp("2025-12-31 23:55:00"), metric_id=metric_id
        )
        for metric_id in ("epsilon_hl", "epsilon_hl_t0_proxy", "cop_cyc_k")
    }
    historical = {
        "v2.6.8": _table("v2.6.8").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
    }

    figure = module._baseline_boundary_sensitivity_figure(metrics, historical)
    labels = [
        label.get_text()
        for axis in figure.axes
        if axis.get_legend() is not None
        for label in axis.get_legend().texts
    ]
    labels.extend(text.get_text() for axis in figure.axes for text in axis.texts)
    labels.extend(text.get_text() for text in figure.texts)
    rendered = "\n".join(labels).lower()

    assert "fixed-9 stable-to-stable" in rendered
    assert "leading recovery + heating + prep/d" in rendered
    assert "only future recovery excluded" in rendered
    assert "ts-dependent 9/13 min" in rendered
    assert "not plotted" in rendered or "not available" in rendered
    plt.close(figure)


def test_v27_formula_figure_states_all_directions_and_recovery_boundaries() -> None:
    module = _module()

    figure = module._metric_formula_boundary_figure()
    rendered = "\n".join(text.get_text() for text in figure.texts)
    rendered += "\n" + "\n".join(
        text.get_text() for axis in figure.axes for text in axis.texts
    )

    for metric_id in (
        "eta_e_cyc",
        "cop_e",
        "epsilon_hl",
        "epsilon_hl_t0_proxy",
        "cop_cyc_k",
        "epsilon_hl_2a",
    ):
        assert metric_id in rendered
    assert rendered.count("MAXIMIZE") == 3
    assert rendered.count("MINIMIZE") == 3
    assert "only future recovery is excluded" in rendered
    assert "leading recovery: included" in rendered
    plt.close(figure)


def test_v27_shape_comparison_draws_one_and_five_percent_guides() -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    table = pd.DataFrame(
        {
            "cycle_name": ["cycle_001"] * 2,
            "cycle_start": [start] * 2,
            "candidate_time": pd.date_range(
                start + pd.Timedelta(minutes=10), periods=2, freq="min"
            ),
            "relative_optimality_gap": [0.01, 0.0],
        }
    )

    figure = module._normalized_gap_figure({"eta_e_cyc": table})
    guide_levels = {
        float(np.asarray(line.get_ydata())[0])
        for line in figure.axes[0].lines
        if np.asarray(line.get_ydata()).size > 1
        and np.all(np.asarray(line.get_ydata()) == np.asarray(line.get_ydata())[0])
    }

    assert {1.0, 5.0} <= guide_levels
    plt.close(figure)


def test_v27_cost_curve_accepts_mixed_fractional_timestamp_formats() -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    metric = pd.DataFrame(
        {
            "cycle_name": ["cycle_001"] * 2,
            "cycle_start": [start] * 2,
            "candidate_time": [
                "2026-01-01 00:10:00.000000000",
                "2026-01-01 00:11:00",
            ],
            "objective_value": [0.8, 0.9],
            "relative_optimality_gap": [0.1, 0.0],
            "optimization_eligible": [True, True],
            "optimization_direction": ["max", "max"],
            "t_star": ["2026-01-01 00:11:00"] * 2,
        }
    )

    figure = module._cost_curve_figure({"eta_e_cyc": metric}, "cycle_001")

    assert len(figure.axes[0].lines) >= 1
    plt.close(figure)



def test_v27_comparison_marks_cycle_without_valid_metric_curve() -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    table = pd.DataFrame(
        {
            "cycle_name": ["cycle_001"],
            "experiment_id": ["exp_20260101"],
            "cycle_start": [start],
            "candidate_time": [start + pd.Timedelta(minutes=10)],
            "objective_value": [1.0],
            "optimization_eligible": [False],
            "optimization_direction": ["max"],
            "t_star": [pd.NaT],
            "t_RB": [pd.NaT],
            "rb_status": ["not_triggered"],
        }
    )

    figure = module._comparison_figure({"eta_e_cyc": table}, ("eta_e_cyc",))

    labels = {collection.get_label() for collection in figure.axes[0].collections}
    assert any("no diagnostic minimum" in label for label in labels)
    plt.close(figure)




def test_v27_publications_keep_no_valid_metric_slots(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    metrics = {
        metric_id: pd.DataFrame(
            {
                "cycle_name": ["cycle_001"],
                "experiment_id": ["exp_20260101"],
                "cycle_start": [start],
                "candidate_time": [start + pd.Timedelta(minutes=10)],
                "objective_value": [1.0],
                "optimization_eligible": [metric_id == "cop_e"],
                "optimization_direction": ["min" if metric_id == "epsilon_hl" else "max"],
                "objective_label": [metric_id],
                "objective_unit": ["-"],
                "t_star": [start + pd.Timedelta(minutes=11) if metric_id == "cop_e" else pd.NaT],
            }
        )
        for metric_id in ("epsilon_hl", "cop_e")
    }
    captured: list[tuple[Path, pd.DataFrame, dict[str, object]]] = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, curve, _images, path, **kwargs: captured.append(
            (path, curve, kwargs)
        ),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    module._render_cycle_sets(
        metrics,
        Loader(),
        {"cycle_001": {"cycle_name": "cycle_001"}},
        tmp_path,
    )

    by_metric = {
        path.parent.name.rsplit("/", 1)[-1]: (curve, kwargs)
        for path, curve, kwargs in captured
    }
    assert set(by_metric) == {"epsilon_hl", "cop_e"}
    assert by_metric["epsilon_hl"][1]["minimum_label"] == "Minimum"
    assert by_metric["cop_e"][1]["minimum_label"] == "Maximum"
    assert by_metric["epsilon_hl"][0]["display_only_objective"].notna().all()


def test_parallel_policy_cycle_reuses_publication_renderer_with_c_h_o(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    times = pd.date_range(start, periods=4, freq="min")
    metrics = {
        metric_id: pd.DataFrame(
            {
                "cycle_name": "cycle_001",
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric_id, values in {
            "cop_cyc_evt": [3.0, 2.99, 2.97, 2.7],
            "h_abs_rate": [8.0, 8.1, 8.2, 8.3],
            "o_abs_rate": [5.0, 4.8, 4.6, 4.4],
        }.items()
    }
    captured = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda *_args, **kwargs: captured.append(kwargs),
    )

    class Loader:
        load_cycle = staticmethod(lambda _cycle: pd.DataFrame({"timestamp": times}))
        load_image_metadata = load_cycle_images = staticmethod(lambda _cycle: pd.DataFrame())

    module._render_parallel_policy_cycles(
        metrics,
        Loader(),
        {
            "cycle_001": {
                "cycle_name": "cycle_001",
                "boundaries": {"stable_heating_start": times[1]},
            }
        },
        tmp_path,
    )

    assert set(captured[0]["parallel_curves"]) == {"C", "H", "O"}
    assert all(
        curve["candidate_time"].tolist() == times[1:].tolist()
        for curve in captured[0]["parallel_curves"].values()
    )
    assert captured[0]["pareto_guardrail"] == 0.05
    assert captured[0]["pareto_selector"] == "knee"
    candidates = pd.read_csv(tmp_path / "pareto_knee_candidates.csv")
    assert candidates["pareto_knee"].sum() == 1
    assert candidates.loc[candidates["pareto_knee"], "candidate_time"].tolist() == [
        str(times[2])
    ]


def test_parallel_policy_cycle_uses_full_pareto_when_five_percent_intersection_is_empty(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    times = pd.date_range("2026-01-01", periods=2, freq="min")
    metrics = {
        metric_id: pd.DataFrame(
            {
                "cycle_name": "cycle_001",
                "candidate_time": times,
                "objective_value": values,
                "optimization_eligible": True,
            }
        )
        for metric_id, values in {
            "cop_cyc_evt": [10.0, 1.0],
            "h_abs_rate": [1.0, 10.0],
            "o_abs_rate": [5.0, 4.0],
        }.items()
    }
    captured = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda *_args, **_kwargs: captured.append(True),
    )

    class Loader:
        load_cycle = staticmethod(lambda _cycle: pd.DataFrame({"timestamp": times}))
        load_image_metadata = load_cycle_images = staticmethod(lambda _cycle: pd.DataFrame())

    result = module._render_parallel_policy_cycle(
        "cycle_001",
        metrics,
        Loader(),
        {"cycle_name": "cycle_001"},
        tmp_path,
    )

    assert captured == [True]
    assert result["pareto_latest"].sum() == 1
    assert result.loc[result["pareto_latest"], "candidate_time"].tolist() == [times[1]]
    assert result["pareto_knee"].sum() == 1
    assert result.loc[result["pareto_knee"], "candidate_time"].tolist() == [times[0]]


def test_historical_and_v27_share_one_cycle_publication_registration(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    v1_path = tmp_path / "cost_function_v1.csv"
    v27_path = tmp_path / "cost_function_v2.7.0.csv"
    v1 = _table("v1")
    v1.to_csv(v1_path, index=False)
    v1.assign(
        algorithm="v2.7.0",
        metric_id="eta_e_cyc",
        objective_value=v1["inverse_cop"],
        optimization_direction="max",
        relative_optimality_gap=v1["relative_regret"],
        supported=True,
        physical_valid=True,
        identifiable=True,
    ).to_csv(v27_path, index=False)

    captured: list[set[str]] = []
    monkeypatch.setattr(module, "_write_v27_summaries", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_render_cost_curve_comparisons", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_render_cycle_sets",
        lambda tables, *_args, **_kwargs: captured.append(set(tables)),
    )

    class Loader:
        @staticmethod
        def get_cycle_record(_cycle_name: str) -> dict[str, object]:
            return {"boundaries": {"start_time": pd.Timestamp("2025-12-31 23:55")}}

    module.generate_cost_function_figures(
        {"v1": v1_path, "v2.7.0": v27_path}, Loader(), tmp_path / "figures"
    )

    assert captured == [{"v1", "eta_e_cyc"}]


def test_cost_curve_panel_shows_distinct_one_and_five_percent_basin_guides() -> None:
    module = _module()
    table = _table("v2.6.5").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))

    figure = module._cost_curve_figure({"v2.6.5": table}, "cycle_003")

    axis = figure.axes[1]
    guides = {
        float(np.asarray(line.get_ydata())[0]): line.get_linestyle()
        for line in axis.lines
        if np.asarray(line.get_ydata()).size > 1
        and np.all(np.asarray(line.get_ydata()) == np.asarray(line.get_ydata())[0])
    }
    assert {1.0, 5.0} <= set(guides)
    assert guides[1.0] != guides[5.0]
    assert axis.get_ylim()[1] > 5.0
    plt.close(figure)


def test_v27_front_image_matching_uses_optimal_target_not_rule_row(monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "match_decision_rgb_images",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "target_type": ["rb", "optimal"],
                "available": [False, True],
                "status": ["rb_right_censored", "matched"],
            }
        ),
    )
    curve = pd.DataFrame(
        {
            "cycle_name": ["cycle_001"],
            "t_star": [pd.Timestamp("2026-01-01 00:10")],
            "optimization_eligible": [True],
            "optimization_direction": ["max"],
        }
    )

    matched = module._match_optimal_front_images(
        {"eta_e_cyc": curve},
        "cycle_001",
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert matched["eta_e_cyc"]["available"] is True
    assert matched["eta_e_cyc"]["status"] == "matched"


def test_v27_rgb_page_passes_cycle_stable_start_to_decision_image(monkeypatch) -> None:
    module = _module()
    stable = pd.Timestamp("2026-01-01 00:09:00")
    captured: list[pd.Timestamp] = []
    monkeypatch.setattr(
        module,
        "_plot_decision_image",
        lambda _axis, _info, _label, _origin, stable_arg: captured.append(stable_arg),
    )

    for figure in module._optimal_rgb_figures(
            {
                "eta_e_cyc": {
                    "target_time": pd.Timestamp("2026-01-01 00:10"),
                    "available": True,
                    "image_time": pd.Timestamp("2026-01-01 00:12:00"),
                    "offset_seconds": 0.0,
                }
            },
            ("eta_e_cyc",),
            "cycle_001",
            pd.Timestamp("2026-01-01"),
            stable,
        ):
        plt.close(figure)

    assert captured == [stable]


def test_v268_cost_figure_draws_supported_and_outside_domain_segments() -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    table = pd.DataFrame(
        {
            "cycle_name": ["cycle_003"] * 8,
            "algorithm": ["v2.6.8"] * 8,
            "experiment_id": ["exp_20260101"] * 8,
            "cycle_start": [start] * 8,
            "candidate_time": pd.date_range(start + pd.Timedelta(minutes=1), periods=8, freq="min"),
            "t_star": [start + pd.Timedelta(minutes=5)] * 8,
            "t_RB": [start + pd.Timedelta(minutes=7)] * 8,
            "rb_status": ["triggered"] * 8,
            "J_model": np.linspace(0.5, 0.3, 8),
            "inverse_cop": np.linspace(0.5, 0.3, 8),
            "relative_regret": [np.nan, 0.5, 0.3, 0.1, 0.0, 0.1, 0.2, np.nan],
            "optimization_eligible": [False, True, True, True, True, True, True, False],
            "supported": [False, True, True, True, True, True, True, False],
            "model_supported": [False, True, True, True, True, True, True, False],
            "cycle_status": ["identified_curve"] * 8,
        }
    )

    figure = module._cost_curve_figure({"v2.6.8": table}, "cycle_003")
    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert "V2.6.8" in labels
    assert "V2.6.8 outside applicability domain" in labels
    plt.close(figure)


def test_v268_standard_comparisons_and_curve_family_have_requested_names(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    sources = {}
    for algorithm in ("v1", "v2.5", "v2.6.7", "v2.6.8"):
        path = tmp_path / f"{algorithm}.csv"
        _table(algorithm).assign(cycle_status="identified_curve").to_csv(path, index=False)
        sources[algorithm] = path

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"boundaries": {"start_time": "2026-01-01"}}

    saved: list[str] = []
    family: list[str] = []
    monkeypatch.setattr(
        module,
        "_save_png",
        lambda figure, path: (saved.append(path.name), plt.close(figure)),
    )
    monkeypatch.setattr(module, "_render_cycle_sets", lambda *args, **kwargs: None)
    module.generate_cost_function_figures(sources, Loader(), tmp_path / "figures")

    assert "comparison_v2.6.8_RB.png" in saved
    assert "comparison_v1_v2.5_v2.6.7_v2.6.8_RB.png" in saved

    monkeypatch.setattr(
        module,
        "_render_cost_curve_comparisons",
        lambda tables, loader, output, **kwargs: family.append(output.name),
    )
    module.generate_cost_function_figures(
        sources,
        Loader(),
        tmp_path / "figures",
        curves_only=True,
    )
    assert family == ["cost_function_v1_v2.5_v2.6.7_v2.6.8_cycle"]


def test_v268_publication_cycle_displays_all_noneligible_model_values(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    table = _table("v2.6.8").loc[lambda values: values["cycle_name"].eq("cycle_003")].copy()
    table["J_model"] = table["inverse_cop"]
    table["supported"] = [True, True]
    table["model_supported"] = table["supported"]
    table["optimization_eligible"] = [True, False]
    table["cycle_status"] = "identified_curve"

    class Loader:
        @staticmethod
        def load_cycle(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min")})

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

        @staticmethod
        def load_cycle_images(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

    seen = {}

    def render(frame, record, curve, images, output, **kwargs) -> None:
        seen.update(curve=curve, output=output, kwargs=kwargs)

    monkeypatch.setattr(module, "render_decision_publication", render)
    monkeypatch.setattr(module, "_decision_images", lambda *args: {})
    module._render_cycle_sets(
        {"v2.6.8": table},
        Loader(),
        {"cycle_003": {"cycle_name": "cycle_003"}},
        tmp_path,
    )

    assert seen["output"].parent.name == "cost_function_v2.6.8_cycle"
    assert seen["kwargs"]["display_metric"] == module.V268_DISPLAY_METRIC
    assert seen["kwargs"]["display_label"] == "Non-eligible V2.6.8 model curve, display only"
    assert (
        seen["curve"]
        .loc[seen["curve"]["optimization_eligible"].eq(False), module.V268_DISPLAY_METRIC]
        .notna()
        .all()
    )


def test_water_reference_curve_uses_its_own_selected_time() -> None:
    module = _module()
    table = _table("v1")

    curve = module._publication_curve(table, "water_reference")

    pd.testing.assert_series_equal(
        curve["t_star"], table["water_reference_t_star"], check_names=False
    )


def test_cost_comparison_exports_three_grids_and_three_png_cycle_sets(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    v1_path, v2_path = tmp_path / "v1.csv", tmp_path / "v2.csv"
    for algorithm, path in (("v1", v1_path), ("v2", v2_path)):
        table = _table(algorithm)
        excluded = pd.concat(
            [
                table.iloc[:2].assign(cycle_name="cycle_007", valid=False),
                table.iloc[:2].assign(cycle_name="cycle_009", is_censored=True),
            ],
            ignore_index=True,
        )
        pd.concat([table, excluded], ignore_index=True).to_csv(path, index=False)
    seen: list[tuple[str, int, str, str, list[str], list[str]]] = []
    candidate_heights: list[list[float]] = []
    original_save = module._save_png

    def capture(fig: plt.Figure, path: Path) -> None:
        axis = fig.axes[0]
        seen.append(
            (
                path.name,
                len(fig.axes),
                axis.get_xlabel(),
                axis.get_ylabel(),
                [tick.get_text() for tick in axis.get_xticklabels()],
                [label.get_text() for label in axis.texts],
            )
        )
        candidate_heights.append([bar.get_height() for bar in axis.containers[0]])
        original_save(fig, path)

    rendered: list[Path] = []
    decision_times: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    optimal_labels: dict[str, str] = {}

    def render(_frame, _record, _curve, _images, output, **_kwargs) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        rendered.append(output)
        if output.is_relative_to(tmp_path / "figures"):
            relative = output.relative_to(tmp_path / "figures").as_posix()
            decision_times[relative] = (
                pd.Timestamp(_images["optimal"]["target_time"]),
                pd.Timestamp(_images["rb"]["target_time"]),
            )
            optimal_labels[relative] = _kwargs["optimal_label"]

    monkeypatch.setattr(module, "_save_png", capture)
    monkeypatch.setattr(module, "render_decision_publication", render)
    loads: list[tuple[str, str]] = []

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            loads.append(("record", cycle_name))
            return {
                "cycle_name": cycle_name,
                "boundaries": {
                    "start_time": "2025-12-31 23:55:00",
                    "stable_heating_start": "2026-01-01",
                },
            }

        @staticmethod
        def load_cycle(cycle_name: str) -> pd.DataFrame:
            loads.append(("frame", cycle_name))
            return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min")})

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            loads.append(("metadata", cycle_name))
            return pd.DataFrame(columns=["camera_role", "file_name", "image_time"])

        @staticmethod
        def load_cycle_images(cycle_name: str) -> pd.DataFrame:
            loads.append(("images", cycle_name))
            return pd.DataFrame(columns=["camera_role", "file_name", "path"])

    output = tmp_path / "figures"
    module.generate_cost_function_figures({"v2": v2_path, "v1": v1_path}, Loader(), output)

    assert seen == [
        (
            "comparison_v1_RB.png",
            1,
            "Cycle ID",
            "Minutes from cycle start",
            ["3", "5", "9"],
            ["01-01", "01-02", "01-01"],
        ),
        (
            "comparison_v2_RB.png",
            1,
            "Cycle ID",
            "Minutes from cycle start",
            ["3", "5", "9"],
            ["01-01", "01-02", "01-01"],
        ),
        (
            "comparison_v1_v2_RB.png",
            1,
            "Cycle ID",
            "Minutes from cycle start",
            ["3", "5", "9"],
            ["01-01", "01-02", "01-01"],
        ),
    ]
    assert candidate_heights == [[21, 25, 21], [21, 25, 21], [21, 25, 21]]
    assert {path.relative_to(output).as_posix() for path in rendered} == {
        f"{directory}/cycle_00{cycle}_publication.png"
        for directory in (
            "水侧制热量_cycle",
            "cost_function_v1_cycle",
            "cost_function_v2_cycle",
        )
        for cycle in (3, 5, 9)
    }
    assert decision_times["cost_function_v2_cycle/cycle_005_publication.png"] == (
        pd.Timestamp("2026-01-01 00:12"),
        pd.Timestamp("2026-01-01 00:14"),
    )
    assert {
        optimal_labels[f"{directory}/cycle_003_publication.png"]
        for directory in (
            "水侧制热量_cycle",
            "cost_function_v1_cycle",
            "cost_function_v2_cycle",
        )
    } == {
        "Water-heat optimum",
        "Unit-heat V1 optimum",
        "Updated V2 optimum",
    }
    assert not list(output.rglob("*.svg"))
    assert not list(output.rglob("*.pdf"))
    assert sorted(loads) == sorted(
        (kind, cycle)
        for kind in ("record", "frame", "metadata", "images")
        for cycle in ("cycle_003", "cycle_005", "cycle_009")
    )

    rendered.clear()
    v1_output = tmp_path / "v1_only"
    module.generate_cost_function_figures({"anything": v1_path}, Loader(), v1_output)
    assert {path.name for path in v1_output.glob("comparison*.png")} == {"comparison_v1_RB.png"}
    assert {path.parent.name for path in rendered} == {
        "水侧制热量_cycle",
        "cost_function_v1_cycle",
    }

    rendered.clear()
    v2_output = tmp_path / "v2_only"
    module.generate_cost_function_figures({"anything": v2_path}, Loader(), v2_output)
    assert {path.name for path in v2_output.glob("comparison*.png")} == {"comparison_v2_RB.png"}
    assert {path.parent.name for path in rendered} == {"cost_function_v2_cycle"}


def test_v2_variants_use_existing_v1_comparison_plotter(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    sources = {}
    for algorithm in ("v1", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6"):
        path = tmp_path / f"{algorithm}.csv"
        _table(algorithm).to_csv(path, index=False)
        sources[algorithm] = path

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"boundaries": {"start_time": "2025-12-31 23:55:00"}}

    monkeypatch.setattr(
        module,
        "_render_cycle_sets",
        lambda *_args: (_ for _ in ()).throw(AssertionError("comparison only")),
    )
    module.generate_cost_function_figures(sources, Loader(), tmp_path, comparison_only=True)

    assert (tmp_path / "comparison_v1_v2.1_RB.png").is_file()
    assert (tmp_path / "comparison_v1_v2.1_v2.2_RB.png").is_file()
    assert (tmp_path / "comparison_v1_v2.2_v2.3_RB.png").is_file()
    assert (tmp_path / "comparison_v1_v2.3_v2.4_RB.png").is_file()
    assert (tmp_path / "comparison_v1_v2.3_v2.5_RB.png").is_file()
    assert (tmp_path / "comparison_v2.5_v2.6_RB.png").is_file()
    assert (tmp_path / "comparison_v1_v2.5_v2.6_RB.png").is_file()


def test_v2_variants_render_publication_cycles(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    sources = {}
    for algorithm in ("v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"):
        path = tmp_path / f"{algorithm}.csv"
        _table(algorithm).to_csv(path, index=False)
        sources[algorithm] = path

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {
                "cycle_name": cycle_name,
                "boundaries": {"start_time": "2025-12-31 23:55:00"},
            }

        @staticmethod
        def load_cycle(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

        @staticmethod
        def load_image_metadata(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame()

    rendered = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, curve, _images, output, **kwargs: rendered.append(
            (curve["algorithm"].iloc[0], output.parent.name, kwargs["optimal_label"])
        ),
    )

    module.generate_cost_function_figures(sources, Loader(), tmp_path / "figures")

    expected = {
        (algorithm, f"cost_function_{algorithm}_cycle", f"{algorithm.upper()} optimum")
        for algorithm in ("v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6")
    }
    expected.add(("v3", "cost_function_v3_cycle", "V3 offline decision"))
    assert set(rendered) == expected


def test_v261_uses_dotted_comparison_and_cycle_directory(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    path = tmp_path / "v2.6.1.csv"
    _table("v2.6.1").to_csv(path, index=False)

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {
                "cycle_name": cycle_name,
                "boundaries": {"start_time": "2025-12-31 23:55:00"},
            }

        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    rendered: list[Path] = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, _curve, _images, output, **_kwargs: rendered.append(output),
    )

    output = tmp_path / "figures"
    module.generate_cost_function_figures({"v2.6.1": path}, Loader(), output)

    assert (output / "comparison_v2.6.1_RB.png").is_file()
    assert {item.relative_to(output).as_posix() for item in rendered} == {
        "cost_function_v2.6.1_cycle/cycle_003_publication.png",
        "cost_function_v2.6.1_cycle/cycle_005_publication.png",
    }


def test_render_all_cost_curves_writes_curve_and_paginated_rgb_plates(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    algorithms = ("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6")
    tables = {
        algorithm: _table(algorithm).assign(
            cycle_start=lambda values: values["cycle_name"].map(
                {
                    "cycle_003": pd.Timestamp("2025-12-31 23:55:00"),
                    "cycle_005": pd.Timestamp("2025-12-31 23:50:00"),
                }
            )
        )
        for algorithm in algorithms
    }
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))

    class Loader:
        @staticmethod
        def load_image_metadata(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "camera_role": ["front", "front"],
                    "file_name": ["front.jpg", "front.jpg"],
                    "image_time": [
                        pd.Timestamp("2026-01-01 00:10:00"),
                        pd.Timestamp("2026-01-01 00:12:00"),
                    ],
                }
            )

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "camera_role": ["front"],
                    "file_name": ["front.jpg"],
                    "path": [image_path],
                }
            )

    rendered = []
    original_save = module._save_png

    def capture(figure: plt.Figure, path: Path) -> None:
        rendered.append(
            (
                path.relative_to(tmp_path).as_posix(),
                len(figure.axes),
                figure.axes[0].get_ylabel(),
                figure.axes[-1].get_xlabel(),
                figure.axes[0].get_title(loc="left"),
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            original_save(figure, path)

    monkeypatch.setattr(module, "_save_png", capture)
    module._render_cost_curve_comparisons(tables, Loader(), tmp_path)

    assert rendered == [
        (
            "cycle_003_cost_curves.png",
            2,
            "Cost J = 1/COP",
            "Minutes from cycle start",
            "Cycle 3: cost-function variants",
        ),
        (
            "optimal_rgb/cycle_003_optimal_rgb_01.png",
            4,
            "",
            "",
            "V1 optimum\n15.0 min · image offset 0 s",
        ),
        (
            "optimal_rgb/cycle_003_optimal_rgb_02.png",
            4,
            "",
            "",
            "V2.3 optimum\n15.0 min · image offset 0 s",
        ),
        (
            "cycle_005_cost_curves.png",
            2,
            "Cost J = 1/COP",
            "Minutes from cycle start",
            "Cycle 5: cost-function variants",
        ),
        (
            "optimal_rgb/cycle_005_optimal_rgb_01.png",
            4,
            "",
            "",
            "V1 optimum\n22.0 min · image offset 0 s",
        ),
        (
            "optimal_rgb/cycle_005_optimal_rgb_02.png",
            4,
            "",
            "",
            "V2.3 optimum\n22.0 min · image offset 0 s",
        ),
    ]


def test_optimal_rgb_figures_paginate_four_methods_per_page() -> None:
    module = _module()
    algorithms = tuple(name for name in module.STYLES if name != "RB")
    images = {
        algorithm: {
            "available": False,
            "status": "physical_image_missing",
            "target_time": pd.Timestamp("2026-01-01 00:10:00"),
        }
        for algorithm in algorithms
    }

    pages = list(
        module._optimal_rgb_figures(
            images,
            algorithms,
            "cycle_003",
            pd.Timestamp("2026-01-01"),
        )
    )

    assert len(pages) == 3
    assert [sum(axis.get_visible() for axis in figure.axes) for figure in pages] == [
        4,
        4,
        2,
    ]
    for figure in pages:
        plt.close(figure)


def test_five_method_v267_family_uses_one_front_image_plate() -> None:
    module = _module()
    algorithms = ("v1", "v2.5", "v2.6.5", "v2.6.6", "v2.6.7")
    images = {algorithm: {"available": False, "status": "missing"} for algorithm in algorithms}

    pages = list(
        module._optimal_rgb_figures(images, algorithms, "cycle_003", pd.Timestamp("2026-01-01"))
    )

    assert len(pages) == 1
    assert sum(axis.get_visible() for axis in pages[0].axes) == 5
    plt.close(pages[0])

    v267_page = next(
        module._optimal_rgb_figures(
            {
                "v2.6.7": {
                    "available": False,
                    "status": "no_valid_optimal",
                    "target_status": "model_support_limited",
                }
            },
            ("v2.6.7",),
            "cycle_003",
            pd.Timestamp("2026-01-01"),
        )
    )
    assert "model support limited · no eligible diagnostic minimum" in v267_page.axes[0].get_title(
        loc="left"
    )
    plt.close(v267_page)


def test_v266_rgb_support_and_page_title_follow_cycle_status(monkeypatch) -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_status="measurement_limited",
        t_star_model_supported=True,
    )
    monkeypatch.setattr(
        module,
        "match_decision_rgb_images",
        lambda *_args: pd.DataFrame(
            {"target_type": ["optimal"], "available": [True], "offset_seconds": [0]}
        ),
    )

    matched = module._match_optimal_front_images(
        {"v2.6.6": table}, "cycle_003", pd.DataFrame(), pd.DataFrame()
    )
    page = next(
        module._optimal_rgb_figures(matched, ("v2.6.6",), "cycle_003", pd.Timestamp("2026-01-01"))
    )

    assert matched["v2.6.6"]["target_supported"] is False
    assert "measurement limited" in page.axes[0].get_title(loc="left")
    assert "selected/diagnostic cost-function times" in page._suptitle.get_text()
    plt.close(page)

    unknown = module._match_optimal_front_images(
        {"v2.6.6": table.assign(t_star_model_supported=np.nan)},
        "cycle_003",
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert unknown["v2.6.6"]["target_supported"] is None


def test_v266_overview_encodes_each_nonidentified_status() -> None:
    module = _module()
    rows = []
    for index, status in enumerate(
        ("measurement_limited", "component_extrapolated", "right_censored"), start=1
    ):
        rows.append(
            _table("v2.6.6")
            .iloc[:1]
            .assign(
                cycle_name=f"cycle_{index:03d}",
                cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
                cycle_status=status,
                t_star_model_supported=True,
            )
        )
    figure = module._comparison_figure({"v2.6.6": pd.concat(rows, ignore_index=True)}, ("v2.6.6",))

    labels = {collection.get_label() for collection in figure.axes[0].collections}
    assert {
        "V2.6.6 diagnostic minimum (measurement-limited)",
        "V2.6.6 diagnostic minimum (component-extrapolated)",
        "V2.6.6 diagnostic minimum (right-censored)",
    } <= labels
    plt.close(figure)


def test_v266_overview_rejects_unrecognized_status() -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="new_unmapped_status",
    )

    with pytest.raises(ValueError, match="unrecognized V2.6.6 cycle_status"):
        module._comparison_figure({"v2.6.6": table}, ("v2.6.6",))


def test_cost_curve_family_rejects_mismatched_cycle_sets(tmp_path: Path) -> None:
    module = _module()
    tables = {
        "v1": _table("v1").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
        "v2.6.6": _table("v2.6.6")
        .loc[lambda values: values["cycle_name"].eq("cycle_003")]
        .assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
    }

    with pytest.raises(ValueError, match="identical cycle sets"):
        module._render_cost_curve_comparisons(tables, object(), tmp_path)


def test_v266_publication_label_includes_cycle_status(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    table = _table("v2.6.6").assign(cycle_status="right_censored")
    labels = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda *_args, **kwargs: labels.append(kwargs["optimal_label"]),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    records = {cycle: {} for cycle in table["cycle_name"].unique()}
    module._render_cycle_sets({"v2.6.6": table}, Loader(), records, tmp_path)

    assert labels == [
        "V2.6.6 diagnostic identification minimum (right censored)",
        "V2.6.6 diagnostic identification minimum (right censored)",
    ]


def test_v267_nonidentified_publication_preserves_support_and_labels_cycle_limit(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_status="measurement_limited",
        measurement_eligible=True,
        model_supported=True,
        t_star_model_supported=True,
        heating_electricity_kwh=0.2,
        unit_heating_kwh=0.4,
        E_T_hat_kwh=0.1,
        Q_T_hat_kwh=0.2,
    )
    seen = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, curve, _images, _output, **kwargs: seen.append(
            (
                curve["model_supported"].tolist(),
                kwargs["display_metric"],
                kwargs["minimum_label"],
                kwargs["minimum_support_label"],
            )
        ),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    records = {cycle: {} for cycle in table["cycle_name"].unique()}
    module._render_cycle_sets({"v2.6.7": table}, Loader(), records, tmp_path)

    assert (
        seen
        == [
            (
                [True, True],
                module.V267_DISPLAY_METRIC,
                "Diagnostic/raw minimum",
                "measurement limited",
            )
        ]
        * 2
    )


def test_all_cost_curves_use_distinct_colors_and_line_styles() -> None:
    module = _module()
    algorithms = ("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6")
    tables = {
        algorithm: _table(algorithm).assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
        for algorithm in algorithms
    }

    figure = module._cost_curve_figure(tables, "cycle_003")
    labels = set(map(str.upper, algorithms))
    lines = [line for line in figure.axes[0].lines if line.get_label() in labels]
    colors = [to_rgb(line.get_color()) for line in lines]
    distances = [
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5
        for left, right in combinations(colors, 2)
    ]

    assert min(distances) > 0.2
    assert len({line.get_linestyle() for line in lines}) >= 4
    plt.close(figure)


def test_cost_curve_selected_marker_uses_true_regret() -> None:
    module = _module()
    table = _table("v2.6.5").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
    selected = table["cycle_name"].eq("cycle_003") & table["candidate_time"].eq(
        pd.Timestamp("2026-01-01 00:16:00")
    )
    table.loc[table["cycle_name"].eq("cycle_003"), "t_star"] = pd.Timestamp("2026-01-01 00:16:00")
    table.loc[selected, "relative_regret"] = 0.009208

    figure = module._cost_curve_figure({"v2.6.5": table}, "cycle_003")

    marker_y = float(figure.axes[1].collections[0].get_offsets()[0, 1])
    assert marker_y == pytest.approx(0.9208)
    plt.close(figure)


def test_comparison_marks_extrapolated_renewal_optima() -> None:
    module = _module()
    table = _table("renewal_water").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        t_star_model_supported=lambda values: values["cycle_name"].ne("cycle_003"),
    )

    figure = module._comparison_figure({"renewal_water": table}, ("renewal_water",))

    labels = {collection.get_label() for collection in figure.axes[0].collections}
    assert "Renewal-water optimum (extrapolated)" in labels
    plt.close(figure)


def test_cost_curve_rgb_fetches_only_missing_optimal_front_members(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    tables = {
        algorithm: _table(algorithm).assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
        for algorithm in ("v1", "v2")
    }
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))

    class Loader:
        dataset_root = tmp_path / "dataset"

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            optimum = 10 if cycle_name == "cycle_003" else 12
            return pd.DataFrame(
                {
                    "camera_role": ["front"],
                    "file_name": [f"front_{optimum}.jpg"],
                    "image_time": [pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=optimum)],
                }
            )

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["camera_role", "file_name", "path"])

    requested = []

    @contextmanager
    def materialize(_dataset, cycle_name, names, **options):
        requested.append((cycle_name, tuple(names), options))
        yield tmp_path / "range"

    def scan(_dataset, cycle_name, metadata, *, cycle_dir):
        return pd.DataFrame(
            {
                "camera_role": ["front"],
                "file_name": [metadata["file_name"].iloc[0]],
                "path": [image_path],
            }
        )

    monkeypatch.setattr(module, "materialize_cycle_image_members", materialize)
    monkeypatch.setattr(module, "scan_cycle_images", scan)
    monkeypatch.setattr(module, "_save_png", lambda figure, _path: plt.close(figure))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module._render_cost_curve_comparisons(
            tables, Loader(), tmp_path, fetch_cloud=True, minimum_free_gib=5
        )

    assert requested == [
        ("cycle_003", ("front_10.jpg",), {"fetch_cloud": True, "minimum_free_gib": 5}),
        ("cycle_005", ("front_12.jpg",), {"fetch_cloud": True, "minimum_free_gib": 5}),
    ]


def test_v268_publication_cycle_can_fetch_its_missing_front_image(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    table = _table("v2.6.8").loc[lambda values: values["cycle_name"].eq("cycle_003")].copy()
    table["J_model"] = table["inverse_cop"]
    table["supported"] = True
    table["model_supported"] = True
    table["cycle_status"] = "identified_curve"
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))

    class Loader:
        dataset_root = tmp_path / "dataset"

        @staticmethod
        def load_cycle(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min")})

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "camera_role": ["front"],
                    "file_name": ["front_10.jpg"],
                    "image_time": [pd.Timestamp("2026-01-01 00:10:00")],
                }
            )

        @staticmethod
        def load_cycle_images(cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["camera_role", "file_name", "path"])

    requested = []

    @contextmanager
    def materialize(_dataset, cycle_name, names, **options):
        requested.append((cycle_name, tuple(names), options))
        yield tmp_path / "range"

    def scan(_dataset, cycle_name, metadata, *, cycle_dir):
        return pd.DataFrame(
            {"camera_role": ["front"], "file_name": ["front_10.jpg"], "path": [image_path]}
        )

    available = []
    monkeypatch.setattr(module, "materialize_cycle_image_members", materialize)
    monkeypatch.setattr(module, "scan_cycle_images", scan)
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda frame, record, curve, images, output, **kwargs: available.append(
            images["optimal"]["available"]
        ),
    )
    module._render_cycle_sets(
        {"v2.6.8": table},
        Loader(),
        {"cycle_003": {"cycle_name": "cycle_003"}},
        tmp_path,
        fetch_cloud=True,
        minimum_free_gib=5,
    )

    assert requested == [
        ("cycle_003", ("front_10.jpg",), {"fetch_cloud": True, "minimum_free_gib": 5})
    ]
    assert available == [True]


def test_v267_is_registered_and_keeps_cycles_without_an_eligible_candidate(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "v267.csv"
    table = _table("v2.6.7").assign(
        cycle_status=lambda values: values["cycle_name"].map(
            {"cycle_003": "identified_curve", "cycle_005": "model_support_limited"}
        ),
        model_supported=lambda values: values["cycle_name"].eq("cycle_003"),
        t_star_model_supported=lambda values: values["cycle_name"].eq("cycle_003"),
    )
    table.loc[table["cycle_name"].eq("cycle_005"), ["valid", "optimization_eligible", "t_star"]] = [
        False,
        False,
        pd.NaT,
    ]
    table.to_csv(path, index=False)

    loaded = module._read_tables({"V2.6.7": path})["v2.6.7"]

    assert "v2.6.7" in module.V26_PATCHES
    assert set(loaded["cycle_name"]) == {"cycle_003", "cycle_005"}


def test_v267_overview_rejects_unrecognized_status() -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="new_unmapped_status",
    )

    with pytest.raises(ValueError, match="unrecognized V2.6.7 cycle_status"):
        module._comparison_figure({"v2.6.7": table}, ("v2.6.7",))


def test_v267_cost_curve_draws_unsupported_extension_without_selecting_it() -> None:
    module = _module()
    table = (
        _table("v2.6.7")
        .loc[lambda values: values["cycle_name"].eq("cycle_003")]
        .assign(
            cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
            cycle_status="model_support_limited",
            measurement_eligible=True,
            model_supported=False,
            optimization_eligible=False,
            valid=False,
            t_star=pd.NaT,
            heating_electricity_kwh=[0.2, 0.4],
            unit_heating_kwh=[0.4, 0.8],
            E_T_hat_kwh=[0.1, 0.1],
            Q_T_hat_kwh=[0.2, 0.2],
        )
    )

    figure = module._cost_curve_figure({"v2.6.7": table}, "cycle_003")

    labels = [line.get_label() for line in figure.axes[0].lines]
    assert "V2.6.7 unsupported model extension, display only" in labels
    assert not figure.axes[0].collections
    assert "model support limited" in figure.axes[0].get_title(loc="left")
    plt.close(figure)


def test_v267_display_extension_preserves_unknown_model_support() -> None:
    module = _module()
    curve = pd.DataFrame(
        {
            "heating_electricity_kwh": [0.2, 0.2, 0.2],
            "unit_heating_kwh": [0.4, 0.4, 0.4],
            "E_T_hat_kwh": [0.1, 0.1, 0.1],
            "Q_T_hat_kwh": [0.2, 0.2, 0.2],
            "measurement_eligible": [True, True, True],
            "model_supported": pd.Series([True, False, np.nan], dtype=object),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        display = module._with_v267_display_extension(curve)[module.V267_DISPLAY_METRIC]

    assert display.iloc[:2].isna().tolist() == [True, False]
    assert display.iloc[1] == pytest.approx(0.5)
    assert pd.isna(display.iloc[2])


def test_v267_overview_marks_missing_minimum_off_the_data_axis() -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="model_support_limited",
        t_star_model_supported=False,
    )
    table.loc[table["cycle_name"].eq("cycle_003"), "t_star"] = pd.NaT

    figure = module._comparison_figure({"v2.6.7": table}, ("v2.6.7",))

    axis = figure.axes[0]
    marker = next(
        item
        for item in axis.collections
        if item.get_label() == "V2.6.7 diagnostic minimum (no diagnostic minimum)"
    )
    assert marker.get_offsets().tolist() == [[0.0, -0.04]]
    assert marker.get_offset_transform() == axis.get_xaxis_transform()
    plt.close(figure)


def test_comparison_rejects_mismatched_cycle_sets() -> None:
    module = _module()
    tables = {
        "v1": _table("v1").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
        "v2.6.7": _table("v2.6.7")
        .loc[lambda values: values["cycle_name"].eq("cycle_003")]
        .assign(
            cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
            cycle_status="identified_curve",
        ),
    }

    with pytest.raises(ValueError, match="identical cycle sets"):
        module._comparison_figure(tables, ("v1", "v2.6.7"))


def test_v267_evidence_writes_separate_bootstrap_and_loeo_pngs(tmp_path: Path) -> None:
    module = _module()
    bootstrap = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000003", "frost_cycle_000006"],
            "experiment_id": ["exp_20260101", "exp_20260102"],
            "two_candidate_repeat_fraction": [0.9, 0.7],
            "argmin_in_original_5pct_basin_fraction": [0.8, 0.6],
        }
    )
    loeo = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000003", "frost_cycle_000006"] * 2,
            "experiment_id": ["exp_20260101", "exp_20260102"] * 2,
            "target": ["E_T", "E_T", "Q_T", "Q_T"],
            "observed_kwh": [0.2, 0.4, 0.5, 0.8],
            "loeo_prediction_kwh": [0.21, 0.38, 0.52, 0.76],
            "training_mean_kwh": [0.3, 0.3, 0.65, 0.65],
            "supported": [True, False, True, False],
            "training_event_count": [10, 10, 8, 8],
            "training_experiment_count": [1, 1, 1, 1],
        }
    )

    module.generate_v267_evidence(bootstrap, loeo, tmp_path)

    assert {path.name for path in tmp_path.glob("*.png")} == {
        "bootstrap_stability_by_cycle.png",
        "ticket_E_T_loeo.png",
        "ticket_Q_T_loeo.png",
    }


def test_bootstrap_title_and_experiment_bars_follow_global_gate() -> None:
    module = _module()
    bootstrap = pd.DataFrame(
        {
            "cycle_name": [f"frost_cycle_{value:06d}" for value in range(1, 5)],
            "experiment_id": ["exp_20260101", "exp_20260101", "exp_20260102", "exp_20260102"],
            "two_candidate_repeat_fraction": [0.9, 0.9, 0.9, 0.9],
            "argmin_in_original_5pct_basin_fraction": [0.9, 0.7, 0.9, 0.9],
        }
    )

    figure = module._plot_bootstrap_stability(bootstrap)

    assert "passes the hard-label gate" in figure._suptitle.get_text()
    assert "3/4 stable (75.0%)" in figure._suptitle.get_text()
    assert "median basin hit 90.0%" in figure._suptitle.get_text()
    experiment_axis = figure.axes[1]
    assert "descriptive" in experiment_axis.get_title().lower()
    assert [to_rgb(bar.get_facecolor()) for bar in experiment_axis.patches[:2]] == [
        to_rgb("#C6C6CC"),
        to_rgb("#7884B4"),
    ]
    plt.close(figure)
