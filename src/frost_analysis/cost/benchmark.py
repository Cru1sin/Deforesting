"""Compare defrost metrics by the decisions they induce."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from frost_analysis.cost.evaluation import finalize_metric_curve
from frost_analysis.cost.outcome import (
    DYNAMIC_8,
    _long_support_runs,
    _prediction_support,
    fit_outcome_fold,
)

FINAL_METRICS = ("cop_cyc_evt", "eta_h_cyc", "eta_e_cyc")
PHYSICAL_COLUMNS = (
    "instant_water_cop",
    "instant_unit_cop",
    "instant_water_heat_kw",
    "instant_evaporator_capacity_kw",
    "heating_attenuation_fraction",
    "power_total",
    "compressor_power",
    "compressor_frequency",
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "ambient_temperature",
)


def outdoor_event_model_ablation(validation: pd.DataFrame) -> pd.DataFrame:
    """Compare direct and component-wise outdoor-event heat on identical LOEO events."""
    events = validation.loc[
        validation["event_valid"].eq(True) & validation["model_name"].eq("dynamic_8")
    ].copy()
    if events["cycle_name"].duplicated().any():
        raise ValueError("dynamic_8 validation must contain one row per event")
    events["Qe_T_observed_kwh"] = (
        events["Q_T_observed_kwh"] - events["E_comp_T_observed_kwh"]
    )
    events["Qe_T_component_prediction_kwh"] = (
        events["Q_T_prediction_kwh"] - events["E_comp_T_prediction_kwh"]
    )
    events["Qe_T_component_supported"] = (
        events["supported"].eq(True) & events["v27_E_comp_T_supported"].eq(True)
    )
    direct = []
    for experiment, heldout in events.groupby("experiment_id", sort=False):
        model = fit_outcome_fold(events, str(experiment), DYNAMIC_8, "Qe_T_observed_kwh")
        supported, _ = _prediction_support(model, heldout)
        direct.append(
            pd.DataFrame(
                {
                    "Qe_T_direct_prediction_kwh": model.predict(heldout),
                    "Qe_T_direct_supported": supported,
                    "Qe_T_direct_alpha": model.alpha,
                },
                index=heldout.index,
            )
        )
    return events.join(pd.concat(direct).sort_index())


def ch_tradeoff_diagnostic(
    c_metric: pd.DataFrame,
    h_metric: pd.DataFrame,
    *,
    epsilon_c: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05),
) -> pd.DataFrame:
    """Quantify the oracle H value available inside each native C tolerance band."""
    rows: list[dict[str, object]] = []
    for cycle, c_curve in c_metric.groupby("cycle_name", sort=False):
        h_curve = h_metric.loc[h_metric["cycle_name"].eq(cycle)]
        if h_curve.empty:
            continue
        columns = [
            "candidate_time",
            "objective_value",
            "optimization_eligible",
            "relative_optimality_gap",
            "t_star",
        ]
        values = c_curve[columns].rename(
            columns={column: f"C_{column}" for column in columns if column != "candidate_time"}
        ).merge(
            h_curve[columns].rename(
                columns={
                    column: f"H_{column}" for column in columns if column != "candidate_time"
                }
            ),
            on="candidate_time",
            validate="one_to_one",
        )
        c_time = pd.to_datetime(values["C_t_star"].iloc[0], errors="coerce")
        h_time = pd.to_datetime(values["H_t_star"].iloc[0], errors="coerce")
        values["candidate_time"] = pd.to_datetime(values["candidate_time"], errors="coerce")
        if pd.isna(c_time) or pd.isna(h_time):
            continue
        at_c = values.loc[values["candidate_time"].eq(c_time)]
        at_h = values.loc[values["candidate_time"].eq(h_time)]
        if (
            at_c.empty
            or at_h.empty
            or not bool(at_c["H_optimization_eligible"].iloc[0])
            or not bool(at_h["C_optimization_eligible"].iloc[0])
        ):
            continue
        h_at_c = float(at_c["H_objective_value"].iloc[0])
        h_at_h = float(at_h["H_objective_value"].iloc[0])
        for tolerance in epsilon_c:
            compatible = (
                values["C_optimization_eligible"].eq(True)
                & values["C_relative_optimality_gap"].le(tolerance)
                & values["H_optimization_eligible"].eq(True)
            )
            if not compatible.any() or not np.isfinite(h_at_c) or h_at_c == 0:
                continue
            best_compatible_h = float(values.loc[compatible, "H_objective_value"].max())
            gain = best_compatible_h - h_at_c
            available_gain = h_at_h - h_at_c
            rows.append(
                {
                    "cycle_name": cycle,
                    "experiment_id": c_curve.get("experiment_id", pd.Series([np.nan])).iloc[0],
                    "epsilon_C": tolerance,
                    "H_gain_upper_bound": gain / abs(h_at_c),
                    "H_gain_upper_bound_kw": gain,
                    "H_gap_recovered_fraction": (
                        gain / available_gain if available_gain > 0 else np.nan
                    ),
                    "H_regret_at_C": float(at_c["H_relative_optimality_gap"].iloc[0]),
                    "C_regret_at_H": float(at_h["C_relative_optimality_gap"].iloc[0]),
                    "compatible_candidate_count": int(compatible.sum()),
                }
            )
    return pd.DataFrame(rows)


def ch_high_value_overlap(
    c_metric: pd.DataFrame,
    h_metric: pd.DataFrame,
    *,
    epsilon_c: tuple[float, ...] = (0.005, 0.01, 0.02),
    epsilon_h: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> pd.DataFrame:
    """Describe overlap between independent native C/H near-optimal sets without selecting."""
    rows: list[dict[str, object]] = []
    for cycle, c_curve in c_metric.groupby("cycle_name", sort=False):
        h_curve = h_metric.loc[h_metric["cycle_name"].eq(cycle)]
        if (
            h_curve.empty
            or not c_curve["t_star"].notna().any()
            or not h_curve["t_star"].notna().any()
        ):
            continue
        values = c_curve[
            ["candidate_time", "optimization_eligible", "relative_optimality_gap"]
        ].rename(
            columns={"optimization_eligible": "C_eligible", "relative_optimality_gap": "C_gap"}
        ).merge(
            h_curve[["candidate_time", "optimization_eligible", "relative_optimality_gap"]].rename(
                columns={
                    "optimization_eligible": "H_eligible",
                    "relative_optimality_gap": "H_gap",
                }
            ),
            on="candidate_time",
            validate="one_to_one",
        )
        values["candidate_time"] = pd.to_datetime(values["candidate_time"], errors="coerce")
        values = values.sort_values("candidate_time", kind="stable")
        cadence = values["candidate_time"].diff().dt.total_seconds().dropna()
        cadence = float(cadence.loc[cadence.gt(0)].median())
        for c_tolerance in epsilon_c:
            c_band = values["C_eligible"].eq(True) & values["C_gap"].le(c_tolerance)
            for h_tolerance in epsilon_h:
                overlap = c_band & values["H_eligible"].eq(True) & values["H_gap"].le(h_tolerance)
                times = values.loc[overlap, "candidate_time"]
                if times.empty:
                    longest = np.nan
                else:
                    runs = times.groupby(times.diff().dt.total_seconds().gt(1.5 * cadence).cumsum())
                    longest = float(
                        runs.apply(lambda run: (run.max() - run.min()).total_seconds() / 60).max()
                    )
                rows.append(
                    {
                        "cycle_name": cycle,
                        "experiment_id": c_curve.get("experiment_id", pd.Series([np.nan])).iloc[0],
                        "epsilon_C": c_tolerance,
                        "epsilon_H": h_tolerance,
                        "longest_overlap_minutes": longest,
                        "overlap_candidate_count": int(overlap.sum()),
                        "overlap_fraction_of_C_band": (
                            float(overlap.sum() / c_band.sum()) if c_band.any() else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def absolute_rate_metric_tables(
    retention_metrics: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build counterfactual-free H/O rate ablations on the same candidate grid."""
    definitions = {
        "h_abs_rate": (
            "eta_h_cyc",
            lambda table: table["water_heating_kwh"] + table["Q_T_hat_kwh"],
            ("Q_T_supported", "D_T_supported"),
            ("water_heating_measurement_eligible",),
            "Delivered heating rate",
        ),
        "o_abs_rate": (
            "eta_e_cyc",
            lambda table: table["evaporator_heating_kwh"]
            + table["Q_T_hat_kwh"]
            - table["E_comp_T_hat_kwh"],
            ("Q_T_supported", "D_T_supported", "E_comp_T_supported"),
            (
                "water_heating_measurement_eligible",
                "heating_compressor_measurement_eligible",
            ),
            "Outdoor heat-extraction rate",
        ),
    }
    result: dict[str, pd.DataFrame] = {}
    for metric, (source, numerator_fn, support_columns, measurement_columns, label) in (
        definitions.items()
    ):
        table = retention_metrics[source].copy()
        candidate = pd.to_datetime(table["candidate_time"], errors="coerce")
        stable = pd.to_datetime(table["stable_start_fixed9"], errors="coerce")
        duration = (candidate - stable).dt.total_seconds() / 3600 + pd.to_numeric(
            table["D_T_hat_minutes"], errors="coerce"
        ) / 60
        numerator = pd.to_numeric(numerator_fn(table), errors="coerce")
        supported = table[list(support_columns)].fillna(False).all(axis=1)
        measurement = table[list(measurement_columns)].fillna(False).all(axis=1)
        table = table.assign(
            metric_id=metric,
            objective_label=label,
            objective_unit="kW",
            objective_value=numerator / duration,
            supported=supported,
            model_supported=supported,
            physical_valid=duration.gt(0) & numerator.notna(),
        )
        pieces = [
            finalize_metric_curve(curve, "max", measurement.loc[curve.index])
            for _, curve in table.groupby("cycle_name", sort=False)
        ]
        result[metric] = pd.concat(
            [piece.dropna(axis=1, how="all") for piece in pieces],
            ignore_index=True,
            sort=False,
        )
    return result


def bootstrap_absolute_rate_trajectories(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Evaluate absolute H/O rates under each existing bootstrap model refit."""
    columns = (
        "replicate_id",
        "cycle_name",
        "candidate_time",
        "stable_start_fixed9",
        "pre_action_window_valid",
        "water_heating_kwh",
        "heating_compressor_electricity_kwh",
        "Q_T",
        "E_comp_T",
        "D_T",
        "support_Q_T",
        "support_D_T",
        "support_E_comp_T",
        "eta_h_cyc_measurement_eligible",
        "eta_e_cyc_measurement_eligible",
    )
    result = trajectories[list(columns)].copy()
    candidate = pd.to_datetime(result["candidate_time"], errors="coerce")
    stable = pd.to_datetime(result["stable_start_fixed9"], errors="coerce")
    duration = (candidate - stable).dt.total_seconds() / 3600 + pd.to_numeric(
        result["D_T"], errors="coerce"
    ) / 60
    numerators = {
        "h_abs_rate": result["water_heating_kwh"] + result["Q_T"],
        "o_abs_rate": result["water_heating_kwh"]
        - result["heating_compressor_electricity_kwh"]
        + result["Q_T"]
        - result["E_comp_T"],
    }
    supports = {
        "h_abs_rate": result[["support_Q_T", "support_D_T"]].all(axis=1),
        "o_abs_rate": result[
            ["support_Q_T", "support_D_T", "support_E_comp_T"]
        ].all(axis=1),
    }
    measurements = {
        "h_abs_rate": result["eta_h_cyc_measurement_eligible"],
        "o_abs_rate": result["eta_e_cyc_measurement_eligible"],
    }
    for metric, numerator in numerators.items():
        objective = pd.to_numeric(numerator, errors="coerce") / duration
        base = (
            supports[metric].fillna(False)
            & measurements[metric].fillna(False)
            & result["pre_action_window_valid"].fillna(False)
            & duration.gt(0)
            & objective.notna()
        )
        eligible = pd.Series(False, index=result.index)
        for _, indices in result.groupby(["replicate_id", "cycle_name"], sort=False).groups.items():
            eligible.loc[indices] = _long_support_runs(
                result.loc[indices, "candidate_time"], base.loc[indices]
            ).to_numpy()
        result[metric] = objective
        result[f"{metric}_eligible"] = eligible
    return result


def final_metric_tables(
    complete_cycle: pd.DataFrame,
    heating: pd.DataFrame,
    outdoor: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the three larger-is-better curves on the shared candidate grid."""
    cop = complete_cycle.copy()
    observed = heating.drop_duplicates(["cycle_name", "candidate_time"])[
        [
            "cycle_name",
            "candidate_time",
            *[column for column in PHYSICAL_COLUMNS if column in heating],
        ]
    ]
    cop = cop.drop(columns=list(PHYSICAL_COLUMNS), errors="ignore").merge(
        observed,
        on=["cycle_name", "candidate_time"],
        how="left",
        validate="one_to_one",
    )
    cop["metric_id"] = "cop_cyc_evt"
    cop["objective_value"] = pd.to_numeric(cop["cycle_cop"], errors="coerce")
    cop["optimization_direction"] = "max"
    cop["objective_label"] = "Complete-cycle COP"
    cop["objective_unit"] = "-"
    best = cop["objective_value"].where(cop["optimization_eligible"].fillna(False)).groupby(
        cop["cycle_name"]
    ).transform("max")
    cop["relative_optimality_gap"] = (best - cop["objective_value"]) / best.abs()
    cop["display_only_objective"] = cop["objective_value"].where(
        ~cop["optimization_eligible"].fillna(False)
    )
    return {
        "cop_cyc_evt": cop,
        "eta_h_cyc": heating.loc[heating["metric_id"].eq("eta_h_cyc")].copy(),
        "eta_e_cyc": outdoor.loc[outdoor["metric_id"].eq("eta_e_cyc")].copy(),
    }


def benchmark_table(metrics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize each cycle's optimum, basin widths, support, and selected state."""
    rows: list[dict[str, object]] = []
    for metric_id, table in metrics.items():
        for cycle_name, curve in table.groupby("cycle_name", sort=False):
            curve = curve.copy()
            curve["candidate_time"] = pd.to_datetime(curve["candidate_time"], errors="coerce")
            target = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
            selected = (
                curve.loc[(curve["candidate_time"] - target).abs().idxmin()]
                if pd.notna(target)
                else curve.iloc[0]
            )
            cycle_start = pd.to_datetime(selected.get("cycle_start"), errors="coerce")
            actual = pd.to_datetime(selected.get("actual_preparation_time"), errors="coerce")
            stable = pd.to_datetime(selected.get("stable_start_fixed9"), errors="coerce")
            valid = curve.get("pre_action_window_valid", pd.Series(True, index=curve.index))
            eligible = curve["optimization_eligible"].fillna(False)
            eligible_times = curve.loc[eligible, "candidate_time"].dropna()
            location = (
                "unidentified"
                if pd.isna(target) or eligible_times.empty
                else "left_boundary"
                if target == eligible_times.min()
                else "right_boundary"
                if target == eligible_times.max()
                else "interior"
            )
            row = {
                "cycle_name": cycle_name,
                "experiment_id": selected.get("experiment_id"),
                "metric_id": metric_id,
                "t_star": target,
                "t_star_cycle_minutes": (
                    (target - cycle_start).total_seconds() / 60
                    if pd.notna(target) and pd.notna(cycle_start)
                    else np.nan
                ),
                "minutes_before_actual_defrost": (
                    (actual - target).total_seconds() / 60
                    if pd.notna(target) and pd.notna(actual)
                    else np.nan
                ),
                "frosting_progress": (
                    (target - stable) / (actual - stable)
                    if pd.notna(target) and pd.notna(stable) and actual > stable
                    else np.nan
                ),
                "W1_minutes": _width(curve, 1),
                "W2_minutes": _width(curve, 2),
                "W5_minutes": _width(curve, 5),
                "extreme_location": location,
                "support_fraction": float(eligible.sum() / max(valid.fillna(False).sum(), 1)),
                "eligible_candidates": int(eligible.sum()),
                "selected_objective": selected.get("objective_value"),
            }
            row.update({column: selected.get(column, np.nan) for column in PHYSICAL_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def cross_objective_regret(
    metrics: Mapping[str, pd.DataFrame],
    *,
    basin_percents: tuple[int, ...] = (1, 2, 5),
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Evaluate every metric-selected decision under all three objectives."""
    rows: list[dict[str, object]] = []
    cycles = sorted(set.intersection(*(set(table["cycle_name"]) for table in metrics.values())))
    for cycle_name in cycles:
        curves = {
            metric: _indexed_curve(table, cycle_name) for metric, table in metrics.items()
        }
        common = set.intersection(*(set(curve.index) for curve in curves.values()))
        if not common:
            continue
        regret = pd.DataFrame(
            {
                metric: _regret(curves[metric].loc[sorted(common)])
                for metric in metric_order
            },
            index=sorted(common),
        )
        eligible = pd.DataFrame(
            {
                metric: curves[metric].loc[sorted(common), "optimization_eligible"].fillna(False)
                for metric in metric_order
            },
            index=sorted(common),
        )
        if not eligible.any(axis=None):
            continue
        for selector in metric_order:
            point = pd.to_datetime(curves[selector]["t_star"].iloc[0], errors="coerce")
            _append_regrets(
                rows,
                cycle_name,
                selector,
                "point",
                point,
                regret,
                eligible,
                metric_order,
            )
            for percent in basin_percents:
                selector_curve = curves[selector].loc[sorted(common)]
                near = selector_curve.get(
                    f"near_optimal_{percent}pct",
                    _regret(selector_curve).le(percent / 100),
                ).astype("boolean").fillna(False).astype(bool)
                choices = regret.index[eligible[selector] & near]
                if len(choices):
                    _append_regrets(
                        rows,
                        cycle_name,
                        selector,
                        f"latest_W{percent}",
                        choices[-1],
                        regret,
                        eligible,
                        metric_order,
                    )
    return pd.DataFrame(rows)


def regret_coverage(regret: pd.DataFrame) -> pd.DataFrame:
    """Report target availability relative to each selector's own decisions."""
    keys = ["selector_metric", "decision_type"]
    denominators = (
        regret.groupby(keys)["cycle_name"].nunique().rename("selector_decisions")
    )
    available = (
        regret.groupby([*keys, "target_metric"])["cycle_name"]
        .nunique()
        .rename("available_cycles")
        .reset_index()
        .join(denominators, on=keys)
    )
    available["coverage_fraction"] = (
        available["available_cycles"] / available["selector_decisions"]
    )
    return available


def matched_decision_regret(
    regret: pd.DataFrame,
    decision_type: str,
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Keep cycles with every independently selected decision scored by every objective."""
    selected = regret.loc[regret["decision_type"].eq(decision_type)].copy()
    expected = len(metric_order) ** 2
    complete = (
        selected.loc[
            selected["selector_metric"].isin(metric_order)
            & selected["target_metric"].isin(metric_order)
        ]
        .drop_duplicates(["cycle_name", "selector_metric", "target_metric"])
        .groupby("cycle_name")
        .size()
        .eq(expected)
    )
    return selected.loc[selected["cycle_name"].isin(complete.index[complete])].copy()


def pareto_nondominated(consequences: pd.DataFrame) -> pd.Series:
    """Return standard minimization Pareto membership for complete consequence vectors."""
    valid = consequences.dropna()
    result = pd.Series(False, index=consequences.index, dtype=bool)
    for index, row in valid.iterrows():
        others = valid.drop(index)
        dominated = (others.le(row).all(axis=1) & others.lt(row).any(axis=1)).any()
        result.loc[index] = not dominated
    return result


def bootstrap_validity_taxonomy(
    trajectories: pd.DataFrame,
    metrics: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Separate undefined formulas, unsupported curves, and valid extrema."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        eligible_column = f"{metric}_eligible"
        for (replicate, cycle), curve in trajectories.groupby(
            ["replicate_id", "cycle_name"], sort=False
        ):
            values = pd.to_numeric(curve[metric], errors="coerce")
            eligible = curve[eligible_column].fillna(False) & values.notna()
            if not values.notna().any():
                status = "formula_unavailable"
            elif not eligible.any():
                status = "support_or_measurement_limited"
            else:
                supported = curve.loc[eligible].sort_values("candidate_time", kind="stable")
                selected = pd.to_numeric(supported[metric], errors="coerce").idxmax()
                status = (
                    "valid_endpoint"
                    if selected in {supported.index[0], supported.index[-1]}
                    else "valid_interior"
                )
            rows.append(
                {
                    "replicate_id": replicate,
                    "cycle_name": cycle,
                    "metric_id": metric,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_failure_anatomy(  # noqa: C901
    trajectories: pd.DataFrame,
    metrics: tuple[str, ...] = ("eta_h_cyc", "eta_e_cyc"),
) -> pd.DataFrame:
    """Explain each bootstrap curve's estimability without collapsing components."""
    requirements = {
        "eta_h_cyc": ("Q_T", "Qw0", "D_T"),
        "eta_e_cyc": ("Q_T", "Qw0", "D_T", "Pcomp0", "E_comp_T"),
    }
    rows: list[dict[str, object]] = []
    for metric in metrics:
        required = requirements[metric]
        for (replicate, cycle), curve in trajectories.groupby(
            ["replicate_id", "cycle_name"], sort=False
        ):
            values = pd.to_numeric(curve[metric], errors="coerce")
            component_any = {
                name: bool(curve[f"support_{name}"].fillna(False).any())
                for name in required
            }
            missing = [name for name, available in component_any.items() if not available]
            flags = {
                name: bool(curve[f"{metric}_{name}"].fillna(False).any())
                for name in (
                    "model_supported",
                    "measurement_eligible",
                    "physical_valid",
                    "base_eligible",
                    "eligible",
                )
            }
            eligible = curve[f"{metric}_eligible"].fillna(False) & values.notna()
            if not values.notna().any():
                reason = "formula_unavailable"
            elif missing:
                reason = "+".join(f"{name}_support" for name in missing)
            elif not flags["model_supported"]:
                reason = "joint_model_support_fragmented"
            elif not flags["measurement_eligible"]:
                reason = "measurement_failed"
            elif not flags["physical_valid"]:
                reason = "physical_failed"
            elif not flags["base_eligible"]:
                reason = "joint_gate_fragmented"
            elif not eligible.any():
                reason = "continuous_support_lt5min"
            else:
                supported = curve.loc[eligible].sort_values("candidate_time", kind="stable")
                selected = pd.to_numeric(supported[metric], errors="coerce").idxmax()
                reason = (
                    "valid_endpoint"
                    if selected in {supported.index[0], supported.index[-1]}
                    else "valid_interior"
                )
            row: dict[str, object] = {
                "replicate_id": replicate,
                "cycle_name": cycle,
                "experiment_id": curve["experiment_id"].iloc[0],
                "metric_id": metric,
                "valid": reason.startswith("valid_"),
                "failure_reason": reason,
                "formula_available": bool(values.notna().any()),
                **{f"{name}_any": value for name, value in flags.items()},
            }
            for name in requirements["eta_e_cyc"]:
                column = f"support_{name}"
                row[f"support_{name}_any"] = (
                    bool(curve[column].fillna(False).any()) if column in curve else np.nan
                )
                row[f"support_{name}_fraction"] = (
                    float(curve[column].fillna(False).mean()) if column in curve else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ho_cofailure(anatomy: pd.DataFrame) -> pd.DataFrame:
    """Quantify whether H and O become inestimable in the same draws."""
    paired = anatomy.pivot_table(
        index=["replicate_id", "cycle_name"],
        columns="metric_id",
        values="valid",
        aggfunc="first",
    ).dropna(subset=["eta_h_cyc", "eta_e_cyc"])
    h_invalid = ~paired["eta_h_cyc"].astype(bool)
    o_invalid = ~paired["eta_e_cyc"].astype(bool)
    return pd.DataFrame(
        {
            "statistic": (
                "P(H_invalid|O_invalid)",
                "P(O_invalid|H_invalid)",
                "P(H_and_O_invalid)",
            ),
            "value": (
                float(h_invalid[o_invalid].mean()) if o_invalid.any() else np.nan,
                float(o_invalid[h_invalid].mean()) if h_invalid.any() else np.nan,
                float((h_invalid & o_invalid).mean()),
            ),
            "n": (int(o_invalid.sum()), int(h_invalid.sum()), len(paired)),
        }
    )


def experiment_leverage(anatomy: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    """Estimate how omitting one source experiment changes invalidity."""
    merged = anatomy.merge(
        draws,
        left_on=["replicate_id", "experiment_id"],
        right_on=["replicate_id", "heldout_experiment_id"],
        how="inner",
        validate="many_to_many",
    )
    merged["invalid"] = ~merged["valid"].astype(bool)
    merged["present"] = merged["draw_count"].gt(0)
    rows: list[dict[str, object]] = []
    for (metric, source), values in merged.groupby(
        ["metric_id", "source_experiment_id"], sort=False
    ):
        absent = values.loc[~values["present"], "invalid"]
        present = values.loc[values["present"], "invalid"]
        rows.append(
            {
                "metric_id": metric,
                "source_experiment_id": source,
                "absent_n": len(absent),
                "present_n": len(present),
                "invalid_if_absent": float(absent.mean()) if len(absent) else np.nan,
                "invalid_if_present": float(present.mean()) if len(present) else np.nan,
                "leverage": (
                    float(absent.mean() - present.mean())
                    if len(absent) and len(present)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def same_cycle_regret(
    regret: pd.DataFrame,
    decisions: tuple[str, ...] = ("point", "latest_W1", "latest_W2", "latest_W5"),
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Keep one common complete cycle subset for every decision semantic."""
    matched = [matched_decision_regret(regret, decision, metric_order) for decision in decisions]
    cycles = set.intersection(*(set(table["cycle_name"]) for table in matched))
    return pd.concat(matched, ignore_index=True).loc[lambda table: table.cycle_name.isin(cycles)]


def regret_distribution(
    regret: pd.DataFrame,
    decision_type: str = "point",
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Summarize matched-cycle regret tails, not only medians."""
    matched = matched_decision_regret(regret, decision_type, metric_order)
    rows: list[dict[str, object]] = []
    for (selector, target), values in matched.groupby(
        ["selector_metric", "target_metric"], sort=False
    ):
        r = pd.to_numeric(values["cross_objective_regret"], errors="coerce").dropna()
        rows.append(
            {
                "decision_type": decision_type,
                "selector_metric": selector,
                "target_metric": target,
                "n_cycles": len(r),
                "median_regret": float(r.median()),
                "p75_regret": float(r.quantile(0.75)),
                "p90_regret": float(r.quantile(0.90)),
                "p95_regret": float(r.quantile(0.95)),
                "P_regret_lt_0.5pct": float(r.lt(0.005).mean()),
                "P_regret_lt_1pct": float(r.lt(0.01).mean()),
                "P_regret_lt_2pct": float(r.lt(0.02).mean()),
            }
        )
    return pd.DataFrame(rows)


def ho_paired_decisions(
    regret: pd.DataFrame,
    *,
    h_metric: str = "eta_h_cyc",
    o_metric: str = "eta_e_cyc",
    c_metric: str = "cop_cyc_evt",
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Compare H and O point decisions cycle-by-cycle on common evidence."""
    point = matched_decision_regret(regret, "point", metric_order)
    values = point.pivot_table(
        index="cycle_name",
        columns=["selector_metric", "target_metric"],
        values="cross_objective_regret",
        aggfunc="first",
    )
    times = point.pivot_table(
        index="cycle_name",
        columns="selector_metric",
        values="decision_time",
        aggfunc="first",
    ).apply(pd.to_datetime)
    rows: list[dict[str, object]] = []
    for cycle in values.index.intersection(times.index):
        h_time = times.loc[cycle, h_metric]
        o_time = times.loc[cycle, o_metric]
        h_in_o = float(values.loc[cycle, (h_metric, o_metric)])
        o_in_h = float(values.loc[cycle, (o_metric, h_metric)])
        row: dict[str, object] = {
            "cycle_name": cycle,
            "t_H": h_time,
            "t_O": o_time,
            "delta_t_O_minus_H_minutes": (o_time - h_time).total_seconds() / 60,
            "abs_delta_t_minutes": abs((o_time - h_time).total_seconds() / 60),
            "C_regret_at_H": float(values.loc[cycle, (h_metric, c_metric)]),
            "C_regret_at_O": float(values.loc[cycle, (o_metric, c_metric)]),
            "H_regret_at_O": o_in_h,
            "O_regret_at_H": h_in_o,
        }
        row["delta_C_regret_O_minus_H"] = row["C_regret_at_O"] - row["C_regret_at_H"]
        for percent in (1, 2, 5):
            row[f"H_in_O_W{percent}"] = h_in_o <= percent / 100
            row[f"O_in_H_W{percent}"] = o_in_h <= percent / 100
        rows.append(row)
    return pd.DataFrame(rows)


def stability_to_basin_ratio(
    benchmark: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    """Compare optimum-location uncertainty with each cycle's 5% value basin."""
    result = stability.merge(
        benchmark[["cycle_name", "metric_id", "W5_minutes"]],
        on=["cycle_name", "metric_id"],
        how="left",
        validate="one_to_one",
    )
    width = pd.to_numeric(result["W5_minutes"], errors="coerce")
    result["rho_IQR_over_W5"] = pd.to_numeric(
        result["IQR_tau_minutes"], errors="coerce"
    ) / width.where(width.gt(0))
    result["uncertainty_within_W5"] = result["rho_IQR_over_W5"].lt(1)
    return result


def bootstrap_fixed_support_stability(
    trajectories: pd.DataFrame,
    point_metrics: Mapping[str, pd.DataFrame],
    metrics: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Diagnostic bootstrap with candidate support frozen to the point estimate."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        point_table = point_metrics[metric]
        for cycle, values in trajectories.groupby("cycle_name", sort=False):
            point = point_table.loc[point_table["cycle_name"].eq(cycle)].copy()
            point["candidate_time"] = pd.to_datetime(point["candidate_time"], errors="coerce")
            allowed = set(point.loc[point["optimization_eligible"].fillna(False), "candidate_time"])
            point_time = pd.to_datetime(point["t_star"].iloc[0], errors="coerce")
            optima: list[pd.Timestamp] = []
            self_regrets: list[float] = []
            for _, replicate in values.groupby("replicate_id", sort=False):
                candidate_time = pd.to_datetime(replicate["candidate_time"], errors="coerce")
                objective = pd.to_numeric(replicate[metric], errors="coerce")
                valid = candidate_time.isin(allowed) & objective.notna()
                if not valid.any():
                    continue
                curve = replicate.loc[valid].assign(
                    _time=candidate_time[valid], _value=objective[valid]
                )
                best_index = curve["_value"].idxmax()
                best_time = pd.Timestamp(curve.loc[best_index, "_time"])
                best_value = float(curve.loc[best_index, "_value"])
                optima.append(best_time)
                if pd.notna(point_time):
                    nearest = (curve["_time"] - point_time).abs().idxmin()
                    self_regrets.append(
                        (best_value - float(curve.loc[nearest, "_value"])) / abs(best_value)
                    )
            minutes = np.array([time.value / 60e9 for time in optima])
            rows.append(
                {
                    "cycle_name": cycle,
                    "metric_id": metric,
                    "bootstrap_mode": "fixed_point_support",
                    "valid_replicates": len(optima),
                    "valid_fraction": len(optima) / max(values["replicate_id"].nunique(), 1),
                    "MAD_tau_minutes": (
                        float(np.median(np.abs(minutes - np.median(minutes))))
                        if len(minutes)
                        else np.nan
                    ),
                    "IQR_tau_minutes": (
                        float(np.quantile(minutes, 0.75) - np.quantile(minutes, 0.25))
                        if len(minutes)
                        else np.nan
                    ),
                    "median_self_regret": (
                        float(np.median(self_regrets)) if self_regrets else np.nan
                    ),
                    "p90_self_regret": (
                        float(np.quantile(self_regrets, 0.9)) if self_regrets else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def local_ratio_attribution(
    curves: pd.DataFrame,
    window_minutes: float = 5.0,
) -> pd.DataFrame:
    """Exactly split local changes in E/Q into heating and event components."""
    rows: list[dict[str, object]] = []
    for cycle, curve in curves.groupby("cycle_name", sort=False):
        values = curve.loc[curve["optimization_eligible"].fillna(False)].copy()
        values["candidate_time"] = pd.to_datetime(values["candidate_time"], errors="coerce")
        values = values.dropna(subset=["candidate_time"]).sort_values("candidate_time")
        optimum = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
        if values.empty or pd.isna(optimum):
            continue

        def nearest(target: pd.Timestamp, source: pd.DataFrame = values) -> pd.Series:
            return source.loc[(source["candidate_time"] - target).abs().idxmin()]

        center = nearest(optimum)
        endpoints = (
            ("before_to_optimum", nearest(optimum - pd.Timedelta(minutes=window_minutes)), center),
            ("optimum_to_after", center, nearest(optimum + pd.Timedelta(minutes=window_minutes))),
        )
        for segment, first, second in endpoints:
            duration = (second["candidate_time"] - first["candidate_time"]).total_seconds() / 60
            if duration <= 0:
                continue
            eh1, eh2 = (
                float(first["heating_electricity_kwh"]),
                float(second["heating_electricity_kwh"]),
            )
            et1, et2 = (float(first["E_T_hat_kwh"]), float(second["E_T_hat_kwh"]))
            qh1, qh2 = (float(first["water_heating_kwh"]), float(second["water_heating_kwh"]))
            qt1, qt2 = (float(first["Q_T_hat_kwh"]), float(second["Q_T_hat_kwh"]))
            e1, e2, q1, q2 = eh1 + et1, eh2 + et2, qh1 + qt1, qh2 + qt2
            if min(q1, q2) <= 0 or not np.isfinite([e1, e2, q1, q2]).all():
                continue
            inverse_heat = 0.5 * (1 / q1 + 1 / q2)
            heat_total = 0.5 * (e1 + e2) * (1 / q2 - 1 / q1)
            delta_q = q2 - q1
            heating_heat = heat_total * (qh2 - qh1) / delta_q if delta_q else 0.0
            event_heat = heat_total - heating_heat
            rows.append(
                {
                    "cycle_name": cycle,
                    "segment": segment,
                    "start_time": first["candidate_time"],
                    "end_time": second["candidate_time"],
                    "duration_minutes": duration,
                    "delta_inverse_cop": e2 / q2 - e1 / q1,
                    "heating_energy_contribution": (eh2 - eh1) * inverse_heat,
                    "event_energy_contribution": (et2 - et1) * inverse_heat,
                    "heating_heat_contribution": heating_heat,
                    "event_heat_contribution": event_heat,
                }
            )
    return pd.DataFrame(rows)


def cycle_trigger_validation(
    predictions: pd.DataFrame,
    metrics: Mapping[str, pd.DataFrame],
    *,
    metric_order: tuple[str, ...] = FINAL_METRICS,
    threshold: float = 0.5,
    consecutive: int = 3,
    max_gap_seconds: float = 90.0,
) -> pd.DataFrame:
    """Convert held-out frame probabilities into cycle trigger decisions."""
    indexed = {
        (metric, cycle): _indexed_curve(table, cycle)
        for metric, table in metrics.items()
        for cycle in table["cycle_name"].unique()
    }
    primary = metric_order[0]
    rows: list[dict[str, object]] = []
    group_columns = [column for column in ("modality", "cycle_name") if column in predictions]
    for keys, source in predictions.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metadata = dict(zip(group_columns, keys, strict=True))
        values = source.loc[
            source.get("fold_evaluable", pd.Series(True, index=source.index)).fillna(False)
            & source["decision_score"].notna()
        ].copy()
        values["image_time"] = pd.to_datetime(values["image_time"], errors="coerce")
        values = values.dropna(subset=["image_time"]).sort_values("image_time", kind="stable")
        run = 0
        previous = pd.NaT
        trigger = pd.NaT
        for _, frame in values.iterrows():
            current = frame["image_time"]
            continuous = (
                pd.isna(previous)
                or (current - previous).total_seconds() <= max_gap_seconds
            )
            run = run + 1 if bool(frame["decision_score"] > threshold) and continuous else int(
                frame["decision_score"] > threshold
            )
            previous = current
            if run >= consecutive:
                trigger = current
                break
        primary_curve = indexed.get((primary, str(metadata["cycle_name"])))
        oracle = (
            pd.to_datetime(primary_curve["t_star"].iloc[0], errors="coerce")
            if primary_curve is not None and not primary_curve.empty
            else pd.NaT
        )
        row: dict[str, object] = {
            **metadata,
            "threshold": threshold,
            "consecutive_frames": consecutive,
            "trigger_time": trigger,
            "oracle_time": oracle,
            "triggered": pd.notna(trigger),
            "signed_error_minutes": (
                (trigger - oracle).total_seconds() / 60
                if pd.notna(trigger) and pd.notna(oracle)
                else np.nan
            ),
        }
        row["absolute_error_minutes"] = abs(row["signed_error_minutes"])
        if primary_curve is not None:
            for percent in (1, 2, 5):
                start, end = _basin_bounds(primary_curve, percent)
                row[f"W{percent}_hit"] = bool(
                    pd.notna(trigger) and pd.notna(start) and start <= trigger <= end
                )
        for metric in metric_order:
            curve = indexed.get((metric, str(metadata["cycle_name"])))
            row[f"regret_{metric}"] = _regret_at_time(curve, trigger)
            row[f"oracle_C_regret_{metric}"] = _regret_at_time(curve, oracle)
            row[f"implementation_penalty_{metric}"] = (
                row[f"regret_{metric}"] - row[f"oracle_C_regret_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _basin_bounds(curve: pd.DataFrame, percent: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_column = f"basin_{percent}pct_start"
    end_column = f"basin_{percent}pct_end"
    if start_column in curve and end_column in curve:
        return (
            pd.to_datetime(curve[start_column].iloc[0], errors="coerce"),
            pd.to_datetime(curve[end_column].iloc[0], errors="coerce"),
        )
    optimum = pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
    near = (
        curve["optimization_eligible"].fillna(False)
        & _regret(curve).le(percent / 100)
    ).to_numpy()
    if pd.isna(optimum) or not near.any():
        return pd.NaT, pd.NaT
    center = int(np.abs((curve.index - optimum).total_seconds()).argmin())
    if not near[center]:
        return pd.NaT, pd.NaT
    left = right = center
    while left and near[left - 1]:
        left -= 1
    while right + 1 < len(near) and near[right + 1]:
        right += 1
    return pd.Timestamp(curve.index[left]), pd.Timestamp(curve.index[right])


def _regret_at_time(curve: pd.DataFrame | None, decision: pd.Timestamp) -> float:
    if curve is None or curve.empty or pd.isna(decision):
        return np.nan
    position = np.abs((curve.index - decision).total_seconds()).argmin()
    if abs((curve.index[position] - decision).total_seconds()) > 61:
        return np.nan
    selected = curve.iloc[position]
    if not bool(selected["optimization_eligible"]):
        return np.nan
    return float(_regret(curve).iloc[position])


def bootstrap_stability(
    trajectories: pd.DataFrame,
    point_metrics: Mapping[str, pd.DataFrame],
    metrics: tuple[str, ...] = FINAL_METRICS,
) -> pd.DataFrame:
    """Compute optimum-time dispersion and point-decision self-regret."""
    point_times = {
        (metric, cycle): pd.to_datetime(curve["t_star"].iloc[0], errors="coerce")
        for metric, table in point_metrics.items()
        for cycle, curve in table.groupby("cycle_name", sort=False)
    }
    rows: list[dict[str, object]] = []
    for metric in metrics:
        eligible_column = f"{metric}_eligible"
        for cycle, values in trajectories.groupby("cycle_name", sort=False):
            optima: list[pd.Timestamp] = []
            self_regrets: list[float] = []
            point = point_times.get((metric, cycle), pd.NaT)
            for _, replicate in values.groupby("replicate_id", sort=False):
                valid = replicate[eligible_column].fillna(False) & replicate[metric].notna()
                if not valid.any():
                    continue
                curve = replicate.loc[valid].sort_values("candidate_time", kind="stable")
                best_index = curve[metric].idxmax()
                best = float(curve.loc[best_index, metric])
                optima.append(pd.Timestamp(curve.loc[best_index, "candidate_time"]))
                if pd.notna(point):
                    nearest = (pd.to_datetime(curve["candidate_time"]) - point).abs().idxmin()
                    self_regrets.append((best - float(curve.loc[nearest, metric])) / abs(best))
            minutes = np.array([time.value / 60e9 for time in optima])
            rows.append(
                {
                    "cycle_name": cycle,
                    "metric_id": metric,
                    "valid_replicates": len(optima),
                    "valid_fraction": len(optima) / max(values["replicate_id"].nunique(), 1),
                    "MAD_tau_minutes": (
                        float(np.median(np.abs(minutes - np.median(minutes))))
                        if len(minutes)
                        else np.nan
                    ),
                    "IQR_tau_minutes": (
                        float(np.quantile(minutes, 0.75) - np.quantile(minutes, 0.25))
                        if len(minutes)
                        else np.nan
                    ),
                    "median_self_regret": (
                        float(np.median(self_regrets)) if self_regrets else np.nan
                    ),
                    "p90_self_regret": (
                        float(np.quantile(self_regrets, 0.9)) if self_regrets else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _width(curve: pd.DataFrame, percent: int) -> float:
    column = f"basin_{percent}pct_width_minutes"
    if column in curve:
        return float(pd.to_numeric(curve[column], errors="coerce").iloc[0])
    near = _regret(curve).le(percent / 100) & curve["optimization_eligible"].fillna(False)
    times = pd.to_datetime(curve.loc[near, "candidate_time"], errors="coerce")
    return (times.max() - times.min()).total_seconds() / 60 if len(times) else np.nan


def _indexed_curve(table: pd.DataFrame, cycle_name: str) -> pd.DataFrame:
    curve = table.loc[table["cycle_name"].eq(cycle_name)].copy()
    curve["candidate_time"] = pd.to_datetime(curve["candidate_time"], errors="coerce")
    return curve.dropna(subset=["candidate_time"]).drop_duplicates("candidate_time").set_index(
        "candidate_time"
    )


def _regret(curve: pd.DataFrame) -> pd.Series:
    if "relative_optimality_gap" in curve:
        return pd.to_numeric(curve["relative_optimality_gap"], errors="coerce")
    objective = pd.to_numeric(curve["objective_value"], errors="coerce")
    best = objective.loc[curve["optimization_eligible"].fillna(False)].max()
    return (best - objective) / abs(best)


def _append_regrets(
    rows: list[dict[str, object]],
    cycle: str,
    selector: str,
    decision_type: str,
    decision: pd.Timestamp,
    regret: pd.DataFrame,
    eligible: pd.DataFrame,
    metric_order: tuple[str, ...] = FINAL_METRICS,
) -> None:
    if (
        pd.isna(decision)
        or decision not in regret.index
        or not eligible.loc[decision, selector]
    ):
        return
    for target in metric_order:
        if not eligible.loc[decision, target]:
            continue
        rows.append(
            {
                "cycle_name": cycle,
                "selector_metric": selector,
                "target_metric": target,
                "decision_type": decision_type,
                "decision_time": decision,
                "cross_objective_regret": float(regret.loc[decision, target]),
            }
        )
