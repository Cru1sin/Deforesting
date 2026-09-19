"""Held-out event electricity and effective heating COP calibration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defrost_event_models.ridge_models import OUTCOME_TARGETS


def _export(figure: plt.Figure, stem: Path, formats: tuple[str, ...] = ("png",)) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        figure.savefig(
            stem.with_suffix(f".{figure_format}"),
            dpi=300 if figure_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def event_calibration_metrics(predictions, targets, *, bootstrap_replicates=0):
    """Summarize OOF calibration without pooling repeated seeds as new events."""
    group_columns = [
        *(["panel"] if "panel" in predictions else []),
        "representation",
        *(["seed"] if "seed" in predictions else []),
    ]
    records = []
    for keys, rows in predictions.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group = dict(zip(group_columns, keys, strict=True))
        for target in targets:
            observed = rows[target].to_numpy(dtype=float)
            predicted = rows[f"predicted_{target}"].to_numpy(dtype=float)
            valid = np.isfinite(observed) & np.isfinite(predicted)
            observed, predicted = observed[valid], predicted[valid]

            def summarize(observed_values, predicted_values):
                slope, intercept = (
                    np.polyfit(predicted_values, observed_values, 1)
                    if len(predicted_values) > 1 and np.ptp(predicted_values) > 0
                    else (np.nan, np.nan)
                )
                residual = predicted_values - observed_values
                return np.array(
                    [
                        residual.mean(),
                        np.abs(residual).mean(),
                        np.sqrt(np.square(residual).mean()),
                        intercept,
                        slope,
                    ]
                )

            estimates = summarize(observed, predicted)
            intervals = np.full((2, 5), np.nan)
            if bootstrap_replicates and len(observed):
                valid_rows = rows.loc[valid].reset_index(drop=True)
                experiments = valid_rows.experiment_id.drop_duplicates().to_numpy()
                rng = np.random.default_rng(0)
                draws = []
                for _ in range(bootstrap_replicates):
                    sampled = rng.choice(experiments, len(experiments), replace=True)
                    sampled_rows = pd.concat(
                        [
                            valid_rows.loc[valid_rows.experiment_id.eq(experiment)]
                            for experiment in sampled
                        ],
                        ignore_index=True,
                    )
                    draws.append(
                        summarize(
                            sampled_rows[target].to_numpy(dtype=float),
                            sampled_rows[f"predicted_{target}"].to_numpy(dtype=float),
                        )
                    )
                intervals = np.nanquantile(np.asarray(draws), [0.025, 0.975], axis=0)
            records.append(
                {
                    **group,
                    "target": target,
                    "events": len(observed),
                    "experiments": rows.loc[valid, "experiment_id"].nunique(),
                    **{
                        name: estimates[index]
                        for index, name in enumerate(
                            (
                                "bias",
                                "mae",
                                "rmse",
                                "calibration_intercept",
                                "calibration_slope",
                            )
                        )
                    },
                    **{
                        f"{name}_ci_{bound}": intervals[index, position]
                        for index, bound in enumerate(("low", "high"))
                        for position, name in enumerate(
                            (
                                "bias",
                                "mae",
                                "rmse",
                                "calibration_intercept",
                                "calibration_slope",
                            )
                        )
                    },
                }
            )
    return pd.DataFrame(records)


def render_calibration_figures(
    predictions,
    event_starts,
    output,
    *,
    targets=None,
    labels=None,
    bootstrap_replicates=2000,
    focus_model="ridge_dynamic_state_8",
):
    """Render OOF event and observed-action C/H calibration with true denominators."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = list(OUTCOME_TARGETS.values()) if targets is None else targets
    labels = {} if labels is None else labels
    if focus_model is not None:
        predictions = predictions.loc[
            predictions.representation.isin([focus_model, "experiment_balanced_mean"])
        ].copy()
    metrics = event_calibration_metrics(
        predictions, targets, bootstrap_replicates=bootstrap_replicates
    )
    metrics.to_csv(output / "event_calibration_metrics.csv", index=False)
    predictions.to_csv(output / "event_calibration_source.csv", index=False)

    for panel, panel_rows in predictions.groupby("panel", sort=False):
        if focus_model is not None:
            if len(targets) != 1 or panel_rows.seed.nunique() != 1:
                raise ValueError(
                    "Focused calibration requires one target and one prediction per event"
                )
            target = targets[0]
            rows = panel_rows.loc[panel_rows.representation.eq(focus_model)]
            baseline = panel_rows.loc[panel_rows.representation.eq("experiment_balanced_mean")]
            statistic = metrics.loc[
                metrics.panel.eq(panel) & metrics.representation.eq(focus_model)
            ].iloc[0]
            # LOEO has one training mean per held-out experiment; the line summarizes them.
            mean_line = baseline.groupby("experiment_id")[f"predicted_{target}"].mean().mean()
            baseline.groupby("experiment_id")[f"predicted_{target}"].mean().rename(
                "training_mean_kwh"
            ).to_csv(output / f"mean_baseline_{panel}.csv")
            with plt.rc_context(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "DejaVu Sans"],
                    "font.size": 7,
                    "axes.linewidth": 0.65,
                    "pdf.fonttype": 42,
                    "svg.fonttype": "none",
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.grid": False,
                }
            ):
                figure, axis = plt.subplots(figsize=(3.5, 3.55))
                values = rows[[target, f"predicted_{target}"]].to_numpy()
                low = np.floor((np.nanmin(values) - 0.01) * 20) / 20
                high = np.ceil((np.nanmax(values) + 0.01) * 20) / 20
                axis.plot(
                    [low, high], [low, high], color="#484848", lw=0.7, ls=(0, (4, 3)), zorder=1
                )
                axis.axhline(mean_line, color="#A0A0A0", lw=0.7, ls=(0, (4, 3)), zorder=1)
                axis.scatter(
                    rows[target],
                    rows[f"predicted_{target}"],
                    s=16,
                    facecolors="none",
                    edgecolors="#337D89",
                    linewidths=0.65,
                    alpha=0.75,
                    zorder=2,
                )
                axis.set(
                    xlim=(low, high),
                    ylim=(low, high),
                    aspect="equal",
                    xlabel="Observed event electricity (kWh)",
                    ylabel="LOEO predicted event electricity (kWh)",
                )
                axis.set_title("Actual-event calibration", loc="left", fontsize=8, pad=9)
                axis.tick_params(direction="out", width=0.65, length=3, labelsize=7)
                axis.text(
                    0.03,
                    0.97,
                    "8D Ridge",
                    transform=axis.transAxes,
                    va="top",
                    color="#337D89",
                    fontsize=7,
                )
                axis.text(
                    0.97,
                    0.97,
                    f"{int(statistic.events)} events, {int(statistic.experiments)} experiments\n"
                    f"MAE = {statistic.mae:.4f} kWh\nRMSE = {statistic.rmse:.4f} kWh",
                    transform=axis.transAxes,
                    va="top",
                    ha="right",
                    fontsize=6.5,
                    linespacing=1.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=2),
                )
                span = high - low
                axis.text(
                    low + 0.07 * span,
                    low + 0.10 * span,
                    "identity",
                    rotation=45,
                    color="#484848",
                    fontsize=6,
                )
                axis.text(
                    high - 0.025 * span,
                    mean_line - 0.025 * span,
                    "mean baseline (LOEO average)",
                    ha="right",
                    va="top",
                    fontsize=5.8,
                    color="#777777",
                )
                figure.tight_layout(pad=0.9)
                _export(
                    figure, output / f"event_calibration_{panel}", formats=("png", "pdf", "svg")
                )
            continue
    return metrics, pd.DataFrame()


def render_effective_cop_calibration(loader, events, predictions, output):
    """Compare measured and LOEO event cost at the same actual action and heat ledger."""
    from defrost_decision.candidate_quantities import build_measured_candidate_quantities
    from defrost_decision.performance_objectives import calculate_cycle_cop

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = OUTCOME_TARGETS["event_electricity"]
    rows = predictions.loc[predictions.representation.eq("ridge_dynamic_state_8")].merge(
        events[["event_id", "defrost_preparation_start"]], on="event_id", validate="one_to_one"
    )
    records = []
    for event in rows.itertuples(index=False):
        bounds = loader.get_cycle_record(event.cycle_name)["boundaries"]
        start = pd.to_datetime(bounds.get("stable_heating_start"))
        if pd.isna(start):
            records.append(
                dict(
                    event_id=event.event_id,
                    experiment_id=event.experiment_id,
                    observed_c=np.nan,
                    predicted_c=np.nan,
                    status="missing_heating_recovery",
                )
            )
            continue
        candidates = pd.DataFrame(
            dict(
                candidate_defrost_time=[event.defrost_preparation_start],
                heating_accounting_start=[start],
            )
        )
        measured = build_measured_candidate_quantities(
            loader, event.cycle_name, candidates, heat_column="heating_capacity"
        )
        values = []
        valid = []
        for field in (target, f"predicted_{target}"):
            curve = measured.assign(
                defrost_event_electricity_kwh=getattr(event, field),
                defrost_event_net_heat_kwh=0.0,
                defrost_event_electricity_prediction_available=True,
                defrost_event_net_heat_prediction_available=True,
                defrost_event_electricity_in_training_domain=True,
                defrost_event_net_heat_in_training_domain=True,
            )
            cop = calculate_cycle_cop(curve, effective=True, allow_model_extrapolation=True)
            values.append(cop.cycle_cop.iloc[0])
            valid.append(bool(cop.cycle_cop_eligible.iloc[0]))
        records.append(
            dict(
                event_id=event.event_id,
                experiment_id=event.experiment_id,
                observed_c=values[0],
                predicted_c=values[1],
                status="evaluated" if all(valid) else "invalid_cop_ledger",
            )
        )
    source = pd.DataFrame(records)
    source.to_csv(output / "effective_heating_cop_source.csv", index=False)
    rows = source.loc[source.status.eq("evaluated")].assign(
        panel="effective_heating", representation="ridge_dynamic_state_8", seed=0
    )
    metric = event_calibration_metrics(rows.assign(c=rows.observed_c), ["c"]).iloc[0]
    metric.to_frame().T.to_csv(output / "effective_heating_cop_metrics.csv", index=False)
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(3.5, 3.55))
        values = rows[["observed_c", "predicted_c"]].to_numpy()
        low, high = np.nanmin(values), np.nanmax(values)
        padding = 0.05 * (high - low)
        low, high = low - padding, high + padding
        axis.scatter(
            rows.observed_c,
            rows.predicted_c,
            s=16,
            facecolors="none",
            edgecolors="#337D89",
            linewidths=0.65,
            alpha=0.75,
        )
        axis.plot([low, high], [low, high], color="#484848", lw=0.7, ls=(0, (4, 3)))
        axis.set(
            xlim=(low, high),
            ylim=(low, high),
            aspect="equal",
            xlabel="Observed effective heating COP",
            ylabel="Ridge effective heating COP",
        )
        axis.set_title("effective heating COP", loc="left", fontsize=8, pad=9)
        axis.tick_params(direction="out", width=0.65, length=3, labelsize=7)
        axis.text(
            0.03,
            0.97,
            f"n={int(metric.events)}\nMAE = {metric.mae:.4f}\n"
            f"Bias = {metric.bias:+.4f}\nSlope = {metric.calibration_slope:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=6.5,
            linespacing=1.5,
        )
        figure.tight_layout(pad=0.9)
        _export(figure, output / "effective_heating_cop", formats=("png", "pdf", "svg"))
    return source
