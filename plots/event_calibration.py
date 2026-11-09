"""Held-out event electricity and effective heating COP calibration."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_layout import PROJECT
from plots.cop import COP_COLOR, _plot_supported_curve
from plots.defrost_decision import _plot_decision_markers, _plot_effective_cop_panel
from plots.publication import (
    _RGB_PANEL_FONT,
    _plot_cycle_panel,
    _shade_cycle_stages,
    _stage_spans,
    save_publication,
)


def _export(figure: plt.Figure, stem: Path, formats: tuple[str, ...] = ("png",)) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for figure_format in formats:
        figure.savefig(
            stem.with_suffix(f".{figure_format}"),
            dpi=600 if figure_format in {"png", "tiff"} else None,
            bbox_inches=None,
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
                    "mse": float(np.square(predicted - observed).mean()),
                    "r2": float(
                        1
                        - np.square(predicted - observed).sum()
                        / np.square(observed - observed.mean()).sum()
                    )
                    if np.ptp(observed) > 0
                    else np.nan,
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
    targets = ["E_prep_D_kwh"] if targets is None else targets
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
            if panel_rows.seed.nunique() != 1:
                raise ValueError("Focused calibration requires one prediction per event")
            rows = panel_rows.loc[panel_rows.representation.eq(focus_model)]
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
                figure, axes = plt.subplots(
                    1,
                    len(targets),
                    squeeze=False,
                    figsize=((89 if len(targets) == 1 else 183) / 25.4, 110 / 25.4),
                )
                for index, (target, axis) in enumerate(zip(targets, axes[0], strict=True)):
                    valid = rows.dropna(subset=[target, f"predicted_{target}"])
                    statistic = metrics.loc[
                        metrics.panel.eq(panel)
                        & metrics.representation.eq(focus_model)
                        & metrics.target.eq(target)
                    ].iloc[0]
                    values = valid[[target, f"predicted_{target}"]].to_numpy()
                    padding = 0.08 * np.ptp(values)
                    low, high = np.nanmin(values) - padding, np.nanmax(values) + padding
                    axis.plot(
                        [low, high], [low, high], color="#484848", lw=0.7, ls=(0, (4, 3)), zorder=1
                    )
                    support = valid.get(
                        "event_electricity_in_training_domain", pd.Series(True, index=valid.index)
                    ).eq(True)
                    for mask, color, marker, label in (
                        (support, "#337D89", "o", "Within training domain"),
                        (~support, "#B08342", "^", "Extrapolation"),
                    ):
                        axis.scatter(
                            valid.loc[mask, target],
                            valid.loc[mask, f"predicted_{target}"],
                            s=12,
                            facecolors="none",
                            edgecolors=color,
                            marker=marker,
                            linewidths=0.65,
                            alpha=0.8,
                            zorder=2,
                            label=label,
                        )
                    label = labels.get(target, "event electricity [kWh]")
                    axis.set(
                        xlim=(low, high),
                        ylim=(low, high),
                        aspect="equal",
                        xlabel=f"Observed {label}",
                        ylabel=f"Predicted {label}",
                    )
                    axis.set_anchor("N")
                    axis.set_title(
                        "Cycle total electricity"
                        if target == "E_total_kwh"
                        else "Preparation + defrost electricity",
                        loc="left",
                        fontsize=8,
                        pad=10,
                    )
                    axis.text(
                        -0.14,
                        1.06,
                        chr(97 + index),
                        transform=axis.transAxes,
                        weight="bold",
                        fontsize=9,
                    )
                    axis.tick_params(direction="out", width=0.65, length=3, labelsize=7)
                    axis.text(
                        0.03,
                        0.97,
                        f"{int(statistic.events)} events / {int(statistic.experiments)} experiments",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=6.5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=2),
                    )
                    r2_digits = 6 if target == "E_total_kwh" else 3
                    annotation = (
                        f"MAE = {statistic.mae:.4f} kWh\n"
                        f"RMSE = {statistic.rmse:.4f} kWh\n"
                        f"$R^2$ = {statistic.r2:.{r2_digits}f}"
                    )
                    if target == "E_total_kwh":
                        relative = (valid[f"predicted_{target}"] - valid[target]).abs() / valid[
                            target
                        ]
                        annotation += (
                            f"\nMedian absolute relative error = {100 * relative.median():.2f}%"
                            f"\n95th percentile = {100 * relative.quantile(0.95):.2f}%"
                        )
                    axis.text(
                        0.5,
                        -0.24,
                        annotation,
                        transform=axis.transAxes,
                        va="top",
                        ha="center",
                        fontsize=6.5,
                        linespacing=1.25,
                    )
                figure.legend(
                    *axes[0, 0].get_legend_handles_labels(),
                    loc="lower center",
                    ncol=2,
                    frameon=False,
                    fontsize=7,
                    bbox_to_anchor=(0.5, 0),
                )
                figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.38, wspace=0.35)
                _export(
                    figure,
                    output / f"event_calibration_{panel}",
                    formats=("png", "pdf", "svg", "tiff"),
                )
            continue
    return metrics, pd.DataFrame()


def measured_preparation_start_electricity(loader, event):
    """Use the shared cycle ledger at the actual preparation-start boundary."""
    from defrost_event_models.training_data import candidate_integral_table

    frame = loader.load_cycle_measurements(event.cycle_name)
    measured = candidate_integral_table(
        frame,
        pd.Timestamp(event.heating_start),
        [pd.Timestamp(event.defrost_preparation_start)],
        "power_total",
    ).iloc[0]
    return {
        "event_id": event.event_id,
        "measured_pre_defrost_electricity_kwh": measured.energy if measured.valid else np.nan,
        "total_electricity_measurement_valid": bool(measured.valid),
    }


def render_frost_event_cost(output):
    """Separate visual-state associations with predicted, observed and phase costs."""
    output = Path(output)
    rows = pd.read_csv(output / "event_source.csv").dropna(subset=["visual_frost_score"])
    summary = pd.read_csv(output / "association_summary.csv").set_index("metric")
    panels = [
        ("predicted_E_prep_D_kwh", "Ridge预测：准备＋正式除霜"),
        ("E_prep_D_kwh", "实测：准备＋正式除霜"),
        ("E_D_kwh", "实测：正式除霜"),
        ("E_prep_kwh", "实测：准备阶段"),
    ]
    caption = (
        "Native global Harmony + Slingshot；霜态坐标不是实测霜厚。\n"
        "每点为一个真实事件；预测为跨实验留出值；95% CI采用实验成组bootstrap。"
    )
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.9))
        for i, ((metric, title), ax) in enumerate(zip(panels, axes.flat)):
            for temperature, color in [(50, "#789DA8"), (55, "#BBA2C8"), (60, "#D3A781")]:
                group = rows.loc[rows.water_temperature_setpoint.eq(temperature)]
                ax.scatter(
                    group.visual_frost_score,
                    group[metric],
                    s=17,
                    alpha=0.7,
                    c=color,
                    linewidths=0,
                    label=f"{temperature}°C (n={len(group)})",
                )
            stats = summary.loc[metric]
            ax.set_title(f"{chr(97 + i)}  {title}", loc="left", fontweight="bold")
            ax.set_xlabel("图像霜态坐标（0–1）")
            ax.set_ylabel("事件电耗（kWh）")
            ax.text(
                0.03,
                0.97,
                f"Spearman ρ={stats.spearman:.3f}\n95% CI [{stats.ci_low:.3f}, {stats.ci_high:.3f}]\n同实验内秩相关={stats.within_experiment_rank_correlation:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
            ax.set_ylim(0, rows[metric].max() * 1.25)
            ax.grid(alpha=0.15)
        axes[0, 0].legend(loc="lower right", fontsize=7, frameon=False)
        save_publication(fig, output, "霜态与除霜成本", caption, dpi=400)
    (output / "figure_caption.txt").write_text(
        caption + "\n样本数见association_summary.csv；未匹配事件不参与相关分析。\n"
        "同实验内秩相关：在各实验内部排序并减去各实验均值后的合并相关；仍不能排除实验内控制和温度变化。"
    )


def render_event_prediction_cycle(cycle_frame, cycle_record, curve, output_path):
    """Event electricity versus candidate time, with shared cycle panels and markers."""

    frame = cycle_frame.sort_values("timestamp").copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    curve = curve.sort_values("candidate_defrost_time").copy()
    curve["candidate_defrost_time"] = pd.to_datetime(curve.candidate_defrost_time)
    origin = frame.timestamp.min()
    minutes = (frame.timestamp - origin).dt.total_seconds() / 60
    candidate_minutes = (curve.candidate_defrost_time - origin).dt.total_seconds() / 60
    spans = _stage_spans(frame, minutes)
    optimum = pd.Timestamp(curve.t_star.dropna().iloc[0])
    peak_minute = (optimum - origin).total_seconds() / 60
    markers = {
        "optimal": {"target_time": optimum},
        "rb": {"target_time": curve.t_RB.dropna().iloc[0] if curve.t_RB.notna().any() else pd.NaT},
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    for column in range(2):
        _plot_cycle_panel(
            axes[0, column],
            frame,
            minutes,
            ("heating_capacity", "power_total"),
            "Capacity / power [kW]",
            spans,
            [],
        )
        axis = axes[1, column]
        supported = curve.defrost_event_electricity_in_training_domain.fillna(False)
        _plot_supported_curve(
            axis,
            candidate_minutes,
            curve.defrost_event_electricity_kwh,
            supported,
            color=COP_COLOR,
            label="Predicted event E (Ridge)",
        )
        _shade_cycle_stages(axis, spans, [])
        axis.set_ylabel("Predicted event electricity [kWh]", fontsize=8)
        axis.grid(axis="x", alpha=0.12)
        axis.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.01))
        relative = _plot_effective_cop_panel(axes[2, column], curve, origin, spans)
        _plot_decision_markers(
            list(axes[:, column]), markers, origin, "Effective COP maximum", target_times=True
        )
        limits = (
            (0, minutes.max())
            if column == 0
            else (max(0, peak_minute - 5), min(minutes.max(), peak_minute + 5))
        )
        for axis in axes[:, column]:
            axis.set_xlim(*limits)
            axis.tick_params(labelsize=7)
        if column == 1:
            local = candidate_minutes.between(*limits)
            for axis, values in (
                (axes[1, column], curve.defrost_event_electricity_kwh),
                (axes[2, column], curve.cycle_cop),
            ):
                values = values.loc[local].dropna()
                if len(values):
                    pad = max((values.max() - values.min()) * 0.12, 0.001)
                    axis.set_ylim(values.min() - pad, values.max() + pad)
            best = curve.loc[curve.cycle_cop_eligible, "cycle_cop"].max()
            relative.set_ylim(*(100 * np.array(axes[2, column].get_ylim()) / best))
        axes[2, column].set_xlabel("Time from cycle start [min]", fontsize=8)
    axes[0, 0].set_title("Full cycle", fontsize=10, pad=30)
    axes[0, 1].set_title("Within 5 minutes of COP optimum", fontsize=10, pad=30)
    fig.suptitle(
        f"{cycle_record.get('cycle_name', 'Cycle')} | Event electricity prediction",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Orange dash-dot: COP optimum; green dashed: RB. "
        "Dashed E curve: outside model domain. Event Q is fixed at zero (not predicted).",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.08, top=0.88, hspace=0.5, wspace=0.42)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    import json

    from output_layout import analysis_output

    parser = argparse.ArgumentParser(description="Render saved three-input Ridge LOEO calibration.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--total-electricity", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    bundle = json.loads((args.model / "candidate_model_parameters.json").read_text())
    predictions = pd.read_csv(args.model / "model_validation.csv")
    predictions["model_version"] = args.model.name
    predictions["measurement_processing"] = bundle["measurement_processing"]
    targets = ["E_prep_D_kwh"]
    if args.total_electricity:
        from joblib import Parallel, delayed, parallel_config

        from dataset_tools import DatasetLoader

        loader = DatasetLoader(args.dataset)
        if loader.registry.get("measurement_processing") != bundle["measurement_processing"]:
            raise ValueError("Dataset processing differs from the saved model")
        with parallel_config(backend="loky", inner_max_num_threads=1):
            measured = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(measured_preparation_start_electricity)(loader, event)
                for event in predictions.itertuples(index=False)
            )
        predictions = predictions.merge(
            pd.DataFrame(measured), on="event_id", validate="one_to_one"
        )
        predictions["E_total_kwh"] = (
            predictions.measured_pre_defrost_electricity_kwh + predictions.E_prep_D_kwh
        )
        predictions["predicted_E_total_kwh"] = (
            predictions.measured_pre_defrost_electricity_kwh + predictions.predicted_E_prep_D_kwh
        )
        targets.append("E_total_kwh")
    baseline = predictions.copy()
    baseline["predicted_E_prep_D_kwh"] = baseline.experiment_id.map(
        {
            experiment: parameters["E_prep_D_kwh"]["mean_baseline"]
            for experiment, parameters in bundle["phase_models"]["folds"].items()
        }
    )
    baseline["representation"] = "experiment_balanced_mean"
    if args.total_electricity:
        baseline["predicted_E_total_kwh"] = (
            baseline.measured_pre_defrost_electricity_kwh + baseline.predicted_E_prep_D_kwh
        )
    render_calibration_figures(
        pd.concat([predictions, baseline], ignore_index=True),
        None,
        analysis_output(args.output),
        targets=targets,
        bootstrap_replicates=0,
        focus_model="ridge_joint_three_input",
        labels={
            "E_prep_D_kwh": "preparation + defrost electricity [kWh]",
            "E_total_kwh": "cycle total electricity [kWh]",
        },
    )
