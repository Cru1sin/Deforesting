"""Compare frozen stopping policies on experiment-disjoint external conditions."""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_layout import PROJECT
from plots.publication import save_publication

matplotlib.use("Agg")
from matplotlib.lines import Line2D

from dataset_tools import DatasetLoader
from dataset_tools.cycle_metadata import cycle_operating_condition
from image_models.checkpoints import checkpoint_files
from image_models.cycle_selection import screen_admission
from image_models.policy import summarize_metrics
from output_layout import analysis_output

COLORS = ["#527FAD", "#A58BB8"]
MARKERS = ["o", "s"]


def load_external(evaluations, dataset):
    """Keep each original checkpoint score and audit the actual training conditions."""
    loader = DatasetLoader(dataset)
    screening = screen_admission(Path(dataset) / "training_data/settings.json")
    selected = set(screening.loc[screening.selected, "cycle_name"])
    metrics, summaries, training, sources = [], [], [], []
    cycle_conditions = {}
    for i, folder in enumerate(evaluations):
        folder = Path(folder)
        settings = json.loads((folder / "settings.json").read_text())
        rows = pd.read_csv(folder / "cycle_metrics.csv", float_precision="round_trip")
        if not set(rows.cycle_name) <= selected:
            raise ValueError(
                "External metrics contain cycles outside current authoritative admission"
            )
        if rows.duplicated(["checkpoint", "cycle_name"]).any():
            raise ValueError("Repeated checkpoint/cycle scores")
        if rows[["selected_cop", "rb_cop", "optimal_cop"]].isna().any().any():
            raise ValueError("Report incomplete COP/RB support before drawing this comparison")
        names = set(rows.cycle_name)
        if metrics and names != set(metrics[0].cycle_name):
            raise ValueError("Compare policies on the same external cycles")
        resets = rows.reset_history.unique()
        if len(resets) != 1:
            raise ValueError("Each evaluation must have one history recipe")
        method = "Memory reset" if resets[0] else "Memory retained"
        trained = set()
        paths = checkpoint_files(Path(settings["source_run"]))
        if {p.stem for p in paths} != set(rows.checkpoint):
            raise ValueError("External scores must include every frozen checkpoint")
        for path in paths:
            model = pickle.load(path.open("rb"))["checkpoint"]
            if names & set(model["training_cycles"]) or set(rows.experiment_id) & set(
                model["training_experiments"]
            ):
                raise ValueError("External cycles overlap model training")
            trained.update(model["training_cycles"])
        for name in sorted(trained):
            if name not in cycle_conditions:
                cycle_conditions[name] = cycle_operating_condition(loader.load_cycle(name))
            training.append(
                dict(method=method, cycle_name=name, operating_condition=cycle_conditions[name])
            )
        for name in names:
            actual = cycle_operating_condition(loader.load_cycle(name))
            if not rows.loc[rows.cycle_name.eq(name), "operating_condition"].eq(actual).all():
                raise ValueError("Saved external condition differs from Dataset")
            if actual in {cycle_conditions[n] for n in trained}:
                raise ValueError("Requested external condition already appears in training")
        rows = rows.assign(method=method, method_index=i)
        metrics.append(rows)
        for (checkpoint, condition), group in rows.groupby(["checkpoint", "operating_condition"]):
            summaries.append(
                dict(
                    method=method,
                    method_index=i,
                    checkpoint=checkpoint,
                    operating_condition=condition,
                    **summarize_metrics(group),
                )
            )
        sources.append(str(folder.resolve()))
    return (
        pd.concat(metrics, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(training),
        sources,
    )


def render_external(metrics, summaries, training, output):
    """Show all cycles; intervals describe frozen-model variation, never sample CIs."""
    conditions = sorted(metrics.operating_condition.unique())
    methods = metrics[["method_index", "method"]].drop_duplicates().sort_values("method_index")
    order = (
        metrics[["operating_condition", "cycle_name"]]
        .drop_duplicates()
        .sort_values(["operating_condition", "cycle_name"])
    )
    cycle_stats = []
    for (method, condition, name), rows in metrics.groupby(
        ["method", "operating_condition", "cycle_name"]
    ):
        record = dict(
            method=method,
            operating_condition=condition,
            cycle_name=name,
            checkpoints=rows.checkpoint.nunique(),
            **summarize_metrics(rows),
        )
        record["gain_p10"], record["gain_p90"] = rows.gain_vs_rb_pct.quantile([0.1, 0.9])
        cycle_stats.append(record)
    cycle_stats = pd.DataFrame(cycle_stats)
    cycle_stats.to_csv(output / "cycle_summary.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(183 / 25.4, 160 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.97,
        bottom=0.18,
        top=0.82,
        wspace=0.37,
        hspace=0.62,
        height_ratios=[1.45, 1],
    )
    ax_gain, ax_hit, ax_cop, ax_capture = [
        fig.add_subplot(grid[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
    for y, row in enumerate(order.itertuples()):
        for method in methods.itertuples():
            i = method.method_index
            rows = metrics.loc[
                metrics.method.eq(method.method) & metrics.cycle_name.eq(row.cycle_name)
            ]
            stat = cycle_stats.loc[
                cycle_stats.method.eq(method.method) & cycle_stats.cycle_name.eq(row.cycle_name)
            ].iloc[0]
            offset = (i - (len(methods) - 1) / 2) * 0.23
            ax_gain.scatter(
                rows.gain_vs_rb_pct,
                y + offset + np.linspace(-0.045, 0.045, len(rows)),
                s=4,
                color=COLORS[i],
                alpha=0.12,
                linewidths=0,
            )
            ax_gain.plot([stat.gain_p10, stat.gain_p90], [y + offset] * 2, color=COLORS[i], lw=1.5)
            ax_gain.plot(
                stat.gain_vs_rb_pct,
                y + offset,
                MARKERS[i],
                color=COLORS[i],
                ms=4,
                mec="white",
                mew=0.4,
            )
            ax_hit.plot(stat.hit_rate_all_pct, y + offset, MARKERS[i], color=COLORS[i], ms=4)
    labels = [
        f"{r.cycle_name.rsplit('_', 1)[-1]}  |  {r.operating_condition.replace('C_', '°C ')}"
        for r in order.itertuples()
    ]
    for ax in [ax_gain, ax_hit]:
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)))
        ax.grid(axis="x", color="#E7E9EC", linewidth=0.5)
        ax.set_axisbelow(True)
    ax_gain.set_yticklabels(labels, fontsize=6)
    ax_hit.set_yticklabels([])
    ax_gain.axvline(0, color="#67717B", lw=0.7, ls="--")
    ax_hit.set_xlim(-3, 104)
    ax_hit.set_xticks([0, 25, 50, 75, 100])
    ax_gain.set(xlabel="COP gain over frozen RB (%)", title="Individual external cycles")
    ax_hit.set(xlabel="Within 1% of supported COP peak (%)", title="Near-optimal performance")
    condition_labels = []
    for j, condition in enumerate(conditions):
        part = metrics.loc[metrics.operating_condition.eq(condition)]
        n = part.cycle_name.nunique()
        condition_labels.append(
            condition.replace("C_", "°C ")
            + f"\n$n={n}$ cycle"
            + ("s" if n != 1 else " (case study)")
        )
        baseline = part.drop_duplicates("cycle_name")
        ax_cop.plot(j - 0.26, baseline.rb_cop.mean(), "_", color="#505862", ms=9, mew=1.5)
        ax_cop.plot(j + 0.26, baseline.optimal_cop.mean(), "D", color="#B89D68", ms=4)
        for method in methods.itertuples():
            i = method.method_index
            group = summaries.loc[
                summaries.method.eq(method.method) & summaries.operating_condition.eq(condition)
            ]
            x = j + (i - (len(methods) - 1) / 2) * 0.16
            for ax, field in [(ax_cop, "mean_cycle_cop"), (ax_capture, "headroom_captured_pct")]:
                values = group[field]
                low, high = values.quantile([0.1, 0.9])
                ax.plot([x, x], [low, high], color=COLORS[i], lw=1.5)
                ax.plot(x, values.mean(), MARKERS[i], color=COLORS[i], ms=4)
    for ax in [ax_cop, ax_capture]:
        ax.set_xticks(range(len(conditions)), condition_labels, fontsize=6.5)
        ax.set_xlim(-0.55, len(conditions) - 0.45)
        ax.grid(axis="y", color="#E7E9EC", linewidth=0.5)
        ax.set_axisbelow(True)
    ax_cop.set(ylabel="Mean effective heating COP", title="Absolute performance and references")
    ax_capture.set(
        ylabel="Available COP gain captured (%)", title="Fraction of improvement opportunity"
    )
    ax_capture.axhline(100, lw=0.7, ls="--", color="#67717B")
    ax_capture.set_ylim(bottom=0)
    for letter, ax in zip("abcd", [ax_gain, ax_hit, ax_cop, ax_capture]):
        ax.text(-0.14, 1.035, letter, transform=ax.transAxes, weight="bold", fontsize=9)
    handles = [
        Line2D(
            [],
            [],
            marker=MARKERS[r.method_index],
            color=COLORS[r.method_index],
            linestyle="none",
            label=r.method,
            ms=4,
        )
        for r in methods.itertuples()
    ]
    handles += [
        Line2D([], [], marker="_", color="#505862", linestyle="none", label="Frozen RB", ms=9),
        Line2D(
            [],
            [],
            marker="D",
            color="#B89D68",
            linestyle="none",
            label="Supported COP maximum",
            ms=4,
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.925), ncol=4, fontsize=6.5
    )
    fig.text(
        0.12,
        0.965,
        "Can frozen LSTM FiLM transfer to unseen fan conditions?",
        fontsize=10,
        weight="bold",
    )
    trained = training.groupby("method").cycle_name.nunique().iloc[0]
    counts = metrics.groupby("method").checkpoint.nunique().unique()
    count_text = "/".join(map(str, counts))
    fig.text(
        0.12,
        0.855,
        f"Training pool: {trained} medium-fan cycles  |  External: {len(order)} independent cycles  |  {count_text} frozen models per version",
        fontsize=6.5,
    )
    caption = (
        "Points: model means; lines: 10–90% range across frozen historical models (not confidence intervals).\n"
        "a: faint dots are all model predictions. b: fraction of models attaining COP ≥99% of the supported peak.\n"
        "c–d: equal weight per cycle within each condition; complete COP/RB support. Low-fan evidence is one case.\n"
        "No retraining, tuning or model selection; exact labels and the frozen 0.5 threshold are unchanged.\n"
        "Evidence supports near-optimal transfer on the high-fan cases; the low-fan case is less consistent."
    )
    fig.text(0.12, 0.025, caption, fontsize=6, va="bottom", linespacing=1.5)
    # Check page geometry numerically; image inspection requires separate user permission.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if (
                bounds.x0 < 0
                or bounds.y0 < 0
                or bounds.x1 > fig.bbox.width
                or bounds.y1 > fig.bbox.height
            ):
                raise ValueError(f"Figure text outside page: {text.get_text()}")
    save_publication(
        fig,
        output,
        "unseen_conditions",
        "",
        layout=False,
        dpi=600,
        formats=("png", "pdf", "svg", "tiff"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = analysis_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, summaries, training, sources = load_external(args.evaluations, args.dataset)
    metrics.to_csv(output / "cycle_predictions.csv", index=False)
    summaries.to_csv(output / "checkpoint_conditions.csv", index=False)
    training.to_csv(output / "training_conditions.csv", index=False)
    render_external(metrics, summaries, training, output)
    (output / "settings.json").write_text(
        json.dumps(
            dict(
                evaluations=sources,
                dataset=str(args.dataset.resolve()),
                independent_cycles=metrics.cycle_name.nunique(),
                interval="10–90 percentile across frozen models; not a confidence interval",
                statistics="Descriptive; no significance test; no low-fan population inference",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Published external-condition figure and source data: {output}", flush=True)


def stopping_change_source(metrics, decision_view):
    """Summarize the saved cycle metrics with the original three-seed definitions."""
    frequency = decision_view == "frequency"
    joint = decision_view == "heating_electricity"
    heating = decision_view == "heating" or joint
    power = decision_view == "power"
    electricity = decision_view in ("electricity", "power") or joint
    if power:
        metrics = metrics.loc[metrics.power_duration_available.eq(True)].copy()
    if frequency:
        metrics = metrics.copy()
        durations = metrics[["cycle_duration_minutes", "rb_cycle_duration_minutes"]]
        if not np.isfinite(durations).all().all() or not durations.gt(0).all().all():
            raise ValueError("Equivalent trigger rates require positive paired intervals")
        metrics["rate_reduction_per_hour"] = (
            60 / metrics.rb_cycle_duration_minutes - 60 / metrics.cycle_duration_minutes
        )
        metrics["rate_reduction_pct"] = 100 * (
            1 - metrics.rb_cycle_duration_minutes / metrics.cycle_duration_minutes
        )
    if heating:
        metrics = metrics.copy()
        durations = metrics[["heating_duration_minutes", "rb_heating_duration_minutes"]]
        if not np.isfinite(durations).all().all() or not durations.gt(0).all().all():
            raise ValueError("Stable-heating comparison requires positive paired durations")
        metrics["stable_heating_increase_minutes"] = (
            metrics.heating_duration_minutes - metrics.rb_heating_duration_minutes
        )
        metrics["stable_heating_increase_pct"] = (
            100
            * metrics.stable_heating_increase_minutes
            / metrics.rb_heating_duration_minutes
        )
    metric_names = (
        ["stable_heating_increase_minutes", "stable_heating_increase_pct"]
        if heating
        else ["rate_reduction_per_hour", "rate_reduction_pct"]
        if frequency
        else ["delta_cop", "gain_vs_rb_pct"]
    )
    figure_name = (
        "Fig2b_ranked_stable_heating_increase"
        if heating
        else "Fig2b_ranked_frequency_reduction"
        if frequency
        else "Fig2a_ranked_cycle_gains"
    )
    if electricity:
        metrics = metrics.copy()
        metrics["defrost_electricity_change_wh"] = (
            1000 * metrics.defrost_electricity_change_kwh
        )
        metric_names = ["defrost_electricity_change_wh", "defrost_electricity_change_pct"]
        figure_name = "Fig2c_ranked_defrost_electricity_change"
    if power:
        metric_names = ["defrost_mean_power_change_w", "defrost_mean_power_change_pct"]
        figure_name = "Fig2d_ranked_defrost_mean_power_change"
    if joint:
        metric_names = [
            "stable_heating_increase_pct",
            "defrost_electricity_change_pct",
            "stable_heating_increase_minutes",
            "defrost_electricity_change_wh",
        ]
        figure_name = "Fig2e_heating_electricity_relationship"
    # Use the saved per-cycle relative metric; denominators differ across cycles.
    assert metrics.rb_cop.gt(0).all()
    assert np.allclose(metrics.delta_cop, metrics.selected_cop - metrics.rb_cop)
    assert np.allclose(metrics.gain_vs_rb_pct, 100 * metrics.delta_cop / metrics.rb_cop)
    assert metrics.groupby("cycle_name").rb_cop.nunique().eq(1).all()
    tables = []
    for metric in metric_names:
        values = metrics.pivot(index="cycle_name", columns="seed", values=metric)
        assert set(values.columns) == {42, 2026, 1010} and values.notna().all().all()
        table = pd.DataFrame(
            {
                f"mean_{metric}": values.mean(axis=1),
                f"minimum_{metric}": values.min(axis=1),
                f"maximum_{metric}": values.max(axis=1),
            }
        )
        tables.append(table.join(values.add_prefix(f"{metric}_seed_")))
    source = pd.concat(tables, axis=1).reset_index()
    source = source.sort_values([f"mean_{metric_names[1]}", "cycle_name"]).reset_index(
        drop=True
    )
    source["rank"] = np.arange(1, len(source) + 1)
    return source, metric_names, figure_name


def plot_stopping_change(axis, source, metric_names, decision_view):
    """Draw one comparison on supplied axes for both standalone and combined figures."""
    color, pale = "#436B85", "#B4CDDA"
    if decision_view == "heating_electricity":
        x = source.mean_stable_heating_increase_pct
        y = source.mean_defrost_electricity_change_pct
        axis.errorbar(
            x, y,
            xerr=np.vstack([x - source.minimum_stable_heating_increase_pct,
                            source.maximum_stable_heating_increase_pct - x]).clip(min=0),
            yerr=np.vstack([y - source.minimum_defrost_electricity_change_pct,
                            source.maximum_defrost_electricity_change_pct - y]).clip(min=0),
            fmt="o", color=color, markersize=3, markeredgewidth=0.3,
            markeredgecolor="white", ecolor=pale, elinewidth=0.6,
        )
        axis.axhline(0, color="0.55", lw=0.6, ls="--")
        axis.axvline(0, color="0.55", lw=0.6, ls="--")
        axis.set(xlabel="Heating duration change (%)",
                 ylabel="Predicted event electricity\nchange (%)")
        axis.margins(x=0.08, y=0.12)
        axes = [axis]
    else:
        right = axis.twinx()
        right.spines["right"].set_visible(True)
        axis.set_zorder(right.get_zorder() + 1)
        axis.patch.set_visible(False)
        labels = {
            "gains": ("COP change", "COP change (%)"),
            "heating": ("Heating duration change (min)", "Heating duration change (%)"),
            "electricity": ("Predicted event electricity\nchange (Wh)",
                            "Predicted event electricity\nchange (%)"),
            "frequency": ("Trigger rate reduction (events/h)", "Trigger rate reduction (%)"),
            "power": ("Predicted mean power change (W)", "Predicted mean power change (%)"),
        }[decision_view]
        maxima = [source[f"maximum_{metric}"].max() for metric in metric_names]
        lower = min(0, *(source[f"minimum_{metric}"].min() / maximum
                        for metric, maximum in zip(metric_names, maxima, strict=True))) - 0.06
        # Separate absolute/relative scales are aligned at zero, not converted units.
        for index, (current, metric, maximum, label) in enumerate(
            zip([axis, right], metric_names, maxima, labels, strict=True)
        ):
            current.set_ylim(lower * maximum, 1.25 * maximum)
            current.set_ylabel(label, color="black", labelpad=4)
            current.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
            current.vlines(source["rank"], source[f"minimum_{metric}"],
                           source[f"maximum_{metric}"], color=color if index == 0 else pale,
                           alpha=0.65, lw=0.6, zorder=2)
            if index == 0:
                current.scatter(source["rank"], source[f"mean_{metric}"], s=9,
                                color=color, edgecolors="white", linewidths=0.2, zorder=3)
            else:
                current.bar(source["rank"], source[f"mean_{metric}"],
                            width=0.7, color=pale, edgecolor="none", zorder=1)
        axis.axhline(0, color="0.45", lw=0.6, zorder=1)
        axis.set(xlim=(0, len(source) + 1),
                 xticks=[1, 25, 50, 75, len(source)], xlabel="Cycle rank")
        axis.legend(handles=[
            Line2D([], [], marker="o", lw=0, color=color, markersize=3,
                   label="Absolute (left)"),
            Line2D([], [], marker="s", lw=0, color=pale, markersize=4,
                   label="Relative (right)"),
        ], loc="upper left", ncol=2, frameon=False, fontsize=7.5,
           handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
        axes = [axis, right]
    for current in axes:
        current.tick_params(axis="both", colors="black", length=3, width=0.6)
        current.xaxis.label.set_color("black")
        current.yaxis.label.set_color("black")
        current.xaxis.get_offset_text().set_color("black")
        current.yaxis.get_offset_text().set_color("black")
        for spine in current.spines.values():
            spine.set_color("black")
    return axes


def render_stopping_results(output: Path, *, figures=None, decision_view="triptych") -> None:
    """Publication figures from one shared saved evaluation, not new COP formulas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    metrics = pd.read_csv(output / "cycle_results.csv")
    model_color, rb_color = "#436B85", "#969696"
    notes = []

    def paired_points(axis, part, color=model_color):
        improved = part.delta_cop.gt(0)
        worse = part.delta_cop.lt(0)
        axis.scatter(
            part.loc[improved, "rb_cop"],
            part.loc[improved, "selected_cop"],
            s=15,
            color=color,
            alpha=0.7,
            label="Higher COP than RB",
        )
        axis.scatter(
            part.loc[worse, "rb_cop"],
            part.loc[worse, "selected_cop"],
            s=22,
            marker="x",
            color="#B56E57",
            linewidths=1,
            label="Lower COP than RB",
        )
        equal = part.delta_cop.eq(0)
        if equal.any():
            axis.scatter(
                part.loc[equal, "rb_cop"],
                part.loc[equal, "selected_cop"],
                s=15,
                facecolors="none",
                edgecolors=rb_color,
                label="Equal COP",
            )

    def save(figure, name, caption, formats=("png", "pdf", "svg", "tiff")):
        if figures is not None and name not in figures:
            plt.close(figure)
            return
        FigureCanvasAgg(figure).draw()
        canvas = figure.bbox
        texts = [*figure.texts]
        for axis in figure.axes:
            texts.extend([axis.xaxis.label, axis.yaxis.label, axis.title, *axis.texts])
        for text in texts:
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(figure.canvas.get_renderer())
                if (
                    box.width
                    and box.height
                    and (
                        box.x0 < 0
                        or box.y0 < 0
                        or box.x1 > canvas.width + 1
                        or box.y1 > canvas.height + 1
                    )
                ):
                    raise ValueError(f"Text outside figure bounds: {text.get_text()}")
        for extension in formats:
            figure.savefig(output / f"{name}.{extension}", dpi=600, facecolor="white")
        if caption:
            notes.append(name + "\n" + caption)
        plt.close(figure)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    ):
        if decision_view in (
            "gains", "frequency", "heating", "electricity", "power", "heating_electricity",
            "operations",
        ):
            views = ["gains", "heating", "electricity", "heating_electricity"]
            titles = ["COP improvement", "Stable-heating duration",
                      "Predicted event electricity", "Duration and electricity"]
            combined = decision_view == "operations"
            if combined:
                plt.rcParams["font.size"] = 8.5
                figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
                figure.get_layout_engine().set(w_pad=0.08, h_pad=0.05, wspace=0.04, hspace=0.08)
                axes = axes.flat
            else:
                figure, axis = plt.subplots(figsize=(7.2, 3.5), layout="constrained")
                axes = [axis]
                views = [decision_view]
            for index, (axis, view) in enumerate(zip(axes, views, strict=True)):
                source, metric_names, figure_name = stopping_change_source(metrics, view)
                source.to_csv(output / f"{figure_name}.csv", index=False)
                plot_stopping_change(axis, source, metric_names, view)
                if combined:
                    axis.set_title(f"({chr(97 + index)}) {titles[index]}",
                                   loc="left", fontsize=9, fontweight="bold", pad=7)
            name = "operation_results" if combined else figure_name
            save(figure, name, None, formats=("png", "pdf"))
            return
        if decision_view == "paired":
            seeds = [s for s in [42, 2026, 1010] if s in metrics.seed.unique()]
            assert set(seeds) == set(metrics.seed.unique())
            figure, axes = plt.subplots(
                1,
                len(seeds),
                figsize=(240 / 25.4, 97 / 25.4),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            figure.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.80, wspace=0.14)
            low = min(metrics.rb_cop.min(), metrics.selected_cop.min()) - 0.04
            high = max(metrics.rb_cop.max(), metrics.selected_cop.max()) + 0.04
            for axis, seed in zip(axes.flat, seeds, strict=True):
                part = metrics.loc[metrics.seed.eq(seed)]
                assert part.cycle_name.is_unique
                axis.fill_between(
                    [low, high], [low, high], high, color=model_color, alpha=0.035, zorder=0
                )
                axis.plot([low, high], [low, high], color=rb_color, lw=0.9, ls="--", zorder=1)
                paired_points(axis, part)
                axis.set(
                    xlim=(low, high),
                    ylim=(low, high),
                    xlabel="RB effective cycle COP",
                    aspect="equal",
                )
                axis.set_title(f"Seed {seed}", fontsize=9, fontweight="bold", loc="left")
                axis.text(
                    0.04,
                    0.96,
                    f"{part.delta_cop.gt(0).sum()} improved\n{part.delta_cop.lt(0).sum()} worsened",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8,
                )
                summary = pd.read_csv(output / "evaluation_summary.csv")
                row = summary.loc[
                    summary.subset.eq("all") & summary.setpoint_c.eq("all") & summary.seed.eq(seed)
                ].iloc[0]
                axis.text(
                    0.96,
                    0.05,
                    f"Mean ΔCOP: +{row.delta_cop_vs_rb:.4f}\n$H_{{RB}}$: {row.headroom_captured_pct:.2f}%",
                    transform=axis.transAxes,
                    ha="right",
                    fontsize=7.5,
                )
                axis.grid(color="#DDE2E6", lw=0.4, alpha=0.55, zorder=0)
            axes[0, 0].set_ylabel("LSTM + FiLM effective cycle COP")
            figure.text(
                0.07,
                0.955,
                "LSTM + FiLM: paired COP comparison with RB",
                fontsize=11,
                fontweight="bold",
            )
            figure.text(
                0.07,
                0.895,
                "101 cycles • 29 experiment-level LOEO folds • One cycle per point; above the dashed line = higher COP than RB",
                fontsize=7.5,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.53, 0.015),
                ncol=len(labels),
                frameon=False,
                fontsize=8,
            )
            save(
                figure,
                "Fig2a_paired_decisions",
                "Question: does LSTM + FiLM improve each cycle relative to RB?\n"
                "Evidence: 101 held-out cycles in each independent seed; blue circles improve, orange crosses worsen, diagonal denotes equal COP.\n"
                "Conclusion: 85–87 cycles improve and 14–16 worsen per seed; no averaging of seed-specific stopping decisions. COP is retrospective, not measured savings.",
            )
            (output / "Fig2a_paired_decisions_caption.txt").write_text(
                "\n".join(notes) + "\n", encoding="utf-8"
            )
            return
        figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.87, wspace=0.55)
        low = min(metrics.rb_cop.min(), metrics.selected_cop.min()) - 0.04
        high = max(metrics.rb_cop.max(), metrics.selected_cop.max()) + 0.04
        seeds = sorted(metrics.seed.unique())
        colors = ["#B39AD6", "#9273B6", "#6F4D96"] if len(seeds) > 1 else [model_color]
        axes[0].plot([low, high], [low, high], color=rb_color, lw=0.8, ls="--")
        counts = []
        for seed, color in zip(seeds, colors, strict=True):
            part = metrics.loc[metrics.seed.eq(seed)]
            improved = part.delta_cop.gt(0)
            worse = part.delta_cop.lt(0)
            counts.append(f"{seed}: {improved.sum()} / {worse.sum()}")
            paired_points(axes[0], part, color)
            delta = np.sort(part.delta_cop.to_numpy())
            axes[1].step(
                np.r_[delta[0], delta],
                np.r_[0, 100 * np.arange(1, len(delta) + 1) / len(delta)],
                where="post",
                color=color,
                lw=1.2,
            )
            regret = np.sort(part.regret_pct.to_numpy())
            axes[2].step(
                np.r_[0, regret],
                np.r_[0, 100 * np.arange(1, len(regret) + 1) / len(regret)],
                where="post",
                label=f"Seed {seed}" if len(seeds) > 1 else "Model",
                color=color,
                lw=1.2,
            )
        axes[0].set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="RB effective cycle COP",
            ylabel="Model effective cycle COP",
        )
        axes[0].set_title("a  Paired decisions", loc="left", fontweight="bold", fontsize=8)
        annotation = (
            ("Improved / worsened\n" + "\n".join(counts))
            if len(seeds) > 1
            else f"{improved.sum()} improved\n{worse.sum()} worsened"
        )
        axes[0].text(0.04, 0.96, annotation, transform=axes[0].transAxes, va="top", fontsize=6.3)
        axes[1].axvline(0, color=rb_color, ls="--", lw=0.7)
        axes[1].set(xlabel="Model − RB COP", ylabel="Cumulative cycles (%)", ylim=(0, 103))
        axes[1].set_title("b  Gains and adverse cases", loc="left", fontweight="bold", fontsize=8)
        reference = metrics.loc[metrics.seed.eq(seeds[0])]
        for _, part in metrics.groupby("seed"):
            assert np.allclose(
                part.set_index("cycle_name").loc[reference.cycle_name, "rb_cop"], reference.rb_cop
            )
        values = np.sort(reference.rb_regret_pct.to_numpy())
        axes[2].step(
            np.r_[0, values],
            np.r_[0, 100 * np.arange(1, len(values) + 1) / len(values)],
            where="post",
            label="RB",
            color=rb_color,
            lw=1.4,
        )
        axes[2].axvline(1, color="#808080", ls="--", lw=0.7)
        axes[2].set(xlabel="COP gap to optimum (%)", ylabel="Cumulative cycles (%)", ylim=(0, 103))
        axes[2].legend(fontsize=6.3, loc="lower right", frameon=False)
        axes[2].set_title("c  Proximity to optimum", loc="left", fontweight="bold", fontsize=8)
        gain = 100 * (metrics.selected_cop.mean() / metrics.rb_cop.mean() - 1)
        figure.text(
            0.08,
            0.105,
            f"{len(reference)} paired cycles • {reference.experiment_id.nunique()} experiment-level LOEO folds • {len(seeds)} seed(s); separate predictions, no ensemble.\n"
            f"Mean COP +{metrics.delta_cop.mean():.4f} / +{gain:.2f}%; exact 1% near-optimal hits {100 * metrics.near_optimal_hit.mean():.2f}%.",
            fontsize=6.3,
            va="top",
        )
        save(
            figure,
            "Fig2_decision_benefit",
            "Question: does the model improve on RB? Evidence: independent seed-specific outer tests, with adverse cases retained.\n"
            "COP uses the fixed retrospective cost model, not measured energy savings. ECDFs use 101 cycles per seed, not 303 independent cycles.\n"
            "Gap denominator is each cycle optimum; gain denominator is RB. Three-seed summaries do not define an ensemble policy.",
        )
        if figures is not None and set(figures) == {"Fig2_decision_benefit"}:
            (output / "figure_captions.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
            return

        uncertainty = pd.read_csv(output / "experiment_bootstrap.csv").iloc[0]
        history = pd.read_csv(output / "inner_learning_curves.csv")
        traces = pd.read_csv(output / "case_predictions.csv")
        cases = pd.read_csv(output / "case_selection.csv")

        days = (
            metrics.groupby("experiment_id", sort=True)
            .agg(delta_cop=("delta_cop", "mean"), cycles=("cycle_name", "size"))
            .reset_index()
        )
        days.to_csv(output / "experiment_effects_source.csv", index=False)
        groups = []
        for temp, g in metrics.groupby("setpoint_c", sort=True):
            groups.append(
                dict(
                    setpoint_c=temp,
                    cycles=len(g),
                    mean_delta_cop=g.delta_cop.mean(),
                    pooled_gain_pct=100 * (g.selected_cop.mean() / g.rb_cop.mean() - 1),
                )
            )
        groups = pd.DataFrame(groups)
        groups.to_csv(output / "setpoint_effects_source.csv", index=False)
        figure, axes = plt.subplots(
            1, 2, figsize=(7.2, 3.65), gridspec_kw={"width_ratios": [1.8, 1]}
        )
        figure.subplots_adjust(left=0.085, right=0.98, bottom=0.32, top=0.82, wspace=0.32)
        ordered = days.sort_values(["delta_cop", "experiment_id"]).reset_index(drop=True)
        x = np.arange(len(ordered))
        colors = np.where(ordered.delta_cop.ge(0), "#7884B4", "#C98B79")
        bars = axes[0].bar(
            x,
            ordered.delta_cop,
            width=0.77,
            color=colors,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        for bar, negative in zip(bars, ordered.delta_cop.lt(0)):
            if negative:
                bar.set_hatch("///")
        axes[0].axhline(0, color="#606060", lw=0.7, zorder=4)
        axes[0].axhline(
            uncertainty.experiment_equal_delta_cop, color="#484878", ls="--", lw=1, zorder=4
        )
        axes[0].set(
            xticks=x,
            xticklabels=ordered.experiment_id.str.replace("exp_2026", "", regex=False),
            ylabel="Mean COP difference (Model − RB)",
            xlim=(-0.8, len(ordered) - 0.2),
        )
        axes[0].tick_params(axis="x", labelsize=5.5, rotation=90, length=0, pad=4)
        axes[0].set_title(
            "a  Improvement across experiments", loc="left", fontsize=8, fontweight="bold", pad=12
        )
        axes[0].text(
            0.02,
            0.96,
            f"{int(uncertainty.positive_experiments)}/{len(days)} experiments improved",
            transform=axes[0].transAxes,
            va="top",
            fontsize=7,
            color="#484878",
        )
        samples = [
            metrics.loc[metrics.setpoint_c.eq(row.setpoint_c), "delta_cop"].to_numpy()
            for row in groups.itertuples()
        ]
        violins = axes[1].violinplot(
            samples,
            positions=np.arange(len(groups)),
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violins["bodies"]:
            body.set_facecolor("#B4C0E4")
            body.set_edgecolor("#7884B4")
            body.set_linewidth(0.5)
            body.set_alpha(0.45)
        for index, (row, values) in enumerate(zip(groups.itertuples(), samples)):
            # Fixed horizontal displacement separates points without changing COP.
            jitter = np.random.default_rng(index).uniform(-0.15, 0.15, len(values))
            axes[1].scatter(
                index + jitter, values, s=8, color="#7884B4", alpha=0.6, linewidths=0, zorder=3
            )
            axes[1].scatter(
                index,
                row.mean_delta_cop,
                marker="D",
                s=28,
                color="#484878",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
            axes[1].text(
                index,
                0.195,
                f"+{row.mean_delta_cop:.4f}\n+{row.pooled_gain_pct:.2f}%",
                ha="center",
                va="top",
                fontsize=6.5,
                color="#484878",
            )
        axes[1].axhline(0, color="#606060", lw=0.7)
        axes[1].set(
            xticks=[0, 1, 2],
            xticklabels=[f"{r.setpoint_c:g} °C\n(n={int(r.cycles)})" for r in groups.itertuples()],
            ylim=(
                min(-0.06, metrics.delta_cop.min() * 1.1),
                max(0.21, metrics.delta_cop.max() * 1.3),
            ),
            ylabel="Cycle COP difference (Model − RB)",
        )
        axes[1].set_title(
            "b  Variation within each setpoint", loc="left", fontsize=8, fontweight="bold", pad=12
        )
        for axis in axes:
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E7E7EC", lw=0.45)
            axis.spines["left"].set_color("#A8A8A8")
            axis.spines["bottom"].set_color("#A8A8A8")
        figure.text(
            0.085,
            0.95,
            "COP improvement is positive in 27 of 29 experiments",
            fontsize=10,
            fontweight="bold",
            color="#484878",
            va="top",
        )
        figure.text(
            0.085,
            0.13,
            f"Question: is improvement consistent? Evidence: {len(days)} experiment blocks, {len(metrics)} paired cycles.\n"
            f"Experiment-equal mean {uncertainty.experiment_equal_delta_cop:+.4f}; 95% block-bootstrap CI "
            f"[{uncertainty.bootstrap_lower_95:+.4f}, {uncertainty.bootstrap_upper_95:+.4f}] (2,000 resamples).\n"
            "Bars ordered by effect; dashed line: experiment mean. Dots: cycles; diamonds: cycle means; violins: distributions, not CIs.",
            fontsize=6.5,
            va="top",
            linespacing=1.5,
        )
        save(
            figure,
            "Fig3_experiment_robustness",
            "2000 bootstrap resamples of entire experiments, equal experiment weighting, seed 0. Different from the cycle-equal overall estimate. No per-experiment confidence intervals; points are descriptive means.",
        )

        curve = (
            history.groupby("epoch")
            .agg(
                training_mean=("training_loss", "mean"),
                training_p25=("training_loss", lambda x: x.quantile(0.25)),
                training_p75=("training_loss", lambda x: x.quantile(0.75)),
                validation_mean=("validation_cdf_loss", "mean"),
                validation_p25=("validation_cdf_loss", lambda x: x.quantile(0.25)),
                validation_p75=("validation_cdf_loss", lambda x: x.quantile(0.75)),
                histories=("history_source", "nunique"),
            )
            .reset_index()
        )
        curve.to_csv(output / "loss_summary.csv", index=False)
        epochs = history[["outer_experiment", "selected_epoch"]].drop_duplicates()
        epochs.to_csv(output / "selected_epochs.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.29, top=0.86, wspace=0.45)
        for prefix, label, color in [
            ("training", "Training", rb_color),
            ("validation", "Validation", model_color),
        ]:
            axes[0].plot(curve.epoch, curve[prefix + "_mean"], label=label, color=color, lw=1.3)
            axes[0].fill_between(
                curve.epoch, curve[prefix + "_p25"], curve[prefix + "_p75"], color=color, alpha=0.15
            )
        axes[0].set(xlabel="Epoch", ylabel="Cycle-balanced CDF BCE")
        axes[0].legend(frameon=False)
        axes[0].set_title(
            "a  Inner-fold learning curves", loc="left", fontsize=8, fontweight="bold"
        )
        axes[1].hist(
            epochs.selected_epoch, bins=np.arange(0, 66, 5), color=model_color, rwidth=0.85
        )
        axes[1].set(xlabel="Selected epoch", ylabel="Outer folds")
        axes[1].set_title(
            "b  Validation-selected duration", loc="left", fontsize=8, fontweight="bold"
        )
        figure.text(
            0.09,
            0.13,
            f"Question: does the training objective converge? Evidence: {history.history_source.nunique()} inner histories across {len(epochs)} outer folds.\n"
            "Lines: equal-history means; bands: interquartile ranges, not confidence intervals.\n"
            f"History count per epoch: {curve.histories.min()}–{curve.histories.max()}. Test data did not select epochs.",
            fontsize=6.5,
            va="top",
        )
        save(
            figure,
            "Fig4a_training",
            "Training curves are inner fits, not outer refits or test losses. Interquartile bands describe overlapping fold histories; they are not independent-replicate uncertainty.",
        )

        figure, axes = plt.subplots(len(cases), 2, figsize=(7.2, 6.7))
        figure.subplots_adjust(
            left=0.09, right=0.98, bottom=0.16, top=0.95, wspace=0.34, hspace=0.6
        )
        for index, case in cases.iterrows():
            row = metrics.loc[metrics.cycle_name.eq(case.cycle_name)].iloc[0]
            trace = traces.loc[traces.cycle_name.eq(case.cycle_name)].copy()
            time = pd.to_datetime(trace.candidate_defrost_time)
            origin = pd.Timestamp(row.cycle_start)
            minutes = (time - origin).dt.total_seconds() / 60
            near = trace.near_optimal.eq(True)
            for axis in axes[index]:
                for minute in minutes[near]:
                    axis.axvspan(minute - 0.25, minute + 0.25, color="#DBE7ED", lw=0, zorder=0)
            axes[index, 0].plot(
                minutes,
                trace.cycle_cop.where(trace.cycle_cop_eligible.eq(True)),
                color=model_color,
                lw=1,
            )
            axes[index, 0].axhline(0.99 * row.optimal_cop, color=rb_color, ls=":", lw=0.7)
            axes[index, 0].set_ylabel("Effective cycle COP")
            axes[index, 1].plot(
                minutes,
                trace.stopping_cdf_target,
                color=rb_color,
                ls="--",
                lw=1,
                label="Training target",
            )
            axes[index, 1].plot(
                minutes, trace.probability, color=model_color, lw=1.3, label="Predicted CDF"
            )
            axes[index, 1].axhline(0.5, color=rb_color, ls=":", lw=0.7)
            axes[index, 1].set(ylim=(-0.03, 1.03), ylabel="Cumulative trigger probability")
            for axis in axes[index]:
                for column, color, style in [
                    ("t_RB", "#4C8A83", "--"),
                    ("selected_time", "#B08342", "-"),
                    ("optimal_time", "#333333", ":"),
                ]:
                    axis.axvline(
                        (pd.Timestamp(row[column]) - origin).total_seconds() / 60,
                        color=color,
                        ls=style,
                        lw=0.8,
                    )
                axis.set_xlabel("Time from cycle start (min)")
            title = f"{case.cycle_name.replace('frost_cycle_', 'Cycle ')}\n{case.role}"
            axes[index, 0].set_title(title, fontsize=7, loc="left")
            axes[index, 1].set_title(
                f"COP gap {case.regret_pct:.3f}% | near components {int(case.near_components)}",
                fontsize=7,
                loc="left",
            )
        axes[0, 1].legend(fontsize=6, frameon=False, loc="upper left")
        figure.text(
            0.09,
            0.09,
            "Question: how does the learned trigger relate to COP? Evidence: cases selected by the recorded numerical criteria.\n"
            "Blue shading: exact near-optimal action slots. Vertical lines: RB (green dashed), model (ochre solid), optimum (black dotted).\n"
            "Training ramps span internal holes; evaluation uses exact COP. Saved outer-test predictions; no new energy calculation.",
            fontsize=6.5,
            va="top",
        )
        save(
            figure,
            "Fig4b_decision_cases",
            "Cases selected without visual inspection: largest near span; most near components if more than one exists, otherwise largest gain over RB; largest decision regret among remaining cycles. CDF reconstructed using the existing trigger interface. Time-grid shading is not a new continuous near-set definition.",
        )
    if figures is None:
        (output / "figure_captions.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
        render_stopping_timing(output)


def render_stopping_timing(output: Path) -> None:
    """Paired trigger intervals; reciprocal means are descriptive equivalent rates."""
    metrics = pd.read_csv(output / "cycle_results.csv")
    timing = (
        metrics[
            [
                "cycle_name",
                "experiment_id",
                "setpoint_c",
                "rb_cycle_duration_minutes",
                "cycle_duration_minutes",
                "delay_vs_rb_minutes",
            ]
        ]
        .dropna()
        .copy()
    )
    if not timing[["rb_cycle_duration_minutes", "cycle_duration_minutes"]].gt(0).all().all():
        raise ValueError("Equivalent trigger rates require positive paired intervals")
    timing.to_csv(output / "trigger_timing_source.csv", index=False)
    summary = []
    for label, g in [("All", timing), *timing.groupby("setpoint_c", sort=True)]:
        rb, model = g.rb_cycle_duration_minutes.mean(), g.cycle_duration_minutes.mean()
        summary.append(
            dict(
                group=str(label),
                cycles=len(g),
                rb_interval_minutes=rb,
                model_interval_minutes=model,
                delay_minutes=model - rb,
                interval_increase_pct=100 * (model / rb - 1),
                rb_rate_per_hour=60 / rb,
                model_rate_per_hour=60 / model,
                rate_difference_per_hour=60 / model - 60 / rb,
                rate_reduction_pct=100 * (1 - rb / model),
                later_cycles=int(g.delay_vs_rb_minutes.gt(0).sum()),
                earlier_cycles=int(g.delay_vs_rb_minutes.lt(0).sum()),
            )
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "trigger_timing_summary.csv", index=False)
    overall = summary.iloc[0]
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
        axes = axes.ravel()
        figure.subplots_adjust(
            left=0.09, right=0.98, bottom=0.2, top=0.92, wspace=0.35, hspace=0.65
        )
        low = 0
        high = 1.05 * max(
            timing.rb_cycle_duration_minutes.max(), timing.cycle_duration_minutes.max()
        )
        axes[0].plot([low, high], [low, high], color="#969696", ls="--", lw=0.7)
        later = timing.delay_vs_rb_minutes.gt(0)
        for mask, color, marker in [(later, "#436B85", "o"), (~later, "#B56E57", "x")]:
            axes[0].scatter(
                timing.loc[mask, "rb_cycle_duration_minutes"],
                timing.loc[mask, "cycle_duration_minutes"],
                color=color,
                marker=marker,
                s=12,
                alpha=0.75,
            )
        axes[0].set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="RB trigger interval (min)",
            ylabel="Model trigger interval (min)",
        )
        axes[0].set_title("a  Paired trigger intervals", loc="left", fontsize=8, fontweight="bold")
        axes[0].text(
            0.04,
            0.96,
            f"{int(overall.later_cycles)} later\n{int(overall.earlier_cycles)} earlier",
            transform=axes[0].transAxes,
            va="top",
        )
        delta = np.sort(timing.delay_vs_rb_minutes.to_numpy())
        axes[1].step(
            np.r_[delta[0], delta],
            np.r_[0, 100 * np.arange(1, len(delta) + 1) / len(delta)],
            where="post",
            color="#436B85",
            lw=1.3,
        )
        axes[1].axvline(0, color="#969696", ls="--", lw=0.7)
        axes[1].set(
            xlabel="Model − RB trigger time (min)", ylabel="Cumulative cycles (%)", ylim=(0, 103)
        )
        axes[1].set_title(
            "b  Distribution of postponement", loc="left", fontsize=8, fontweight="bold"
        )
        x = np.arange(len(summary))
        labels = ["All", *[f"{float(g):g} °C" for g in summary.group.iloc[1:]]]
        labels = [f"{label}\n(n={int(n)})" for label, n in zip(labels, summary.cycles)]
        for offset, prefix, label, color in [
            (-0.18, "rb", "RB", "#969696"),
            (0.18, "model", "Model", "#436B85"),
        ]:
            axes[2].bar(
                x + offset,
                summary[prefix + "_interval_minutes"],
                width=0.32,
                color=color,
                label=label,
            )
            axes[3].bar(
                x + offset, summary[prefix + "_rate_per_hour"], width=0.32, color=color, label=label
            )
        for index, row in summary.iterrows():
            axes[2].text(
                index,
                115,
                f"+{row.delay_minutes:.2f} min\n+{row.interval_increase_pct:.2f}%",
                ha="center",
                va="top",
                fontsize=6,
            )
            axes[3].text(
                index,
                1.07,
                f"{row.rate_difference_per_hour:.3f}/h\n−{row.rate_reduction_pct:.2f}%",
                ha="center",
                va="top",
                fontsize=6,
            )
        axes[2].set(
            xticks=x, xticklabels=labels, ylabel="Mean trigger interval (min)", ylim=(0, 120)
        )
        axes[3].set(
            xticks=x, xticklabels=labels, ylabel="Equivalent triggers per hour", ylim=(0, 1.12)
        )
        handles, labels_legend = axes[2].get_legend_handles_labels()
        figure.legend(
            handles,
            labels_legend,
            frameon=False,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.995),
            ncol=2,
        )
        axes[2].set_title(
            "c  Trigger intervals by setpoint", loc="left", fontsize=8, fontweight="bold", pad=12
        )
        axes[3].set_title(
            "d  Equivalent rates by setpoint", loc="left", fontsize=8, fontweight="bold", pad=12
        )
        figure.text(
            0.09,
            0.115,
            f"Question: does the model postpone RB defrost triggers? Evidence: {len(timing)} paired outer-test cycles.\n"
            f"Conclusion: mean interval {overall.rb_interval_minutes:.2f} → {overall.model_interval_minutes:.2f} min "
            f"(+{overall.delay_minutes:.2f} min / +{overall.interval_increase_pct:.2f}%).\n"
            "Rate = 60 / mean interval from recorded cycle start to trigger. Excludes subsequent preparation/defrost.\n"
            "Retrospective equivalent rate; not measured defrost counts. Longer intervals alone do not establish better efficiency.",
            fontsize=6.5,
            va="top",
            linespacing=1.4,
        )
        figure.canvas.draw()
        for extension in ("png", "pdf", "svg", "tiff"):
            figure.savefig(
                output / f"Fig5_defrost_trigger_timing.{extension}", dpi=600, facecolor="white"
            )
        plt.close(figure)


if __name__ == "__main__":
    main()
