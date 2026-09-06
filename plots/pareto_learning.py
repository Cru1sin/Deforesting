"""First-frame decision evidence for the shared economic × relation experiment.

Cycle-level timing/C/H remain separate; unknown economic points are not safe points.
Figures use observed held-out streams, with no smoothing or threshold selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_models.evaluation import classification_metrics
from plots.image_models import _export, plot_trigger_error_figures, trigger_error_table
from plots.pareto_selection import plot_cop_heating_rate_pareto

STYLES = {
    "baseline": ("#666666", "o"), "economic": ("#3B7E9B", "s"),
    "relation": ("#A37CAA", "^"), "combined": ("#365A83", "D"),
    "nonvisual": ("#C29565", "v"),
}
IDENTITY = ["method", "seed"]


def _label(method, seed):
    return f"{method} · seed {seed}"


def _groups(frame):
    return sorted(frame.groupby(IDENTITY, sort=False),
                  key=lambda item: (list(STYLES).index(item[0][0]), item[0][1]))


def _supported(row, objective):
    value = row[f"{objective}_eligible_without_extrapolation"]
    return pd.notna(value) and bool(value)


def _median(values):
    finite = values.dropna()
    return finite.median() if len(finite) else np.nan


def _domain(row):
    if not np.isfinite([row.economic_c, row.economic_h]).all():
        return "unknown"
    names = [
        f"defrost_event_{quantity}_in_training_domain"
        for quantity in ("electricity", "net_heat", "duration")
    ]
    if any(name in row and pd.notna(row[name]) and not bool(row[name]) for name in names):
        return "outside_training_domain"
    if all(_supported(row, name) for name in ("cycle_cop", "cycle_heating_rate_kw")):
        return "in_training_domain"
    return "unknown"


def _adapt(predictions):
    return predictions.assign(
        camera="front", image_feature="dinov2", classifier="pareto_boundary",
        input_feature=[_label(m, s) for m, s in predictions[IDENTITY].itertuples(index=False)],
        decision_score=predictions.logit,
    )


def evaluate_predictions(predictions, teachers):
    """Return cycle decisions, grouped frame metrics, and per-method/seed summaries."""
    cycle_rows, frame_rows = [], []
    for (method, seed, heldout), stream in predictions.groupby(
        [*IDENTITY, "heldout_experiment"], sort=True
    ):
        reference = teachers.loc[teachers.heldout_experiment.eq(heldout)]
        decisions = stream[["cycle_name", "teacher_time"]].drop_duplicates().rename(
            columns={"teacher_time": "selected_defrost_time"}
        ).assign(is_selected=True)
        errors = trigger_error_table(_adapt(stream), decisions, threshold=0.)
        errors = errors.loc[errors.strategy.eq("first_positive")]
        reference = reference.set_index("cycle_name")
        for error in errors.to_dict("records"):
            cycle = stream.loc[stream.cycle_name.eq(error["cycle_name"])].sort_values("image_time")
            teacher_time = pd.to_datetime(cycle.teacher_time.iloc[0])
            native_times = pd.to_datetime(cycle.image_time)
            first = (native_times.iloc[0] - teacher_time).total_seconds() / 60
            last = (native_times.iloc[-1] - teacher_time).total_seconds() / 60
            relative = (native_times - teacher_time).dt.total_seconds() / 60
            nearest_pre = relative.loc[relative.le(0)].max()
            nearest_post = relative.loc[relative.ge(0)].min()
            observation_status = (
                "teacher_absent" if pd.isna(teacher_time) else
                "starts_after_knee" if first > 0 else
                "ends_before_knee" if last < 0 else "brackets_knee"
            )
            row = {
                **error, "method": method, "seed": seed, "heldout_experiment": heldout,
                "persistent_high": bool(cycle.logit.ge(0).all()),
                "persistent_low": bool(cycle.logit.lt(0).all()),
                "trigger_domain": "no_trigger", "delta_c": np.nan, "delta_h": np.nan,
                "first_native_relative_teacher_minutes": first,
                "last_native_relative_teacher_minutes": last,
                "observation_status": observation_status,
                "observed_post_knee": bool(last > 0),
                "nearest_pre_knee_relative_minutes": nearest_pre,
                "nearest_post_knee_relative_minutes": nearest_post,
                "trigger_extra_observed_delay_minutes": (
                    error["trigger_error_minutes"] - nearest_post
                    if error["trigger_error_minutes"] >= 0 else np.nan
                ),
            }
            covered = error["cycle_name"] in reference.index
            row["teacher_covered"] = covered
            knee = reference.loc[error["cycle_name"]] if covered else pd.Series({
                "economic_c": np.nan, "economic_h": np.nan,
                "cycle_cop_eligible_without_extrapolation": False,
                "cycle_heating_rate_kw_eligible_without_extrapolation": False,
            })
            row.update(teacher_c=knee.economic_c, teacher_h=knee.economic_h)
            if pd.notna(error["trigger_time"]):
                trigger = cycle.loc[cycle.image_time.eq(error["trigger_time"])].iloc[0]
                row.update(trigger_domain=_domain(trigger), trigger_c=trigger.economic_c,
                           trigger_h=trigger.economic_h)
                for short, objective in (("c", "cycle_cop"), ("h", "cycle_heating_rate_kw")):
                    quantities = [trigger[f"economic_{short}"], knee[f"economic_{short}"]]
                    if (_supported(trigger, objective) and _supported(knee, objective)
                            and np.isfinite(quantities).all()):
                        row[f"delta_{short}"] = quantities[0] - quantities[1]
            cycle_rows.append(row)
        hard = stream.loc[stream.target.isin([0, 1])]
        frame_rows.append({
            "method": method, "seed": seed, "heldout_experiment": heldout,
            "frame_count": len(hard),
            **(classification_metrics(
                hard.target.astype(int), hard.prediction.astype(int), "binary"
            ) if len(hard) else {key: np.nan for key in (
                "accuracy", "balanced_accuracy", "macro_f1"
            )}),
        })
    cycles = pd.DataFrame(cycle_rows)
    summaries = []
    for (method, seed), group in cycles.groupby(IDENTITY, sort=True):
        errors = group.trigger_error_minutes
        triggered = group.trigger_time.notna()
        trigger_domains = group.loc[triggered, "trigger_domain"]
        bracketed = group.loc[group.observation_status.eq("brackets_knee")]
        bracketed_errors = bracketed.loc[bracketed.trigger_time.notna(), "trigger_error_minutes"]
        summaries.append({
            "method": method, "seed": seed, "cycles": len(group),
            "teacher_covered_cycles": int(group.teacher_covered.sum()),
            "triggered_cycles": int(triggered.sum()),
            "no_trigger_fraction": float((~triggered).mean()),
            "median_error_minutes": _median(errors),
            "median_absolute_error_minutes": _median(errors.abs()),
            "p10_error_minutes": errors.quantile(.1), "p90_error_minutes": errors.quantile(.9),
            "outside_domain_trigger_fraction": trigger_domains.eq("outside_training_domain").mean(),
            "unknown_domain_trigger_fraction": trigger_domains.eq("unknown").mean(),
            "persistent_high_fraction": group.persistent_high.mean(),
            "persistent_low_fraction": group.persistent_low.mean(),
            "delta_c_evaluable_cycles": int(group.delta_c.notna().sum()),
            "delta_h_evaluable_cycles": int(group.delta_h.notna().sum()),
            "median_delta_c": _median(group.delta_c), "median_delta_h": _median(group.delta_h),
            **{f"{status}_cycles": int(group.observation_status.eq(status).sum()) for status in (
                "teacher_absent", "starts_after_knee", "ends_before_knee", "brackets_knee"
            )},
            "brackets_knee_trigger_fraction": bracketed.trigger_time.notna().mean(),
            "brackets_knee_median_error_minutes": _median(bracketed_errors),
            "brackets_knee_median_absolute_error_minutes": _median(bracketed_errors.abs()),
        })
    return cycles, pd.DataFrame(frame_rows), pd.DataFrame(summaries)


def _design(output):
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    axis.axis("off")
    for method, x, y, text in (
        ("baseline", .05, .56, "Economic OFF · relation OFF\nBCE"),
        ("economic", .54, .56, "Economic ON · relation OFF\nBCE"),
        ("relation", .05, .15, "Economic OFF · relation ON\nBCE + same-side |logit| rank"),
        ("combined", .54, .15, "Economic ON · relation ON\nBCE + same-side |logit| rank"),
    ):
        axis.text(x, y, f"{method.capitalize()}\n{text}", transform=axis.transAxes,
                  fontsize=9, va="bottom", color=STYLES[method][0],
                  bbox={"facecolor": "#F5F5F5", "edgecolor": "none", "pad": 10})
    axis.set_title("Two factors, one model and one first-frame controller", fontsize=11)
    figure.text(.5, .01, "Shared frozen front DINOv2 + causal sensor Sin-MLP; s ≥ 0 triggers.\n"
                "G, teacher and preprocessing are experiment-isolated. Nonvisual zeros RGB.",
                ha="center", fontsize=7)
    _export(figure, output / "experiment_design")


def _loss_plot(losses, output):
    settings = _groups(losses)
    figure, axes = plt.subplots(len(settings), 3, figsize=(7.2, 2 * len(settings)), squeeze=False)
    for axes_row, ((method, seed), rows) in zip(axes, settings, strict=True):
        for axis, split in zip(axes_row, ("train_side", "val_side", "train_rank"), strict=True):
            for _, fold in rows.loc[rows.split.eq(split)].groupby("heldout_experiment"):
                axis.plot(fold.epoch, fold.loss, color=STYLES[method][0], lw=.7, alpha=.45)
            title = split.replace("_", " ")
            if split == "train_rank":
                title += "\n" + ("optimized" if method in {"relation", "combined"}
                                  else "diagnostic only")
            axis.set(xlabel="Epoch", title=title)
        axes_row[0].set_ylabel(_label(method, seed))
    figure.suptitle("Grouped inner loss: each line is one held-out-experiment run", fontsize=10)
    figure.tight_layout()
    _export(figure, output / "losses")


def _summary_plot(cycles, output):
    settings = _groups(cycles)
    figure, axes = plt.subplots(1, 3, figsize=(9, 3.7))
    for axis, field, title in zip(
        axes, ("trigger_error_minutes", "delta_c", "delta_h"),
        ("Trigger − knee [min]", "Δ cycle COP [−]", "Δ heating rate [kW]"), strict=True
    ):
        for index, ((method, _seed), rows) in enumerate(settings):
            values = rows[field].dropna().to_numpy()
            x = index + np.linspace(-.14, .14, len(values))
            axis.scatter(x, values, s=12, color=STYLES[method][0], alpha=.65)
            if len(values):
                axis.plot([index - .2, index + .2], [np.median(values)] * 2, color="#333333", lw=1)
        axis.axhline(0, color="#777777", lw=.7)
        labels = [_label(*key) + f"\nmiss {g.trigger_time.isna().sum()}/{len(g)}"
                  + f"; ref {g.teacher_covered.sum()}/{len(g)}" for key, g in settings]
        axis.set(title=title, xticks=range(len(settings)), xticklabels=labels)
        axis.tick_params(axis="x", rotation=35, labelsize=6)
    figure.suptitle("First native-frame activation; C and H are evaluated separately", fontsize=10)
    figure.text(.5, .01, "Dots: cycles; lines: medians. Misses and absent references are counted.\n"
                "C/H differences require evaluable, in-domain trigger and exact knee.",
                ha="center", fontsize=7)
    figure.tight_layout(rect=(0, .10, 1, .95))
    _export(figure, output / "decision_summary")


def _representatives(predictions, cycles, output, teacher_curves=None):
    available = cycles.dropna(subset=["trigger_error_minutes"]).sort_values("trigger_error_minutes")
    names = (available.iloc[[0, len(available) // 2, -1]].cycle_name.drop_duplicates().tolist()
             if len(available) else cycles.cycle_name.drop_duplicates().head(3).tolist())
    columns = 2 if teacher_curves is not None else 1
    figure, axes = plt.subplots(len(names), columns,
                               figsize=(11 if columns == 2 else 7.2, 3.4 * len(names)),
                               squeeze=False)
    source = predictions.loc[predictions.cycle_name.isin(names)].copy()
    source.to_csv(output / "representative_streams.csv", index=False)
    pareto_sources = []
    for axis, name in zip(axes[:, 0], names, strict=True):
        selected = source.loc[source.cycle_name.eq(name)]
        origin = pd.Timestamp(selected.heating_start.iloc[0])
        stable = (pd.Timestamp(selected.stable_heating_start.iloc[0]) - origin).total_seconds() / 60
        end = (pd.to_datetime(selected.image_time).max() - origin).total_seconds() / 60
        axis.axvspan(0, stable, color="#78A6BC", alpha=.15)
        axis.axvspan(stable, end, color="#F2A35E", alpha=.10)
        for (method, seed), rows in _groups(selected):
            rows = rows.sort_values("image_time")
            minute = (pd.to_datetime(rows.image_time) - origin).dt.total_seconds() / 60
            segments = pd.to_datetime(rows.image_time).diff().gt(pd.Timedelta(seconds=45)).cumsum()
            for segment, chunk in rows.groupby(segments, sort=False):
                axis.plot(minute.loc[chunk.index], chunk.logit, color=STYLES[method][0],
                          lw=.85, marker=".", markersize=2,
                          label=_label(method, seed) if segment == 0 else "_nolegend_")
            hits = minute.loc[rows.logit.ge(0)]
            if len(hits):
                axis.axvline(hits.iloc[0], color=STYLES[method][0], lw=.8, ls=":")
        if pd.notna(selected.teacher_time.iloc[0]):
            knee = (pd.Timestamp(selected.teacher_time.iloc[0]) - origin).total_seconds() / 60
            axis.axvline(knee, color="black", ls="--", lw=1, label="Teacher knee")
        axis.axhline(0, color="#777777", lw=.7)
        axis.set(title=name, xlabel="Time from cycle start [min]", ylabel="Raw decision logit")
        if teacher_curves is not None:
            panel = axes[names.index(name), 1]
            grid = teacher_curves.loc[
                teacher_curves.cycle_name.eq(name)
                & teacher_curves.heldout_experiment.eq(selected.heldout_experiment.iloc[0])
                & teacher_curves.is_teacher_candidate
            ].copy()
            grid["is_cop_heating_rate_pareto_point"] = grid.pareto_selection_score.notna()
            grid["is_selected_pareto_point"] = grid.is_knee
            plot_cop_heating_rate_pareto(panel, grid, origin)
            fields = ["cycle_name", "heldout_experiment", "candidate_defrost_time",
                      "cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw",
                      "cycle_cop_eligible", "cycle_heating_rate_kw_eligible",
                      "pareto_selection_score", "is_knee"]
            pareto_sources.append(grid.loc[:, grid.columns.intersection(fields)])
            for (method, seed), stream in _groups(selected):
                hit = stream.sort_values("image_time").loc[lambda x: x.logit.ge(0)].head(1)
                if not hit.empty and np.isfinite(hit[["economic_c", "economic_h"]]).all().all():
                    x, y = hit[["economic_c", "economic_h"]].iloc[0]
                    panel.scatter(x, y, color=STYLES[method][0], marker=STYLES[method][1],
                                  s=45, edgecolors="white", linewidths=.6, zorder=6,
                                  label=_label(method, seed))
                    for value, getter, setter in ((x, panel.get_xlim, panel.set_xlim),
                                                   (y, panel.get_ylim, panel.set_ylim)):
                        low, high = getter()
                        span = max(high, value) - min(low, value)
                        setter(min(low, value - .03 * span), max(high, value + .03 * span))
            panel.legend(fontsize=5.5, ncol=2)
    if pareto_sources:
        pd.concat(pareto_sources).to_csv(output / "representative_pareto.csv", index=False)
        cycles.loc[cycles.cycle_name.isin(names), [
            "method", "seed", "heldout_experiment", "cycle_name", "trigger_time",
            "trigger_c", "trigger_h", "trigger_domain"
        ]].to_csv(output / "representative_pareto_triggers.csv", index=False)
    axes[0, 0].legend(ncol=3, fontsize=6)
    figure.suptitle("Full streams: blue recovery, amber frosting; dotted triggers\n"
                   "Gaps: no recent RGB (>45 s)", fontsize=9)
    figure.tight_layout()
    _export(figure, output / "representative_streams")


def _pair_order_plot(pairs, output):
    selected = pairs.loc[pairs.split.isin(["outer_train", "outer_test"])
                         & pairs.pairs.gt(0)].dropna(subset=["pair_order_accuracy"])
    if selected.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    settings = _groups(selected)
    for index, ((method, _seed), rows) in enumerate(settings):
        for offset, split, marker in ((-.15, "outer_train", "o"), (.15, "outer_test", "^")):
            values = rows.loc[rows.split.eq(split)].groupby(
                "heldout_experiment").pair_order_accuracy.mean()
            axis.scatter(np.full(len(values), index + offset), values, marker=marker,
                         color=STYLES[method][0], s=18, alpha=.65)
            axis.plot([index + offset - .08, index + offset + .08], [values.mean()] * 2,
                      color="black", lw=1.4)
            axis.text(index + offset, 1.03, f"n={len(values)}", ha="center", fontsize=6)
    axis.axhline(.5, color="#999999", ls="--", lw=.8)
    axis.set(ylim=(0, 1.1), ylabel="Same-side pair-order accuracy", xticks=range(len(settings)),
             xticklabels=[_label(*key) for key, _ in settings],
             title="Does improved training order generalize to unseen experiments?")
    axis.tick_params(axis="x", labelsize=7)
    figure.text(.5, .01, "Left circles: outer train; right triangles: outer test. "
                "Black: fold-equal mean (test: held-out experiments); 0.5: chance reference.",
                ha="center", fontsize=7)
    figure.tight_layout(rect=(0, .06, 1, 1))
    _export(figure, output / "pair_order_diagnostics")


def render_figures(predictions, teachers, losses, pairs, output, teacher_curves=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cycles, frames, summary = evaluate_predictions(predictions, teachers)
    for name, table in (("cycle_decisions", cycles), ("frame_metrics", frames),
                        ("summary", summary), ("losses", losses), ("pair_diagnostics", pairs)):
        table.to_csv(output / f"{name}.csv", index=False)
    _design(output)
    if not losses.empty:
        _loss_plot(losses, output)
    _summary_plot(cycles, output)
    _representatives(predictions, cycles, output, teacher_curves)
    if not pairs.empty:
        _pair_order_plot(pairs, output)
    adapted = _adapt(predictions)
    decisions = predictions[["cycle_name", "teacher_time"]].drop_duplicates().rename(
        columns={"teacher_time": "selected_defrost_time"}
    ).assign(is_selected=True)
    styles = {_label(method, seed): STYLES[method] for method, seed in
              predictions[IDENTITY].drop_duplicates().itertuples(index=False)}
    plot_trigger_error_figures(
        predictions=adapted, decisions=decisions, output=output, source_output=output,
        image_feature="dinov2", classifier="pareto_boundary", continuous_stream=True,
        method_styles=styles, policies=("first_positive",), threshold=0., flat_output=True,
        error_quantile=.90,
    )
    return cycles, frames, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = {name: [] for name in ("predictions", "losses", "pair_metrics")}
    for run in args.runs:
        seed = json.loads((run / "settings.json").read_text())["seed"]
        for name in tables:
            path = run / (f"{name}.parquet" if name == "predictions" else f"{name}.csv")
            frame = pd.read_parquet(path) if name == "predictions" else pd.read_csv(path)
            tables[name].append(frame.assign(seed=seed))
    predictions, losses, pairs = [pd.concat(tables[name], ignore_index=True) for name in tables]
    base = pd.read_parquet(args.data / "base.parquet")
    teachers, teacher_curves = [], []
    for heldout in predictions.heldout_experiment.unique():
        fold = pd.read_parquet(args.data / "teachers" / f"{heldout}.parquet")
        overlap = base.columns.intersection(fold.columns).difference(["row_id"])
        values = base.drop(columns=overlap).merge(fold, on="row_id", validate="one_to_one")
        selected = values.loc[values.is_knee & values.experiment_id.eq(heldout)]
        teachers.append(selected.assign(heldout_experiment=heldout))
        teacher_curves.append(values.loc[
            values.is_teacher_candidate & values.experiment_id.eq(heldout)
        ].assign(heldout_experiment=heldout))
    render_figures(predictions, pd.concat(teachers), losses, pairs, args.output,
                   teacher_curves=pd.concat(teacher_curves))


if __name__ == "__main__":
    main()
