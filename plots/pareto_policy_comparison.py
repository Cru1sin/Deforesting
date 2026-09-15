"""Diagnostic comparison of nine independent and Pareto timing policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

from defrost_decision.selection_results import five_minute_support_runs  # noqa: E402

METHOD_LABELS = (
    "C",
    "H",
    "O",
    "CH knee",
    "CO knee",
    "HO knee",
    "CH → max O",
    "CO → max H",
    "HO → max C",
)
OBJECTIVES = {
    "C": "cycle_cop",
    "H": "cycle_heating_rate_kw",
    "O": "cycle_evaporator_capacity_kw",
}
METHOD_COLORS = {
    "C": "#A9BDE1",
    "H": "#F1C27D",
    "O": "#9FD3C7",
    "CH knee": "#3F5F9B",
    "CO knee": "#287F88",
    "HO knee": "#7A5C88",
    "CH → max O": "#E6C3CC",
    "CO → max H": "#DFA9B8",
    "HO → max C": "#CF839A",
}
PAIR_METHODS = {
    "CH knee": ("knee", "C", "H", None),
    "CO knee": ("knee", "C", "O", None),
    "HO knee": ("knee", "H", "O", None),
    "CH → max O": ("third", "C", "H", "O"),
    "CO → max H": ("third", "C", "O", "H"),
    "HO → max C": ("third", "H", "O", "C"),
}


def _eligible_masks(candidates: pd.DataFrame) -> dict[str, pd.Series]:
    minimum = pd.Timestamp(candidates["stable_heating_start"].iloc[0])
    masks = {}
    for symbol, column in OBJECTIVES.items():
        raw = (
            pd.to_datetime(candidates["candidate_defrost_time"]).ge(minimum)
            & candidates[f"{column}_eligible"].fillna(False)
            & np.isfinite(candidates[column])
        )
        masks[symbol] = raw & five_minute_support_runs(
            candidates["candidate_defrost_time"], raw
        )
    return masks


def _earliest_max(candidates: pd.DataFrame, positions: np.ndarray, column: str) -> int:
    values = candidates.loc[positions, column]
    return int(values.loc[values.eq(values.max())].index[0])


def _pareto_front(
    candidates: pd.DataFrame, masks: dict[str, pd.Series], left: str, right: str
) -> np.ndarray:
    positions = candidates.index[masks[left] & masks[right]].to_numpy()
    if not len(positions):
        return positions
    x = candidates.loc[positions, OBJECTIVES[left]].to_numpy(dtype=float)
    y = candidates.loc[positions, OBJECTIVES[right]].to_numpy(dtype=float)
    front = np.ones(len(positions), dtype=bool)
    # ponytail: quadratic scan is simplest and remains cheap for <1,000 candidates/cycle.
    for index, (x_value, y_value) in enumerate(zip(x, y, strict=True)):
        front[index] = not np.any(
            (x >= x_value) & (y >= y_value) & ((x > x_value) | (y > y_value))
        )
    return positions[front]


def _pareto_knee(
    candidates: pd.DataFrame, positions: np.ndarray, left: str, right: str
) -> int:
    x = candidates.loc[positions, OBJECTIVES[left]].to_numpy(dtype=float)
    y = candidates.loc[positions, OBJECTIVES[right]].to_numpy(dtype=float)
    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    scores = -np.hypot(
        (x.max() - x) / max(abs(float(x.max())), 1e-12),
        (y.max() - y) / max(abs(float(y.max())), 1e-12),
    )
    if len(positions) >= 3 and x_span > 0 and y_span > 0:
        chord = (
            (x - x.min()) / x_span + (y - y.min()) / y_span - 1
        ) / np.sqrt(2)
        if float(chord.max()) > 1e-12:
            scores = chord
    return int(positions[np.flatnonzero(scores == scores.max())[0]])


def _select_policy_indices(candidates: pd.DataFrame) -> dict[str, int | None]:
    rows = candidates.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
    masks = _eligible_masks(rows)
    selected: dict[str, int | None] = {}
    for symbol in OBJECTIVES:
        positions = rows.index[masks[symbol]].to_numpy()
        selected[symbol] = (
            _earliest_max(rows, positions, OBJECTIVES[symbol]) if len(positions) else None
        )
    for method, (kind, left, right, third) in PAIR_METHODS.items():
        front = _pareto_front(rows, masks, left, right)
        if kind == "third" and len(front):
            assert third is not None
            front = front[masks[third].iloc[front].to_numpy()]
        if not len(front):
            selected[method] = None
        elif kind == "knee":
            selected[method] = _pareto_knee(rows, front, left, right)
        else:
            assert third is not None
            selected[method] = _earliest_max(rows, front, OBJECTIVES[third])
    return selected


def select_policy_times(candidates: pd.DataFrame) -> dict[str, pd.Timestamp | pd.NaT]:
    """Return the nine diagnostic selections on one cycle's frozen candidate grid."""
    rows = candidates.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
    indices = _select_policy_indices(rows)
    return {
        method: (
            pd.Timestamp(rows.loc[index, "candidate_defrost_time"])
            if index is not None
            else pd.NaT
        )
        for method, index in indices.items()
    }


def load_out_of_fold_candidates(data_dir: Path) -> pd.DataFrame:
    """Join each cycle only to the teacher that excluded its experiment."""
    base_columns = [
        "row_id",
        "cycle_name",
        "experiment_id",
        "candidate_defrost_time",
        "is_teacher_candidate",
        "stable_heating_start",
        "observed_defrost_preparation_start",
    ]
    base = pd.read_parquet(data_dir / "base.parquet", columns=base_columns)
    teacher_columns = [
        "row_id",
        *OBJECTIVES.values(),
        *(f"{column}_eligible" for column in OBJECTIVES.values()),
        "is_knee",
    ]
    parts = []
    for experiment_id, rows in base.groupby("experiment_id", sort=False):
        teacher = pd.read_parquet(
            data_dir / "teachers" / f"{experiment_id}.parquet",
            columns=teacher_columns,
        )
        parts.append(rows.merge(teacher, on="row_id", validate="one_to_one"))
    result = pd.concat(parts, ignore_index=True)
    result = result.loc[result["is_teacher_candidate"]].copy()
    result["candidate_defrost_time"] = pd.to_datetime(result["candidate_defrost_time"])
    return result


def evaluate_policies(candidates: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    """Evaluate selections without relabelling any historical method."""
    rb_lookup = rb.set_index("cycle_name")
    records = []
    for cycle_name, group in candidates.groupby("cycle_name", sort=True):
        rows = group.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
        masks = _eligible_masks(rows)
        indices = _select_policy_indices(rows)
        maxima = {}
        for symbol, column in OBJECTIVES.items():
            positions = rows.index[masks[symbol]].to_numpy()
            maxima[symbol] = (
                _earliest_max(rows, positions, column) if len(positions) else None
            )
        stored_knee = rows.index[rows["is_knee"].fillna(False)].to_numpy()
        if indices["CH knee"] is not None and (
            len(stored_knee) != 1 or indices["CH knee"] != int(stored_knee[0])
        ):
            raise ValueError(f"generalized CH knee differs from frozen selector: {cycle_name}")
        observed = pd.Timestamp(rows["observed_defrost_preparation_start"].iloc[0])
        for method in METHOD_LABELS:
            index = indices[method]
            if index is None:
                continue
            selected_time = pd.Timestamp(rows.loc[index, "candidate_defrost_time"])
            record: dict[str, object] = {
                "cycle_name": cycle_name,
                "experiment_id": str(rows["experiment_id"].iloc[0]),
                "method": method,
                "selected_time": selected_time,
                "minutes_before_observed_preparation": (
                    observed - selected_time
                ).total_seconds()
                / 60,
            }
            if cycle_name in rb_lookup.index:
                rb_row = rb_lookup.loc[cycle_name]
                triggered = str(rb_row["rb_status"]) == "triggered" and pd.notna(rb_row["t_RB"])
                record["rb_status"] = str(rb_row["rb_status"])
                record["rb_time"] = pd.Timestamp(rb_row["t_RB"]) if triggered else pd.NaT
                record["delta_vs_rb_minutes"] = (
                    (selected_time - pd.Timestamp(rb_row["t_RB"])).total_seconds() / 60
                    if triggered
                    else np.nan
                )
            else:
                record.update(
                    rb_status="unavailable", rb_time=pd.NaT, delta_vs_rb_minutes=np.nan
                )
            for symbol, column in OBJECTIVES.items():
                optimum = maxima[symbol]
                best = float(rows.loc[optimum, column]) if optimum is not None else np.nan
                value = float(rows.loc[index, column])
                record[f"{symbol}_value"] = value
                record[f"{symbol}_regret_percent"] = (
                    100 * (best - value) / abs(best)
                    if np.isfinite(best) and abs(best) > 1e-12
                    else np.nan
                )
            records.append(record)
    return pd.DataFrame(records)


def _family_bands(axis: plt.Axes) -> None:
    for low, high, color in (
        (5.5, 8.5, "#F2F5FA"),
        (2.5, 5.5, "#EEF7F6"),
        (-0.5, 2.5, "#FBF1F3"),
    ):
        axis.axhspan(low, high, color=color, zorder=0)


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        figure.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def plot_timing(results: pd.DataFrame, output_dir: Path) -> None:
    paired = results.dropna(subset=["delta_vs_rb_minutes"])
    figure, axis = plt.subplots(figsize=(7.2, 5.05))
    _family_bands(axis)
    positions = np.arange(len(METHOD_LABELS))[::-1]
    rng = np.random.default_rng(20260906)
    for position, method in zip(positions, METHOD_LABELS, strict=True):
        values = paired.loc[
            paired["method"].eq(method), "delta_vs_rb_minutes"
        ].to_numpy()
        violin = axis.violinplot(
            values,
            positions=[position],
            orientation="horizontal",
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.35,
        )["bodies"][0]
        violin.set_facecolor(METHOD_COLORS[method])
        violin.set_edgecolor(METHOD_COLORS[method])
        violin.set_alpha(0.32)
        jitter = np.clip(rng.normal(0, 0.065, len(values)), -0.16, 0.16)
        axis.scatter(
            values,
            position + jitter,
            s=7,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.6,
            zorder=3,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axis.plot([q1, q3], [position, position], color="#30343B", lw=3.2, zorder=4)
        axis.plot(
            median,
            position,
            marker="o",
            ms=3.8,
            color="white",
            mec="#30343B",
            mew=0.8,
            zorder=5,
        )
        axis.plot(values.mean(), position, marker="D", ms=3.1, color="#30343B", zorder=5)
    axis.axvline(0, color="#30343B", lw=0.9, ls=(0, (3, 2)), zorder=2)
    axis.set_yticks(positions, METHOD_LABELS)
    axis.set_xlabel("Selected defrost time relative to recorded RB trigger (min)")
    axis.set_title(
        "Timing policies redistribute, rather than simply reproduce, RB triggers",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=10,
    )
    axis.text(
        0,
        1.015,
        "paired triggered cycles, n = 65  |  circle: median  |  diamond: mean  |  bar: IQR",
        transform=axis.transAxes,
        color="#60656F",
        fontsize=6.4,
    )
    axis.text(
        0.01,
        -0.11,
        "Earlier than RB",
        transform=axis.transAxes,
        ha="left",
        color="#5A6F95",
        fontsize=6.5,
    )
    axis.text(
        0.99,
        -0.11,
        "Later than RB  →",
        transform=axis.transAxes,
        ha="right",
        color="#A85A70",
        fontsize=6.5,
    )
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#D7DBE2", lw=0.45, alpha=0.7)
    figure.subplots_adjust(left=0.23, right=0.98, top=0.88, bottom=0.17)
    _save(figure, output_dir, "timing_vs_rb")


def plot_regret(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        results.groupby("method")[
            [f"{symbol}_regret_percent" for symbol in OBJECTIVES]
        ]
        .median()
        .reindex(METHOD_LABELS)
        .rename(
            columns={
                f"{symbol}_regret_percent": symbol for symbol in OBJECTIVES
            }
        )
    )
    matrix = summary.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 5.05))
    _family_bands(axis)
    positions = np.arange(len(METHOD_LABELS))[::-1]
    maximum = max(1.0, float(np.nanmax(matrix)))
    norm = mpl.colors.PowerNorm(gamma=0.55, vmin=0, vmax=maximum)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "regret", ["#F6F8FB", "#A9C4DD", "#D69A9E", "#9A4252"]
    )
    for row, (position, _method) in enumerate(
        zip(positions, METHOD_LABELS, strict=True)
    ):
        for column, _symbol in enumerate(OBJECTIVES):
            value = matrix[row, column]
            size = 45 + 520 * np.sqrt(value / max(float(np.nanmax(matrix)), 1e-12))
            axis.scatter(
                column,
                position,
                s=size,
                color=cmap(norm(value)),
                edgecolor="#30343B",
                linewidth=0.55,
            )
            axis.text(
                column,
                position,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if norm(value) > 0.55 else "#20242A",
                fontweight="bold",
            )
    axis.set_xticks(range(3), ["C efficiency", "H heat rate", "O evaporator"])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=6)
    axis.set_yticks(positions, METHOD_LABELS)
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(-0.55, 2.55)
    axis.set_ylim(-0.55, 8.55)
    axis.set_title(
        "Median loss from each objective's independently supported optimum",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=52,
    )
    axis.text(
        0,
        1.095,
        "all selectable cycles, n = 96  |  circle area and colour encode regret"
        "  |  lower is better",
        transform=axis.transAxes,
        color="#60656F",
        fontsize=6.4,
    )
    for x in (0.5, 1.5):
        axis.axvline(x, color="white", lw=1.5, zorder=0)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    figure.subplots_adjust(left=0.23, right=0.98, top=0.75, bottom=0.08)
    _save(figure, output_dir, "objective_regret")
    return summary


def selection_similarity(results: pd.DataFrame) -> pd.DataFrame:
    """Return exact selected-time agreement without duplicating diagonal columns."""
    decisions = results.pivot(
        index="cycle_name", columns="method", values="selected_time"
    )
    similarity = pd.DataFrame(index=METHOD_LABELS, columns=METHOD_LABELS, dtype=float)
    for left in METHOD_LABELS:
        for right in METHOD_LABELS:
            available = decisions[left].notna() & decisions[right].notna()
            similarity.loc[left, right] = (
                100 * decisions.loc[available, left].eq(decisions.loc[available, right]).mean()
                if available.any()
                else np.nan
            )
    return similarity


def plot_similarity(results: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    similarity = selection_similarity(results)
    values = similarity.to_numpy(dtype=float)
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    shown = np.ma.array(values, mask=mask)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "similarity", ["#F7F8FA", "#C6D6EA", "#6E8FBD", "#334A7D"]
    ).with_extremes(bad="white")
    figure, axis = plt.subplots(figsize=(7.2, 5.9))
    image = axis.imshow(shown, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    for row in range(len(METHOD_LABELS)):
        for column in range(row + 1):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value >= 68 else "#30343B",
                fontweight="bold" if value >= 95 else "normal",
            )
    axis.set_xticks(
        range(len(METHOD_LABELS)),
        METHOD_LABELS,
        rotation=42,
        ha="left",
        rotation_mode="anchor",
    )
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(METHOD_LABELS)), METHOD_LABELS)
    axis.tick_params(length=0, pad=4)
    for boundary in (2.5, 5.5):
        axis.axhline(boundary, color="white", lw=2)
        axis.axvline(boundary, color="white", lw=2)
    axis.add_patch(
        mpl.patches.Rectangle(
            (5.5, 5.5), 3, 3, fill=False, edgecolor="#B64362", lw=1.6
        )
    )
    axis.text(
        7.0,
        8.72,
        "95/96 cycles share one decision",
        ha="center",
        va="top",
        color="#9A4258",
        fontsize=6.7,
        fontweight="bold",
    )
    axis.set_title(
        "Exact selected-time agreement exposes policy collapse",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=92,
    )
    axis.text(
        0,
        1.18,
        "all selectable cycles, n = 96  |  cells show identical selected times (%)",
        transform=axis.transAxes,
        color="#60656F",
        fontsize=6.4,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.035)
    colorbar.set_label("Exact agreement (%)")
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.25, right=0.91, top=0.64, bottom=0.06)
    _save(figure, output_dir, "selection_similarity")
    return similarity


def build_figures(data_dir: Path, rb_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_out_of_fold_candidates(data_dir)
    rb = pd.read_csv(rb_path, usecols=["cycle_name", "t_RB", "rb_status"])
    rb["t_RB"] = pd.to_datetime(rb["t_RB"], errors="coerce")
    results = evaluate_policies(candidates, rb)
    results.to_csv(output_dir / "policy_cycle_results.csv", index=False)
    timing = results.dropna(subset=["delta_vs_rb_minutes"])
    timing.to_csv(output_dir / "timing_vs_rb_source.csv", index=False)
    plot_timing(results, output_dir)
    regret = plot_regret(results, output_dir)
    regret.to_csv(output_dir / "objective_regret_source.csv")
    similarity = plot_similarity(results, output_dir)
    similarity.to_csv(output_dir / "selection_similarity_source.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figures(args.data, args.rb, args.output)


if __name__ == "__main__":
    main()
