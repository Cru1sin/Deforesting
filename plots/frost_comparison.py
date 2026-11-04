"""Native frost-state grades and supported COP coverage."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from image_models.frost_progression import match_trajectory_frames
from output_layout import PROJECT
from plots.cop import COP_COLOR, plot_normalized
from plots.frost_cop import plot_overlap
from plots.publication import _RGB_PANEL_FONT, save_publication

matplotlib.use("Agg")
from joblib import Parallel, delayed, parallel_config

OLD, NEW, STATE, GRAY = COP_COLOR, "#C77836", "#287D68", "#A8B1B9"
TEMPERATURES = {50: "#547DA1", 55: "#A88865", 60: "#698F80"}
KEY = ["cycle_name", "candidate_defrost_time"]
STYLE = {
    "font.family": _RGB_PANEL_FONT,
    "font.size": 8,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "legend.frameon": False,
}


def select_frost_overlap_images(
    curves,
    intervals,
    target_grade,
    loader,
    *,
    all_cycles=False,
    images_dir=None,
    available_only=False,
):
    """Nearest observed frame inside a near component covering the target grade."""
    hits = intervals.loc[intervals.left.le(target_grade) & intervals.right.ge(target_grade)]
    records = []
    names = sorted(curves.cycle_name.unique()) if all_cycles else sorted(hits.cycle_name.unique())
    for name in names:
        parts = hits.loc[hits.cycle_name.eq(name)]
        g = curves.loc[curves.cycle_name.eq(name)].copy()
        if not intervals.cycle_name.eq(name).any():
            if available_only:
                continue
            records.append(
                dict(
                    cycle_name=name,
                    setpoint_c=g.Ts.median(),
                    target_grade=target_grade,
                    observed_grade=np.nan,
                    grade_distance=np.nan,
                    exact_grade=False,
                    interval_hit=False,
                    near_grade_available=False,
                    candidate_defrost_time=pd.NaT,
                    image_time=pd.NaT,
                    file_name=np.nan,
                    image_path="",
                    available=False,
                    cop_fraction=np.nan,
                )
            )
            continue
        member = pd.Series(parts.empty, index=g.index)
        for part in parts.itertuples():
            member |= g.candidate_defrost_time.between(part.start, part.end)
        peak = g.raw_effective_cop.max()
        eligible = member & g.raw_effective_cop.notna() & g.frost_grade.notna()
        if not parts.empty:
            eligible &= g.raw_effective_cop.ge(0.99 * peak)
        g = g.loc[eligible].copy()
        if images_dir is None:
            physical = loader.load_cycle_images(name)
            paths = (
                physical.loc[physical.camera_role.eq("front")].set_index("file_name").path.to_dict()
            )
            g["image_path"] = g.file_name.map(paths).fillna("")
        else:
            g["image_path"] = g.file_name.map(
                lambda file: str(images_dir / name / "front" / file) if pd.notna(file) else ""
            )
        g["available"] = g.image_path.map(lambda p: bool(p) and Path(p).is_file())
        g["grade_distance"] = (g.frost_grade - target_grade).abs()
        # Availability breaks only equal-grade-distance ties, never moves the target.
        row = g.sort_values(
            ["grade_distance", "available", "candidate_defrost_time"], ascending=[True, False, True]
        ).iloc[0]
        if available_only and not row.available:
            continue
        records.append(
            dict(
                cycle_name=name,
                setpoint_c=row.Ts,
                target_grade=target_grade,
                observed_grade=row.frost_grade,
                grade_distance=row.grade_distance,
                exact_grade=bool(row.grade_distance == 0),
                interval_hit=not parts.empty,
                candidate_defrost_time=row.candidate_defrost_time,
                near_grade_available=bool(intervals.cycle_name.eq(name).any()),
                image_time=row.image_time,
                file_name=row.file_name,
                image_path=row.image_path,
                available=bool(row.available),
                cop_fraction=row.raw_effective_cop / peak,
            )
        )
    return pd.DataFrame(records)


def select_frost_summary_images(
    curves, values, coverage, statistic, loader, output, images_dir=None, available_only=False
):
    """Three distinct observed cycles at each 99% boundary and the aggregate maximum."""
    from dataset_tools.cycle_metadata import image_root

    column = "mean_observed_normalized_cop" if statistic == "mean" else "median_normalized_cop"
    near = coverage.loc[coverage[column].ge(0.99), "grade"]
    peak = int(coverage.loc[coverage[column].idxmax(), "grade"])
    targets = [("left", int(near.min())), ("peak", peak), ("right", int(near.max()))]
    curves = curves.assign(
        cop_fraction=curves.raw_effective_cop
        / curves.groupby("cycle_name").raw_effective_cop.transform("max")
    )
    cached = {path.name: path for path in output.parent.rglob("RGB_*.jpg")}
    shared = PROJECT / "output/90_复现材料/05_霜态轨迹/结霜代表照片"
    records = []
    for position, grade in targets:
        pool = curves.loc[
            curves.frost_grade.eq(grade)
            & curves.raw_effective_cop.notna()
            & curves.file_name.notna()
        ].copy()
        if available_only:
            root = images_dir if images_dir is not None else image_root(loader.dataset_root)
            pool = pool.loc[
                [
                    bool((root / row.cycle_name / "front" / row.file_name).is_file())
                    for row in pool.itertuples()
                ]
            ]
        cycle_values = values.loc[values.grade.eq(grade)].set_index("cycle_name").normalized_cop
        pool["cycle_grade_cop"] = pool.cycle_name.map(cycle_values)
        pool["frame_distance"] = (pool.cop_fraction - pool.cycle_grade_cop).abs()
        pool = pool.sort_values(["frame_distance", "candidate_defrost_time"]).drop_duplicates(
            "cycle_name"
        )
        pool["representative_distance"] = (
            pool.cycle_grade_cop - coverage.loc[coverage.grade.eq(grade), column].iloc[0]
        ).abs()
        pool = pool.sort_values(["representative_distance", "cycle_name"])
        if pool.Ts.nunique() == 3:
            chosen = pool.groupby("Ts", sort=True).head(1).sort_values("Ts")
        else:
            chosen = pool.head(3)
        for rank, row in enumerate(chosen.itertuples()):
            paths = (
                [images_dir / row.cycle_name / "front" / row.file_name]
                if images_dir is not None
                else [
                    image_root(loader.dataset_root) / row.cycle_name / "front" / row.file_name,
                    shared / row.cycle_name / "front" / row.file_name,
                    cached.get(f"RGB_{row.cycle_name}_{row.file_name}"),
                ]
            )
            path = next(
                (p for p in paths if p is not None and p.is_file()),
                paths[0]
                if images_dir is not None
                else shared / row.cycle_name / "front" / row.file_name,
            )
            records.append(
                dict(
                    position=position,
                    rank=rank,
                    target_grade=grade,
                    observed_grade=row.frost_grade,
                    cycle_name=row.cycle_name,
                    setpoint_c=row.Ts,
                    image_time=row.image_time,
                    candidate_defrost_time=row.candidate_defrost_time,
                    file_name=row.file_name,
                    cop_fraction=row.cop_fraction,
                    cycle_grade_cop=row.cycle_grade_cop,
                    image_path=str(path),
                    available=path.is_file(),
                )
            )
    return pd.DataFrame(records)


def render_cost_grade_comparison(source, progression, screening, output):
    """Four independent COP near sets on one native grade axis and main cohort."""
    from image_models.overlap import interval_cycle_counts, reference_basin_intervals

    settings = json.loads((progression.parent / "run_settings.json").read_text())
    frames = pd.read_csv(progression)
    if (
        settings.get("coordinate_correction") != "harmony"
        or not frames.reference_method.eq("slingshot").all()
    ):
        raise ValueError("Cost-grade comparison requires native global Harmony + Slingshot")
    curves = pd.read_parquet(source / "source_data.parquet")
    cost_settings = json.loads((source / "settings.json").read_text())
    sample_label = (
        f" | {cost_settings['dataset_interval_seconds']} s Dataset"
        if "dataset_interval_seconds" in cost_settings
        else ""
    )
    names = set(screening.loc[screening.selected, "cycle_name"])
    if set(curves.cycle_name) != names:
        raise ValueError("Cost curves must cover exactly the re-screened main cohort")
    curves[KEY[1]] = pd.to_datetime(curves[KEY[1]])
    matched = match_trajectory_frames(
        curves[KEY].drop_duplicates(), frames, metric="reference_index"
    )
    curves = curves.merge(
        matched[KEY + ["reference_index", "image_time", "file_name"]],
        on=KEY,
        how="left",
        validate="many_to_one",
    ).rename(columns={"reference_index": "frost_grade"})
    curves["Ts"] = curves.cycle_name.map(screening.set_index("cycle_name").setpoint_c)
    intervals = []
    for recipe, group in curves.groupby("cost_recipe", sort=False):
        spans = reference_basin_intervals(
            group,
            objectives=[(recipe, "cycle_cop")],
            coordinates=["frost_grade"],
            near_column="near_optimal",
        )
        intervals.append(spans.assign(kind="feasible"))
    intervals = pd.concat(intervals, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    curves.to_parquet(output / "figure_actions.parquet", index=False)
    intervals.to_csv(output / "near_optimal_grade_intervals.csv", index=False)
    recipes = [
        ("zero", "Zero prep+D"),
        ("mean", "Mean prep+D"),
        ("new", "New Ridge"),
        ("old", "Old Ridge"),
    ]
    grade_max = int(frames.reference_index.max())
    grades = np.arange(grade_max + 1)
    temperature = screening.set_index("cycle_name").loc[sorted(names), "setpoint_c"]
    cohorts = [
        ("all", sorted(names)),
        *[(f"{ts:g}C", sorted(group.index)) for ts, group in temperature.groupby(temperature)],
    ]
    summaries, coverage_tables = [], []
    for cohort, cycles in cohorts:
        with plt.rc_context(
            {
                **STYLE,
                "font.size": 7,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "axes.spines.right": False,
            }
        ):
            figure = plt.figure(figsize=(14.4, max(6.0, len(cycles) * 0.11 + 3.0)), dpi=300)
            subfigures = figure.subfigures(1, 4, wspace=0.015)
            for (recipe, label), subfigure in zip(recipes, subfigures, strict=True):
                spans = intervals.loc[
                    intervals.method.eq(recipe) & intervals.cycle_name.isin(cycles)
                ]
                counts = interval_cycle_counts(spans, grades, closed=True)
                coverage = pd.DataFrame(
                    dict(
                        grade=grades,
                        left=grades - 0.5,
                        right=grades + 0.5,
                        n_near_optimal=counts,
                        total_cycles=len(cycles),
                        coverage=counts / len(cycles),
                    )
                )
                coverage_tables.append(coverage.assign(cohort=cohort, cost_recipe=recipe))
                evidence = screening.loc[screening.cycle_name.isin(cycles)].copy()
                evidence["reference_available"] = evidence.cycle_name.isin(
                    curves.loc[curves.frost_grade.notna(), "cycle_name"]
                )
                plot_overlap(
                    evidence,
                    spans,
                    coverage,
                    output,
                    cycle_names=cycles,
                    cohort_name=label,
                    grade_max=grade_max,
                    source_label="Native Harmony + Slingshot",
                    figure=subfigure,
                )
                summaries.append(
                    dict(
                        cohort=cohort,
                        cost_recipe=recipe,
                        cycles=len(cycles),
                        cycles_with_near_grade=spans.cycle_name.nunique(),
                        maximum_hits=int(counts.max()),
                        maximum_overlap=counts.max() / len(cycles),
                        maximum_grades=",".join(map(str, grades[counts == counts.max()])),
                    )
                )
            figure.suptitle(
                f"{'All conditions' if cohort == 'all' else cohort} | {len(cycles)} cycles{sample_label}",
                y=0.995,
                fontsize=11,
            )
            save_publication(
                figure, output, f"cost_grade_intervals_{cohort}", "", layout=False, dpi=300
            )
        print(f"{cohort}: four cost recipes, {len(cycles)} cycles", flush=True)
    pd.concat(coverage_tables, ignore_index=True).to_csv(output / "grade_coverage.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "maximum_overlap.csv", index=False)


def render_frost_overlap(
    source,
    progression,
    output,
    admission_settings=None,
    image_gallery=False,
    fetch_images=False,
    n_jobs=6,
    all_screened_images=False,
    image_scope="setpoint",
    summary_metric="coverage",
    frequency_styles="all",
    mean_only=False,
    images_dir=None,
    cycle_output=None,
    available_only=False,
    grade_scale="reference",
    render_cohorts=None,
):
    """Current COP and native grades through the shared two-panel overlap template."""
    from dataset_tools import DatasetLoader
    from defrost_decision.performance_objectives import calculate_cycle_cop
    from image_models.cycle_selection import screen_cv_cycles
    from image_models.overlap import (
        interval_cycle_counts,
        normalized_cop_by_grade,
        reference_basin_intervals,
    )

    frames = pd.read_csv(progression)
    if images_dir is not None and fetch_images:
        raise ValueError("Explicit local RGB input must not enable cloud downloads")
    settings = json.loads((progression.parent / "run_settings.json").read_text())
    if (
        settings.get("coordinate_correction") != "harmony"
        or not frames.reference_method.isin(
            [
                "slingshot",
                "mst_polyline",
                "mst_guided_visual_segments",
            ]
        ).all()
    ):
        raise ValueError("Slingshot overlap requires native global Harmony + Slingshot")
    source_label = "Native Harmony + Slingshot"
    if settings.get("feature_pooling") == "patch_mean":
        source_label = "Patch mean · native Harmony + Slingshot"
        if settings.get("spline_df") is not None:
            source_label += (
                f" · {settings.get('spline_knot_placement', 'quantile')} df{settings['spline_df']}"
            )
    if settings.get("method") == "mst_guided_visual_segments":
        groups = pd.read_csv(progression.parent / "fine_stages.csv").kind.eq("visual_stage").sum()
        source_label = (
            f"Native Harmony + Slingshot MST · {groups} groups + observed tail · polyline"
        )
    if grade_scale == "vertices":
        from image_models.overlap import vertex_grades

        if image_gallery or summary_metric not in ["coverage", "observed-cop"]:
            raise ValueError(
                "Vertex grades support native Slingshot coverage and observed COP without images"
            )
        geometry_source = Path(settings.get("geometry_source", progression.parent))
        vertices = pd.read_csv(geometry_source / "trajectory_vertices.csv").phase.to_numpy()
        frames["reference_index"] = vertex_grades(frames.visual_frost_score, vertices)
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            dict(
                left_grade=np.arange(1, len(vertices)),
                right_grade=np.arange(2, len(vertices) + 1),
                arc_boundary=(vertices[:-1] + vertices[1:]) / 2,
            )
        ).to_csv(output / "grade_boundaries.csv", index=False)
        frames.to_csv(output / "discrete_progression.csv", index=False)
        source_label = (
            f"Native Harmony + Slingshot MST · {len(vertices)} visual grades · arc midpoints"
        )
        (output / "run_settings.json").write_text(
            json.dumps(
                dict(
                    settings,
                    grade_scale=grade_scale,
                    progression_source=str(progression),
                    near_grade_rule="observed COP >= 0.99 own maximum; no grade interpolation",
                    mean_rule="within-cycle observed grade mean, then equal-cycle mean; available cycles only",
                ),
                indent=2,
            )
            + "\n"
        )
    decisions = pd.read_csv(source / "candidate_decisions.csv", low_memory=False)
    source_settings = json.loads((source / "run_settings.json").read_text())
    quality = None
    options = {}
    if admission_settings is not None:
        admission = json.loads(admission_settings.read_text())
        assert (
            Path(admission["decisions"]).resolve() == (source / "candidate_decisions.csv").resolve()
        )
        quality = pd.read_csv(admission["quality"])
        options = {
            key: admission[key]
            for key in [
                "exclude_rb",
                "minimum_terminal_decline",
                "rb_minimum_terminal_decline",
                "input_rule",
                "cohort_rule",
                "excluded_cycles",
                "maximum_per_setpoint",
                "post_near_minutes_by_setpoint",
            ]
            if key in admission
        }
        options["maximum_interpolated_fraction"] = admission["limits"][
            "maximum_interpolated_fraction"
        ]
    loader = DatasetLoader(Path(source_settings["dataset"]))
    screening = screen_cv_cycles(decisions, loader.list_cycles(), quality=quality, **options)
    output.mkdir(parents=True, exist_ok=True)
    screening.to_csv(output / "cycle_screening.csv", index=False)
    names = set(screening.loc[screening.selected, "cycle_name"])
    curves = calculate_cycle_cop(decisions.loc[decisions.cycle_name.isin(names)], effective=True)
    curves[KEY[1]] = pd.to_datetime(curves[KEY[1]], format="mixed")
    curves["raw_effective_cop"] = curves.cycle_cop.where(curves.cycle_cop_eligible)
    curves["Ts"] = curves.water_temperature_setpoint
    matched = match_trajectory_frames(curves[KEY], frames, metric="reference_index")
    curves = curves.merge(
        matched[KEY + ["reference_index", "image_time", "file_name"]],
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    curves = curves.rename(columns={"reference_index": "frost_grade"})
    curves["frost_grade_method"] = "slingshot"
    curves.to_parquet(output / "figure_actions.parquet", index=False)
    if summary_metric == "frequency-cop":
        from plots.frequency import render_cop_slope

        cohort = screening.loc[screening.selected].copy()
        records = cohort.set_index("cycle_name")
        curves = curves.sort_values(KEY).copy()
        origin = pd.to_datetime(curves.measured_cycle_start, format="mixed")
        end = pd.to_datetime(curves.cycle_name.map(records.end_time), format="mixed")
        curves["elapsed_minutes"] = (curves.candidate_defrost_time - origin).dt.total_seconds() / 60
        curves["cycle_progress"] = curves.elapsed_minutes / ((end - origin).dt.total_seconds() / 60)
        curves["relative_cop"] = curves.raw_effective_cop / curves.groupby(
            "cycle_name"
        ).raw_effective_cop.transform("max")
        curves["normalized_cop_percent"] = 100 * curves.relative_cop
        curves.to_parquet(output / "figure_actions.parquet", index=False)
        styles = [(False, "both"), (True, "both"), (True, "increase"), (True, "decrease")]
        if frequency_styles == "changes":
            styles = styles[1:]
        if frequency_styles == "full-rise-fall":
            styles = [styles[0], *styles[2:]]
        for y_column in ["cycle_cop", "relative_cop"]:
            render_cop_slope(
                output,
                curves=curves,
                cohort=cohort,
                rise_markers="none",
                include_all=True,
                all_styles=True,
                cop_lower=1.3,
                x_column="frost_grade",
                y_column=y_column,
                styles=styles,
                source_label=source_label,
                n_jobs=n_jobs,
            )
        print(f"Saved {len(styles) * 8} frost-state frequency-coloured COP PNGs", flush=True)
        return curves
    if summary_metric == "per-cycle-cop":
        from plots.frost_trajectory import plot_cycle_frost_time

        cycle_output = output if cycle_output is None else cycle_output
        for name in sorted(names):
            (cycle_output / name).mkdir(parents=True, exist_ok=True)
        with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
            Parallel(verbose=5)(
                delayed(render_frost_state_normalized_cop)(
                    curve,
                    cycle_output / name / "frost_state_normalized_cop.png",
                    grade_max=int(frames.reference_index.max()),
                    source_label=source_label,
                )
                for name, curve in curves.groupby("cycle_name", sort=True)
            )
            Parallel(verbose=5)(
                delayed(plot_cycle_frost_time)(
                    group, cycle_output / name, source_label=source_label
                )
                for name, group in frames.loc[frames.cycle_name.isin(names)].groupby(
                    "cycle_name", sort=True
                )
            )
        print(
            f"Saved {curves.cycle_name.nunique()} screened-cycle frost-state COP PNGs", flush=True
        )
        return curves
    if grade_scale == "vertices":
        from image_models.overlap import observed_grade_intervals

        spans = observed_grade_intervals(curves)
    else:
        spans = reference_basin_intervals(
            curves, objectives=[("effective_cop", "raw_effective_cop")], coordinates=["frost_grade"]
        )
    spans.to_csv(output / "near_optimal_grade_intervals.csv", index=False)
    grade_max = len(vertices) if grade_scale == "vertices" else int(frames.reference_index.max())
    grades = np.arange(1 if grade_scale == "vertices" else 0, grade_max + 1)
    observed_metric = summary_metric in ["regret", "observed-cop", "mapped-cop"]
    if summary_metric != "coverage":
        mapping = (
            "interpolated-grade"
            if summary_metric in ["interpolated-cop", "mapped-cop"]
            else ("observed-grade" if observed_metric else "first-crossing")
        )
        cop_by_grade = normalized_cop_by_grade(curves, grades, mapping=mapping)
        cop_by_grade["cop_regret"] = 1 - cop_by_grade.normalized_cop
        cop_by_grade.to_csv(output / "normalized_cop_by_cycle_grade.csv", index=False)
    summaries = []
    image_rows = []
    figure_summaries = []
    summary_images = []
    # Group by cycle-level setpoint, so a cycle cannot enter multiple denominators.
    temperatures = curves.groupby("cycle_name").Ts.median()
    cohorts = [(ts, sorted(group.index)) for ts, group in temperatures.groupby(temperatures)]
    cohorts.append((None, sorted(names)))
    for ts, cycles in cohorts:
        suffix = "all" if ts is None else f"{ts:g}C"
        label = "All screened cycles" if ts is None else f"{ts:g} °C"
        intervals = spans.loc[spans.cycle_name.isin(cycles)].assign(kind="feasible")
        if summary_metric != "coverage":
            values = cop_by_grade.loc[cop_by_grade.cycle_name.isin(cycles)]
            coverage = (
                values.groupby("grade")
                .normalized_cop.agg(n_evaluated="count", mean_normalized_cop="mean")
                .reindex(grades)
            )
            if mean_only:
                coverage["median_normalized_cop"] = (
                    values.groupby("grade").normalized_cop.median().reindex(grades)
                )
            coverage["common_support"] = coverage.n_evaluated.eq(len(cycles))
            if observed_metric:
                metric_name = (
                    "mean_cop_regret"
                    if summary_metric == "regret"
                    else "mean_observed_normalized_cop"
                )
                coverage[metric_name] = (
                    1 - coverage.mean_normalized_cop
                    if summary_metric == "regret"
                    else coverage.mean_normalized_cop
                )
                coverage["n_evaluated"] = coverage.n_evaluated.fillna(0).astype(int)
                coverage = coverage.drop(columns="mean_normalized_cop")
                if summary_metric == "mapped-cop":
                    coverage["n_interpolated_cycles"] = (
                        values.loc[values.n_interpolated_segments.gt(0)]
                        .groupby("grade")
                        .cycle_name.nunique()
                        .reindex(grades, fill_value=0)
                    )
                    coverage["grade_mapping"] = "local-linear-interpolation"
            else:
                metric_name = "mean_normalized_cop"
                coverage["mean_normalized_cop"] = coverage.mean_normalized_cop.where(
                    coverage.common_support
                )
                if summary_metric == "interpolated-cop":
                    coverage["n_interpolated_cycles"] = (
                        values.loc[values.n_interpolated_segments.gt(0)]
                        .groupby("grade")
                        .cycle_name.nunique()
                        .reindex(grades, fill_value=0)
                    )
                    coverage["grade_mapping"] = "local-linear-interpolation"
                    metric_name = "mean_interpolated_normalized_cop"
            coverage = coverage.reset_index().assign(
                left=grades - 0.5, right=grades + 0.5, total_cycles=len(cycles)
            )
            coverage.to_csv(output / f"{metric_name}_{suffix}.csv", index=False)
        else:
            counts = interval_cycle_counts(intervals, grades, closed=True)
            coverage = pd.DataFrame(
                dict(
                    grade=grades,
                    left=grades - 0.5,
                    right=grades + 0.5,
                    n_near_optimal=counts,
                    total_cycles=len(cycles),
                    coverage=counts / len(cycles),
                )
            )
            coverage.to_csv(output / f"grade_coverage_{suffix}.csv", index=False)
        coverage["grade_scale"] = grade_scale
        evidence = screening.loc[screening.cycle_name.isin(cycles)].copy()
        available = curves.loc[curves.frost_grade.notna(), "cycle_name"]
        evidence["reference_available"] = evidence.cycle_name.isin(available)
        render = render_cohorts is None or suffix in render_cohorts
        with plt.rc_context(
            {**STYLE, "font.size": 7, "axes.spines.right": False, "axes.linewidth": 0.6}
        ):
            if not render:
                pass
            elif mean_only and summary_metric in ["observed-cop", "mapped-cop"]:
                from plots.frost_cop import plot_mean_frost_cop

                for statistic in ["mean"] if grade_scale == "vertices" else ["mean", "median"]:
                    pictures = None
                    if image_gallery:
                        pictures = select_frost_summary_images(
                            curves.loc[curves.cycle_name.isin(cycles)],
                            values,
                            coverage,
                            statistic,
                            loader,
                            output,
                            images_dir=images_dir,
                            available_only=available_only,
                        )
                        if fetch_images:
                            from dataset_tools.cloud_images import materialize_image_members

                            missing = pictures.loc[~pictures.available]
                            requests = missing.groupby("cycle_name").file_name.agg(list).to_dict()
                            if requests:
                                with materialize_image_members(
                                    loader.dataset_root,
                                    requests,
                                    images_dir=(
                                        PROJECT / "output/90_复现材料/05_霜态轨迹/结霜代表照片"
                                    ),
                                    n_jobs=n_jobs,
                                    minimum_free_gib=40,
                                ) as completed:
                                    for _ in completed:
                                        pass
                            pictures["available"] = pictures.image_path.map(
                                lambda p: Path(p).is_file()
                            )
                        summary_images.append(pictures.assign(cohort=suffix, statistic=statistic))
                    record = plot_mean_frost_cop(
                        coverage,
                        output,
                        cohort_label=f"{label} · {len(cycles)} cycles",
                        name=f"{statistic}_frost_cop_{suffix}",
                        source_label=source_label,
                        statistic=statistic,
                        images=pictures,
                    )
                    figure_summaries.append(dict(cohort=suffix, **record))
            elif summary_metric == "mapped-cop":
                from plots.frost_cop import plot_frost_cop_mapping

                plot_frost_cop_mapping(
                    curves.loc[curves.cycle_name.isin(cycles)],
                    values,
                    coverage,
                    output,
                    cohort_label=label,
                    name=f"frost_cop_mapping_{suffix}",
                    source_label=source_label,
                    cycle_gallery=ts is None,
                )
            else:
                plot_overlap(
                    evidence,
                    intervals,
                    coverage,
                    output,
                    cycle_names=cycles,
                    cohort_name=f"{label} · n = {len(cycles)}",
                    grade_max=grade_max,
                    source_label=source_label,
                    name="frost_grade_"
                    + (metric_name + "_" if summary_metric != "coverage" else "overlap_")
                    + suffix,
                )
        if observed_metric:
            summary = dict(
                cohort=suffix,
                total_cycles=len(cycles),
                grades_with_data=int(coverage[metric_name].notna().sum()),
                common_grades=",".join(map(str, coverage.loc[coverage.common_support, "grade"])),
            )
            if summary_metric == "mapped-cop" or (mean_only and summary_metric == "observed-cop"):
                best = coverage.loc[coverage[metric_name].eq(coverage[metric_name].max())]
                summary.update(
                    maximum_mean_normalized_cop=float(best[metric_name].max()),
                    maximum_grades=",".join(map(str, best.grade)),
                    n_at_maximum=",".join(map(str, best.n_evaluated)),
                )
            summaries.append(summary)
            print(summary, flush=True)
            continue
        if summary_metric in ["normalized-cop", "interpolated-cop"]:
            peak = coverage.mean_normalized_cop.max()
            summary = dict(
                cohort=suffix,
                total_cycles=len(cycles),
                common_grades=",".join(map(str, coverage.loc[coverage.common_support, "grade"])),
                maximum_mean_normalized_cop=peak,
                maximum_grades=",".join(
                    map(str, coverage.loc[coverage.mean_normalized_cop.eq(peak), "grade"])
                ),
            )
            summaries.append(summary)
            print(summary, flush=True)
            continue
        summary = dict(
            setpoint_c=ts,
            total_cycles=len(cycles),
            cycles_with_near_grade=intervals.cycle_name.nunique(),
            maximum_hits=int(counts.max()),
            maximum_overlap=float(counts.max() / len(cycles)),
            maximum_grades=",".join(map(str, grades[counts == counts.max()])),
        )
        if ts is None:
            pooled_summary = summary
            pooled_grade = int(
                grades[counts == counts.max()][(np.sum(counts == counts.max()) - 1) // 2]
            )
            pd.DataFrame([summary]).drop(columns="setpoint_c").to_csv(
                output / "maximum_overlap_all.csv", index=False
            )
            print(f"Slingshot {label}: {summary}", flush=True)
        else:
            summaries.append(summary)
        if image_gallery and ((ts is None) == (image_scope == "pooled")):
            peaks = grades[counts == counts.max()]
            image_rows.append(
                select_frost_overlap_images(
                    curves.loc[curves.cycle_name.isin(cycles)],
                    intervals,
                    int(peaks[(len(peaks) - 1) // 2]),
                    loader,
                    all_cycles=all_screened_images,
                    images_dir=images_dir,
                    available_only=available_only,
                )
            )
    if summary_metric != "coverage":
        if figure_summaries:
            pd.DataFrame(figure_summaries).to_csv(
                output / "frost_cop_statistics_summary.csv", index=False
            )
        if summary_images:
            pd.concat(summary_images, ignore_index=True).to_csv(
                output / "frost_cop_summary_images.csv", index=False
            )
        table_name = (
            (
                "cop_regret_support"
                if summary_metric == "regret"
                else "observed_normalized_cop_support"
            )
            if observed_metric
            else "maximum_mean_normalized_cop"
        )
        pd.DataFrame(summaries).to_csv(output / f"{table_name}.csv", index=False)
        return dict(curves=curves, summary=summaries[-1])
    pd.DataFrame(summaries).to_csv(output / "maximum_overlap.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    if image_gallery:
        rows = pd.concat(image_rows, ignore_index=True)
        for index, row in rows.loc[~rows.available & (images_dir is None)].iterrows():
            path = output / f"RGB_{row.cycle_name}_{row.file_name}"
            if not path.is_file() and pd.notna(row.file_name):
                path = next(output.parent.rglob(path.name), path)
            if path.is_file():
                rows.loc[index, "image_path"] = str(path)
                rows.loc[index, "available"] = True
        if fetch_images and (~rows.available).any():
            import shutil

            from dataset_tools.cloud_images import materialize_image_members

            missing = rows.loc[~rows.available & rows.file_name.notna()]
            requests = missing.groupby("cycle_name").file_name.agg(list).to_dict()
            download_root = Path("temp/test/frost_overlap_images")
            with materialize_image_members(
                loader.dataset_root, requests, images_dir=download_root, n_jobs=n_jobs
            ) as completed:
                for _ in completed:
                    pass
            for index, row in missing.iterrows():
                path = output / f"RGB_{row.cycle_name}_{row.file_name}"
                shutil.move(download_root / row.cycle_name / "front" / row.file_name, path)
                rows.loc[index, "image_path"] = str(path)
                rows.loc[index, "available"] = path.is_file()
        image_name = (
            "pooled_maximum_coverage_frost_images"
            if image_scope == "pooled"
            else (
                "screened_cycles_frost_images"
                if all_screened_images
                else "maximum_overlap_frost_images"
            )
        )
        table_name = (
            "pooled_maximum_coverage_images"
            if image_scope == "pooled"
            else ("screened_cycle_images" if all_screened_images else "maximum_overlap_images")
        )
        rows.to_csv(output / f"{table_name}.csv", index=False)
        from plots.frost_cop import plot_frost_overlap_images

        with plt.rc_context({**STYLE, "font.size": 7}):
            if all_screened_images:
                photo_summary = (
                    pd.Series(pooled_summary)
                    if image_scope == "pooled"
                    else pd.DataFrame(summaries)
                )
                plot_frost_overlap_images(
                    rows,
                    photo_summary,
                    output,
                    combined=True,
                    name=f"{image_name}",
                    common_grade=image_scope == "pooled",
                    source_label=source_label,
                )
            else:
                for ts, group in rows.groupby("setpoint_c"):
                    plot_frost_overlap_images(
                        group,
                        pd.DataFrame(summaries).set_index("setpoint_c").loc[ts],
                        output,
                        name=f"maximum_overlap_frost_images_{ts:g}C",
                        source_label=source_label,
                    )
        print(
            rows.groupby("setpoint_c")
            .agg(
                images=("cycle_name", "size"),
                exact_grade=("exact_grade", "sum"),
                available=("available", "sum"),
            )
            .to_string(),
            flush=True,
        )
    return dict(
        curves=curves,
        summary=pooled_summary,
        target_grade=pooled_grade,
        images=rows if image_gallery and image_scope == "pooled" else None,
        source_label=source_label,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--frost-overlap",
        action="store_true",
        help="Current COP near-grade overlap by setpoint; source is full candidate directory",
    )
    parser.add_argument(
        "--frost-images",
        action="store_true",
        help="All-cycle image plates at maximum overlap grades",
    )
    parser.add_argument(
        "--all-screened-images",
        action="store_true",
        help="One image plate containing every screened cycle, including interval misses",
    )
    parser.add_argument(
        "--frost-image-scope",
        choices=["setpoint", "pooled"],
        default="setpoint",
        help="Use condition-specific or pooled maximum coverage grade for the image plate",
    )
    parser.add_argument(
        "--frost-grade-scale",
        choices=["reference", "vertices"],
        default="reference",
        help="Original reference grades or discrete grades at trajectory arc midpoints",
    )
    parser.add_argument(
        "--frost-combine-conditions",
        action="store_true",
        help="Compose condition comparisons from saved vertex-grade coverage and mean sources",
    )
    parser.add_argument(
        "--frost-pooled-only",
        action="store_true",
        help="Save all cohort tables but render only the pooled figure",
    )
    parser.add_argument(
        "--frost-summary",
        choices=[
            "coverage",
            "normalized-cop",
            "regret",
            "observed-cop",
            "interpolated-cop",
            "mapped-cop",
            "per-cycle-cop",
            "frequency-cop",
        ],
        default="coverage",
        help="Coverage, grade statistics, time-to-state mapping, or original-template COP figures",
    )
    parser.add_argument(
        "--frost-mean-only",
        action="store_true",
        help="Render grade mean COP and participation only",
    )
    parser.add_argument(
        "--frost-frequency-styles",
        choices=["all", "changes", "full-rise-fall"],
        default="all",
        help="All four styles, change/rise/fall, or full/rise/fall",
    )
    parser.add_argument(
        "--admission-settings", type=Path, help="Training input admission recipe for frost overlap"
    )
    parser.add_argument(
        "--fetch-images", action="store_true", help="Fetch only missing optimum RGB frames"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Read RGB only from this existing image root; no downloads or cache fallback",
    )
    parser.add_argument(
        "--available-images-only",
        action="store_true",
        help="Choose photos only from available local RGB; keep the statistical cohort unchanged",
    )
    parser.add_argument(
        "--cycle-output", type=Path, help="Parent folder for one directory per cycle"
    )
    parser.add_argument(
        "--progression", type=Path, help="Saved Slingshot progression.csv with reference_index"
    )
    args = parser.parse_args()
    from output_layout import prepare_output

    prepare_output(args, training=False)
    if args.frost_combine_conditions and not args.frost_overlap:
        from plots.frost_cop import render_frost_condition_comparison

        with plt.rc_context(STYLE):
            render_frost_condition_comparison(
                args.source,
                images_dir=args.images_dir,
                metric="normalized-cop" if args.frost_summary == "observed-cop" else "coverage",
            )
        return
    if args.frost_overlap:
        if args.progression is None:
            parser.error("--progression is required for frost overlap")
        if args.frost_mean_only and args.frost_summary not in ["observed-cop", "mapped-cop"]:
            parser.error("--frost-mean-only requires --frost-summary observed-cop or mapped-cop")
        if args.frost_summary != "coverage" and (
            args.all_screened_images or (args.frost_images and not args.frost_mean_only)
        ):
            parser.error(
                "Normalized COP currently renders the two-panel figures; omit image-gallery options"
            )
        options = dict(
            admission_settings=args.admission_settings,
            image_gallery=args.frost_images or args.all_screened_images,
            fetch_images=args.fetch_images,
            n_jobs=args.n_jobs,
            all_screened_images=args.all_screened_images,
            image_scope=args.frost_image_scope,
            summary_metric=args.frost_summary,
            frequency_styles=args.frost_frequency_styles,
            mean_only=args.frost_mean_only,
            images_dir=args.images_dir,
            cycle_output=args.cycle_output,
            available_only=args.available_images_only,
            grade_scale=args.frost_grade_scale,
            render_cohorts=["all"] if args.frost_pooled_only else None,
        )
        render_frost_overlap(
            args.source,
            args.progression,
            args.output,
            **options,
        )
        if args.frost_combine_conditions:
            from plots.frost_cop import render_frost_condition_comparison

            render_frost_condition_comparison(
                args.output,
                images_dir=args.images_dir,
                metric="normalized-cop" if args.frost_summary == "observed-cop" else "coverage",
            )
        return
    parser.error("Choose --frost-overlap or --frost-combine-conditions")


def render_frost_state_normalized_cop(
    curve, output_path, *, grade_max=99, source_label="Native Harmony + Slingshot"
):
    """Reuse the effective-COP panel style with chronological time-to-state pairs."""
    curve = curve.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
    curve[
        [
            "cycle_name",
            "candidate_defrost_time",
            "cycle_cop",
            "cycle_cop_eligible",
            "frost_grade",
            "image_time",
            "file_name",
        ]
    ].to_csv(Path(output_path).with_suffix(".csv"), index=False)
    origin = pd.Timestamp(curve.candidate_defrost_time.min())
    with plt.rc_context(
        {
            "font.family": _RGB_PANEL_FONT,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(6.4, 3.8))
        for percent, alpha in ((5, 0.07), (2, 0.10), (1, 0.14)):
            axis.axhspan(
                1 - percent / 100,
                1,
                color=COP_COLOR,
                alpha=alpha,
                lw=0,
                label=f"Within {percent}% of best COP",
            )
        plot_normalized(
            axis, curve, origin, metrics=("cycle_cop",), x_column="frost_grade", percent=False
        )
        axis.set(
            xlabel="Frost-state grade",
            ylabel=r"Normalized cycle COP ($COP/COP_{\max}$)",
            xlim=(0, grade_max),
            ylim=(0.9, 1),
        )
        axis.grid(axis="x", alpha=0.12, lw=0.5)
        figure.text(
            0.13,
            0.93,
            f"{curve.cycle_name.iloc[0]}  |  {curve.water_temperature_setpoint.iloc[0]:g} °C",
            fontweight="bold",
            fontsize=10,
        )
        figure.text(0.13, 0.87, source_label, color="#64717D", fontsize=7)
        axis.legend(ncol=2, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.01))
        figure.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.70)
        figure.savefig(output_path, dpi=600, facecolor="white")
        plt.close(figure)


if __name__ == "__main__":
    main()
