"""Publish screened trajectories and frozen DINO inputs inside Dataset."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from defrost_decision.performance_objectives import calculate_cycle_cop
from image_models.control_context import COMPENSATION, load_control_context, stopping_inputs
from image_models.cycle_selection import (
    input_quality_limits,
    legal_quality_actions,
    rb_failure_cases,
    screen_cv_cycles,
)
from image_models.inputs import RGB
from image_models.labels import label_actions
from output_layout import PROJECT


def publish_screening_inputs(dataset, settings, camera_role="front"):
    """Publish only measured screening evidence; retain labels, cohorts and source settings."""
    training = Path(dataset) / "training_data"
    suffix = "" if camera_role == "front" else f"__{camera_role}"
    quality = pd.read_csv(settings["quality"])
    inputs = pd.read_parquet(
        Path(settings["quality"]).parent / "decision_inputs.parquet",
        columns=["cycle_name", "candidate_defrost_time", "input_valid"],
    )
    candidates = pd.read_parquet(training / "candidate_decisions.parquet", columns=["cycle_name"])
    if set(quality.cycle_name) != set(candidates.cycle_name):
        raise ValueError("Screening evidence must cover the full published candidate set")
    quality.to_csv(training / f"quality_metrics{suffix}.csv", index=False)
    inputs.to_parquet(training / f"action_validity{suffix}.parquet", index=False)


def publish_training_data(
    dataset,
    admission_run,
    *,
    n_jobs=6,
    feature_caches=(),
    figures=False,
    append=False,
    camera_role="front",
    refresh_camera=False,
):
    """Recompute the authoritative cohort; copy frozen inputs, never train a model."""
    dataset, admission_run = Path(dataset), Path(admission_run)
    training, features = dataset / "training_data", dataset / "image_DINO_feature"
    if training.exists() and not append:
        raise FileExistsError("Archive the published training_data before replacing them")
    prior = pd.read_csv(training / "cohort.csv") if append and training.exists() else pd.DataFrame()
    if len(prior) and "camera_role" not in prior:
        prior["camera_role"] = "front"
    prior_camera = prior.loc[prior.camera_role.eq(camera_role)] if len(prior) else prior
    if refresh_camera and len(prior):
        prior = prior.loc[prior.camera_role.ne(camera_role)].copy()
        prior_camera = prior.iloc[:0]
    suffix = "" if camera_role == "front" else f"__{camera_role}"
    settings = json.loads((admission_run / "admission_settings.json").read_text())
    # Record the current authoritative implementation, not stale saved threshold prose.
    settings["limits"] = input_quality_limits(
        settings["input_rule"], settings["limits"].get("maximum_interpolated_fraction", 0.30)
    )
    path = Path(settings["decisions"])
    full = (
        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    )
    loader = DatasetLoader(dataset)
    processing = loader.registry.get("measurement_processing", "dataset_causal_60s")
    if not full.measurement_processing.eq(processing).all():
        raise ValueError(
            "Publish only COP decisions matching the current Dataset measurement processing"
        )
    source = Path(settings["quality"]).parent
    inputs = pd.read_parquet(source / "decision_inputs.parquet")
    quality = pd.read_csv(settings["quality"])
    screening = screen_cv_cycles(
        full,
        loader.list_cycles(),
        quality=quality,
        exclude_rb=settings["exclude_rb"],
        minimum_terminal_decline=settings["minimum_terminal_decline"],
        rb_minimum_terminal_decline=settings.get("rb_minimum_terminal_decline"),
        input_rule=settings["input_rule"],
        cohort_rule=settings["cohort_rule"],
        action_inputs=inputs,
        maximum_interpolated_fraction=settings["limits"].get("maximum_interpolated_fraction", 0.30),
        excluded_cycles=settings.get("excluded_cycles", []),
        maximum_per_setpoint=settings.get("maximum_per_setpoint"),
        post_near_minutes_by_setpoint=settings.get("post_near_minutes_by_setpoint"),
    )
    cohort = screening.loc[screening.selected].copy()
    if len(prior_camera):
        old_names = set(
            pd.read_parquet(
                training / "candidate_decisions.parquet", columns=["cycle_name"]
            ).cycle_name
        )
        if set(cohort.loc[cohort.cycle_name.isin(old_names), "cycle_name"]) != set(
            prior_camera.cycle_name
        ):
            raise ValueError(
                "Append would change historical admission; inspect the changed rule before publishing"
            )
        cohort = cohort.loc[~cohort.cycle_name.isin(prior_camera.cycle_name)].copy()
    if figures and camera_role == "front":
        from plots.dataset import render_effective_cop_assets

        render_effective_cop_assets(dataset, path, n_jobs=n_jobs)
    if cohort.empty:
        if len(prior_camera):
            print("[publish] No additional admitted cycles; historical inputs retained", flush=True)
            return prior_camera, prior_camera.loc[prior_camera.evaluation_eligible]
        raise ValueError("The chosen screening rule has no training cycles")
    cohort["operating_condition"] = cohort.cycle_name.map(
        loader.list_cycles()
        .set_index("cycle_name")
        .get("operating_condition", pd.Series(dtype=str))
    )
    evaluation = cohort.loc[
        cohort.get("selected_evaluation", pd.Series(True, index=cohort.index))
    ].copy()
    reference = full.loc[full.cycle_name.isin(cohort.cycle_name)].copy()
    for key in ["candidate_defrost_time", "t_RB", "stable_heating_start"]:
        reference[key] = pd.to_datetime(reference[key], format="mixed")
    reference = calculate_cycle_cop(reference, effective=True)
    ends = cohort.set_index("cycle_name").end_time
    reference["on_action_grid"] = legal_quality_actions(
        reference
    ) & reference.candidate_defrost_time.lt(reference.cycle_name.map(ends))
    rows, intervals = label_actions(
        reference, label_rule="stopping_latch", near_set_rule=settings.get("near_set_rule", "exact")
    )
    keys = ["cycle_name", "candidate_defrost_time"]
    frequency = stopping_inputs("headroom_fraction", "changes")["RGB_response"]
    rows = rows.merge(
        inputs[keys + ["file_name", "image_time", "input_valid", *frequency]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    rows["input_valid"] = rows.input_valid.fillna(False).astype(bool)
    rows["visual_frost_score"] = np.nan  # Shared metric schema, never a DINO input.
    cohort["terminal_time"] = cohort.end_time.where(cohort.end_source.eq("actual_preparation"))
    cohort["terminal_cop"] = cohort.end_cop.where(cohort.terminal_time.notna())
    complete = rows.loc[rows.input_valid].groupby("cycle_name").cycle_cop_eligible.all()
    cohort["complete_reward"] = (
        cohort.cycle_name.map(complete).fillna(False).astype(bool)
        & cohort.terminal_time.notna()
        & cohort.terminal_cop.notna()
    )
    rows = rows.merge(
        cohort[["cycle_name", "terminal_time", "terminal_cop"]],
        on="cycle_name",
        validate="many_to_one",
    )
    # Original controller snapshots are causal inputs; teacher temperature statistics are not.
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        contexts = Parallel()(
            delayed(load_control_context)(dataset, name, group[keys + ["stable_heating_start"]])
            for name, group in rows.groupby("cycle_name")
        )
    control = pd.concat(contexts, ignore_index=True).rename(
        columns={
            "water_temperature_setpoint": "control_setpoint_c",
            "water_out_temperature": "control_outlet_c",
        }
    )
    added = [
        c for c in COMPENSATION if c not in ("control_setpoint_c", "frequency_headroom_fraction")
    ] + ["control_ambient_c"]
    rows = rows.merge(
        control[keys + ["control_timestamp", "control_setpoint_c", "control_outlet_c", *added]],
        on=keys,
        validate="one_to_one",
    )
    # Copy all existing frozen action features, including cycles rejected by this publication.
    features.mkdir(parents=True, exist_ok=True)
    for name, group in inputs.loc[inputs.file_name.notna()].groupby("cycle_name"):
        if len(prior) and name not in set(cohort.cycle_name):
            continue
        vectors = (
            group[["cycle_name", "file_name", *RGB]]
            .drop_duplicates("file_name")
            .assign(camera_role=camera_role)
        )
        if vectors[RGB].notna().any(axis=1).sum() == 0:
            continue
        path = features / f"{name}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            overlap = vectors.merge(
                existing, on=["cycle_name", "camera_role", "file_name"], suffixes=("", "_saved")
            )
            if len(overlap) and not np.allclose(
                overlap[RGB], overlap[[c + "_saved" for c in RGB]], equal_nan=True
            ):
                raise ValueError(f"Conflicting frozen DINO values: {name}")
            vectors = pd.concat([existing, vectors]).drop_duplicates(
                ["cycle_name", "camera_role", "file_name"]
            )
        vectors.to_parquet(path, index=False)
    for cache in feature_caches:
        for source_path in sorted(Path(cache).glob("*.parquet")):
            path = features / source_path.name
            extra = pd.read_parquet(
                source_path, columns=["cycle_name", "camera_role", "file_name", *RGB]
            )
            extra = extra.loc[extra.camera_role.eq(camera_role) & extra[RGB].notna().all(axis=1)]
            if extra.empty:
                continue
            if path.exists():
                saved = pd.read_parquet(path)
                extra = pd.concat([saved, extra]).drop_duplicates(
                    ["cycle_name", "camera_role", "file_name"]
                )
            extra.to_parquet(path, index=False)
    feature_counts = [
        dict(cycle_name=p.stem, frames=len(pd.read_parquet(p, columns=["file_name"])))
        for p in sorted(features.glob("frost_cycle_*.parquet"))
    ]
    pd.DataFrame(feature_counts).to_csv(features / "cohort.csv", index=False)
    (features / "README.md").write_text(
        "# 冻结 DINOv2 特征\n\ndinov2_vits14，384 维；RGB、双三次 Resize256、CenterCrop224、ImageNet 归一化。每循环一个 parquet，camera_role 区分机位；保留已有特征，不训练骨干。\n"
    )
    cohort["evaluation_eligible"] = cohort.cycle_name.isin(evaluation.cycle_name)
    cohort["camera_role"] = camera_role
    training.mkdir(exist_ok=append)
    if (training / "reference.parquet").exists():
        reference = pd.concat(
            [pd.read_parquet(training / "reference.parquet"), reference], ignore_index=True
        ).drop_duplicates(["cycle_name", "candidate_defrost_time"], keep="first")
    if len(prior):
        cohort = pd.concat([prior, cohort], ignore_index=True)
    cohort.to_csv(training / "cohort.csv", index=False)
    reference.to_parquet(training / "reference.parquet", index=False)
    for name, table in rows.groupby("cycle_name"):
        table.to_parquet(training / f"{name}{suffix}.parquet", index=False)
    evaluation_note = (
        "evaluation_eligible 与训练准入一致，全部循环参加实验级留出回放；不另筛评价集。"
        if settings["cohort_rule"] == "observed"
        else "evaluation_eligible 标记完整评价资格。"
    )
    if camera_role == "front":
        (training / "README.md").write_text(
            "# 训练与评价数据\n\n每个 frost_cycle_*.parquet 保存该循环的动作输入与标签。cohort.csv 的"
            + evaluation_note
            + "complete_reward 仅标记完整收益资格。camera_role 区分机位，非 front 动作文件使用 __<camera_role> 后缀；DatasetLoader 按机位加载。分折按实验生成；实际训练、验证、测试角色由每一折决定。reference.parquet 保存共享参考曲线。\n"
        )
    (training / f"settings{suffix}.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    )
    screening.to_csv(training / f"cycle_screening{suffix}.csv", index=False)
    intervals.to_csv(training / f"near_optimal_intervals{suffix}.csv", index=False)
    # Full candidates remain in Dataset for subsequent authoritative re-screening.
    full.to_parquet(training / "candidate_decisions.parquet", index=False)
    publish_screening_inputs(dataset, settings, camera_role)
    if figures and camera_role == "front":
        publish_cycle_figures(dataset, full, screening, n_jobs=n_jobs)
    cohort = cohort.loc[cohort.camera_role.eq(camera_role)].copy()
    evaluation = cohort.loc[cohort.evaluation_eligible].copy()
    print(
        f"[publish] {len(full.cycle_name.unique())} source cycles -> {len(cohort)} training; "
        f"{len(evaluation)} evaluation eligible; {int(cohort.complete_reward.sum())} complete reward",
        flush=True,
    )
    return cohort, evaluation


def _render_cycle_pair(dataset, curve, output, near_set_rule, frost_frames=None):
    from plots.cycle import render_cycle_publication
    from plots.defrost_decision import render_current_decision_cycle

    loader = DatasetLoader(dataset)
    curve = curve.copy()
    for column in curve:
        if column.endswith(("_time", "_start", "_end")) or column in {"t_star", "t_RB"}:
            curve[column] = pd.to_datetime(curve[column], format="mixed")
    name = curve.cycle_name.iloc[0]
    render_current_decision_cycle(
        curve,
        loader,
        output / f"{name}_effective_COP.png",
        near_set_rule=near_set_rule,
        frost_frames=frost_frames,
    )
    if not (output / f"{name}.png").is_file():
        render_cycle_publication(
            loader.load_cycle(name), loader.get_cycle_record(name), output / f"{name}.png"
        )
    print(f"[figures] {output.name}/{name}: COP + overview", flush=True)


def publish_cycle_figures(
    dataset,
    decisions,
    screening,
    *,
    n_jobs=6,
    near_set_rule="connected",
    early_review_rule="all_cycles",
    redraw=False,
):
    """Add paired figures without rewriting published model inputs or labels."""
    dataset = Path(dataset)
    training = dataset / "training_data"
    cohort = DatasetLoader(dataset).training_cohort()
    if set(cohort.cycle_name) != set(screening.loc[screening.selected, "cycle_name"]):
        raise ValueError(
            "Published membership differs from current screening; inspect before adding figures"
        )
    failures = rb_failure_cases(decisions, screening, early_review_rule=early_review_rule)
    destinations = {name: training for name in cohort.cycle_name}
    review_destinations = {}
    for kind in ["早除霜", "迟除霜"]:
        output = dataset / "RB失败循环" / kind
        output.mkdir(parents=True, exist_ok=True)
        cases = failures.loc[failures.failure_type.eq(kind)]
        if (output / "cohort.csv").exists() and "review_rule" not in pd.read_csv(
            output / "cohort.csv"
        ).columns:
            shutil.copy2(output / "cohort.csv", output / "cohort_strict.csv")
        cases.to_csv(output / "cohort.csv", index=False)
        review_destinations.update({name: output for name in cases.cycle_name})
        print(f"[review] {kind}: {len(cases)}", flush=True)
    # Render a cycle once; an overlapping review case uses the identical figure pair.
    from plots.defrost_decision import load_native_frost_frames

    frames = load_native_frost_frames()
    frost = dict(tuple(frames.groupby("cycle_name")))
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel(verbose=5)(
            delayed(_render_cycle_pair)(
                dataset,
                curve,
                destinations.get(name, review_destinations.get(name)),
                near_set_rule,
                frost.get(name, frames.iloc[:0]),
            )
            for name, curve in decisions.loc[
                decisions.cycle_name.isin(set(destinations) | set(review_destinations))
            ].groupby("cycle_name")
            if redraw
            or not all(
                (
                    destinations.get(name, review_destinations.get(name)) / f"{name}{suffix}"
                ).is_file()
                for suffix in [".png", "_effective_COP.png"]
            )
        )
    for name, output in review_destinations.items():
        if name in destinations:
            for suffix in [".png", "_effective_COP.png"]:
                shutil.copy2(training / f"{name}{suffix}", output / f"{name}{suffix}")
    from plots.dataset import refresh_effective_decision_figures

    sources = {
        name: output / f"{name}_effective_COP.png"
        for name, output in (review_destinations | destinations).items()
    }
    refresh_effective_decision_figures(dataset, sources)


def refresh_control_inputs(dataset, columns=None):
    """Add causal compensation columns to published inputs without changing support or labels."""
    from image_models.control_context import COMPENSATION

    dataset = Path(dataset)
    columns = columns or [
        c for c in COMPENSATION if c not in ("control_setpoint_c", "frequency_headroom_fraction")
    ] + ["control_ambient_c"]
    cohort = DatasetLoader(dataset).training_cohort()
    for name in cohort.cycle_name:
        source = pd.read_parquet(dataset / "training_data" / f"{name}.parquet")
        keys = ["candidate_defrost_time", "stable_heating_start"]
        control = load_control_context(dataset, name, source[keys])
        rows = source.drop(columns=columns, errors="ignore")
        rows = rows.merge(
            control[["candidate_defrost_time", *columns]],
            on="candidate_defrost_time",
            how="left",
            validate="one_to_one",
        )
        rows.to_parquet(dataset / "training_data" / f"{name}.parquet", index=False)


def exclude_training_cycles(dataset, names, reason, output):
    """Update published membership through authoritative screening; retain observations."""
    import shutil

    dataset, output = Path(dataset), Path(output)
    training = dataset / "training_data"
    settings = json.loads((training / "settings.json").read_text())
    all_cohorts = pd.read_csv(training / "cohort.csv")
    prior = DatasetLoader(dataset).training_cohort()
    excluded = sorted(set(settings.get("excluded_cycles", [])) | set(names))
    loader = DatasetLoader(dataset)
    full, quality, inputs = loader.load_screening_inputs()
    screening = screen_cv_cycles(
        full,
        loader.list_cycles(),
        quality=quality,
        exclude_rb=settings["exclude_rb"],
        minimum_terminal_decline=settings["minimum_terminal_decline"],
        rb_minimum_terminal_decline=settings.get("rb_minimum_terminal_decline"),
        input_rule=settings["input_rule"],
        cohort_rule=settings["cohort_rule"],
        maximum_interpolated_fraction=settings["limits"].get("maximum_interpolated_fraction", 0.30),
        action_inputs=inputs,
        excluded_cycles=excluded,
        maximum_per_setpoint=settings.get("maximum_per_setpoint"),
        post_near_minutes_by_setpoint=settings.get("post_near_minutes_by_setpoint"),
    )
    selected = set(screening.loc[screening.selected, "cycle_name"])
    if selected != set(prior.cycle_name) - set(names):
        raise ValueError(
            "Authoritative screening changed beyond requested exclusions; inspect before updating publication"
        )
    output.mkdir(parents=True, exist_ok=False)
    for filename in ["cohort.csv", "settings.json", "cycle_screening.csv"]:
        shutil.copy2(training / filename, output / ("before_" + filename))
    reasons = settings.setdefault("exclusion_reasons_by_cycle", {})
    for name in names:
        reasons[name] = reason
    settings["excluded_cycles"] = excluded
    screening.to_csv(training / "cycle_screening.csv", index=False)
    keep = all_cohorts.cycle_name.isin(selected)
    if "camera_role" in all_cohorts:
        keep |= all_cohorts.camera_role.ne("front")
    all_cohorts.loc[keep].to_csv(training / "cohort.csv", index=False)
    (training / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    )
    screening.loc[screening.cycle_name.isin(names)].assign(reason=reason).to_csv(
        output / "excluded_cycles.csv", index=False
    )
    print(
        f"Published training cohort: {len(prior)} -> {len(selected)}; observations and historical results retained"
    )


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser(
        "training-data", help="publish screened model inputs inside Dataset"
    )
    training.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    training.add_argument("--admission-run", type=Path, required=True)
    training.add_argument("--n-jobs", type=int, default=6)
    training.add_argument("--feature-caches", type=Path, nargs="*", default=[])
    training.add_argument(
        "--figures", action="store_true", help="Also generate cycle figures after publication"
    )
    training.add_argument(
        "--append",
        action="store_true",
        help="Publish new admitted cycles while retaining historical inputs",
    )
    screening = actions.add_parser(
        "screening-inputs", help="copy quality evidence into an existing training publication"
    )
    screening.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    screening.add_argument("--camera-role", default="front")
    context = actions.add_parser(
        "control-inputs", help="refresh causal compensation features in published inputs"
    )
    context.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    context.add_argument("--columns", nargs="+", help="refresh only these causal control columns")

    exclude = actions.add_parser(
        "exclude-training-cycles", help="exclude published cycles through authoritative screening"
    )
    exclude.add_argument("cycles", nargs="+")
    exclude.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    exclude.add_argument("--reason", required=True)
    exclude.add_argument("--output", type=Path, required=True)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.action == "screening-inputs":
        from .load_dataset import DatasetLoader

        loader = DatasetLoader(args.dataset)
        publish_screening_inputs(
            loader.dataset_root, loader.camera_training_settings(args.camera_role), args.camera_role
        )
        return 0
    if args.action == "exclude-training-cycles":
        exclude_training_cycles(args.dataset, args.cycles, args.reason, args.output)
        return 0
    if args.action == "control-inputs":
        refresh_control_inputs(args.dataset, columns=args.columns)
        return 0

    if args.action == "training-data":
        publish_training_data(
            args.dataset,
            args.admission_run,
            n_jobs=args.n_jobs,
            feature_caches=args.feature_caches,
            figures=args.figures,
            append=args.append,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
