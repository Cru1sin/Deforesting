"""Teacher-independent student inputs and cycle-level quality measurements.

No model predictions, optimal-time positions or performance targets enter this audit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from dataset_tools import DatasetLoader
from dataset_tools.cycle_metadata import image_root
from image_models.control_context import load_control_context, stopping_inputs
from image_models.cycle_selection import legal_quality_actions
from image_models.dinov2_features import merge_cached_features
from image_models.frost_progression import match_trajectory_frames
from image_models.inputs import RGB

FREQUENCY = stopping_inputs("headroom_fraction", "changes")["RGB_response"]


def trace_sources(actions, dataset, caches, output, camera_role="front"):
    """Distinguish time-matching, cache coverage and local-file absence numerically."""
    records = []
    for name, g in actions.groupby("cycle_name"):
        missing = g.loc[
            g.rgb_time_valid.eq(True) & g.dino_valid.eq(False), "file_name"
        ].drop_duplicates()
        valid_later = set()
        for root in caches:
            path = root / f"{name}.parquet"
            if len(missing) and path.exists():
                f = pd.read_parquet(path, columns=["camera_role", "file_name", *RGB])
                valid_later.update(
                    f.loc[
                        f.camera_role.eq(camera_role) & np.isfinite(f[RGB]).all(axis=1), "file_name"
                    ]
                )
        records.append(
            dict(
                cycle_name=name,
                matched_frames_missing_dino=len(missing),
                missing_dino_local_rgb=sum(
                    (image_root(Path(dataset)) / name / camera_role / f).is_file() for f in missing
                ),
                missing_dino_finite_later_cache=sum(f in valid_later for f in missing),
                max_image_age_s=(g.candidate_defrost_time - g.image_time).dt.total_seconds().max(),
                max_control_age_s=(g.candidate_defrost_time - g.control_timestamp)
                .dt.total_seconds()
                .max(),
            )
        )
    result = pd.DataFrame(records)
    result.to_csv(Path(output) / "source_trace.csv", index=False)
    return result


def audit_cycle(name, reference, frames, catalog, dataset, caches, camera_role="front"):
    reference = reference.copy()
    for key in ["candidate_defrost_time", "t_RB", "stable_heating_start"]:
        reference[key] = pd.to_datetime(reference[key], format="mixed")
    end = pd.to_datetime(catalog.defrost_preparation_start)
    actual = pd.notna(end)
    if not actual:
        end = pd.to_datetime(catalog.end_time)
    rows = (
        reference.loc[legal_quality_actions(reference) & reference.candidate_defrost_time.lt(end)]
        .sort_values("candidate_defrost_time")
        .copy()
    )
    record = dict(
        cycle_name=name,
        camera_role=camera_role,
        experiment_id=catalog.experiment_id,
        actions=len(rows),
        actual_endpoint=actual,
        endpoint_time=end,
    )
    # Entire published cycle, not just the near-optimal window. Missing flags are unknown.
    cycle = DatasetLoader(dataset).load_cycle(name)
    flags = cycle.get("is_interpolated")
    record["interpolated_fraction"] = (
        float(flags.mean()) if flags is not None and len(flags) and flags.notna().all() else np.nan
    )
    endpoint = reference.loc[reference.candidate_defrost_time.eq(end)]
    record["endpoint_reward_valid"] = bool(len(endpoint) and endpoint.cycle_cop_eligible.iloc[0])
    rb = reference.loc[reference.candidate_defrost_time.eq(reference.t_RB)]
    record["rb_reward_valid"] = bool(len(rb) and rb.cycle_cop_eligible.iloc[0])
    if rows.empty:
        return rows, pd.DataFrame([record]), np.array([])
    frames = frames.sort_values("image_time").drop_duplicates(
        ["cycle_name", "camera_role", "file_name"]
    )
    keys = frames[["cycle_name", "camera_role", "file_name"]]
    features = merge_cached_features(keys, [root / f"{name}.parquet" for root in caches])
    frames = frames.merge(
        features, on=["cycle_name", "camera_role", "file_name"], validate="one_to_one"
    )
    frames["dino_valid"] = np.isfinite(frames[RGB]).all(axis=1)
    local = {
        p.name for p in (image_root(Path(dataset)) / name / camera_role).glob("*") if p.is_file()
    }
    record["unindexed_local_rgb_files"] = len(local - set(frames.file_name))
    record["local_rgb_files"] = len(local)
    record["cached_finite_frames"] = int(frames.dino_valid.sum())
    matched = match_trajectory_frames(rows, frames, metric="dino_valid", max_age_seconds=30)
    matched = matched.merge(
        features.drop(columns="camera_role"),
        on=["cycle_name", "file_name"],
        how="left",
        validate="many_to_one",
    )
    # The latest captured frame is matched first; a bad vector is not replaced by an older one.
    matched["rgb_time_valid"] = matched.image_time.notna()
    matched["dino_valid"] = np.isfinite(matched[RGB]).all(axis=1)
    control = load_control_context(
        dataset, name, rows[["cycle_name", "candidate_defrost_time", "stable_heating_start"]]
    )
    fields = ["cycle_name", "candidate_defrost_time", "control_timestamp", *FREQUENCY]
    matched = matched.merge(
        control[fields], on=["cycle_name", "candidate_defrost_time"], validate="one_to_one"
    )
    matched["control_valid"] = np.isfinite(matched[FREQUENCY]).all(axis=1)
    matched["image_age_seconds"] = (
        matched.candidate_defrost_time - matched.image_time
    ).dt.total_seconds()
    matched["control_age_seconds"] = (
        matched.candidate_defrost_time - matched.control_timestamp
    ).dt.total_seconds()
    matched["input_valid"] = matched.rgb_time_valid & matched.dino_valid & matched.control_valid
    # Deliberately absent: Harmony, COP/reward validity, pressure, temperature and power gates.
    start = rows.candidate_defrost_time.min()
    period = frames.loc[frames.image_time.between(start, end, inclusive="left")]
    gaps = period.image_time.diff().dt.total_seconds().dropna().to_numpy()
    record.update(
        rgb_period_frames=len(period),
        rgb_total_frames=len(frames),
        rgb_local_period_frames=sum(
            (image_root(Path(dataset)) / name / camera_role / f).is_file() for f in period.file_name
        ),
        frame_gap_median_s=np.median(gaps) if len(gaps) else np.nan,
        frame_gap_p95_s=np.quantile(gaps, 0.95) if len(gaps) else np.nan,
        max_internal_image_gap_s=max(gaps) if len(gaps) else np.nan,
        leading_image_gap_s=(period.image_time.min() - start).total_seconds()
        if len(period)
        else (end - start).total_seconds(),
        trailing_image_gap_s=(end - period.image_time.max()).total_seconds()
        if len(period)
        else (end - start).total_seconds(),
        raw_rgb_coverage=matched.rgb_time_valid.mean(),
        dino_action_coverage=matched.dino_valid.mean(),
        control_coverage=matched.control_valid.mean(),
        action_coverage=matched.input_valid.mean(),
        dino_frame_coverage=period.dino_valid.mean(),
        max_image_age_s=matched.image_age_seconds.max(),
        max_control_age_s=matched.control_age_seconds.max(),
        heat_coverage=rows.pre_defrost_heat_measurement_valid.eq(True).mean(),
        electricity_coverage=rows.pre_defrost_electricity_measurement_valid.eq(True).mean(),
        ridge_prediction_coverage=rows.defrost_event_electricity_prediction_available.eq(
            True
        ).mean(),
        ridge_domain_coverage=rows.defrost_event_electricity_in_training_domain.eq(True).mean(),
        cop_candidate_coverage=rows.cycle_cop_eligible.eq(True).mean(),
        available_actions=int(matched.input_valid.sum()),
        complete_stopping_reward=bool(
            matched.input_valid.any()
            and matched.loc[matched.input_valid, "cycle_cop_eligible"].all()
        ),
        observable_reward_coverage=matched.loc[matched.input_valid, "cycle_cop_eligible"].mean(),
    )
    for field in FREQUENCY:
        record[field + "_coverage"] = matched[field].notna().mean()
    return matched, pd.DataFrame([record]), gaps


def audit_inputs(
    reference, catalog, dataset, caches, output, n_jobs=6, legacy_run=None, camera_role="front"
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = DatasetLoader(dataset).load_image_metadata()
    frames = frames.loc[frames.camera_role.eq(camera_role)].copy()
    frames["image_time"] = pd.to_datetime(frames.image_time)
    catalog = catalog.set_index("cycle_name")
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        results = Parallel(max_nbytes=None)(
            delayed(audit_cycle)(
                name,
                cycle,
                frames.loc[frames.cycle_name.eq(name)],
                catalog.loc[name],
                dataset,
                caches,
                camera_role,
            )
            for name, cycle in reference.groupby("cycle_name")
        )
    actions = pd.concat([x[0] for x in results], ignore_index=True)
    quality = pd.concat([x[1] for x in results], ignore_index=True)
    if legacy_run:
        old = pd.read_parquet(Path(legacy_run) / "decision_inputs.parquet")
        keys = ["cycle_name", "candidate_defrost_time"]
        old["old_dino_finite"] = np.isfinite(old[RGB]).all(axis=1)
        old["old_harmony_finite"] = np.isfinite(old.visual_frost_score)
        old["old_control_finite"] = np.isfinite(old[FREQUENCY]).all(axis=1)
        trace = actions[
            keys + ["input_valid", "rgb_time_valid", "dino_valid", "control_valid"]
        ].merge(
            old[
                keys
                + ["input_valid", "old_dino_finite", "old_harmony_finite", "old_control_finite"]
            ],
            on=keys,
            how="left",
            suffixes=("", "_old"),
        )
        trace["recovered_action"] = trace.input_valid & trace.input_valid_old.eq(False)
        trace["old_action_known"] = trace.input_valid_old.notna()
        trace.to_parquet(output / "input_chain_trace.parquet", index=False)
        totals = (
            trace.groupby("cycle_name")[
                [
                    "input_valid_old",
                    "old_dino_finite",
                    "old_harmony_finite",
                    "old_control_finite",
                    "recovered_action",
                    "old_action_known",
                ]
            ]
            .sum()
            .reset_index()
        )
        quality = quality.merge(totals, on="cycle_name", how="left", validate="one_to_one")
    actions.to_parquet(output / "decision_inputs.parquet", index=False)
    trace_sources(actions, dataset, caches, output, camera_role)
    quality.to_csv(output / "quality_metrics.csv", index=False)
    distribution = (
        quality.select_dtypes(include="number")
        .quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
        .T
    )
    distribution.to_csv(output / "quality_distributions.csv")
    gaps = np.concatenate([x[2] for x in results])
    pd.DataFrame(
        dict(
            quantile=[0, 0.01, 0.05, 0.5, 0.95, 0.99, 1],
            seconds=np.quantile(gaps, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]),
        )
    ).to_csv(output / "camera_sampling.csv", index=False)
    return actions, quality
