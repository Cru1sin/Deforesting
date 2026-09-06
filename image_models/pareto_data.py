"""Reusable measured rows and experiment-isolated economic Pareto teachers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from defrost_decision.candidate_quantities import build_measured_candidate_quantities
from defrost_decision.pareto_selection import select_cop_heating_rate_pareto_knee
from defrost_decision.performance_objectives import calculate_performance_objectives
from defrost_event_models.ridge_models import (
    DYNAMIC_STATE_8,
    OUTCOME_TARGETS,
    fit_model_on_all_experiments,
    model_to_parameters,
    predict_with_model_parameters,
    select_events_complete_for_all_outcomes,
)
from defrost_event_models.training_data import build_candidate_boundaries
from image_models.sensor_features import CURRENT_SENSORS, build_past_only_sensor_statistics

BASE_VERSION = "pareto_measured_union_stat6_v1"


def build_cycle_base_table(loader, cycle_name, cache_dir=None, *, candidate_step_seconds=10):
    """Cache G-free native frames plus the unchanged candidate grid per cycle."""
    record = loader.get_cycle_record(cycle_name)
    boundary = record.get("boundaries", record)
    start = pd.Timestamp(boundary["heating_start"])
    end = pd.Timestamp(boundary["defrost_preparation_start"])
    assets = record["assets"]
    paths = [loader.dataset_root / str(assets[key]) for key in ("original_csv", "parquet")]
    images = loader.load_image_metadata(cycle_name)
    images["image_time"] = pd.to_datetime(images["image_time"], format="mixed")
    images = images.loc[
        images.camera_role.eq("front") & images.image_time.ge(start) & images.image_time.lt(end)
    ]
    images = images.sort_values(["image_time", "file_name"]).drop_duplicates("image_time")
    signature = {
        "version": BASE_VERSION,
        "candidate_step_seconds": candidate_step_seconds,
        "boundaries": boundary,
        "experiment_id": str(record["experiment_id"]),
        "registry": loader.registry,
        "images": images[["file_name", "image_time"]].astype(str).to_dict("records"),
        "sources": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
    }
    destination = Path(cache_dir) / f"{cycle_name}.parquet" if cache_dir is not None else None
    metadata = destination.with_suffix(".json") if destination else None
    if (
        destination
        and destination.exists()
        and metadata.exists()
        and json.loads(metadata.read_text()) == signature
    ):
        return pd.read_parquet(destination)
    grid = build_candidate_boundaries(
        cycle_name, str(record["experiment_id"]), start, end, step_seconds=candidate_step_seconds
    )
    times = (
        pd.DatetimeIndex(grid.candidate_defrost_time)
        .union(pd.DatetimeIndex(images.image_time))
        .sort_values()
    )
    rows = grid.iloc[[0]].reindex(np.zeros(len(times), dtype=int)).reset_index(drop=True)
    rows["candidate_defrost_time"] = times
    rows["minutes_since_heating_start"] = (times - start).total_seconds() / 60
    rows = build_measured_candidate_quantities(loader, cycle_name, rows)
    rows["image_time"] = times
    rows["row_id"] = [f"{cycle_name}:{value.value}" for value in times]
    rows["is_frame"] = times.isin(images.image_time)
    rows["is_teacher_candidate"] = times.isin(grid.candidate_defrost_time)
    rows["stable_heating_start"] = pd.Timestamp(boundary["stable_heating_start"])
    rgb = images[["image_time", "file_name"]].rename(columns={"image_time": "rgb_image_time"})
    rows = pd.merge_asof(
        rows,
        rgb,
        left_on="image_time",
        right_on="rgb_image_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=45),
    )
    rows["camera_role"] = "front"
    rows["rgb_available"] = rows.file_name.notna()
    columns = [
        "cycle_name",
        "timestamp",
        *CURRENT_SENSORS,
        *[f"{name}__imputed" for name in CURRENT_SENSORS],
    ]
    statistics = build_past_only_sensor_statistics(
        loader.load_cycle(cycle_name, columns=columns),
        bucket_seconds=int(loader.registry["resample_interval_seconds"]),
    )
    rows = pd.merge_asof(
        rows,
        statistics.drop(columns="cycle_name"),
        left_on="image_time",
        right_on="sensor_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=15),
    )
    for column in rows.select_dtypes(include="datetime"):
        rows[column] = rows[column].astype("datetime64[ns]")
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(destination, index=False)
        metadata.write_text(json.dumps(signature, indent=2))
    return rows


def fit_fold_parameters(training_events, excluded_experiments):
    """Fit all four dynamic8 targets to one explicitly isolated complete cohort."""
    selected = select_events_complete_for_all_outcomes(training_events)
    selected = selected.loc[
        ~selected.experiment_id.astype(str).isin(set(map(str, excluded_experiments)))
    ]
    if selected.experiment_id.nunique() < 3:
        raise ValueError("fold G requires at least three training experiments for inner LOEO")
    return {
        name: model_to_parameters(fit_model_on_all_experiments(selected, DYNAMIC_STATE_8, target))
        for name, target in OUTCOME_TARGETS.items()
    }


def apply_fold_teacher(base_table, parameters, *, allow_model_extrapolation=False):
    """Apply the same fold G to every row; only original grid rows select knee.

    Economic C/H values are unmasked online measurements; eligibility, relation
    support and labels are offline metadata and must not enter model features.
    """
    tables = []
    for _, base in base_table.groupby("cycle_name", sort=False):
        rows = base.sort_values("candidate_defrost_time").reset_index(drop=True).copy()
        for name in OUTCOME_TARGETS:
            prediction = predict_with_model_parameters(parameters[name], rows)
            field = "defrost_" + name
            unit = "minutes" if name == "event_duration" else "kwh"
            rows[f"{field}_{unit}"] = prediction.prediction.to_numpy()
            rows[f"{field}_prediction_available"] = np.isfinite(prediction.prediction.to_numpy())
            rows[f"{field}_in_training_domain"] = prediction.support_distance.le(
                prediction.support_threshold
            ).to_numpy()
            rows[f"{field}_training_distance"] = prediction.support_distance.to_numpy()
        values = calculate_performance_objectives(
            rows, allow_model_extrapolation=allow_model_extrapolation
        )
        values["economic_c"] = values.cycle_cop
        values["economic_h"] = values.cycle_heating_rate_kw
        # Recompute support on the frozen grid: native frames cannot bridge gaps.
        grid = calculate_performance_objectives(
            rows.loc[rows.is_teacher_candidate], allow_model_extrapolation=allow_model_extrapolation
        )
        teacher = select_cop_heating_rate_pareto_knee(
            grid, minimum_time=rows.stable_heating_start.iloc[0]
        )
        common = teacher.cycle_cop_eligible & teacher.cycle_heating_rate_kw_eligible
        domain = (
            teacher.defrost_event_electricity_in_training_domain
            & teacher.defrost_event_net_heat_in_training_domain
            & teacher.defrost_event_duration_in_training_domain
        )
        breaks = (
            ~common
            | ~common.shift(fill_value=False)
            | domain.ne(domain.shift())
            | teacher.candidate_defrost_time.diff().gt(pd.Timedelta(seconds=90))
        )
        teacher["relation_support_run"] = breaks.cumsum().astype(str).where(common)
        teacher["relation_branch"] = teacher.pareto_selection_method
        fields = [
            "candidate_defrost_time",
            "pareto_selection_score",
            "relation_support_run",
            "relation_branch",
        ]
        values = values.merge(
            teacher[fields], on="candidate_defrost_time", how="left", validate="one_to_one"
        )
        selected = teacher.loc[teacher.is_selected_pareto_point, "candidate_defrost_time"]
        tau = selected.iloc[0] if len(selected) else pd.NaT
        values["teacher_time"] = tau
        values["target"] = np.nan
        if pd.notna(tau):
            values["target"] = np.where(
                values.candidate_defrost_time.lt(tau),
                0.0,
                np.where(values.candidate_defrost_time.eq(tau), 0.5, 1.0),
            )
        values["is_knee"] = values.candidate_defrost_time.eq(tau) & values.is_teacher_candidate
        values["teacher_coverage_reason"] = "selected" if pd.notna(tau) else "no_common_domain"
        tables.append(values)
    return pd.concat(tables, ignore_index=True)
