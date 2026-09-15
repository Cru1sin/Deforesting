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

BASE_VERSION = "pareto_measured_union_stat6_quality_v2"


def _temporal_features(rows: pd.DataFrame, source: str, prefix: str) -> pd.DataFrame:
    """Current and five-minute past-only trajectory features for one scalar."""
    values = pd.to_numeric(rows[source], errors="coerce")
    indexed = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(rows["candidate_defrost_time"]))
    rolling = indexed.rolling("5min", closed="both", min_periods=1)

    def delta(window):
        valid = window[np.isfinite(window)]
        return valid[-1] - valid[0] if len(valid) >= 2 else np.nan

    def slope(window):
        valid = window.dropna()
        if len(valid) < 2:
            return np.nan
        minutes = (valid.index.asi8 - valid.index.asi8[0]) / 60e9
        centered = minutes - minutes.mean()
        denominator = np.square(centered).sum()
        return (
            np.dot(centered, valid.to_numpy() - valid.mean()) / denominator
            if denominator else np.nan
        )

    result = pd.DataFrame(index=rows.index)
    result[f"{prefix}_current"] = values.to_numpy()
    result[f"{prefix}_mean"] = rolling.mean().to_numpy()
    result[f"{prefix}_std"] = rolling.std().to_numpy()
    result[f"{prefix}_delta"] = rolling.apply(delta, raw=True).to_numpy()
    result[f"{prefix}_slope"] = rolling.apply(slope, raw=False).to_numpy()
    result[f"{prefix}_valid_count"] = rolling.count().to_numpy()
    last_valid = pd.Series(indexed.index.where(indexed.notna()), index=indexed.index).ffill()
    result[f"{prefix}_age_seconds"] = (
        pd.Series(indexed.index, index=indexed.index) - last_valid
    ).dt.total_seconds().to_numpy()
    for column in result.columns:
        if column.endswith(("valid_count", "age_seconds")):
            continue
        result[f"{column}_missing"] = result[column].isna()
    return result


def add_online_economic_features(
    rows: pd.DataFrame, *, feature_prefix: str = "online"
) -> pd.DataFrame:
    """Calculate deployable C/H/O from causal accounting; never move the teacher."""
    tables = []
    replacements = {
        "pre_defrost_electricity_kwh": "online_pre_defrost_electricity_kwh",
        "pre_defrost_heat_kwh": "online_pre_defrost_heat_kwh",
        "pre_defrost_compressor_electricity_kwh": (
            "online_pre_defrost_compressor_electricity_kwh"
        ),
        "pre_defrost_electricity_measurement_valid": (
            "online_pre_defrost_electricity_measurement_valid"
        ),
        "pre_defrost_heat_measurement_valid": "online_pre_defrost_heat_measurement_valid",
        "pre_defrost_compressor_electricity_measurement_valid": (
            "online_pre_defrost_compressor_measurement_valid"
        ),
    }
    for _, cycle in rows.groupby("cycle_name", sort=False):
        cycle = cycle.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
        online = cycle.copy()
        for target, source in replacements.items():
            online[target] = online[source]
        objectives = calculate_performance_objectives(online, allow_model_extrapolation=True)
        names = {
            "c": "cycle_cop", "h": "cycle_heating_rate_kw",
            "o": "cycle_evaporator_capacity_kw",
        }
        validity = []
        for short, name in names.items():
            pointwise = (
                objectives[f"{name}_measurements_valid"].fillna(False)
                & objectives[f"{name}_physically_valid"].fillna(False)
                & objectives["pre_defrost_feature_window_valid"].fillna(False)
                & np.isfinite(objectives[name])
            )
            objectives[f"{feature_prefix}_{short}_pointwise_valid"] = pointwise
            objectives[f"{feature_prefix}_{short}"] = objectives[name].where(pointwise)
            validity.append(pointwise)
        objectives[f"{feature_prefix}_pointwise_valid"] = np.logical_and.reduce(validity)
        features = [
            _temporal_features(
                objectives, f"{feature_prefix}_{short}", f"{feature_prefix}_{short}"
            )
            for short in names
        ]
        tables.append(pd.concat([cycle, *features, objectives[[
            f"{feature_prefix}_c", f"{feature_prefix}_h", f"{feature_prefix}_o",
            f"{feature_prefix}_c_pointwise_valid",
            f"{feature_prefix}_h_pointwise_valid",
            f"{feature_prefix}_o_pointwise_valid",
            f"{feature_prefix}_pointwise_valid",
        ]]], axis=1))
    return pd.concat(tables, ignore_index=True)


def build_cycle_base_table(
    loader,
    cycle_name,
    cache_dir=None,
    *,
    candidate_step_seconds=10,
    allow_measurement_reconstruction=False,
    event_start_only=False,
):
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
        "allow_measurement_reconstruction": allow_measurement_reconstruction,
        "event_start_only": event_start_only,
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
    if event_start_only:
        grid = pd.DataFrame({
            "cycle_name": [cycle_name], "experiment_id": [str(record["experiment_id"])],
            "candidate_defrost_time": [end],
            "minutes_since_heating_start": [(end - start).total_seconds() / 60],
            "heating_accounting_start": [start + pd.Timedelta(minutes=9)],
            "heating_accounting_start_rule": ["fixed_post_defrost_9min"],
            "heating_start": [start], "observed_defrost_preparation_start": [end],
        })
    else:
        grid = build_candidate_boundaries(
            cycle_name, str(record["experiment_id"]), start, end,
            step_seconds=candidate_step_seconds,
        )
    times = (
        pd.DatetimeIndex(grid.candidate_defrost_time)
        .union(pd.DatetimeIndex([] if event_start_only else images.image_time))
        .sort_values()
    )
    rows = grid.iloc[[0]].reindex(np.zeros(len(times), dtype=int)).reset_index(drop=True)
    rows["candidate_defrost_time"] = times
    rows["minutes_since_heating_start"] = (times - start).total_seconds() / 60
    rows = build_measured_candidate_quantities(
        loader,
        cycle_name,
        rows,
        allow_measurement_reconstruction=allow_measurement_reconstruction,
    )
    rows["image_time"] = times
    rows["row_id"] = [f"{cycle_name}:{value.value}" for value in times]
    rows["is_frame"] = times.isin(images.image_time)
    rows["is_teacher_candidate"] = (
        False if event_start_only else times.isin(grid.candidate_defrost_time)
    )
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
    rows["rgb_missing"] = ~rows["rgb_available"]
    rows["rgb_age_seconds"] = (rows["image_time"] - rows["rgb_image_time"]).dt.total_seconds()
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


def add_neural_event_predictions(
    rows: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Put one frozen neural event readout into the shared objective columns."""
    result = rows.merge(predictions, on="row_id", validate="one_to_one")
    for outcome, target in OUTCOME_TARGETS.items():
        unit = "minutes" if outcome == "event_duration" else "kwh"
        field = f"defrost_{outcome}_{unit}"
        result[field] = result[f"predicted_{target}"]
        result[f"defrost_{outcome}_prediction_available"] = np.isfinite(result[field])
        result[f"defrost_{outcome}_in_training_domain"] = pd.Series(
            pd.NA, index=result.index, dtype="boolean"
        )
    return result


def apply_neural_pareto(rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Replace only event outcomes, then reuse the frozen-grid C/H Pareto selector."""
    joined = rows.merge(predictions, on="row_id", validate="one_to_one")
    working = add_neural_event_predictions(rows, predictions).set_index("row_id")
    tables = []
    for _, cycle in joined.groupby("cycle_name", sort=False):
        work = working.loc[cycle.row_id].sort_values(
            "candidate_defrost_time", kind="stable"
        ).reset_index()
        objectives = calculate_performance_objectives(
            work, allow_model_extrapolation=True
        )
        grid = select_cop_heating_rate_pareto_knee(
            objectives.loc[objectives.is_teacher_candidate],
            minimum_time=work.stable_heating_start.iloc[0],
        )
        result = cycle.sort_values("candidate_defrost_time", kind="stable").reset_index(drop=True)
        result["neural_event_prediction_domain"] = "unknown"
        for outcome in OUTCOME_TARGETS:
            unit = "minutes" if outcome == "event_duration" else "kwh"
            field = f"defrost_{outcome}_{unit}"
            result[f"neural_{field}"] = work[field].to_numpy()
            result[f"neural_defrost_{outcome}_prediction_available"] = np.isfinite(
                work[field]
            )
        for name in (
            "cycle_cop", "cycle_heating_rate_kw", "cycle_evaporator_capacity_kw"
        ):
            result[f"neural_{name}"] = objectives[name].to_numpy()
            for suffix in (
                "measurements_valid", "physically_valid", "eligible",
                "eligible_without_extrapolation", "uses_model_extrapolation",
            ):
                result[f"neural_{name}_{suffix}"] = objectives[f"{name}_{suffix}"].to_numpy()
            result[f"neural_{name}_eligible_without_extrapolation"] = pd.NA
            result[f"neural_{name}_uses_model_extrapolation"] = pd.NA
        result["neural_c"] = result["neural_cycle_cop"]
        result["neural_h"] = result["neural_cycle_heating_rate_kw"]
        result["neural_o"] = result["neural_cycle_evaporator_capacity_kw"]
        selection = grid[[
            "candidate_defrost_time", "is_cop_heating_rate_pareto_point",
            "is_selected_pareto_point", "pareto_selection_score", "pareto_selection_method",
        ]].rename(columns={
            "is_cop_heating_rate_pareto_point": "neural_is_pareto",
            "is_selected_pareto_point": "neural_is_knee",
            "pareto_selection_score": "neural_pareto_selection_score",
            "pareto_selection_method": "neural_pareto_selection_method",
        })
        result = result.merge(selection, on="candidate_defrost_time", how="left")
        result["neural_is_pareto"] = result.neural_is_pareto.eq(True)
        result["neural_is_knee"] = result.neural_is_knee.eq(True)
        selected = grid.loc[grid.is_selected_pareto_point, "candidate_defrost_time"]
        result["neural_teacher_time"] = selected.iloc[0] if len(selected) else pd.NaT
        tables.append(result)
    return pd.concat(tables, ignore_index=True)
