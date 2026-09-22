"""Measured sensor/RGB inputs and experiment-isolated effective COP references."""

import json
import os

import numpy as np
import pandas as pd

from dataset_tools import DatasetLoader
from defrost_decision.performance_objectives import calculate_cycle_cop
from defrost_event_models.ridge_models import (
    DYNAMIC_STATE_8,
    OUTCOME_TARGETS,
    fit_model_on_all_experiments,
    model_to_parameters,
    predict_with_model_parameters,
    select_valid_events_for_quantity,
)
from image_models.sensor_features import CURRENT_SENSORS

SENSORS = tuple(c for c in CURRENT_SENSORS if c != "environment_relative_humidity")
STATISTICS = ("current", "mean", "std", "skew", "kurt", "slope", "entropy")
RGB = [f"dinov2_{i:03d}" for i in range(384)]


def reliable_rgb_cohort(dataset, *, source=None, include_rb=False, require_complete_peak=False):
    """Return cycles valid now, before COP filtering, and for front RGB."""
    loader = dataset if isinstance(dataset, DatasetLoader) else DatasetLoader(dataset)
    cycles = loader.list_valid_cycles(include_rb=include_rb)
    if require_complete_peak:
        complete = cycles.cop_peak_quality.map(lambda x: isinstance(x, dict) and x.get("complete") is True)
        cycles = cycles.loc[complete]
    front_valid = cycles.get("front_rgb_valid", pd.Series(index=cycles.index, dtype=object))
    if front_valid.isna().any():
        metadata = loader.load_image_metadata()
        available = set(metadata.loc[metadata.camera_role.eq("front"), "cycle_name"])
        audited = cycles.get("rgb_frost_status", pd.Series(index=cycles.index, dtype=object)).eq("valid")
        front_valid = front_valid.where(front_valid.notna(), cycles.cycle_name.isin(available) & audited)
    cohort = (
        cycles.loc[
            front_valid.eq(True),
            [
                "cycle_name",
                "experiment_id",
            ],
        ]
        .sort_values(["experiment_id", "cycle_name"], kind="stable")
        .reset_index(drop=True)
    )
    if source is None:
        return cohort
    return cohort.merge(source, on=["cycle_name", "experiment_id"], how="inner")


def processing_rows(rows):
    """Select the same fixed 30-second clock used for online replay."""
    first = rows.groupby("cycle_name").candidate_defrost_time.transform("min")
    return (rows.candidate_defrost_time - first).dt.total_seconds().mod(30).eq(0)


def normalize_curve(curve):
    result = curve.copy()
    supported = result.cycle_cop.where(result.cycle_cop_eligible & np.isfinite(result.cycle_cop))
    maximum = supported.max()
    result["reference_max"] = maximum if maximum > 0 else np.nan
    result["target"] = supported / result.reference_max
    result["relative_reference"] = result.cycle_cop / result.reference_max
    return result


def ridge_parameters(events, excluded):
    selected = select_valid_events_for_quantity(events, "event_electricity")
    selected = selected.loc[~selected.experiment_id.isin(excluded)]
    if selected.experiment_id.nunique() < 3:
        raise ValueError(f"insufficient Ridge experiments after exclusions: {sorted(excluded)}")
    return model_to_parameters(
        fit_model_on_all_experiments(
            selected, DYNAMIC_STATE_8, OUTCOME_TARGETS["event_electricity"]
        )
    )


def apply_reference(base, parameters, rgb="on", *, allow_model_extrapolation=False):
    result = base.copy()
    prediction = predict_with_model_parameters(parameters, result)
    result["defrost_event_electricity_kwh"] = prediction.prediction.to_numpy()
    result["defrost_event_electricity_prediction_available"] = np.isfinite(prediction.prediction)
    result["defrost_event_electricity_in_training_domain"] = prediction.support_distance.le(
        prediction.support_threshold
    ).to_numpy()
    result["defrost_event_electricity_support_distance"] = prediction.support_distance.to_numpy()
    result["defrost_event_electricity_support_threshold"] = prediction.support_threshold.to_numpy()
    result["defrost_event_net_heat_kwh"] = 0.0
    result["defrost_event_net_heat_prediction_available"] = True
    result["defrost_event_net_heat_in_training_domain"] = True
    result = calculate_cycle_cop(
        result, effective=True, allow_model_extrapolation=allow_model_extrapolation
    )
    result = normalize_curve(result)
    result["input_available"] = result.sensor_timestamp.notna() & (
        result.rgb_available if rgb == "on" else True
    )
    return result


def build_fold_rows(
    args,
    cohort,
    events,
    excluded,
):
    """Cross-fit every training experiment; excluded experiments never enter any Ridge."""
    tables, parameters = [], {}
    for experiment, cycles in cohort.groupby("experiment_id", sort=True):
        omit = frozenset((*excluded, experiment))
        key = "__".join(sorted(omit))
        model_path = args.output / "ridge" / f"{key or 'full'}.json"
        if model_path.exists():
            model = json.loads(model_path.read_text())
        else:
            model = ridge_parameters(events, omit)
            # Independent workers can produce the same deterministic fit; retain one file.
            temporary = model_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(model))
            temporary.replace(model_path)
        assert not set(model["training_experiment_ids"]) & omit
        parameters[key] = model
        for name in cycles.cycle_name:
            root = args.output / "base"
            base = pd.read_parquet(root / f"{name}.parquet")
            tables.append(
                apply_reference(
                    base,
                    model,
                    "on" if getattr(args, "require_rgb_input", False) else args.rgb,
                    allow_model_extrapolation=getattr(args, "allow_extrapolation", False),
                )
            )
    return pd.concat(tables, ignore_index=True), parameters
