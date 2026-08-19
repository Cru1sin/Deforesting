"""Loader streaming orchestration and Dataset cohort construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..dataset_loader import DatasetLoader
from .contracts import (
    CYCLE_ELIGIBILITY_COLUMNS,
    FEATURE_CYCLE_METRIC_COLUMNS,
    FUTURE_ASSOCIATION_COLUMNS,
    TARGET_AUDIT_COLUMNS,
    TARGET_DEGRADATION_DIRECTION,
    EvidenceBundle,
)
from .metrics import feature_cycle_rows, future_association_rows, pair_cycle_values
from .readiness import (
    audit_performance_target,
    compare_incremental_models,
    summarize_readiness,
)
from .settings import EvidenceSettings
from .summary import feature_pair_similarity, feature_profile, future_horizon_summary


def build_evidence(loader: DatasetLoader, settings: EvidenceSettings) -> EvidenceBundle:
    """Build Evidence tables by streaming valid cycles from ``loader``."""
    features = candidate_features(loader.registry, settings)
    cycles = loader.list_cycles()
    cycle_rows = _cycle_metadata_rows(cycles)
    eligibility = _eligibility_table(cycle_rows, settings)
    eligible_names = set(
        eligibility.loc[eligibility["eligible"], "cycle_name"].astype(str)
    )

    feature_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    pair_inputs: list[tuple[str, str, dict[str, dict[float, float]]]] = []
    readiness_inputs: list[tuple[Mapping[str, object], pd.DataFrame]] = []
    for record, frame in loader.iter_cycle_frames(statuses=set(settings.eligible_statuses)):
        if str(record.get("cycle_name", "")) not in eligible_names:
            continue
        readiness_inputs.append((record, frame))
        metadata = _record_metadata(record)
        feature_rows.extend(feature_cycle_rows(frame, metadata, features, settings))
        future_rows.extend(future_association_rows(frame, metadata, features, settings))
        pair_inputs.append(
            (
                str(metadata["cycle_name"]),
                str(metadata["experiment_date"]),
                pair_cycle_values(frame, features),
            )
        )

    feature_metrics = _frame(feature_rows, FEATURE_CYCLE_METRIC_COLUMNS)
    future_association = _frame(future_rows, FUTURE_ASSOCIATION_COLUMNS)
    future_summary = future_horizon_summary(future_association, features, settings)
    profile = feature_profile(feature_metrics, future_summary, features, settings)
    pair_similarity = feature_pair_similarity(pair_inputs, features, settings)
    target_audit = _frame(
        [
            audit_performance_target(record, frame, target, settings)
            for record, frame in readiness_inputs
            for target in settings.targets
        ],
        TARGET_AUDIT_COLUMNS,
    )
    readiness_split = compare_incremental_models(
        readiness_inputs, features, target_audit, settings
    )
    readiness_summary = summarize_readiness(
        readiness_split, target_audit, feature_metrics, features, settings
    )
    return EvidenceBundle(
        cycle_eligibility=eligibility,
        feature_cycle_metrics=feature_metrics,
        future_association=future_association,
        future_horizon_summary=future_summary,
        feature_profile=profile,
        feature_pair_similarity=pair_similarity,
        target_audit=target_audit,
        readiness_split=readiness_split,
        readiness_summary=readiness_summary,
    )


def candidate_features(
    registry: Mapping[str, object], settings: EvidenceSettings
) -> list[tuple[str, str]]:
    unknown_targets = [
        target for target in settings.targets if target not in TARGET_DEGRADATION_DIRECTION
    ]
    if unknown_targets:
        raise ValueError(f"unknown target: {unknown_targets[0]}")
    raw_channels = registry.get("channels")
    if not isinstance(raw_channels, Mapping):
        raise ValueError("Dataset registry channels must be a mapping")
    features: list[tuple[str, str]] = []
    for raw_name, raw_settings in raw_channels.items():
        if not isinstance(raw_settings, Mapping):
            raise ValueError(f"Dataset registry channel is not a mapping: {raw_name}")
        if not bool(raw_settings.get("analysis_candidate", False)):
            continue
        direction = str(raw_settings.get("expected_frost_direction", ""))
        if direction not in {"increase", "decrease"}:
            raise ValueError(
                f"candidate channel {raw_name} has invalid expected_frost_direction"
            )
        name = str(raw_name)
        if name in settings.targets or str(raw_settings.get("role", "")) == "performance":
            raise ValueError(f"candidate channel {name} conflicts with target/performance")
        features.append((name, direction))
    return features


def _cycle_metadata_rows(cycles: pd.DataFrame) -> list[dict[str, object]]:
    if cycles.empty:
        return []
    return [
        _record_metadata({str(key): value for key, value in row.items()})
        for row in cycles.to_dict(orient="records")
    ]


def _record_metadata(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "cycle_name": _text(record.get("cycle_name")),
        "cycle_uid": _text(record.get("cycle_uid")),
        "experiment_id": _text(record.get("experiment_id")),
        "experiment_date": _date_text(record.get("experiment_date")),
        "status": _text(record.get("status")),
        "analysis_duration_minutes": _analysis_duration_minutes(record),
    }


def _eligibility_table(
    rows: list[dict[str, object]], settings: EvidenceSettings
) -> pd.DataFrame:
    result: list[dict[str, object]] = []
    for row in rows:
        duration = float(
            pd.to_numeric(pd.Series([row["analysis_duration_minutes"]]), errors="coerce").iloc[0]
        )
        status_eligible = row["status"] in settings.eligible_statuses
        duration_eligible = np.isfinite(duration) and duration > settings.minimum_cycle_minutes
        if not status_eligible:
            reason = "cycle_status_not_eligible"
        elif not np.isfinite(duration):
            reason = "cycle_duration_unavailable"
        elif not duration_eligible:
            reason = "cycle_duration_not_over_minimum"
        else:
            reason = ""
        result.append(
            {
                **row,
                "eligible": status_eligible and duration_eligible,
                "exclusion_reason": reason,
            }
        )
    return _frame(result, CYCLE_ELIGIBILITY_COLUMNS)


def _analysis_duration_minutes(record: Mapping[str, object]) -> float:
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, Mapping):
        return np.nan
    start_value = boundaries.get("stable_heating_start")
    end_value = boundaries.get("defrost_start") or boundaries.get("end_time")
    start = (
        pd.to_datetime(str(start_value), errors="coerce")
        if start_value is not None
        else pd.NaT
    )
    end = pd.to_datetime(str(end_value), errors="coerce") if end_value is not None else pd.NaT
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return float((end - start).total_seconds() / 60.0)


def _frame(rows: list[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _date_text(value: object) -> str:
    return _text(value)[:10]
