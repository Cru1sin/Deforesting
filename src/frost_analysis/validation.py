"""Structural checks for the three public pipeline data contracts."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

_STAGES = {"recovery", "frost_development", "defrost", "partial"}
_STATUSES = {"valid", "incomplete", "invalid"}


def validate_prepared(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    _require(frame, ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_progress"])
    _unique(frame, ["experiment_id", "timestamp"])
    _unique(cycle_summary, ["experiment_id", "cycle_id"])
    if not frame["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError("prepared cycle_stage contains an invalid value")
    if "cycle_status" in frame and not frame["cycle_status"].dropna().isin(_STATUSES).all():
        raise ValueError("prepared cycle_status contains an invalid value")
    _validate_progress(frame)
    forbidden = ("baseline", "rolling", "slope", "__imputed")
    if any(any(token in str(column) for token in forbidden) for column in frame.columns):
        raise ValueError("prepared data contains Process-only columns")


def validate_processed(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    _require(frame, ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_progress"])
    _unique(frame, ["experiment_id", "timestamp"])
    _unique(cycle_summary, ["experiment_id", "cycle_id"])
    if not frame["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError("processed cycle_stage contains an invalid value")
    if "cycle_status" in frame and not frame["cycle_status"].dropna().isin(_STATUSES).all():
        raise ValueError("processed cycle_status contains an invalid value")
    _validate_progress(frame)
    for column in frame.columns:
        if str(column).endswith("__imputed") and not is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be boolean")
        if str(column).endswith("__baseline_status"):
            value_column = str(column).removesuffix("__baseline_status") + "__baseline"
            accepted = frame[column].eq("accepted")
            if (
                value_column in frame
                and accepted.any()
                and frame.loc[accepted, value_column].isna().any()
            ):
                raise ValueError(f"accepted baseline has no value: {column}")
    source_suffixes = ("__missing", "__invalid", "__duplicate", "__conflict")
    if any(str(column).endswith(source_suffixes) for column in frame.columns):
        raise ValueError("processed data contains Prepared source-quality columns")


def validate_analysis(evidence: pd.DataFrame) -> None:
    required = [
        "experiment_id",
        "experiment_date",
        "channel",
        "trend_cycle_count",
        "reset_pair_count",
        "future_cycle_count",
        "context_cycle_count",
        "trend_effect",
        "direction_consistency",
        "reset_effect",
        "future_performance_effect",
        "max_abs_context_spearman",
        "decision",
        "reason",
    ]
    _require(evidence, required)
    _unique(evidence, ["experiment_id", "channel"])
    forbidden = {"rank", "weighted_score", "candidate_score"}
    present = sorted(forbidden.intersection(str(column) for column in evidence.columns))
    if present:
        readable = ", ".join(value.replace("_", " ") for value in present)
        raise ValueError(f"analysis contains forbidden ranking fields: {readable}")


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _unique(frame: pd.DataFrame, columns: list[str]) -> None:
    _require(frame, columns)
    if frame.duplicated(columns).any():
        raise ValueError(f"columns must be unique: {columns}")


def _validate_progress(frame: pd.DataFrame) -> None:
    progress = pd.to_numeric(frame["cycle_progress"], errors="coerce").dropna()
    if not progress.between(0, 1).all():
        raise ValueError("cycle_progress must be within [0, 1]")
