"""Structural validators for Prepared, Processed, and Analysis contracts."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

_STAGES = {"recovery", "frost_development", "defrost", "partial"}
_STATUSES = {"valid", "incomplete", "invalid"}
_DECISIONS = {
    "trend_supported_candidate",
    "partial_evidence",
    "insufficient_coverage",
}
_ANALYSIS_FIELDS = {
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
    "reset_evidence_status",
    "reset_evidence_reason",
    "future_performance_association",
    "median_max_abs_context_spearman",
    "decision",
    "reason",
}


def validate_prepared(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    _require(frame, ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_progress"])
    _unique(frame, ["experiment_id", "timestamp"])
    _unique(cycle_summary, ["experiment_id", "cycle_id"])
    _validate_cycle_labels(frame, "prepared")
    _validate_progress(frame)
    _validate_elapsed(frame)
    forbidden_tokens = ("__baseline", "__rolling", "__slope", "__delta_", "__lag_", "__imputed")
    if any(any(token in str(column) for token in forbidden_tokens) for column in frame.columns):
        raise ValueError("prepared data contains Process-only columns")


def validate_processed(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    _require(frame, ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_progress"])
    _unique(frame, ["experiment_id", "timestamp"])
    _unique(cycle_summary, ["experiment_id", "cycle_id"])
    _validate_cycle_labels(frame, "processed")
    if frame["cycle_stage"].eq("partial").any():
        raise ValueError("processed data must not contain partial rows")
    _validate_progress(frame)
    _validate_elapsed(frame)
    for column in frame.columns:
        if str(column).endswith("__imputed") and not is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be boolean")
    _validate_baseline_contract(frame, cycle_summary)
    source_suffixes = ("__missing", "__invalid", "__duplicate", "__conflict")
    if any(str(column).endswith(source_suffixes) for column in frame.columns):
        raise ValueError("processed data contains Prepared source-quality columns")


def validate_analysis(evidence: pd.DataFrame) -> None:
    required = sorted(_ANALYSIS_FIELDS)
    _require(evidence, required)
    _unique(evidence, ["experiment_id", "channel"])
    if not evidence.empty:
        _validate_counts(evidence)
        if not evidence["decision"].isin(_DECISIONS).all():
            raise ValueError("analysis contains an invalid decision")
        if not evidence["reset_evidence_status"].eq("not_evaluated").all():
            raise ValueError("reset evidence must be not_evaluated")
        if not evidence["reset_evidence_reason"].eq(
            "independent_reference_unavailable"
        ).all():
            raise ValueError("reset evidence reason is not fixed")
        if not evidence["reset_pair_count"].eq(0).all() or evidence["reset_effect"].notna().any():
            raise ValueError("reset evidence must be empty")
    forbidden = {"rank", "weighted_score", "candidate_score"}
    present = sorted(forbidden.intersection(str(column) for column in evidence.columns))
    if present:
        readable = ", ".join(value.replace("_", " ") for value in present)
        raise ValueError(f"analysis contains forbidden ranking fields: {readable}")


def _validate_cycle_labels(frame: pd.DataFrame, name: str) -> None:
    if not frame["cycle_stage"].dropna().isin(_STAGES).all():
        raise ValueError(f"{name} cycle_stage contains an invalid value")
    if "cycle_status" in frame and not frame["cycle_status"].dropna().isin(_STATUSES).all():
        raise ValueError(f"{name} cycle_status contains an invalid value")


def _validate_progress(frame: pd.DataFrame) -> None:
    progress = pd.to_numeric(frame["cycle_progress"], errors="coerce")
    finite = progress.notna()
    if not progress.loc[finite].between(0, 1).all():
        raise ValueError("cycle_progress must be within [0, 1]")
    if progress.loc[frame["cycle_stage"].ne("frost_development")].notna().any():
        raise ValueError("cycle_progress is only defined during frost_development")


def _validate_elapsed(frame: pd.DataFrame) -> None:
    if "cycle_elapsed_seconds" not in frame:
        return
    elapsed = pd.to_numeric(frame["cycle_elapsed_seconds"], errors="coerce")
    if elapsed.loc[frame["cycle_stage"].ne("frost_development")].notna().any():
        raise ValueError("cycle_elapsed_seconds is only defined during frost_development")


def _validate_baseline_contract(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    _require(summary, ["baseline_status", "baseline_failure_reason"])
    for _, cycle in summary.iterrows():
        mask = frame["experiment_id"].eq(cycle["experiment_id"]) & frame["cycle_id"].eq(
            cycle["cycle_id"]
        )
        baseline_columns = [
            column for column in frame.columns if str(column).endswith("__baseline")
        ]
        residual_columns = [
            column for column in frame.columns if str(column).endswith("__baseline_residual")
        ]
        if cycle["baseline_status"] != "available":
            if any(
                frame.loc[mask, column].notna().any()
                for column in [*baseline_columns, *residual_columns]
            ):
                raise ValueError("unavailable baseline must leave baseline values NaN")
        elif pd.isna(cycle.get("baseline_start")) or pd.isna(cycle.get("baseline_end")):
            raise ValueError("available baseline must have a window")


def _validate_counts(evidence: pd.DataFrame) -> None:
    columns = [
        "trend_cycle_count",
        "reset_pair_count",
        "future_cycle_count",
        "context_cycle_count",
    ]
    for column in columns:
        values = evidence[column]
        if not is_integer_dtype(values) or values.isna().any() or (values < 0).any():
            raise ValueError(f"{column} must be nonnegative integers")


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _unique(frame: pd.DataFrame, columns: list[str]) -> None:
    _require(frame, columns)
    if frame.duplicated(columns).any():
        raise ValueError(f"columns must be unique: {columns}")
