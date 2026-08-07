"""Scientific contracts for Dataset construction intermediates."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

_STAGES = {"recovery", "frost_development", "defrost", "partial"}
_STATUSES = {"valid", "partial", "incomplete", "invalid"}


def validate_prepared(frame: pd.DataFrame, cycle_summary: pd.DataFrame) -> None:
    _require(frame, ["experiment_id", "timestamp", "cycle_id", "cycle_stage", "cycle_progress"])
    _unique(frame, ["experiment_id", "timestamp"])
    _unique(cycle_summary, ["experiment_id", "cycle_id"])
    _validate_cycle_references(frame, cycle_summary, require_exact=True)
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
    _validate_cycle_references(frame, cycle_summary, require_exact=False)
    _validate_cycle_labels(frame, "processed")
    _validate_progress(frame)
    _validate_elapsed(frame)
    for column in frame.columns:
        if str(column).endswith("__imputed") and not is_bool_dtype(frame[column]):
            raise ValueError(f"{column} must be boolean")
    _validate_baseline_contract(frame, cycle_summary)
    source_suffixes = ("__missing", "__invalid", "__duplicate", "__conflict")
    if any(str(column).endswith(source_suffixes) for column in frame.columns):
        raise ValueError("processed data contains Prepared source-quality columns")


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
    frost_elapsed = elapsed.loc[frame["cycle_stage"].eq("frost_development")].dropna()
    if (frost_elapsed < 0).any():
        raise ValueError("cycle_elapsed_seconds must be nonnegative")


def _validate_cycle_references(
    frame: pd.DataFrame, cycle_summary: pd.DataFrame, *, require_exact: bool
) -> None:
    frame_keys = _cycle_key_set(frame)
    summary_keys = _cycle_key_set(cycle_summary)
    if require_exact:
        missing_from_summary = frame_keys - summary_keys
        summary_only = summary_keys - frame_keys
        if missing_from_summary:
            raise ValueError("Prepared and cycle_summary cycle keys must match")
        if summary_only:
            if "cycle_status" not in cycle_summary:
                raise ValueError("Prepared and cycle_summary cycle keys must match")
            summary_only_rows = cycle_summary.loc[
                cycle_summary.set_index(["experiment_id", "cycle_id"]).index.isin(
                    summary_only
                )
            ]
            if not summary_only_rows["cycle_status"].eq("incomplete").all():
                raise ValueError("Prepared and cycle_summary cycle keys must match")
        return
    if not require_exact and not frame_keys <= summary_keys:
        raise ValueError("Processed cycle keys must be present in cycle_summary")


def _cycle_key_set(frame: pd.DataFrame) -> set[tuple[object, object]]:
    columns = ["experiment_id", "cycle_id"]
    if frame[columns].isna().any().any():
        raise ValueError("cycle keys must not be null")
    return set(
        frame[columns].drop_duplicates().itertuples(index=False, name=None)
    )


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


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _unique(frame: pd.DataFrame, columns: list[str]) -> None:
    _require(frame, columns)
    if frame.duplicated(columns).any():
        raise ValueError(f"columns must be unique: {columns}")
