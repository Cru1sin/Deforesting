"""Task-local qualification for sensor variables and RGB modalities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def evaluate_task_eligibility(  # noqa: C901 - qualification dimensions stay auditable together
    processed_data: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    *,
    task: str,
    required_features: Sequence[str],
    required_targets: Sequence[str],
    required_modalities: Mapping[str, Any] | Sequence[str] | None = None,
    minimum_observed_coverage: float = 0.0,
    minimum_available_coverage: float = 0.0,
    maximum_imputed_fraction: float = 1.0,
) -> pd.DataFrame:
    """Return one qualification row per task, cycle, and required variable.

    Structural cycle status, sensor support, and RGB support are evaluated as
    separate dimensions.  A gap in one sensor therefore only affects rows for
    the variables and tasks that use that sensor.
    """

    if not 0 <= minimum_observed_coverage <= 1:
        raise ValueError("minimum_observed_coverage must be in [0, 1]")
    if not 0 <= minimum_available_coverage <= 1:
        raise ValueError("minimum_available_coverage must be in [0, 1]")
    if not 0 <= maximum_imputed_fraction <= 1:
        raise ValueError("maximum_imputed_fraction must be in [0, 1]")
    if "cycle_id" not in processed_data:
        raise ValueError("task qualification requires cycle_id")

    frame = processed_data.copy()
    summary = cycle_summary.copy()
    summary_status = _summary_status(summary)
    cycle_ids = list(dict.fromkeys(frame["cycle_id"].dropna().astype(str).tolist()))
    cycle_ids.extend(
        cycle_id for cycle_id in summary_status if cycle_id not in cycle_ids
    )
    features = _unique_strings(required_features)
    targets = _unique_strings(required_targets)
    modalities = _normalise_modalities(required_modalities)
    rows: list[dict[str, object]] = []

    for cycle_id in cycle_ids:
        group = frame.loc[frame["cycle_id"].astype(str).eq(cycle_id)]
        status = summary_status.get(cycle_id, _frame_cycle_status(group))
        for variable_role, variables in (("feature", features), ("target", targets)):
            for variable in variables:
                rows.append(
                    _variable_row(
                        group,
                        cycle_id=cycle_id,
                        task=task,
                        variable=variable,
                        variable_role=variable_role,
                        cycle_status=status,
                        minimum_observed_coverage=minimum_observed_coverage,
                        minimum_available_coverage=minimum_available_coverage,
                        maximum_imputed_fraction=maximum_imputed_fraction,
                    )
                )
        for modality, settings in modalities.items():
            if not bool(settings.get("required", False)):
                continue
            if modality == "rgb":
                roles = _string_list(settings.get("required_camera_roles", []))
                for role in roles:
                    rows.append(
                        _rgb_row(
                            group,
                            cycle_id=cycle_id,
                            task=task,
                            role=role,
                            cycle_status=status,
                        )
                    )
            elif modality == "sensor":
                rows.append(
                    _modality_row(
                        cycle_id=cycle_id,
                        task=task,
                        modality=modality,
                        cycle_status=status,
                    )
                )
            else:
                raise ValueError(f"unsupported analysis modality: {modality}")

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_result()
    required_by_cycle = result.groupby("cycle_id", sort=False)["qualified"].all()
    result["task_qualified"] = result["cycle_id"].map(required_by_cycle).fillna(False).astype(bool)
    return result.sort_values(["cycle_id", "modality", "variable"], kind="stable").reset_index(
        drop=True
    )


def _variable_row(
    group: pd.DataFrame,
    *,
    cycle_id: str,
    task: str,
    variable: str,
    variable_role: str,
    cycle_status: str,
    minimum_observed_coverage: float,
    minimum_available_coverage: float,
    maximum_imputed_fraction: float,
) -> dict[str, object]:
    if variable in group:
        values = pd.to_numeric(group[variable], errors="coerce")
    else:
        values = pd.Series(np.nan, index=group.index, dtype=float)
    available = values.notna()
    observed = _audit_flag(group, variable, "observed", available)
    imputed = _audit_flag(group, variable, "imputed", pd.Series(False, index=group.index))
    row_count = len(group)
    available_coverage = float(available.mean()) if row_count else 0.0
    observed_coverage = float(observed.mean()) if row_count else 0.0
    imputed_fraction = float(imputed.mean()) if row_count else 0.0
    qualified = True
    reason = "available"
    if cycle_status != "valid":
        qualified, reason = False, "structural_cycle_not_valid"
    elif not row_count or not available.any():
        qualified, reason = False, "missing_required_variable"
    elif observed_coverage < minimum_observed_coverage:
        qualified, reason = False, "insufficient_observed_coverage"
    elif available_coverage < minimum_available_coverage:
        qualified, reason = False, "insufficient_available_coverage"
    elif imputed_fraction > maximum_imputed_fraction:
        qualified, reason = False, "excessive_imputation"
    elif not _window_support_is_valid(group, variable):
        qualified, reason = False, "invalid_required_window"
    return {
        "task": task,
        "cycle_id": cycle_id,
        "variable": variable,
        "variable_role": variable_role,
        "modality": "sensor",
        "cycle_status": cycle_status,
        "qualified": qualified,
        "task_qualified": False,
        "reason": reason,
        "row_count": row_count,
        "observed_coverage": observed_coverage,
        "available_coverage": available_coverage,
        "imputed_fraction": imputed_fraction,
        "maximum_raw_gap_seconds": _maximum_support_gap(group, variable),
    }


def _rgb_row(
    group: pd.DataFrame,
    *,
    cycle_id: str,
    task: str,
    role: str,
    cycle_status: str,
) -> dict[str, object]:
    column = f"image_{_camera_role_key(role)}_path"
    if column in group:
        paths = group[column].astype("string").str.strip()
        present = paths.notna() & paths.ne("")
    else:
        present = pd.Series(False, index=group.index)
    qualified = cycle_status == "valid" and bool(present.any())
    reason = (
        "available"
        if qualified
        else "structural_cycle_not_valid"
        if cycle_status != "valid"
        else "missing_required_camera_role"
    )
    return {
        "task": task,
        "cycle_id": cycle_id,
        "variable": role,
        "variable_role": "modality",
        "modality": "rgb",
        "cycle_status": cycle_status,
        "qualified": qualified,
        "task_qualified": False,
        "reason": reason,
        "row_count": len(group),
        "observed_coverage": float(present.mean()) if len(group) else 0.0,
        "available_coverage": float(present.mean()) if len(group) else 0.0,
        "imputed_fraction": 0.0,
        "maximum_raw_gap_seconds": np.nan,
    }


def _modality_row(
    *, cycle_id: str, task: str, modality: str, cycle_status: str
) -> dict[str, object]:
    qualified = cycle_status == "valid"
    return {
        "task": task,
        "cycle_id": cycle_id,
        "variable": modality,
        "variable_role": "modality",
        "modality": modality,
        "cycle_status": cycle_status,
        "qualified": qualified,
        "task_qualified": False,
        "reason": "available" if qualified else "structural_cycle_not_valid",
        "row_count": 0,
        "observed_coverage": np.nan,
        "available_coverage": np.nan,
        "imputed_fraction": 0.0,
        "maximum_raw_gap_seconds": np.nan,
    }


def _summary_status(summary: pd.DataFrame) -> dict[str, str]:
    if "cycle_id" not in summary or "cycle_status" not in summary:
        return {}
    return dict(
        zip(
            summary["cycle_id"].astype(str),
            summary["cycle_status"].astype(str),
            strict=False,
        )
    )


def _frame_cycle_status(group: pd.DataFrame) -> str:
    if "cycle_status" not in group or group.empty:
        return "invalid"
    if group["cycle_status"].notna().any():
        return str(group["cycle_status"].dropna().iloc[0])
    return "invalid"


def _audit_flag(
    group: pd.DataFrame, variable: str, suffix: str, fallback: pd.Series
) -> pd.Series:
    column = f"{variable}__{suffix}"
    if column not in group:
        return fallback.astype(bool)
    return group[column].fillna(False).astype(bool)


def _window_support_is_valid(group: pd.DataFrame, variable: str) -> bool:
    columns = [
        column
        for column in group.columns
        if str(column).startswith(f"{variable}__window_") and str(column).endswith("__valid")
    ]
    return not columns or bool(group[columns].fillna(False).astype(bool).any(axis=1).any())


def _maximum_support_gap(group: pd.DataFrame, variable: str) -> float:
    columns = [
        column
        for column in group.columns
        if str(column).startswith(f"{variable}__window_")
        and str(column).endswith("__maximum_raw_gap_s")
    ]
    parts = [pd.to_numeric(group[column], errors="coerce") for column in columns]
    values = pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=float)
    return float(values.max()) if not values.empty else np.nan


def _normalise_modalities(
    value: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {"sensor": {"required": True}}
    if isinstance(value, Mapping):
        return {
            str(name): dict(settings)
            for name, settings in value.items()
            if isinstance(settings, Mapping)
        }
    return {str(name): {"required": True} for name in value}


def _camera_role_key(role: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", role.strip()).strip("_").lower()
    return safe or f"role_{hashlib.sha1(role.encode('utf-8')).hexdigest()[:8]}"


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("required_camera_roles must be a list")
    return _unique_strings([str(item) for item in value])


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "task",
            "cycle_id",
            "variable",
            "variable_role",
            "modality",
            "cycle_status",
            "qualified",
            "task_qualified",
            "reason",
            "row_count",
            "observed_coverage",
            "available_coverage",
            "imputed_fraction",
            "maximum_raw_gap_seconds",
        ]
    )
