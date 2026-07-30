"""Unified, task-agnostic feature registry and physical projections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    canonical_name: str
    raw_source: str | None
    meaning_zh: str
    physical_family: str
    source_type: str
    unit: str
    formula: str
    data_role: str
    availability: str
    deployment_status: str
    confidence: str
    primary_or_validation: str
    analysis_enabled: bool
    notes: str
    data_kind: str = "continuous"
    missing_policy: str = "none"
    resample_method: str = "mean"
    required_for_sensor_quality: bool = False


@dataclass(frozen=True)
class RegistryResult:
    frame: pd.DataFrame
    specs: dict[str, FeatureSpec]
    metadata: pd.DataFrame


# Derived channels are always rebuilt from the current source values.  Keeping
# the dependency graph here makes the rule reusable by both Prepare and
# Process, without teaching missing-data code the names of scientific targets.
DERIVED_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "cop": ("heating_capacity", "power_total"),
    "water_heating_capacity": (
        "water_flow",
        "water_out_temperature",
        "water_in_temperature",
    ),
    "water_cop": ("water_heating_capacity", "power_total"),
    "superheat_calculated": ("suction_temperature", "evaporating_temperature"),
    "pressure_ratio": ("condensing_pressure", "evaporating_pressure"),
    "controller_pressure_ratio_expected": ("pressure_ratio",),
    "controller_pressure_ratio_residual": (
        "controller_pressure_ratio",
        "controller_pressure_ratio_expected",
    ),
    "water_delta_temperature": ("water_out_temperature", "water_in_temperature"),
    "ambient_evaporating_delta": ("ambient_temperature", "evaporating_temperature"),
    "temperature_lift": ("condensing_temperature", "evaporating_temperature"),
    "pressure_lift": ("condensing_pressure", "evaporating_pressure"),
    "controller_pressure_ratio_check": ("controller_pressure_ratio",),
}

DERIVED_FORMULAS: dict[str, str] = {
    "cop": "heating_capacity_div_power_total",
    "water_heating_capacity": "water_flow_times_water_delta_temperature",
    "water_cop": "water_heating_capacity_div_power_total",
    "superheat_calculated": "suction_temperature_minus_evaporating_temperature",
    "pressure_ratio": "condensing_pressure_div_evaporating_pressure",
    "controller_pressure_ratio_expected": "100_times_pressure_ratio",
    "controller_pressure_ratio_residual": (
        "controller_pressure_ratio_minus_controller_pressure_ratio_expected"
    ),
    "water_delta_temperature": "water_out_temperature_minus_water_in_temperature",
    "ambient_evaporating_delta": "ambient_temperature_minus_evaporating_temperature",
    "temperature_lift": "condensing_temperature_minus_evaporating_temperature",
    "pressure_lift": "condensing_pressure_minus_evaporating_pressure",
    "controller_pressure_ratio_check": "controller_pressure_ratio",
}


def load_feature_registry(path: Path) -> dict[str, FeatureSpec]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = loaded.get("features")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature registry must contain a non-empty features list")
    result: dict[str, FeatureSpec] = {}
    allowed_roles = {"X", "C", "M"}
    allowed_data_kinds = {"continuous", "control", "event", "derived"}
    allowed_missing_policies = {"linear", "forward_fill", "none"}
    allowed_resample_methods = {"mean", "last", "none"}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("each feature registry item must be a mapping")
        item = {str(key): value for key, value in raw.items()}
        feature_id = str(item.get("feature_id", "")).strip()
        if not feature_id or feature_id in result:
            raise ValueError(f"invalid or duplicate feature_id: {feature_id}")
        role = str(item.get("data_role", "X"))
        if role not in allowed_roles:
            raise ValueError(f"invalid data_role for {feature_id}: {role}")
        source_type = str(item.get("source_type", "measured"))
        derived_by_shape = item.get("raw_source") is None or source_type == "derived"
        data_kind = str(
            item.get(
                "data_kind",
                "derived" if derived_by_shape else "control" if role == "M" else "continuous",
            )
        )
        missing_policy = str(
            item.get(
                "missing_policy",
                "none"
                if data_kind == "derived"
                else "forward_fill"
                if role == "M"
                else "linear",
            )
        )
        resample_method = str(
            item.get(
                "resample_method",
                "none" if data_kind == "derived" else "last" if role == "M" else "mean",
            )
        )
        if data_kind not in allowed_data_kinds:
            raise ValueError(f"invalid data_kind for {feature_id}: {data_kind}")
        if missing_policy not in allowed_missing_policies:
            raise ValueError(f"invalid missing_policy for {feature_id}: {missing_policy}")
        if resample_method not in allowed_resample_methods:
            raise ValueError(f"invalid resample_method for {feature_id}: {resample_method}")
        result[feature_id] = FeatureSpec(
            feature_id=feature_id,
            canonical_name=str(item.get("canonical_name", feature_id)),
            raw_source=None if item.get("raw_source") is None else str(item["raw_source"]),
            meaning_zh=str(item.get("meaning_zh", "")),
            physical_family=str(item.get("physical_family", "unclassified")),
            source_type=source_type,
            unit=str(item.get("unit", "unknown")),
            formula=str(item.get("formula", "")),
            data_role=role,
            availability=str(item.get("availability", "current_history")),
            deployment_status=str(item.get("deployment_status", "pending")),
            confidence=str(item.get("confidence", "pending")),
            primary_or_validation=str(item.get("primary_or_validation", "primary")),
            analysis_enabled=bool(item.get("analysis_enabled", True)),
            notes=str(item.get("notes", "")),
            data_kind=data_kind,
            missing_policy=missing_policy,
            resample_method=resample_method,
            required_for_sensor_quality=bool(item.get("required_for_sensor_quality", False)),
        )
    return result


def apply_feature_registry(
    frame: pd.DataFrame,
    specs: dict[str, FeatureSpec],
    *,
    heating_mode_value: float = 3,
) -> RegistryResult:
    """Project raw columns into registered fields and derived relationships.

    ``result`` keeps the source rows and receives one canonical column for each
    Registry specification. ``operating_mode`` is the numeric source code;
    ``is_heating`` is the separate nullable predicate used by cycle checks.
    """
    source_frame = frame.copy()
    projected: dict[str, pd.Series] = {}
    for spec in specs.values():
        if spec.raw_source is None:
            continue
        if spec.raw_source not in source_frame:
            # Missing source columns remain explicit NaN fields for auditability.
            projected[spec.canonical_name] = pd.Series(np.nan, index=source_frame.index)
            projected[f"{spec.canonical_name}__missing"] = pd.Series(
                pd.NA, index=source_frame.index, dtype="boolean"
            )
            projected[f"{spec.canonical_name}__invalid"] = pd.Series(
                pd.NA, index=source_frame.index, dtype="boolean"
            )
            projected[f"{spec.canonical_name}__source_state"] = pd.Series(
                "not_sampled", index=source_frame.index, dtype="string"
            )
            if spec.canonical_name == "operating_mode":
                projected["is_heating"] = pd.Series(
                    pd.NA, index=source_frame.index, dtype="boolean"
                )
            continue
        value = source_frame[spec.raw_source]
        for suffix in ("__missing", "__invalid", "__source_state"):
            source_column = f"{spec.raw_source}{suffix}"
            if source_column in source_frame:
                projected[f"{spec.canonical_name}{suffix}"] = source_frame[source_column]
        state_column = f"{spec.canonical_name}__source_state"
        if state_column not in projected:
            state = pd.Series("observed", index=source_frame.index, dtype="string")
            state.loc[value.isna()] = "missing"
            projected[state_column] = state
        missing_column = f"{spec.canonical_name}__missing"
        if missing_column not in projected:
            projected[missing_column] = value.isna().astype("boolean")
        invalid_column = f"{spec.canonical_name}__invalid"
        if invalid_column not in projected:
            projected[invalid_column] = pd.Series(
                False, index=source_frame.index, dtype="boolean"
            )
        values = pd.to_numeric(value, errors="coerce")
        if spec.formula == "scale_0.01":
            values = values * 0.01
        elif spec.formula == "scale_0.001":
            values = values * 0.001
        if spec.canonical_name == "operating_mode":
            # Preserve code 3 as operating_mode; derive the nullable boolean separately.
            projected[spec.canonical_name] = values
            projected["is_heating"] = values.eq(heating_mode_value).astype("boolean").where(
                values.notna()
            )
        elif spec.canonical_name == "defrost_flag":
            projected[spec.canonical_name] = value
        else:
            projected[spec.canonical_name] = values
    result = pd.concat(
        [source_frame.drop(columns=list(projected), errors="ignore"), pd.DataFrame(projected)],
        axis=1,
    )
    _derive(result)
    metadata = _metadata(result, specs)
    return RegistryResult(result, specs, metadata)


def _copy_source_state(frame: pd.DataFrame, source: str, canonical: str) -> None:
    """Project loader provenance columns from a raw name to a Registry name."""
    value = frame[source]
    for suffix in ("__missing", "__invalid", "__source_state"):
        source_column = f"{source}{suffix}"
        target_column = f"{canonical}{suffix}"
        if source_column in frame:
            frame[target_column] = frame[source_column]
    if f"{canonical}__source_state" not in frame:
        state = pd.Series("observed", index=frame.index, dtype="string")
        state.loc[value.isna()] = "missing"
        frame[f"{canonical}__source_state"] = state
    if f"{canonical}__missing" not in frame:
        frame[f"{canonical}__missing"] = value.isna()
    if f"{canonical}__invalid" not in frame:
        frame[f"{canonical}__invalid"] = pd.Series(False, index=frame.index, dtype="boolean")


def derived_feature_names(specs: Mapping[str, FeatureSpec] | None = None) -> set[str]:
    """Return derived names known to the Registry and optional custom specs."""
    names = set(DERIVED_FORMULAS)
    if specs is not None:
        names.update(
            spec.canonical_name for spec in specs.values() if spec.data_kind == "derived"
        )
    return names


def recompute_derived_features(
    frame: pd.DataFrame,
    specs: Mapping[str, FeatureSpec] | None = None,
) -> pd.DataFrame:
    """Rebuild derived channels from the current, already-processed sources.

    Derived values intentionally have no interpolation path.  Their observed
    and imputed flags are propagated from their source channels so downstream
    windows can distinguish measured support from reconstructed support.
    """

    result = frame.copy()

    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(result.get(name, pd.Series(np.nan, index=result.index)), errors="coerce")

    def safe_dividend(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.where(denominator.abs().gt(1e-12))

    values: dict[str, pd.Series] = {
        "cop": safe_dividend(numeric("heating_capacity"), numeric("power_total")),
        "water_delta_temperature": numeric("water_out_temperature")
        - numeric("water_in_temperature"),
        "water_heating_capacity": 1.16278
        * numeric("water_flow")
        * (numeric("water_out_temperature") - numeric("water_in_temperature")),
        "superheat_calculated": numeric("suction_temperature")
        - numeric("evaporating_temperature"),
        "pressure_ratio": safe_dividend(
            numeric("condensing_pressure"), numeric("evaporating_pressure")
        ),
        "ambient_evaporating_delta": numeric("ambient_temperature")
        - numeric("evaporating_temperature"),
        "temperature_lift": numeric("condensing_temperature")
        - numeric("evaporating_temperature"),
        "pressure_lift": numeric("condensing_pressure") - numeric("evaporating_pressure"),
        "controller_pressure_ratio_check": numeric("controller_pressure_ratio"),
    }
    values["water_cop"] = safe_dividend(values["water_heating_capacity"], numeric("power_total"))
    values["controller_pressure_ratio_expected"] = 100.0 * values["pressure_ratio"]
    values["controller_pressure_ratio_residual"] = numeric("controller_pressure_ratio") - values[
        "controller_pressure_ratio_expected"
    ]

    names = [name for name in DERIVED_FORMULAS if name in derived_feature_names(specs)]
    if specs is not None:
        names.extend(
            spec.canonical_name
            for spec in specs.values()
            if spec.data_kind == "derived" and spec.canonical_name not in names
        )
    derived_columns: dict[str, pd.Series] = {}
    observed_by_name: dict[str, pd.Series] = {}
    imputed_by_name: dict[str, pd.Series] = {}
    for name in names:
        if name not in values:
            continue
        value = values[name]
        dependencies = DERIVED_DEPENDENCIES.get(name, ())
        observed_parts = [
            observed_by_name[dependency]
            if dependency in observed_by_name
            else result[f"{dependency}__observed"].astype(bool)
            if f"{dependency}__observed" in result
            else numeric(dependency).notna()
            for dependency in dependencies
        ]
        imputed_parts = [
            imputed_by_name[dependency]
            if dependency in imputed_by_name
            else result[f"{dependency}__imputed"].astype(bool)
            if f"{dependency}__imputed" in result
            else pd.Series(False, index=result.index)
            for dependency in dependencies
        ]
        if observed_parts:
            observed = pd.concat(observed_parts, axis=1).all(axis=1)
            imputed = pd.concat(imputed_parts, axis=1).any(axis=1) & value.notna()
        else:
            observed = value.notna()
            imputed = pd.Series(False, index=result.index)
        observed = observed.astype(bool)
        imputed = imputed.astype(bool)
        observed_by_name[name] = observed
        imputed_by_name[name] = imputed
        derived_columns[name] = value
        derived_columns[f"{name}__observed"] = observed
        derived_columns[f"{name}__imputed"] = imputed
        derived_columns[f"{name}__missing"] = value.isna().astype("boolean")
        derived_columns[f"{name}__invalid"] = pd.Series(
            False, index=result.index, dtype="boolean"
        )
        state = pd.Series("derived", index=result.index, dtype="string")
        derived_columns[f"{name}__source_state"] = state.where(value.notna(), "missing")
    if not derived_columns:
        return result
    return pd.concat(
        [result.drop(columns=list(derived_columns), errors="ignore"), pd.DataFrame(derived_columns)],
        axis=1,
    )


def _derive(frame: pd.DataFrame) -> None:
    derived = recompute_derived_features(frame)
    columns = [
        column
        for name in derived_feature_names()
        for column in (
            name,
            f"{name}__observed",
            f"{name}__imputed",
            f"{name}__missing",
            f"{name}__invalid",
            f"{name}__source_state",
        )
        if column in derived
    ]
    frame.drop(columns=columns, errors="ignore", inplace=True)
    frame[columns] = derived[columns]


def _metadata(frame: pd.DataFrame, specs: dict[str, FeatureSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    derived = {
        "cop": "heating_capacity / power_total",
        "water_heating_capacity": "1.16278 * water_flow * water_delta_temperature",
        "water_cop": "water_heating_capacity / power_total",
        "superheat_calculated": "suction_temperature - evaporating_temperature",
        "pressure_ratio": "condensing_pressure / evaporating_pressure",
        "controller_pressure_ratio_expected": "100 * pressure_ratio",
        "controller_pressure_ratio_residual": (
            "controller_pressure_ratio - controller_pressure_ratio_expected"
        ),
        "water_delta_temperature": "water_out_temperature - water_in_temperature",
        "ambient_evaporating_delta": "ambient_temperature - evaporating_temperature",
        "temperature_lift": "condensing_temperature - evaporating_temperature",
        "pressure_lift": "condensing_pressure - evaporating_pressure",
    }
    for feature_id, spec in specs.items():
        source = (
            frame[spec.canonical_name]
            if spec.canonical_name in frame
            else pd.Series(np.nan, index=frame.index)
        )
        # Event/state fields are often exported as ON/OFF or numeric codes.
        # Their registry coverage must not be measured by coercing strings to
        # numeric values, otherwise a usable event channel appears empty.
        present = source.notna() & source.astype("string").str.strip().ne("")
        values = pd.to_numeric(source, errors="coerce")
        rows.append(
            {
                "feature_id": feature_id,
                "canonical_name": spec.canonical_name,
                "raw_source": spec.raw_source or "",
                "meaning_zh": spec.meaning_zh,
                "physical_family": spec.physical_family,
                "source_type": spec.source_type,
                "unit": spec.unit,
                "formula": spec.formula,
                "data_role": spec.data_role,
                "availability": spec.availability,
                "deployment_status": spec.deployment_status,
                "confidence": spec.confidence,
                "primary_or_validation": spec.primary_or_validation,
                "analysis_enabled": spec.analysis_enabled,
                "observed_count": int(present.sum()),
                "missing_rate": float((~present).mean()),
                "notes": spec.notes,
                "data_kind": spec.data_kind,
                "missing_policy": spec.missing_policy,
                "resample_method": spec.resample_method,
                "required_for_sensor_quality": spec.required_for_sensor_quality,
            }
        )
    for name, formula in derived.items():
        if name not in frame or name in specs:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        rows.append(
            {
                "feature_id": name,
                "canonical_name": name,
                "raw_source": "",
                "meaning_zh": formula,
                "physical_family": (
                    "condenser_cycle_response"
                    if name.startswith("controller_pressure_ratio")
                    else "derived_response"
                ),
                "source_type": "derived",
                "unit": "dimensionless"
                if name in {"cop", "water_cop", "pressure_ratio"}
                else "derived",
                "formula": formula,
                "data_role": "X",
                "availability": "current_history",
                "deployment_status": "confirmed",
                "confidence": "high",
                "primary_or_validation": (
                    "validation" if name.startswith("controller_pressure_ratio") else "primary"
                ),
                "analysis_enabled": not name.startswith("controller_pressure_ratio"),
                "observed_count": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "notes": (
                    "Pr 与绝对压力压比的一致性核查"
                    if name.startswith("controller_pressure_ratio")
                    else "明确物理派生量"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_id").reset_index(drop=True)
