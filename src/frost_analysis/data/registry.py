"""Unified, task-agnostic feature registry and physical projections."""

from __future__ import annotations

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


@dataclass(frozen=True)
class RegistryResult:
    frame: pd.DataFrame
    specs: dict[str, FeatureSpec]
    metadata: pd.DataFrame


def load_feature_registry(path: Path) -> dict[str, FeatureSpec]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = loaded.get("features")
    if not isinstance(rows, list) or not rows:
        raise ValueError("feature registry must contain a non-empty features list")
    result: dict[str, FeatureSpec] = {}
    allowed_roles = {"X", "C", "M"}
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
        result[feature_id] = FeatureSpec(
            feature_id=feature_id,
            canonical_name=str(item.get("canonical_name", feature_id)),
            raw_source=None if item.get("raw_source") is None else str(item["raw_source"]),
            meaning_zh=str(item.get("meaning_zh", "")),
            physical_family=str(item.get("physical_family", "unclassified")),
            source_type=str(item.get("source_type", "measured")),
            unit=str(item.get("unit", "unknown")),
            formula=str(item.get("formula", "")),
            data_role=role,
            availability=str(item.get("availability", "current_history")),
            deployment_status=str(item.get("deployment_status", "pending")),
            confidence=str(item.get("confidence", "pending")),
            primary_or_validation=str(item.get("primary_or_validation", "primary")),
            analysis_enabled=bool(item.get("analysis_enabled", True)),
            notes=str(item.get("notes", "")),
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
    result = frame.copy()
    for spec in specs.values():
        if spec.raw_source is None:
            continue
        if spec.raw_source not in result:
            # Missing source columns remain explicit NaN fields for auditability.
            result[spec.canonical_name] = np.nan
            if spec.canonical_name == "operating_mode":
                result["is_heating"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
            continue
        values = pd.to_numeric(result[spec.raw_source], errors="coerce")
        if spec.formula == "scale_0.01":
            values = values * 0.01
        elif spec.formula == "scale_0.001":
            values = values * 0.001
        if spec.canonical_name == "operating_mode":
            # Preserve code 3 as operating_mode; derive the nullable boolean separately.
            result[spec.canonical_name] = values
            result["is_heating"] = values.eq(heating_mode_value).astype("boolean").where(
                values.notna()
            )
        elif spec.canonical_name == "defrost_flag":
            result[spec.canonical_name] = result[spec.raw_source]
        else:
            result[spec.canonical_name] = values
    _derive(result)
    metadata = _metadata(result, specs)
    return RegistryResult(result, specs, metadata)


def _derive(frame: pd.DataFrame) -> None:
    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce")

    frame["cop"] = numeric("heating_capacity") / numeric("power_total").where(
        numeric("power_total").abs().gt(1e-12)
    )
    frame["water_heating_capacity"] = (
        1.16278
        * numeric("water_flow")
        * (numeric("water_out_temperature") - numeric("water_in_temperature"))
    )
    frame["water_cop"] = frame["water_heating_capacity"] / numeric("power_total").where(
        numeric("power_total").abs().gt(1e-12)
    )
    frame["superheat_calculated"] = numeric("suction_temperature") - numeric(
        "evaporating_temperature"
    )
    # Pc and Pe are treated as absolute-pressure readings in the source export.
    # The controller's Pr is retained separately as a scaled consistency check.
    frame["pressure_ratio"] = numeric("condensing_pressure") / numeric(
        "evaporating_pressure"
    ).where(numeric("evaporating_pressure").abs().gt(1e-12))
    frame["controller_pressure_ratio_expected"] = 100.0 * frame["pressure_ratio"]
    controller = numeric("controller_pressure_ratio")
    frame["controller_pressure_ratio_residual"] = (
        controller - frame["controller_pressure_ratio_expected"]
    )
    frame["water_delta_temperature"] = numeric("water_out_temperature") - numeric(
        "water_in_temperature"
    )
    frame["ambient_evaporating_delta"] = numeric("ambient_temperature") - numeric(
        "evaporating_temperature"
    )
    frame["temperature_lift"] = numeric("condensing_temperature") - numeric(
        "evaporating_temperature"
    )
    frame["pressure_lift"] = numeric("condensing_pressure") - numeric("evaporating_pressure")
    frame["controller_pressure_ratio_check"] = numeric("controller_pressure_ratio")


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
