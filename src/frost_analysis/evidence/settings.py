"""Scientific settings for the Dataset-native Evidence analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_FIELDS = {
    "analysis_version",
    "targets",
    "primary_target",
    "primary_horizon_minutes",
    "horizons_minutes",
    "minimum_feature_points",
    "minimum_feature_coverage",
    "minimum_valid_pairs",
    "minimum_pair_coverage",
    "onset_window_seconds",
    "onset_mad_multiplier",
    "onset_persistence_seconds",
    "aggregation_method",
}
_ANALYSIS_VERSION = "frost-cycle-evidence-v2.2"
_AGGREGATION_METHOD = "date_balanced_median_of_cycle_medians_v1"


@dataclass(frozen=True)
class EvidenceSettings:
    """Immutable, date-independent scientific parameters for Evidence."""

    analysis_version: str
    targets: tuple[str, ...]
    primary_target: str
    primary_horizon_minutes: int
    horizons_minutes: tuple[int, ...]
    minimum_feature_points: int
    minimum_feature_coverage: float
    minimum_valid_pairs: int
    minimum_pair_coverage: float
    onset_window_seconds: float
    onset_mad_multiplier: float
    onset_persistence_seconds: float
    aggregation_method: str

    @classmethod
    def from_yaml(cls, path: Path) -> EvidenceSettings:
        """Load and strictly validate one scientific Evidence YAML file."""
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Evidence settings must be a YAML mapping")
        keys = {str(key) for key in loaded}
        missing = sorted(_REQUIRED_FIELDS - keys)
        if missing:
            raise ValueError(f"Evidence settings missing required fields: {missing}")
        unknown = sorted(keys - _REQUIRED_FIELDS)
        if unknown:
            raise ValueError(f"Evidence settings contain unknown fields: {unknown}")

        analysis_version = _string(loaded["analysis_version"], "analysis_version")
        if analysis_version != _ANALYSIS_VERSION:
            raise ValueError(
                f"analysis_version must be {_ANALYSIS_VERSION!r}, got {analysis_version!r}"
            )
        targets = _string_tuple(loaded["targets"], "targets")
        primary_target = _string(loaded["primary_target"], "primary_target")
        if primary_target not in targets:
            raise ValueError("primary_target must be one of targets")
        primary_horizon = _positive_int(
            loaded["primary_horizon_minutes"], "primary_horizon_minutes"
        )
        horizons = _positive_int_tuple(loaded["horizons_minutes"], "horizons_minutes")
        if primary_horizon not in horizons:
            raise ValueError("primary_horizon_minutes must be one of horizons_minutes")

        minimum_feature_points = _positive_int(
            loaded["minimum_feature_points"], "minimum_feature_points"
        )
        minimum_feature_coverage = _unit_interval(
            loaded["minimum_feature_coverage"], "minimum_feature_coverage"
        )
        minimum_valid_pairs = _positive_int(
            loaded["minimum_valid_pairs"], "minimum_valid_pairs"
        )
        minimum_pair_coverage = _unit_interval(
            loaded["minimum_pair_coverage"], "minimum_pair_coverage"
        )
        onset_window_seconds = _positive_number(
            loaded["onset_window_seconds"], "onset_window_seconds"
        )
        onset_mad_multiplier = _positive_number(
            loaded["onset_mad_multiplier"], "onset_mad_multiplier"
        )
        onset_persistence_seconds = _positive_number(
            loaded["onset_persistence_seconds"], "onset_persistence_seconds"
        )
        aggregation_method = _string(
            loaded["aggregation_method"], "aggregation_method"
        )
        if aggregation_method != _AGGREGATION_METHOD:
            raise ValueError(
                "aggregation_method must be "
                f"{_AGGREGATION_METHOD!r}, got {aggregation_method!r}"
            )
        return cls(
            analysis_version=analysis_version,
            targets=targets,
            primary_target=primary_target,
            primary_horizon_minutes=primary_horizon,
            horizons_minutes=horizons,
            minimum_feature_points=minimum_feature_points,
            minimum_feature_coverage=minimum_feature_coverage,
            minimum_valid_pairs=minimum_valid_pairs,
            minimum_pair_coverage=minimum_pair_coverage,
            onset_window_seconds=onset_window_seconds,
            onset_mad_multiplier=onset_mad_multiplier,
            onset_persistence_seconds=onset_persistence_seconds,
            aggregation_method=aggregation_method,
        )

    def normalized(self) -> dict[str, object]:
        """Return the path-independent mapping used for reproducibility hashing."""
        return {
            "analysis_version": self.analysis_version,
            "targets": list(self.targets),
            "primary_target": self.primary_target,
            "primary_horizon_minutes": self.primary_horizon_minutes,
            "horizons_minutes": list(self.horizons_minutes),
            "minimum_feature_points": self.minimum_feature_points,
            "minimum_feature_coverage": self.minimum_feature_coverage,
            "minimum_valid_pairs": self.minimum_valid_pairs,
            "minimum_pair_coverage": self.minimum_pair_coverage,
            "onset_window_seconds": self.onset_window_seconds,
            "onset_mad_multiplier": self.onset_mad_multiplier,
            "onset_persistence_seconds": self.onset_persistence_seconds,
            "aggregation_method": self.aggregation_method,
        }

    @property
    def sha256(self) -> str:
        """Return the hash of canonical normalized settings."""
        payload = json.dumps(
            self.normalized(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = tuple(_string(item, name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    values = tuple(_positive_int(item, name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


__all__ = ["EvidenceSettings"]
