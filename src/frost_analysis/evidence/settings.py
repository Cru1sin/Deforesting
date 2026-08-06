"""Scientific settings for Dataset-native Evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvidenceSettings:
    """Immutable, date-independent scientific parameters for Evidence."""

    targets: tuple[str, ...]
    primary_target: str
    primary_horizon_minutes: int
    horizons_minutes: tuple[int, ...]
    minimum_feature_points: int
    minimum_feature_coverage: float
    minimum_valid_pairs: int
    minimum_pair_coverage: float
    event_thresholds: tuple[float, ...] = (0.05, 0.10, 0.15)
    primary_event_threshold: float = 0.10
    event_persistence_seconds: int = 120
    signal_reference_minutes: int = 5
    signal_smoothing_seconds: int = 60
    signal_mad_multiplier: float = 3.0
    signal_persistence_seconds: int = 60
    dynamic_window_minutes: int = 5
    ridge_alpha: float = 1.0
    context_features: tuple[str, ...] = (
        "ambient_temperature",
        "environment_relative_humidity",
        "water_in_temperature",
        "water_flow",
        "compressor_frequency",
    )

    def __post_init__(self) -> None:
        if self.primary_target not in self.targets:
            raise ValueError("primary_target must be one of targets")
        if self.primary_horizon_minutes not in self.horizons_minutes:
            raise ValueError("primary_horizon_minutes must be one of horizons_minutes")
        if not 0 <= self.minimum_feature_coverage <= 1:
            raise ValueError("minimum_feature_coverage must be in [0, 1]")
        if self.primary_event_threshold not in self.event_thresholds:
            raise ValueError("primary_event_threshold must be one of event_thresholds")
        if any(value <= 0 for value in self.event_thresholds):
            raise ValueError("event_thresholds must be positive")
        durations = (
            self.event_persistence_seconds,
            self.signal_reference_minutes,
            self.signal_smoothing_seconds,
            self.signal_persistence_seconds,
            self.dynamic_window_minutes,
        )
        if any(value <= 0 for value in durations):
            raise ValueError("readiness durations must be positive")
        if self.signal_mad_multiplier <= 0 or self.ridge_alpha <= 0:
            raise ValueError("signal_mad_multiplier and ridge_alpha must be positive")
        if not self.context_features:
            raise ValueError("context_features must not be empty")

    @classmethod
    def from_yaml(cls, path: Path) -> EvidenceSettings:
        """Load the scientific fields used by Evidence from YAML."""
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Evidence settings must be a YAML mapping")
        return cls(
            targets=tuple(loaded["targets"]),
            primary_target=loaded["primary_target"],
            primary_horizon_minutes=loaded["primary_horizon_minutes"],
            horizons_minutes=tuple(loaded["horizons_minutes"]),
            minimum_feature_points=loaded["minimum_feature_points"],
            minimum_feature_coverage=loaded["minimum_feature_coverage"],
            minimum_valid_pairs=loaded["minimum_valid_pairs"],
            minimum_pair_coverage=loaded["minimum_pair_coverage"],
            event_thresholds=tuple(loaded["event_thresholds"]),
            primary_event_threshold=loaded["primary_event_threshold"],
            event_persistence_seconds=loaded["event_persistence_seconds"],
            signal_reference_minutes=loaded["signal_reference_minutes"],
            signal_smoothing_seconds=loaded["signal_smoothing_seconds"],
            signal_mad_multiplier=loaded["signal_mad_multiplier"],
            signal_persistence_seconds=loaded["signal_persistence_seconds"],
            dynamic_window_minutes=loaded["dynamic_window_minutes"],
            ridge_alpha=loaded["ridge_alpha"],
            context_features=tuple(loaded["context_features"]),
        )

    def normalized(self) -> dict[str, object]:
        """Return the path-independent mapping used for settings hashing."""
        return {
            "targets": list(self.targets),
            "primary_target": self.primary_target,
            "primary_horizon_minutes": self.primary_horizon_minutes,
            "horizons_minutes": list(self.horizons_minutes),
            "minimum_feature_points": self.minimum_feature_points,
            "minimum_feature_coverage": self.minimum_feature_coverage,
            "minimum_valid_pairs": self.minimum_valid_pairs,
            "minimum_pair_coverage": self.minimum_pair_coverage,
            "event_thresholds": list(self.event_thresholds),
            "primary_event_threshold": self.primary_event_threshold,
            "event_persistence_seconds": self.event_persistence_seconds,
            "signal_reference_minutes": self.signal_reference_minutes,
            "signal_smoothing_seconds": self.signal_smoothing_seconds,
            "signal_mad_multiplier": self.signal_mad_multiplier,
            "signal_persistence_seconds": self.signal_persistence_seconds,
            "dynamic_window_minutes": self.dynamic_window_minutes,
            "ridge_alpha": self.ridge_alpha,
            "context_features": list(self.context_features),
        }

    @property
    def sha256(self) -> str:
        """Return the hash of canonical normalized scientific settings."""
        payload = json.dumps(
            self.normalized(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
