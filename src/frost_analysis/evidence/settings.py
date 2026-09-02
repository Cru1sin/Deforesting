"""Scientific settings for Dataset-native Evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

FORECAST_HORIZONS_MINUTES = (5, 10, 20)
PRIMARY_FORECAST_HORIZON_MINUTES = 10
DEGRADATION_THRESHOLDS = (0.05, 0.10, 0.15)
PRIMARY_DEGRADATION_THRESHOLD = 0.10


@dataclass(frozen=True)
class EvidenceSettings:
    """Immutable, date-independent scientific parameters for Evidence."""

    targets: tuple[str, ...] = ("heating_capacity", "cop")
    primary_target: str = "heating_capacity"
    minimum_feature_points: int = 12
    minimum_feature_coverage: float = 0.8
    minimum_valid_pairs: int = 30
    minimum_pair_coverage: float = 0.8
    eligible_statuses: tuple[str, ...] = ("valid",)
    minimum_cycle_minutes: float = 30.0
    primary_horizon_minutes: int = field(
        default=PRIMARY_FORECAST_HORIZON_MINUTES, init=False
    )
    horizons_minutes: tuple[int, ...] = field(default=FORECAST_HORIZONS_MINUTES, init=False)
    event_thresholds: tuple[float, ...] = field(default=DEGRADATION_THRESHOLDS, init=False)
    primary_event_threshold: float = field(
        default=PRIMARY_DEGRADATION_THRESHOLD, init=False
    )
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
        if not 0 <= self.minimum_feature_coverage <= 1:
            raise ValueError("minimum_feature_coverage must be in [0, 1]")
        if not self.eligible_statuses:
            raise ValueError("eligible_statuses must not be empty")
        if self.minimum_cycle_minutes < 0:
            raise ValueError("minimum_cycle_minutes must be nonnegative")
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
            "eligible_statuses": list(self.eligible_statuses),
            "minimum_cycle_minutes": self.minimum_cycle_minutes,
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
