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

    def __post_init__(self) -> None:
        if self.primary_target not in self.targets:
            raise ValueError("primary_target must be one of targets")
        if self.primary_horizon_minutes not in self.horizons_minutes:
            raise ValueError("primary_horizon_minutes must be one of horizons_minutes")
        if not 0 <= self.minimum_feature_coverage <= 1:
            raise ValueError("minimum_feature_coverage must be in [0, 1]")

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
        }

    @property
    def sha256(self) -> str:
        """Return the hash of canonical normalized scientific settings."""
        payload = json.dumps(
            self.normalized(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
