"""Immutable public result model for Dataset-native Evidence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvidenceBundle:
    """The six tables produced by one Evidence analysis."""

    cycle_eligibility: pd.DataFrame
    feature_cycle_metrics: pd.DataFrame
    future_association: pd.DataFrame
    future_horizon_summary: pd.DataFrame
    feature_profile: pd.DataFrame
    feature_pair_similarity: pd.DataFrame


__all__ = ["EvidenceBundle"]
