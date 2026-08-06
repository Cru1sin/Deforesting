"""Output writer and compact reproducibility manifest for Evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..dataset_loader import DatasetLoader
from .contracts import (
    ANALYSIS_VERSION,
    CYCLE_ELIGIBILITY_COLUMNS,
    FEATURE_CYCLE_METRIC_COLUMNS,
    FEATURE_PAIR_SIMILARITY_COLUMNS,
    FEATURE_PROFILE_COLUMNS,
    FUTURE_ASSOCIATION_COLUMNS,
    FUTURE_HORIZON_SUMMARY_COLUMNS,
    EvidenceBundle,
)
from .figures import write_figures
from .settings import EvidenceSettings


def write_evidence(
    bundle: EvidenceBundle,
    output_dir: Path,
    *,
    loader: DatasetLoader,
    settings: EvidenceSettings,
) -> Path:
    """Write six CSV tables, figures, and the compact Evidence manifest."""
    dataset_root = loader.dataset_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == dataset_root or dataset_root in resolved_output.parents:
        raise ValueError("Evidence output directory must be outside the Dataset")
    resolved_output.mkdir(parents=True, exist_ok=False)

    tables = {
        "cycle_eligibility.csv": (bundle.cycle_eligibility, CYCLE_ELIGIBILITY_COLUMNS),
        "feature_cycle_metrics.csv": (
            bundle.feature_cycle_metrics,
            FEATURE_CYCLE_METRIC_COLUMNS,
        ),
        "future_association.csv": (bundle.future_association, FUTURE_ASSOCIATION_COLUMNS),
        "future_horizon_summary.csv": (
            bundle.future_horizon_summary,
            FUTURE_HORIZON_SUMMARY_COLUMNS,
        ),
        "feature_profile.csv": (bundle.feature_profile, FEATURE_PROFILE_COLUMNS),
        "feature_pair_similarity.csv": (
            bundle.feature_pair_similarity,
            FEATURE_PAIR_SIMILARITY_COLUMNS,
        ),
    }
    for filename, (table, columns) in tables.items():
        table.loc[:, columns].to_csv(resolved_output / filename, index=False)

    figure_files = write_figures(resolved_output, bundle, loader, settings)
    output_files = [*tables, *figure_files, "analysis_manifest.json"]
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset_id": loader.manifest["dataset_id"],
        "dataset_schema_version": loader.manifest["dataset_schema_version"],
        "dataset_updated_at": loader.manifest["updated_at"],
        "channel_registry_hash": loader.registry["canonical_hash"],
        "settings_sha256": settings.sha256,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_files": output_files,
        "row_counts": {filename: int(len(table)) for filename, (table, _) in tables.items()},
        "recovery_effect": "not_evaluated",
    }
    (resolved_output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return resolved_output
