"""Small contract and leakage checks for the active candidate-channel outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_phase_invariants(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if "cycle_phase" not in frame:
        return ["cycle_phase column missing"]
    valid = frame.loc[frame["cycle_phase"].notna()]
    phase = pd.to_numeric(valid["cycle_phase"], errors="coerce")
    if not phase.between(-1e-12, 1 + 1e-12).all():
        errors.append("cycle_phase outside [0, 1]")
    if "stage" in valid and not valid["stage"].isin(["stable_clean", "frost_development"]).all():
        errors.append("cycle_phase assigned outside normal heating")
    if "cycle_id" in valid:
        for cycle_id, group in valid.groupby("cycle_id", sort=False):
            if not group.sort_values("sensor_time")["cycle_phase"].is_monotonic_increasing:
                errors.append(f"cycle_phase is not monotonic for {cycle_id}")
    if (
        "cycle_time_s" in valid
        and (pd.to_numeric(valid["cycle_time_s"], errors="coerce") < 0).any()
    ):
        errors.append("negative cycle_time_s")
    return errors


def validate_outputs(
    root: Path,
    required_relative_paths: list[str],
    frame: pd.DataFrame,
    *,
    image_tolerance_s: float | None = None,
) -> list[str]:
    if (root / "prepared_data.parquet").exists() or (root / "processed_data.parquet").exists():
        return validate_stage_outputs(root, frame, required_relative_paths)
    errors = validate_phase_invariants(frame)
    for relative in required_relative_paths:
        alternatives = [root / candidate for candidate in relative.split("|")]
        existing = [path for path in alternatives if path.exists()]
        if not existing:
            errors.append(f"missing required output: {relative}")
        elif all(path.stat().st_size == 0 for path in existing):
            errors.append(f"empty required output: {relative}")
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty and np.isinf(numeric.to_numpy()).any():
        errors.append("infinite numeric value in feature timeseries")
    errors.extend(_validate_active_tables(root))
    errors.extend(_validate_causal_columns(frame))
    errors.extend(validate_image_artifacts(root, tolerance_s=image_tolerance_s))
    manifest = root / "logs" / "run_manifest.json"
    if manifest.exists():
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            if loaded.get("latent_z_trained") is True:
                errors.append("latent z training must be false in phase one")
        except json.JSONDecodeError:
            errors.append("run_manifest.json is not valid JSON")
    return errors


def validate_stage_outputs(  # noqa: C901 - contract checks are intentionally explicit
    root: Path, frame: pd.DataFrame, required_relative_paths: list[str] | None = None
) -> list[str]:
    """Validate the four-file prepare/process/analyze contract."""
    required = required_relative_paths or [
        "prepared_data.parquet",
        "processed_data.parquet",
        "cycle_summary.csv",
        "correlation_results.csv",
    ]
    errors: list[str] = []
    for relative in required:
        path = root / relative
        if not path.is_file():
            errors.append(f"missing required stage output: {relative}")
        elif path.stat().st_size == 0:
            errors.append(f"empty required stage output: {relative}")
    prepared_path = root / "prepared_data.parquet"
    if prepared_path.is_file():
        try:
            prepared = pd.read_parquet(prepared_path)
            required_prepared = {"timestamp", "cycle_id", "cycle_stage", "cycle_status"}
            errors.extend(
                f"prepared data missing column: {column}"
                for column in sorted(required_prepared - set(prepared))
            )
            forbidden = tuple(
                column
                for column in prepared.columns
                if any(token in str(column) for token in ("baseline", "__mean_", "__slope_"))
            )
            if forbidden:
                errors.append(f"prepared data contains processed columns: {list(forbidden[:5])}")
        except Exception as error:
            errors.append(f"prepared data unreadable: {type(error).__name__}: {error}")
    summary_path = root / "cycle_summary.csv"
    if summary_path.is_file():
        summary = _read_csv(summary_path)
        if summary is not None:
            required_summary = {
                "cycle_id",
                "cycle_status",
                "max_sensor_gap_seconds",
                "sensor_interruption_intervals",
                "rgb_max_gap_seconds",
                "rgb_interruption_intervals",
            }
            errors.extend(
                f"cycle summary missing column: {column}"
                for column in sorted(required_summary - set(summary))
            )
    correlation_path = root / "correlation_results.csv"
    if correlation_path.is_file():
        correlation = _read_csv(correlation_path)
        if correlation is not None:
            required_results = {"canonical_name", "candidate_status", "physical_family"}
            errors.extend(
                f"correlation results missing column: {column}"
                for column in sorted(required_results - set(correlation))
            )
            if "candidate_score" in correlation or "rank" in correlation:
                errors.append("correlation results contain deprecated weighted ranking")
    if "cycle_phase" in frame:
        errors.extend(validate_phase_invariants(frame))
    elif "cycle_progress" in frame:
        progress = pd.to_numeric(frame["cycle_progress"], errors="coerce").dropna()
        if not progress.between(-1e-12, 1 + 1e-12).all():
            errors.append("cycle_progress outside [0, 1]")
    return errors


def _validate_active_tables(root: Path) -> list[str]:
    errors: list[str] = []
    registry = _read_csv(root / "tables" / "feature_registry.csv")
    evidence = _read_csv(root / "tables" / "candidate_channel_evidence.csv")
    cycles = _read_csv(root / "tables" / "cycle_summary.csv")
    if registry is not None:
        required = {
            "feature_id",
            "canonical_name",
            "physical_family",
            "data_role",
            "analysis_enabled",
            "raw_source",
        }
        errors.extend(
            f"feature_registry missing column: {column}"
            for column in sorted(required - set(registry))
        )
        if (
            registry.astype(str)
            .apply(lambda column: column.str.contains("CCQ_Comp", regex=False))
            .any()
            .any()
        ):
            errors.append("CCQ_Comp entered active feature registry")
    if evidence is not None:
        required = {
            "canonical_name",
            "physical_family",
            "candidate_status",
            "trend_direction",
            "reset_pair_count",
        }
        errors.extend(
            f"candidate evidence missing column: {column}"
            for column in sorted(required - set(evidence))
        )
        if "candidate_score" in evidence or "rank" in evidence:
            errors.append("candidate evidence contains deprecated weighted ranking")
    if cycles is not None:
        required = {
            "cycle_id",
            "date",
            "cycle_quality",
            "rgb_quality",
            "multimodal_quality",
            "rgb_max_gap_seconds",
            "rgb_interruption_intervals",
        }
        errors.extend(
            f"cycle summary missing column: {column}" for column in sorted(required - set(cycles))
        )
    tables_dir = root / "tables"
    if tables_dir.is_dir():
        allowed = {"feature_registry.csv", "candidate_channel_evidence.csv", "cycle_summary.csv"}
        unexpected = sorted(
            path.name for path in tables_dir.glob("*.csv") if path.name not in allowed
        )
        if unexpected:
            errors.append(f"deprecated analysis tables present: {unexpected}")
    return errors


def _validate_causal_columns(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if "sensor_time" not in frame:
        return errors
    for column in frame.columns:
        if not any(
            token in str(column)
            for token in ("__mean_", "__slope_", "__change_", "__std_", "__auc_")
        ):
            continue
        if column.endswith("__available") or column.endswith("__coverage"):
            continue
        base = str(column).split("__", 1)[0]
        available = f"{base}__window_available"
        if available in frame:
            invalid = frame[available].fillna(False).astype(bool).eq(False) & frame[column].notna()
            if invalid.any():
                errors.append(f"unavailable rolling rows contain values: {column}")
    return errors


def validate_image_artifacts(  # noqa: C901
    root: Path, *, tolerance_s: float | None
) -> list[str]:
    manifest = _read_frame(root / "processed" / "image_manifest")
    alignment = _read_frame(root / "processed" / "image_sensor_alignment")
    multiview = _read_frame(root / "processed" / "multiview_index")
    if manifest is None and alignment is None and multiview is None:
        return []
    errors: list[str] = []
    schemas = {
        "image_manifest": {
            "sample_id",
            "image_time",
            "camera_id",
            "image_path",
            "timestamp_ok",
            "image_ok",
        },
        "image_sensor_alignment": {
            "sample_id",
            "candidate_timestamp",
            "timestamp",
            "time_delta_s",
            "matched",
        },
        "multiview_index": {"group_id", "group_time", "camera_count", "all_cameras_present"},
    }
    for name, table in (
        ("image_manifest", manifest),
        ("image_sensor_alignment", alignment),
        ("multiview_index", multiview),
    ):
        if table is None:
            errors.append(f"missing image artifact: {name}")
        else:
            errors.extend(
                f"image artifact schema missing {name}: {column}"
                for column in sorted(schemas[name] - set(table))
            )
    if manifest is None or alignment is None or multiview is None:
        return errors
    if manifest["sample_id"].duplicated().any() or alignment["sample_id"].duplicated().any():
        errors.append("duplicate sample_id in image artifacts")
    if set(manifest["sample_id"].astype(str)) != set(alignment["sample_id"].astype(str)):
        errors.append("image manifest and alignment sample_id sets differ")
    matched = alignment["matched"].fillna(False).astype(bool)
    if (
        alignment.loc[matched, "timestamp"].isna().any()
        or alignment.loc[~matched, "timestamp"].notna().any()
    ):
        errors.append("image matched state and timestamp disagree")
    if tolerance_s is not None:
        delta = pd.to_numeric(alignment.loc[matched, "time_delta_s"], errors="coerce")
        if delta.isna().any() or delta.abs().gt(tolerance_s + 1e-12).any():
            errors.append("matched image exceeds tolerance")
    cycle_labels = _read_frame(root / "processed" / "cycle_labeled_timeseries")
    if cycle_labels is not None and {"timestamp", "cycle_id"} <= set(cycle_labels):
        labels = cycle_labels[["timestamp", "cycle_id"]].drop_duplicates("timestamp")
        matched_rows = alignment.loc[matched, ["timestamp", "cycle_id"]].merge(
            labels, on="timestamp", how="left", suffixes=("_image", "_sensor")
        )
        if (
            matched_rows["cycle_id_sensor"].notna().any()
            and (
                matched_rows["cycle_id_image"].astype(str)
                != matched_rows["cycle_id_sensor"].astype(str)
            ).any()
        ):
            errors.append("cycle label mismatch")
    used = pd.Series(multiview.filter(like="__sample_id").to_numpy().ravel()).dropna()
    if used.duplicated().any():
        errors.append("reuses sample_id in multiview index")
    return errors


def _read_frame(path_without_suffix: Path) -> pd.DataFrame | None:
    for path in (
        path_without_suffix.with_suffix(".parquet"),
        path_without_suffix.with_suffix(".csv"),
    ):
        if path.exists():
            try:
                return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            except Exception:
                return None
    return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def validate_readme_contract(readme_path: Path, artifact_root: Path) -> list[str]:
    """Keep a small public contract check for downstream callers."""
    if not readme_path.exists():
        return ["README missing"]
    text = readme_path.read_text(encoding="utf-8")
    return (
        ["README does not document candidate_channel_evidence.csv"]
        if "candidate_channel_evidence.csv" not in text
        else []
    )
