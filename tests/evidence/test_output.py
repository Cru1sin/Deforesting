from __future__ import annotations

import json
from pathlib import Path

import pytest

from frost_analysis.evidence import build_evidence, write_evidence

from .conftest import frame_for, settings, write_dataset


def test_writer_rejects_dataset_internal_output_and_preserves_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    loader = write_dataset(dataset, [("c1", "2026-07-01", "valid", frame_for())])
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)

    with pytest.raises(ValueError, match="outside the Dataset"):
        write_evidence(
            bundle,
            dataset / "evidence",
            loader=loader,
            settings=evidence_settings,
        )
    output = write_evidence(
        bundle,
        tmp_path / "evidence",
        loader=loader,
        settings=evidence_settings,
    )

    assert output == (tmp_path / "evidence").resolve()
    expected_tables = {
        "cycle_eligibility.csv",
        "feature_cycle_metrics.csv",
        "future_association.csv",
        "future_horizon_summary.csv",
        "feature_profile.csv",
        "feature_pair_similarity.csv",
        "target_audit.csv",
        "readiness_split.csv",
        "readiness_summary.csv",
    }
    assert expected_tables <= {path.name for path in output.iterdir()}
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.pdf"))) == 4
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "analysis_version",
        "dataset_id",
        "dataset_schema_version",
        "dataset_updated_at",
        "channel_registry_hash",
        "settings_sha256",
        "generated_at",
        "output_files",
        "row_counts",
        "recovery_effect",
    }
    assert manifest["dataset_updated_at"] == "2026-01-01T00:00:00+00:00"
    assert manifest["recovery_effect"] == "not_evaluated"
    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_writer_uses_loader_manifest_without_reading_dataset_manifest(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    loader = write_dataset(dataset, [("c1", "2026-07-01", "valid", frame_for())])
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)
    (dataset / "dataset_manifest.json").unlink()

    output = write_evidence(
        bundle,
        tmp_path / "evidence",
        loader=loader,
        settings=evidence_settings,
    )

    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_updated_at"] == "2026-01-01T00:00:00+00:00"


def test_writer_rejects_existing_output_directory(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    loader = write_dataset(dataset, [("c1", "2026-07-01", "valid", frame_for())])
    evidence_settings = settings(targets=("heating_capacity",))
    bundle = build_evidence(loader, evidence_settings)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        write_evidence(bundle, output, loader=loader, settings=evidence_settings)
