from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.dataset import (
    CYCLE_NAME_WIDTH,
    DATASET_SCHEMA_VERSION,
    SourceRun,
    assign_cycle_names,
    build_cycle_index,
    cycle_sort_key,
    format_cycle_name,
    load_source_run,
    logical_schema_compatible,
    make_cycle_uid,
    scientific_fingerprint,
    source_processed_schema,
    validate_dataset_id,
)
from frost_analysis.images import image_columns, image_roles
from frost_analysis.io import relative_posix_path, sha256_file


def test_shared_file_helpers_hash_and_normalize_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "input" / "camera01" / "image.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")

    assert sha256_file(source) == hashlib.sha256(b"image").hexdigest()
    assert relative_posix_path(source, tmp_path) == "input/camera01/image.jpg"


def test_image_helpers_share_the_process_image_contract() -> None:
    frame = pd.DataFrame(
        columns=[
            "image_left_path",
            "image_left_time",
            "image_left_offset_seconds",
            "image_front_path",
            "image_front_time",
            "image_front_offset_seconds",
            "unrelated",
        ]
    )

    assert image_roles(frame) == ("front", "left")
    assert image_columns("front") == (
        "image_front_path",
        "image_front_time",
        "image_front_offset_seconds",
    )


def test_dataset_identity_and_natural_cycle_order_are_stable() -> None:
    assert DATASET_SCHEMA_VERSION == 1
    assert CYCLE_NAME_WIDTH == 6
    assert make_cycle_uid("frost_0715", "cycle_2") == "frost_0715__cycle_2"
    assert format_cycle_name(12) == "frost_cycle_000012"
    assert cycle_sort_key("2026-07-15", "frost_0715", "cycle_2") < cycle_sort_key(
        "2026-07-15", "frost_0715", "cycle_10"
    )


@pytest.mark.parametrize("value", ["", "Frost", "frost-cycles", "_frost", "frost/cycles"])
def test_dataset_id_rejects_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        validate_dataset_id(value)


def test_dataset_id_accepts_versioned_names() -> None:
    assert validate_dataset_id("frost_cycles_v1") == "frost_cycles_v1"


def test_fingerprint_uses_dataset_inputs_not_audit_metadata(tmp_path: Path) -> None:
    run = SourceRun(
        path=tmp_path / "run",
        experiment_id="frost_0715",
        experiment_date="2026-07-15",
        input_dir=tmp_path / "data",
        prepared_path=tmp_path / "prepared.parquet",
        processed_path=tmp_path / "processed.parquet",
        summary_path=tmp_path / "summary.csv",
        resolved_config_sha256="config",
        prepared_sha256="prepared",
        processed_sha256="processed",
        summary_sha256="summary",
        manifest_sha256="manifest-a",
        git_commit="commit-a",
    )

    first = scientific_fingerprint(run, "images")
    second = scientific_fingerprint(
        SourceRun(**{**run.__dict__, "manifest_sha256": "manifest-b", "git_commit": "commit-b"}),
        "images",
    )

    assert first == second
    assert len(first) == 64
    assert first == second


def test_logical_schema_allows_null_column_to_promote() -> None:
    null_schema = [
        {"name": "signal", "logical_type": "null", "nullable": True},
        {"name": "timestamp", "logical_type": "timestamp[ns]", "nullable": True},
    ]
    concrete_schema = [
        {"name": "signal", "logical_type": "double", "nullable": True},
        {"name": "timestamp", "logical_type": "timestamp[ns]", "nullable": True},
    ]

    merged = logical_schema_compatible(null_schema, concrete_schema)

    assert merged[0]["logical_type"] == "double"
    assert merged[1] == concrete_schema[1]


def test_logical_schema_rejects_real_type_changes() -> None:
    left = [{"name": "signal", "logical_type": "double", "nullable": True}]
    right = [{"name": "signal", "logical_type": "string", "nullable": True}]

    with pytest.raises(ValueError, match="schema"):
        logical_schema_compatible(left, right)


@pytest.mark.parametrize(
    ("expected", "actual", "message_parts"),
    [
        (
            [
                {"name": "signal", "logical_type": "double", "nullable": True},
                {"name": "image_front_path", "logical_type": "string", "nullable": True},
            ],
            [
                {"name": "signal", "logical_type": "double", "nullable": True},
                {
                    "name": "image_unverified_camera_01_path",
                    "logical_type": "string",
                    "nullable": True,
                },
            ],
            (
                "missing columns: ['image_front_path']",
                "extra columns: ['image_unverified_camera_01_path']",
            ),
        ),
        (
            [
                {"name": "first", "logical_type": "double", "nullable": True},
                {"name": "second", "logical_type": "double", "nullable": True},
            ],
            [
                {"name": "second", "logical_type": "double", "nullable": True},
                {"name": "first", "logical_type": "double", "nullable": True},
            ],
            ("order changed: true",),
        ),
        (
            [{"name": "signal", "logical_type": "double", "nullable": True}],
            [{"name": "signal", "logical_type": "string", "nullable": True}],
            ("type changes: ['signal: double -> string']",),
        ),
    ],
)
def test_logical_schema_reports_actionable_difference(
    expected: list[dict[str, object]],
    actual: list[dict[str, object]],
    message_parts: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError) as error:
        logical_schema_compatible(expected, actual)

    message = str(error.value)
    assert "Processed schema mismatch:" in message
    for part in message_parts:
        assert part in message


def test_load_source_run_keeps_paths_and_hashes_only(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    input_dir = tmp_path / "data" / "0715"
    input_dir.mkdir(parents=True)
    run_dir = tmp_path / "outputs" / "runs" / "frost_0715"
    run_dir.mkdir(parents=True)

    prepared = pd.DataFrame(
        {"experiment_id": ["frost_0715"], "timestamp": pd.to_datetime(["2026-07-15"])}
    )
    processed = prepared.assign(cycle_id="cycle_1")
    prepared.to_parquet(run_dir / "prepared_data.parquet", index=False)
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)
    pd.DataFrame(
        {"experiment_id": ["frost_0715"], "cycle_id": ["cycle_1"]}
    ).to_csv(run_dir / "cycle_summary.csv", index=False)
    (run_dir / "candidate_channel_evidence.csv").write_text(
        "experiment_id,channel\n", encoding="utf-8"
    )
    manifest = {
        "experiment_id": "frost_0715",
        "experiment_date": "2026-07-15",
        "config_provenance": {"resolved_config_sha256": "resolved"},
        "resolved_config": {"input_dir": "data/0715"},
        "outputs": {
            "prepared_data": "prepared_data.parquet",
            "processed_data": "processed_data.parquet",
            "cycle_summary": "cycle_summary.csv",
            "candidate_channel_evidence": "candidate_channel_evidence.csv",
        },
        "git_commit": "commit",
    }
    (run_dir / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )

    source = load_source_run(run_dir)

    assert source.path == run_dir.resolve()
    assert source.input_dir == input_dir.resolve()
    assert source.prepared_path.name == "prepared_data.parquet"
    assert source.git_commit == "commit"
    assert source.manifest_sha256


def test_source_processed_schema_reads_arrow_fields(tmp_path: Path) -> None:
    path = tmp_path / "processed.parquet"
    pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-07-15"]), "signal": [1.0]}
    ).to_parquet(path, index=False)

    schema = source_processed_schema(path)

    assert [field["name"] for field in schema] == ["timestamp", "signal"]
    assert schema[1]["logical_type"] == "double"


def test_assign_cycle_names_uses_only_published_cycles_and_natural_order() -> None:
    summary = pd.DataFrame(
        {
            "experiment_id": ["frost_0715", "frost_0715", "frost_0715"],
            "experiment_date": ["2026-07-15"] * 3,
            "cycle_id": ["cycle_10", "cycle_2", "partial_1"],
        }
    )
    processed_counts = {
        ("frost_0715", "cycle_10"): 2,
        ("frost_0715", "cycle_2"): 1,
    }

    names = assign_cycle_names(summary, processed_counts)

    assert names == {
        ("frost_0715", "cycle_2"): "frost_cycle_000001",
        ("frost_0715", "cycle_10"): "frost_cycle_000002",
    }


def test_build_cycle_index_separates_publication_and_analysis_eligibility() -> None:
    summary = pd.DataFrame(
        {
            "experiment_id": ["frost_0715", "frost_0715"],
            "experiment_date": ["2026-07-15", "2026-07-15"],
            "cycle_id": ["cycle_1", "cycle_2"],
            "cycle_status": ["valid", "incomplete"],
            "cycle_status_reason": [pd.NA, "defrost_state_gap"],
            "baseline_status": ["available", "not_applicable"],
            "baseline_failure_reason": [pd.NA, "cycle_not_valid"],
        }
    )

    index = build_cycle_index(
        summary,
        processed_counts={("frost_0715", "cycle_1"): 4},
        cycle_names={("frost_0715", "cycle_1"): "frost_cycle_000001"},
        cycle_files={
            ("frost_0715", "cycle_1"): {
                "data_path": "cycles/frost_cycle_000001.parquet",
                "data_sha256": "sha",
                "data_size_bytes": 12,
                "image_count": 3,
            }
        },
    )

    published = index.loc[index["cycle_id"].eq("cycle_1")].iloc[0]
    excluded = index.loc[index["cycle_id"].eq("cycle_2")].iloc[0]
    assert bool(published["published"])
    assert bool(published["recommended_for_analysis"])
    assert not bool(excluded["published"])
    assert excluded["dataset_exclusion_reason"] == "no_processed_rows"
