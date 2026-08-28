from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import frost_analysis.labels.cost as rgb_cost_labels
from frost_analysis.labels.cost import (
    assign_image_cost_states,
    complete_observed_cycle_names,
)


def _load_build_script() -> ModuleType:
    path = Path("scripts/labels/build_rgb_cost_labels.py")
    spec = importlib.util.spec_from_file_location("build_rgb_cost_labels", path)
    assert spec and spec.loader
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


def _patch_loader(
    build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    catalog: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    class FakeLoader:
        def __init__(self, _root: Path) -> None:
            pass

        def load_image_metadata(self) -> pd.DataFrame:
            return metadata

        def list_cycles(self) -> pd.DataFrame:
            return catalog

    monkeypatch.setattr(build, "DatasetLoader", FakeLoader)


def test_image_states_respect_contiguous_eligible_runs() -> None:
    curve = pd.DataFrame(
        {
            "candidate_time": pd.to_datetime(
                [
                    "2026-01-01 00:10",
                    "2026-01-01 00:20",
                    "2026-01-01 00:30",
                    "2026-01-01 00:40",
                    "2026-01-01 00:50",
                    "2026-01-01 01:00",
                    "2026-01-01 01:10",
                ]
            ),
            "relative_regret": [0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1],
            "optimization_eligible": [True, True, False, True, True, False, True],
        }
    )
    images = pd.to_datetime(
        [
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:15",
            "2026-01-01 00:20",
            "2026-01-01 00:25",
            "2026-01-01 00:30",
            "2026-01-01 00:35",
            "2026-01-01 00:40",
            "2026-01-01 00:45",
            "2026-01-01 00:50",
            "2026-01-01 00:55",
            "2026-01-01 01:00",
            "2026-01-01 01:05",
            "2026-01-01 01:10",
            "2026-01-01 01:15",
        ]
    )

    labels = assign_image_cost_states(images, curve, regret_threshold=0.01)

    expected = pd.Series(
        [
            pd.NA,
            "pre_optimal",
            "pre_optimal",
            "near_optimal",
            pd.NA,
            pd.NA,
            pd.NA,
            "post_optimal",
            "post_optimal",
            "near_optimal",
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ],
        dtype="string",
    )
    pd.testing.assert_series_equal(labels["cost_state"], expected, check_names=False)
    assert labels["three_class_state"].tolist() == labels["cost_state"].tolist()
    assert labels.loc[labels["cost_state"].eq("near_optimal"), "binary_state"].isna().all()
    assert labels.loc[labels["cost_state"].isna(), "relative_regret"].isna().all()


def test_image_states_do_not_envelope_near_optimal_points_within_a_run() -> None:
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="10min"),
            "relative_regret": [0.0, 0.2, 0.0],
            "optimization_eligible": [True, True, True],
        }
    )

    labels = assign_image_cost_states(
        curve["candidate_time"], curve, regret_threshold=0.01
    )

    assert labels["cost_state"].tolist() == [
        "near_optimal",
        "post_optimal",
        "near_optimal",
    ]


def test_image_states_leave_cycle_unlabeled_when_t_star_is_singleton() -> None:
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01 00:10", periods=7, freq="10min"),
            "relative_regret": [0.2, 0.1, np.nan, 0.0, np.nan, 0.1, 0.2],
            "optimization_eligible": [True, True, False, True, False, True, True],
        }
    )

    labels = assign_image_cost_states(
        pd.to_datetime(["2026-01-01 00:15", "2026-01-01 00:40", "2026-01-01 01:05"]),
        curve,
        regret_threshold=0.01,
    )

    assert labels["cost_state"].isna().all()
    assert labels["relative_regret"].isna().all()


def test_image_states_leave_all_ineligible_cycle_unlabeled() -> None:
    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="min"),
            "relative_regret": [0.0, 0.1],
            "optimization_eligible": [False, False],
        }
    )

    labels = assign_image_cost_states(
        pd.date_range("2026-01-01", periods=2, freq="min"),
        curve,
        regret_threshold=0.01,
    )

    assert labels["cost_state"].isna().all()
    assert labels["relative_regret"].isna().all()


def test_curve_label_exclusion_reason_has_a_narrow_public_result() -> None:
    no_candidates = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=2, freq="min"),
            "relative_regret": [0.0, 0.1],
            "optimization_eligible": [False, False],
        }
    )
    singleton_optimum = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "relative_regret": [0.1, 0.0, 0.1],
            "optimization_eligible": [True, False, True],
        }
    )

    assert (
        rgb_cost_labels.curve_label_exclusion_reason(no_candidates)
        == "no_eligible_candidates"
    )
    assert (
        rgb_cost_labels.curve_label_exclusion_reason(singleton_optimum)
        == "t_star_not_in_interpolatable_run"
    )


def test_build_writes_complete_provenance_and_preserves_other_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _load_build_script()
    catalog = pd.DataFrame(
        {
            "cycle_name": [
                "with_images",
                "without_images",
                "without_curve",
                "censored_curve",
            ],
            "experiment_id": [
                "experiment_a",
                "experiment_b",
                "experiment_c",
                "experiment_d",
            ],
            "status": ["valid"] * 4,
            "stable_heating_start": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "defrost_start": [
                "2026-01-01 01:00",
                "2026-01-02 01:00",
                "2026-01-03 01:00",
                "2026-01-04 01:00",
            ],
            "defrost_end": [
                "2026-01-01 01:05",
                "2026-01-02 01:05",
                "2026-01-03 01:05",
                "2026-01-04 01:05",
            ],
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["with_images", "without_curve", "censored_curve"],
            "cycle_stage": ["frost_development"] * 3,
            "image_time": pd.to_datetime(
                [
                    "2026-01-01 00:05",
                    "2026-01-03 00:05",
                    "2026-01-04 00:05",
                ]
            ),
            "camera_role": ["top"] * 3,
            "file_name": [
                "with_images.jpg",
                "without_curve.jpg",
                "censored_curve.jpg",
            ],
        }
    )
    _patch_loader(build, monkeypatch, catalog, metadata)
    cost_root = tmp_path / "cost"
    cost_root.mkdir()
    pd.DataFrame(
        {
            "cycle_name": (
                ["with_images"] * 2
                + ["without_images"] * 2
                + ["censored_curve"] * 2
            ),
            "candidate_time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:10",
                    "2026-01-02 00:00",
                    "2026-01-02 00:10",
                    "2026-01-04 00:00",
                    "2026-01-04 00:10",
                ]
            ),
            "relative_regret": [0.0, 0.2] * 3,
            "optimization_eligible": [True] * 6,
            "is_censored": [False, False, False, False, True, False],
        }
    ).to_csv(cost_root / "cost_function_v1.csv", index=False)
    cost_source = cost_root / "cost_function_v1.csv"
    output = tmp_path / "output"
    output.mkdir()
    (output / "cycle_splits.csv").write_text("keep split edits\n")
    (output / "报告.md").write_text("keep report edits\n")

    build.build_labels(tmp_path / "dataset", cost_source, output)

    provenance = json.loads((output / "label_provenance.json").read_text())
    records = {row["cycle_name"]: row for row in provenance["cycles"]["records"]}
    assert records["with_images"]["reason"] == "labeled"
    assert records["without_images"]["reason"] == "no_interpolatable_image_times"
    assert records["without_curve"] == {
        "cycle_name": "without_curve",
        "included": False,
        "reason": "no_current_curve",
        "labeled_image_count": 0,
    }
    assert records["censored_curve"] == {
        "cycle_name": "censored_curve",
        "included": False,
        "reason": "censored_curve",
        "labeled_image_count": 0,
    }
    assert provenance["cycles"]["included_count"] == 1
    assert provenance["cycles"]["excluded_count"] == 3
    assert provenance["cycles"]["reason_counts"] == {
        "censored_curve": 1,
        "labeled": 1,
        "no_current_curve": 1,
        "no_interpolatable_image_times": 1,
    }
    labels_path = output / "image_cost_labels.parquet"
    labels = pd.read_parquet(labels_path)
    assert "cost_source_sha256" not in labels
    assert provenance["cost_source"] == str(cost_source)
    assert not {
        "cost_source_sha256",
        "output_label_sha256",
        "code_sha256",
    } & set(provenance)
    assert provenance["thresholds"] == list(build.THRESHOLDS)
    assert provenance["git_revision"]
    assert provenance["generated_at_utc"].endswith("+00:00")
    assert (output / "cycle_splits.csv").read_text() == "keep split edits\n"
    assert (output / "报告.md").read_text() == "keep report edits\n"


def test_build_raises_domain_error_when_no_labels_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _load_build_script()
    catalog = pd.DataFrame(
        {
            "cycle_name": ["without_curve"],
            "experiment_id": ["experiment_a"],
            "status": ["valid"],
            "stable_heating_start": ["2026-01-01"],
            "defrost_start": ["2026-01-01 01:00"],
            "defrost_end": ["2026-01-01 01:05"],
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["without_curve"],
            "cycle_stage": ["frost_development"],
            "image_time": pd.to_datetime(["2026-01-01 00:05"]),
            "camera_role": ["top"],
            "file_name": ["image.jpg"],
        }
    )
    _patch_loader(build, monkeypatch, catalog, metadata)
    cost_root = tmp_path / "cost"
    cost_root.mkdir()
    pd.DataFrame(
        {
            "cycle_name": pd.Series(dtype="string"),
            "candidate_time": pd.Series(dtype="datetime64[ns]"),
            "relative_regret": pd.Series(dtype=float),
            "optimization_eligible": pd.Series(dtype=bool),
            "is_censored": pd.Series(dtype=bool),
        }
    ).to_parquet(cost_root / "candidate_cost_curves.parquet", index=False)
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="^no supported RGB labels$"):
        build.build_labels(
            tmp_path / "dataset", cost_root / "candidate_cost_curves.parquet", output
        )
    assert not (output / "image_cost_labels.parquet").exists()


def test_complete_cycles_use_observed_boundaries_and_censoring_not_name() -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["partial_but_complete", "ordinary_censored", "ordinary_no_defrost"],
            "status": ["valid", "valid", "valid"],
            "stable_heating_start": ["2026-01-01"] * 3,
            "defrost_start": ["2026-01-01", "2026-01-01", None],
            "defrost_end": ["2026-01-01 00:05", "2026-01-01 00:05", None],
        }
    )
    curves = pd.DataFrame(
        {
            "cycle_name": [
                "partial_but_complete",
                "ordinary_censored",
                "ordinary_censored",
                "ordinary_no_defrost",
            ],
            "is_censored": [False, True, False, False],
        }
    )

    assert complete_observed_cycle_names(catalog, curves) == ["partial_but_complete"]


def test_complete_cycles_accept_mixed_fractional_timestamp_strings() -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000063", "frost_cycle_000064"],
            "status": ["valid", "valid"],
            "stable_heating_start": [
                "2026-07-30T09:56:11",
                "2026-07-30T12:14:38.999999999",
            ],
            "defrost_start": ["2026-07-30T11:53:09", "2026-07-30T14:03:22"],
            "defrost_end": ["2026-07-30T11:58:11", "2026-07-30T14:08:08"],
        }
    )
    curves = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000063", "frost_cycle_000064"],
            "is_censored": [False, False],
        }
    )

    assert complete_observed_cycle_names(catalog, curves) == [
        "frost_cycle_000063",
        "frost_cycle_000064",
    ]
