from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import frost_analysis.labels.cost as legacy_cost
from labels import build


def _canonical_cost(**columns: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "cycle_name": ["cycle_a", "cycle_a"],
        "candidate_time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10"]),
        "relative_regret": [0.2, 0.0],
        "optimization_eligible": [True, True],
        "is_censored": [False, False],
        "label_eligible": [True, True],
        "variant": [None, None],
    }
    values.update(columns)
    return pd.DataFrame(values)


def test_experiment_split_matches_the_formal_v1_pattern() -> None:
    assert build.experiment_splits(["e5", "e3", "e1", "e4", "e2"]) == {
        "e1": "train",
        "e2": "train",
        "e3": "train",
        "e4": "validation",
        "e5": "test",
    }


def test_threshold_suffixes_are_exact_and_collision_free() -> None:
    assert [build.threshold_suffix(value) for value in (0.01, 0.02, 0.05, 0.10)] == [
        "01pct",
        "02pct",
        "05pct",
        "10pct",
    ]
    assert build.threshold_suffix(0.011) == "01p1pct"
    assert build.threshold_suffix(0.019) == "01p9pct"
    assert build.threshold_suffix(0.29) == "29pct"


def test_build_rejects_threshold_suffix_collision_before_dataset_loading(
    tmp_path: Path,
) -> None:
    thresholds = (0.12345678901231, 0.12345678901232)
    assert len({build.threshold_suffix(value) for value in thresholds}) == 1

    with pytest.raises(ValueError, match="threshold suffix collision"):
        build.build_labels(
            tmp_path / "missing_dataset",
            _canonical_cost(),
            tmp_path / "labels",
            thresholds,
        )


def test_legacy_module_reexports_the_single_label_algorithm_owner() -> None:
    assert legacy_cost._curve_support is build._curve_support
    assert legacy_cost.curve_label_exclusion_reason is build.curve_label_exclusion_reason
    assert legacy_cost.complete_catalog_cycle_names is build.complete_catalog_cycle_names
    assert legacy_cost.complete_observed_cycle_names is build.complete_observed_cycle_names
    assert legacy_cost.assign_image_cost_states is build.assign_image_cost_states


def test_build_rejects_an_all_unsupported_cycle_with_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["unsupported"],
            "experiment_id": ["e1"],
            "status": ["valid"],
            "stable_heating_start": ["2026-01-01 00:00"],
            "defrost_start": ["2026-01-01 00:40"],
            "defrost_end": ["2026-01-01 00:45"],
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["unsupported"],
            "cycle_stage": ["frost_development"],
            "image_time": pd.to_datetime(["2026-01-01 00:10"]),
            "camera_role": ["top"],
            "file_name": ["image.jpg"],
        }
    )

    class FakeLoader:
        def __init__(self, _: Path) -> None:
            pass

        def list_cycles(self) -> pd.DataFrame:
            return catalog

        def load_image_metadata(self) -> pd.DataFrame:
            return metadata

    monkeypatch.setattr(build, "DatasetLoader", FakeLoader)
    cost = pd.DataFrame(
        {
            "cycle_name": ["unsupported"] * 3,
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="10min"),
            "relative_regret": [0.2, 0.0, 0.1],
            "optimization_eligible": [False, True, False],
            "is_censored": [False] * 3,
            "label_eligible": [True] * 3,
            "variant": [None] * 3,
        }
    )

    with pytest.raises(RuntimeError, match="^no supported RGB labels$"):
        build.build_labels(tmp_path / "dataset", cost, tmp_path / "labels", (0.01,))


def test_build_writes_labels_balance_and_cycle_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "no_curve", "censored", "no_images"],
            "experiment_id": ["e1", "e2", "e3", "e4"],
            "status": ["valid"] * 4,
            "stable_heating_start": pd.date_range("2026-01-01", periods=4, freq="D"),
            "defrost_start": pd.date_range("2026-01-01 01:00", periods=4, freq="D"),
            "defrost_end": pd.date_range("2026-01-01 01:05", periods=4, freq="D"),
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "cycle_a", "cycle_a", "no_curve", "censored"],
            "cycle_stage": [
                "frost_development",
                "frost_development",
                "defrost",
                "frost_development",
                "frost_development",
            ],
            "image_time": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:10",
                    "2026-01-01 01:01",
                    "2026-01-02 00:05",
                    "2026-01-03 00:05",
                ]
            ),
            "camera_role": ["top", "top", "top", "left", "front"],
            "file_name": ["a0.jpg", "a1.jpg", "defrost.jpg", "missing.jpg", "censored.jpg"],
        }
    )

    class FakeLoader:
        def __init__(self, _: Path) -> None:
            pass

        def list_cycles(self) -> pd.DataFrame:
            return catalog

        def load_image_metadata(self) -> pd.DataFrame:
            return metadata

    monkeypatch.setattr(build, "DatasetLoader", FakeLoader, raising=False)
    cost = pd.concat(
        [
            _canonical_cost(),
            _canonical_cost(
                cycle_name=["censored", "censored"],
                candidate_time=pd.to_datetime(["2026-01-03 00:00", "2026-01-03 00:10"]),
                is_censored=[True, False],
            ),
            _canonical_cost(
                cycle_name=["no_images", "no_images"],
                candidate_time=pd.to_datetime(["2026-01-04 00:00", "2026-01-04 00:10"]),
            ),
        ],
        ignore_index=True,
    )
    output = tmp_path / "labels"
    output.mkdir()
    (output / "keep.txt").write_text("do not remove\n", encoding="utf-8")

    build.build_labels(tmp_path / "dataset", cost, output, (0.01, 0.10), overwrite=True)

    labels = pd.read_parquet(output / "image_cost_labels.parquet")
    assert labels["cycle_name"].tolist() == ["cycle_a", "cycle_a"]
    assert labels["split"].eq("train").all()
    assert set(labels) >= {
        "cost_state_01pct",
        "three_class_state_01pct",
        "binary_state_01pct",
        "cost_state_10pct",
        "three_class_state_10pct",
        "binary_state_10pct",
    }
    assert labels["cost_state_01pct"].tolist() == ["pre_optimal", "near_optimal"]
    assert pd.isna(labels.loc[1, "binary_state_01pct"])

    audit = pd.read_csv(output / "cycle_audit.csv").set_index("cycle_name")
    assert audit.loc["cycle_a", "reason"] == "labeled"
    assert audit.loc["no_curve", "reason"] == "no_current_curve"
    assert audit.loc["censored", "reason"] == "censored_curve"
    assert audit.loc["no_images", "reason"] == "no_interpolatable_image_times"

    balance = pd.read_csv(output / "label_balance.csv")
    top = balance.loc[balance["regret_threshold"].eq(0.01) & balance["camera_group"].eq("top")]
    assert top["image_count"].sum() == 2
    assert set(balance["camera_group"]) >= {"top", "top_pair", "all"}
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not remove\n"


def test_existing_output_is_rejected_before_dataset_loading(tmp_path: Path) -> None:
    output = tmp_path / "labels"
    output.mkdir()

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        build.build_labels(tmp_path / "missing_dataset", _canonical_cost(), output, (0.01,))
