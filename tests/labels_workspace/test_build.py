from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from image_labels import timing as build


def _canonical_cost(**columns: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "cycle_name": ["cycle_a", "cycle_a"],
        "candidate_defrost_time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:10"]),
        "relative_regret": [0.2, 0.0],
        "optimization_eligible": [True, True],
        "is_censored": [False, False],
        "label_eligible": [True, True],
    }
    values.update(columns)
    return pd.DataFrame(values)


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
            thresholds,
        )


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
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=3, freq="10min"),
            "relative_regret": [0.2, 0.0, 0.1],
            "optimization_eligible": [False, True, False],
            "is_censored": [False] * 3,
            "label_eligible": [True] * 3,
        }
    )

    with pytest.raises(RuntimeError, match="^no supported RGB labels$"):
        build.build_labels(tmp_path / "dataset", cost, (0.01,))


def test_build_returns_labels_balance_and_cycle_audit(
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
                candidate_defrost_time=pd.to_datetime(["2026-01-03 00:00", "2026-01-03 00:10"]),
                is_censored=[True, False],
            ),
            _canonical_cost(
                cycle_name=["no_images", "no_images"],
                candidate_defrost_time=pd.to_datetime(["2026-01-04 00:00", "2026-01-04 00:10"]),
            ),
        ],
        ignore_index=True,
    )
    labels, balance, audit_table = build.build_labels(tmp_path / "dataset", cost, (0.01, 0.10))

    assert labels["cycle_name"].tolist() == ["cycle_a", "cycle_a"]
    assert "split" not in labels
    assert set(labels) >= {
        "timing_state_01pct",
        "three_class_target_01pct",
        "binary_target_01pct",
        "timing_state_10pct",
        "three_class_target_10pct",
        "binary_target_10pct",
    }
    assert labels["timing_state_01pct"].tolist() == ["before_reference", "near_reference"]
    assert pd.isna(labels.loc[1, "binary_target_01pct"])
    assert labels["reference_method"].eq("v1_cost_optimum").all()

    audit = audit_table.set_index("cycle_name")
    assert audit.loc["cycle_a", "reason"] == "labeled"
    assert audit.loc["no_curve", "reason"] == "no_current_curve"
    assert audit.loc["censored", "reason"] == "censored_curve"
    assert audit.loc["no_images", "reason"] == "no_interpolatable_image_times"

    assert "split" not in balance
    top = balance.loc[balance["regret_threshold"].eq(0.01) & balance["camera_group"].eq("top")]
    assert top["image_count"].sum() == 2
    assert set(balance["camera_group"]) >= {"top", "top_pair", "all"}


def test_selected_time_labels_preserve_the_source_selection_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["selected", "abstained"],
            "experiment_id": ["e1", "e2"],
            "status": ["valid", "valid"],
            "stable_heating_start": pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"]),
            "defrost_start": pd.to_datetime(["2026-01-01 00:30", "2026-01-02 00:30"]),
            "defrost_end": pd.to_datetime(["2026-01-01 00:35", "2026-01-02 00:35"]),
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["selected", "selected", "abstained"],
            "cycle_stage": ["frost_development"] * 3,
            "image_time": pd.to_datetime(
                ["2026-01-01 00:09", "2026-01-01 00:11", "2026-01-02 00:10"]
            ),
            "camera_role": ["front"] * 3,
            "file_name": ["before.jpg", "after.jpg", "unused.jpg"],
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
    policy = pd.DataFrame(
        {
            "cycle_name": ["selected", "selected", "abstained"],
            "candidate_defrost_time": pd.to_datetime(
                ["2026-01-01 00:05", "2026-01-01 00:10", "2026-01-02 00:05"]
            ),
            "is_selected": [False, True, False],
            "selected_defrost_time": pd.to_datetime(["2026-01-01 00:10", "2026-01-01 00:10", None]),
            "selection_method": ["future_selected_time_method"] * 3,
            "selection_status": ["selected", "selected", "abstain"],
        }
    )

    labels, balance, audit = build.build_selected_time_labels(tmp_path / "dataset", policy)

    assert labels["timing_state"].tolist() == ["before_reference", "after_reference"]
    assert labels["binary_target"].tolist() == ["before_reference", "after_reference"]
    assert labels["reference_time"].nunique() == 1
    assert labels["reference_method"].eq("future_selected_time_method").all()
    assert set(balance["camera_group"]) >= {"front", "all"}
    reasons = audit.set_index("cycle_name")["reason"]
    assert reasons["selected"] == "labeled"
    assert reasons["abstained"] == "selection_abstained"
