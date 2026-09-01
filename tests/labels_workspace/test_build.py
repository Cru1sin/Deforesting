from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_image_states_interpolate_only_contiguous_runs_with_two_candidates() -> None:
    curve = _canonical_cost(
        cycle_name=["cycle_a"] * 7,
        candidate_time=pd.date_range("2026-01-01 00:10", periods=7, freq="10min"),
        relative_regret=[0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1],
        optimization_eligible=[True, True, False, True, True, False, True],
        is_censored=[False] * 7,
        label_eligible=[True] * 7,
        variant=[None] * 7,
    )
    image_times = pd.to_datetime(
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

    labels = build.assign_image_cost_states(image_times, curve, regret_threshold=0.01)

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
    assert labels.loc[labels["cost_state"].eq("near_optimal"), "binary_state"].isna().all()
    assert labels.loc[labels["cost_state"].isna(), "relative_regret"].isna().all()


def test_image_states_do_not_envelope_near_points() -> None:
    curve = _canonical_cost(
        cycle_name=["cycle_a"] * 3,
        candidate_time=pd.date_range("2026-01-01", periods=3, freq="10min"),
        relative_regret=[0.0, 0.2, 0.0],
        optimization_eligible=[True] * 3,
        is_censored=[False] * 3,
        label_eligible=[True] * 3,
        variant=[None] * 3,
    )

    labels = build.assign_image_cost_states(curve["candidate_time"], curve, regret_threshold=0.01)

    assert labels["cost_state"].tolist() == [
        "near_optimal",
        "post_optimal",
        "near_optimal",
    ]


def test_image_states_require_t_star_in_an_interpolatable_run() -> None:
    curve = _canonical_cost(
        cycle_name=["cycle_a"] * 7,
        candidate_time=pd.date_range("2026-01-01 00:10", periods=7, freq="10min"),
        relative_regret=[0.2, 0.1, np.nan, 0.0, np.nan, 0.1, 0.2],
        optimization_eligible=[True, True, False, True, False, True, True],
        is_censored=[False] * 7,
        label_eligible=[True] * 7,
        variant=[None] * 7,
    )

    labels = build.assign_image_cost_states(
        pd.to_datetime(["2026-01-01 00:15", "2026-01-01 00:40", "2026-01-01 01:05"]),
        curve,
        regret_threshold=0.01,
    )

    assert labels["cost_state"].isna().all()
    assert labels["relative_regret"].isna().all()


def test_complete_cycles_require_observed_stable_preparation_and_defrost_boundaries() -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["complete", "missing_prep", "censored"],
            "status": ["valid", "valid", "valid"],
            "stable_heating_start": ["2026-01-01"] * 3,
            "defrost_preparation_start": ["2026-01-01 00:50", None, "2026-01-01 00:50"],
            "defrost_start": ["2026-01-01 01:00"] * 3,
            "defrost_end": ["2026-01-01 01:05"] * 3,
        }
    )
    cost = pd.DataFrame(
        {
            "cycle_name": ["complete", "missing_prep", "censored"],
            "is_censored": [False, False, True],
        }
    )

    assert build.complete_observed_cycle_names(catalog, cost) == ["complete"]


def test_experiment_split_matches_the_formal_v1_pattern() -> None:
    assert build.experiment_splits(["e5", "e3", "e1", "e4", "e2"]) == {
        "e1": "train",
        "e2": "train",
        "e3": "train",
        "e4": "validation",
        "e5": "test",
    }


def test_build_writes_labels_balance_and_cycle_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = pd.DataFrame(
        {
            "cycle_name": ["cycle_a", "no_curve", "censored", "no_images"],
            "experiment_id": ["e1", "e2", "e3", "e4"],
            "status": ["valid"] * 4,
            "stable_heating_start": pd.date_range("2026-01-01", periods=4, freq="D"),
            "defrost_preparation_start": pd.date_range("2026-01-01 00:50", periods=4, freq="D"),
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


def test_existing_output_is_rejected_before_dataset_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DatasetMustNotLoad:
        def __init__(self, _: Path) -> None:
            raise AssertionError("existing-output check must precede Dataset loading")

    monkeypatch.setattr(build, "DatasetLoader", DatasetMustNotLoad, raising=False)
    output = tmp_path / "labels"
    output.mkdir()

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        build.build_labels(tmp_path / "dataset", _canonical_cost(), output, (0.01,))


def test_dataloader_wrapper_exports_rgb_camera_order() -> None:
    import dataloader.images as images

    assert hasattr(images, "RGB_CAMERA_ORDER")
