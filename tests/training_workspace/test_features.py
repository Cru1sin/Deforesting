from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from image_models import image_features as features


def _image_rows(dataset: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, cycle in (("a", "cycle_a"), ("b", "cycle_b")):
        for index, state in enumerate(("before_reference", "after_reference")):
            relative = Path("images") / cycle / "front" / f"{index}.png"
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 8), (index * 100, 20, 40)).save(path)
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": cycle,
                    "camera_role": "front",
                    "file_name": path.name,
                    "image_path": str(relative),
                    "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5 - 5 * index),
                    "stable_heating_start": pd.Timestamp("2025-12-31 23:50:00"),
                    "relative_regret": 0.2,
                    "timing_state_01pct": state,
                    "target": index,
                }
            )
    return pd.DataFrame(rows)


def test_color_gradient_descriptor_is_the_existing_34_color_gradient_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "red.png"
    Image.new("RGB", (16, 8), (255, 0, 0)).save(path)

    values = features.extract_color_gradient_features(path)

    assert values.shape == (34,)
    assert np.isfinite(values).all()
    assert np.allclose(values[24:27], [1.0, 0.0, 0.0])


def test_two_classifiers_reuse_one_fixed_color_gradient_cache(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dataset = tmp_path / "dataset"
    rows = _image_rows(dataset)
    calls = 0
    original = features.extract_color_gradient_features

    def counted(path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(features, "extract_color_gradient_features", counted)
    cache_root = tmp_path / "output/image_models/_cache"

    first, first_columns = features.prepare_features(
        rows,
        dataset_root=dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )
    second, second_columns = features.prepare_features(
        rows,
        dataset_root=dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )

    assert calls == len(rows)
    assert first_columns == second_columns
    assert len(first_columns) == 35  # 34 image values + one front-camera indicator.
    pd.testing.assert_frame_equal(first, second)
    assert (cache_root / "color_gradient/front/features.parquet").is_file()


def test_elapsed_time_inputs_use_stable_heating_start(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    rows = _image_rows(dataset)
    cache_root = tmp_path / "output/image_models/_cache"

    time_rows, time_columns = features.prepare_features(
        rows,
        dataset_root=dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="elapsed_time_only",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )
    assert not (cache_root / "color_gradient/front/features.parquet").exists()
    image_plus_elapsed_time_rows, image_plus_elapsed_time_columns = features.prepare_features(
        rows,
        dataset_root=dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="image_plus_elapsed_time",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )

    assert time_columns == ["time_minutes"]
    assert sorted(time_rows.groupby("cycle_name")["time_minutes"].apply(list)) == [
        [15.0, 10.0],
        [15.0, 10.0],
    ]
    assert "time_minutes" in image_plus_elapsed_time_columns
    assert len(image_plus_elapsed_time_columns) == 36
    assert image_plus_elapsed_time_rows["time_minutes"].min() == 10.0


def test_cache_reuse_requires_the_same_resolved_dataset_root(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    first_dataset = tmp_path / "first"
    second_dataset = tmp_path / "second"
    first_rows = _image_rows(first_dataset)
    second_rows = _image_rows(second_dataset)
    for relative in second_rows["image_path"]:
        Image.new("RGB", (16, 8), (0, 220, 0)).save(second_dataset / relative)
    calls = 0
    original = features.extract_color_gradient_features

    def counted(path: Path) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(features, "extract_color_gradient_features", counted)
    cache_root = tmp_path / "cache"

    first, columns = features.prepare_features(
        first_rows,
        dataset_root=first_dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )
    second, _ = features.prepare_features(
        second_rows,
        dataset_root=second_dataset,
        image_feature="color_gradient",
        camera="front",
        input_feature="image_only",
        label_column="timing_state_01pct",
        max_images_per_cycle_label=48,
        cache_root=cache_root,
    )

    assert calls == len(first_rows) + len(second_rows)
    assert not np.allclose(first[columns], second[columns])
    cached = pd.read_parquet(cache_root / "color_gradient/front/features.parquet")
    assert cached["absolute_path"].tolist() == [
        str((second_dataset / path).resolve()) for path in second_rows["image_path"]
    ]
    assert "dataset_root" not in cached


def test_dinov2_cache_joins_back_to_the_supplied_labels(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "camera_role": ["front"],
            "file_name": ["0.png"],
            "binary_target_01pct": ["before_reference"],
        }
    )
    pd.DataFrame(
        {
            "cycle_name": ["cycle_a"],
            "camera_role": ["front"],
            "file_name": ["0.png"],
            "dinov2_0": [1.25],
        }
    ).to_parquet(tmp_path / "cycle_a.parquet", index=False)

    result = features.load_dinov2_feature_cache(labels, tmp_path, "dinov2")

    assert result.loc[0, "binary_target_01pct"] == "before_reference"
    assert result.loc[0, "dinov2_0"] == 1.25
