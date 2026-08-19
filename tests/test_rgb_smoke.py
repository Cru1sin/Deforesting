from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from frost_analysis.rgb_smoke import (
    cycle_feature_shard,
    even_sample_groups,
    image_color_features,
    image_feature_matrix,
    selected_names,
)


def test_even_sample_groups_keeps_endpoints(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "group": ["a"] * 10,
            "image_time": pd.date_range("2026-01-01", periods=10, freq="min"),
            "value": range(10),
        }
    )

    sampled = even_sample_groups(frame, ["group"], maximum_per_group=3)

    assert sampled["value"].tolist() == [0, 4, 9]


def test_color_features_are_finite_for_a_constant_rgb_image(tmp_path) -> None:
    path = tmp_path / "red.png"
    Image.new("RGB", (16, 8), (255, 0, 0)).save(path)

    features = image_color_features(path)

    assert features.shape == (34,)
    assert np.isfinite(features).all()
    assert np.allclose(features[24:27], [1.0, 0.0, 0.0])


def test_selected_names_returns_one_requested_model_or_every_model() -> None:
    available = ("logistic", "svm")

    assert selected_names("svm", available) == ("svm",)
    assert selected_names("all", available) == available


def test_image_feature_matrix_excludes_unreadable_images(tmp_path) -> None:
    good = tmp_path / "good.png"
    Image.new("RGB", (16, 8), (0, 255, 0)).save(good)
    rows = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "camera_role": ["top", "left"],
            "file_name": ["good.png", "missing.png"],
            "absolute_path": [str(good), str(tmp_path / "missing.png")],
        }
    )

    features, positions, excluded = image_feature_matrix(rows, ("top", "left"))

    assert features.shape == (1, 36)
    assert positions == [0]
    assert excluded["file_name"].tolist() == ["missing.png"]


def test_cycle_feature_shard_samples_and_embeds_features(tmp_path) -> None:
    cycle_dir = tmp_path / "frost_cycle_000001"
    camera = cycle_dir / "top"
    camera.mkdir(parents=True)
    rows = []
    for index, state in enumerate(("pre_optimal", "pre_optimal", "post_optimal")):
        name = f"{index}.png"
        Image.new("RGB", (8, 8), (index * 50, 0, 0)).save(camera / name)
        rows.append(
            {
                "cycle_name": cycle_dir.name,
                "camera_role": "top",
                "file_name": name,
                "image_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=index),
                "cost_state": state,
            }
        )

    shard, excluded = cycle_feature_shard(
        pd.DataFrame(rows), cycle_dir, ("top", "left"), maximum_per_group=1
    )

    assert len(shard) == 2
    assert shard["cost_state"].tolist() == ["post_optimal", "pre_optimal"]
    assert len([column for column in shard if column.startswith("feature_")]) == 36
    assert excluded.empty
