from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from frost_analysis.rgb_smoke import even_sample_groups, image_color_features


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
