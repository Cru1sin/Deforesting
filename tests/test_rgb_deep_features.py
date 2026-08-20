from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from frost_analysis.rgb_deep_features import (
    add_embedding_columns,
    extract_embeddings,
    extract_representation_matrices,
)


class ChannelMean(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(2, 3))


def test_extract_embeddings_batches_images_in_order(tmp_path) -> None:
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (4, 2), color).save(path)
        paths.append(path)

    matrix = extract_embeddings(
        paths,
        ChannelMean(),
        transforms.ToTensor(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, np.eye(3), atol=1e-6)


def test_add_embedding_columns_keeps_rows_and_names_representations() -> None:
    shard = pd.DataFrame({"file_name": ["a.jpg", "b.jpg"]})

    result = add_embedding_columns(
        shard,
        {
            "dinov2": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "efficientnet": np.asarray([[5.0], [6.0]], dtype=np.float32),
        },
    )

    assert result["file_name"].tolist() == ["a.jpg", "b.jpg"]
    assert result[["dinov2_000", "dinov2_001"]].to_numpy().tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert result["efficientnet_000"].tolist() == [5.0, 6.0]


def test_extract_representation_matrices_keeps_representation_names(tmp_path) -> None:
    path = tmp_path / "red.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(path)

    matrices = extract_representation_matrices(
        [path],
        {"tiny": (ChannelMean(), transforms.ToTensor())},
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert list(matrices) == ["tiny"]
    assert np.allclose(matrices["tiny"], [[1.0, 0.0, 0.0]])
