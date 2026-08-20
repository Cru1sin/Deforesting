"""Frozen neural image embeddings for the streamed RGB cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

DEEP_REPRESENTATIONS = ("dinov2", "efficientnet")
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def extract_embeddings(
    paths: list[Path],
    model: torch.nn.Module,
    transform,  # type: ignore[no-untyped-def]
    *,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return model embeddings for paths in their original order."""
    model = model.eval().to(device)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            values = model(torch.stack(tensors).to(device)).detach().cpu().numpy()
            batches.append(values)
    return np.concatenate(batches).astype(np.float32)


def extract_representation_matrices(
    paths: list[Path],
    extractors: dict[str, tuple[nn.Module, object]],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Apply each frozen extractor to the same ordered image paths."""
    return {
        name: extract_embeddings(
            paths,
            model,
            transform,
            device=device,
            batch_size=batch_size,
        )
        for name, (model, transform) in extractors.items()
    }


def load_frozen_extractors(names: list[str]) -> dict[str, tuple[nn.Module, object]]:
    """Load the two locked pretrained image encoders without extra dependencies."""
    extractors = {}
    if "dinov2" in names:
        extractors["dinov2"] = (
            torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vits14",
                pretrained=True,
                trust_repo=True,
                skip_validation=True,
            ),
            IMAGE_TRANSFORM,
        )
    if "efficientnet" in names:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
        extractors["efficientnet"] = (model, IMAGE_TRANSFORM)
    return extractors


def add_embedding_columns(shard: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    """Append named embedding matrices without changing row order."""
    feature_frames = []
    for name, matrix in matrices.items():
        if len(matrix) != len(shard):
            raise ValueError(f"{name} rows do not match shard")
        feature_frames.append(
            pd.DataFrame(
                matrix,
                columns=[f"{name}_{index:03d}" for index in range(matrix.shape[1])],
            )
        )
    return pd.concat([shard.reset_index(drop=True), *feature_frames], axis=1)
