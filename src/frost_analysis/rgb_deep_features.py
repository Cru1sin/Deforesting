"""Frozen neural image embeddings for the streamed RGB cohort."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

DEEP_REPRESENTATIONS = (
    "dinov2",
    "efficientnet",
    "mobilenet_v3_small",
    "repvit_m0_9",
)
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def illumination_transform(name: str):  # type: ignore[no-untyped-def]
    """Return one deterministic photometric stress followed by model preprocessing."""
    if name == "native":
        return IMAGE_TRANSFORM

    def gamma(image: Image.Image, exponent: float) -> Image.Image:
        table = [round(255 * (value / 255) ** exponent) for value in range(256)]
        return image.point(table * 3)

    def sensor_noise(image: Image.Image) -> Image.Image:
        values = np.asarray(gamma(image, 2.2), dtype=np.float32) / 255
        rng = np.random.default_rng(20260820)
        noisy = rng.poisson(values * 30) / 30 + rng.normal(0, 0.015, values.shape)
        return Image.fromarray(np.uint8(np.clip(noisy, 0, 1) * 255))

    def vignette(image: Image.Image) -> Image.Image:
        values = np.asarray(gamma(image, 1.8), dtype=np.float32)
        height, width = values.shape[:2]
        y, x = np.ogrid[-1:1 : height * 1j, -1:1 : width * 1j]
        mask = np.clip(1 - 0.55 * (x * x + y * y), 0.35, 1)[..., None]
        return Image.fromarray(np.uint8(np.clip(values * mask, 0, 255)))

    changes = {
        "dark_60pct": lambda image: ImageEnhance.Brightness(image).enhance(0.6),
        "bright_140pct": lambda image: ImageEnhance.Brightness(image).enhance(1.4),
        "low_contrast_60pct": lambda image: ImageEnhance.Contrast(image).enhance(0.6),
        "gamma_1p8": lambda image: gamma(image, 1.8),
        "gamma_2p2": lambda image: gamma(image, 2.2),
        "gamma_2p2_sensor_noise": sensor_noise,
        "gamma_1p8_vignette": vignette,
    }
    if name not in changes:
        raise ValueError(f"unknown illumination condition: {name}")
    return transforms.Compose([transforms.Lambda(changes[name]), IMAGE_TRANSFORM])


def cosine_similarity_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return paired cosine similarity for two embedding matrices."""
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return numerator / np.maximum(denominator, np.finfo(float).eps)


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
    """Load the locked pretrained image encoders used by the shared benchmark."""
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
    if "mobilenet_v3_small" in names:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = nn.Identity()
        extractors["mobilenet_v3_small"] = (model, IMAGE_TRANSFORM)
    if "repvit_m0_9" in names:
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("repvit_m0_9 requires the ml extra with timm") from error
        extractors["repvit_m0_9"] = (
            timm.create_model("repvit_m0_9.dist_450e_in1k", pretrained=True, num_classes=0),
            IMAGE_TRANSFORM,
        )
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
