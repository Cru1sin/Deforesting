from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import frost_analysis.rgb_deep_features as rgb_deep_features
from frost_analysis.rgb_deep_features import (
    DEEP_REPRESENTATIONS,
    add_embedding_columns,
    cosine_similarity_rows,
    extract_embeddings,
    extract_representation_matrices,
    illumination_transform,
    load_frozen_extractor,
    preferred_device,
)


def test_lightweight_sota_representations_are_registered() -> None:
    assert DEEP_REPRESENTATIONS == (
        "dinov2",
        "efficientnet",
        "mobilenet_v3_small",
        "repvit_m0_9",
        "convnext_tiny",
        "dinov3",
        "siglip2",
    )


def test_timm_encoders_use_their_pretrained_data_transforms(monkeypatch) -> None:
    import timm

    models = {}
    configs = {}
    transforms = {}

    def create_model(model_id, **kwargs):
        model = torch.nn.Identity()
        models[model_id] = (model, kwargs)
        return model

    def resolve_model_data_config(model):
        config = {"model": next(key for key, value in models.items() if value[0] is model)}
        configs[config["model"]] = model
        return config

    def create_transform(**config):
        transform = object()
        transforms[config["model"]] = transform
        return transform

    monkeypatch.setattr(timm, "create_model", create_model)
    monkeypatch.setattr(timm.data, "resolve_model_data_config", resolve_model_data_config)
    monkeypatch.setattr(timm.data, "create_transform", create_transform)

    dinov3 = load_frozen_extractor("dinov3")
    siglip2 = load_frozen_extractor("siglip2")

    expected_ids = {
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_base_patch16_siglip_224.v2_webli",
    }
    assert set(models) == set(configs) == set(transforms) == expected_ids
    assert all(kwargs == {"pretrained": True, "num_classes": 0} for _, kwargs in models.values())
    assert dinov3 == (
        models["vit_small_patch16_dinov3.lvd1689m"][0],
        transforms["vit_small_patch16_dinov3.lvd1689m"],
    )
    assert siglip2 == (
        models["vit_base_patch16_siglip_224.v2_webli"][0],
        transforms["vit_base_patch16_siglip_224.v2_webli"],
    )


def test_convnext_uses_its_pretrained_weight_transform(monkeypatch) -> None:
    transform = object()
    weights = SimpleNamespace(transforms=lambda: transform)
    model = SimpleNamespace(classifier=[object()])

    monkeypatch.setattr(
        rgb_deep_features,
        "ConvNeXt_Tiny_Weights",
        SimpleNamespace(DEFAULT=weights),
    )
    monkeypatch.setattr(
        rgb_deep_features,
        "convnext_tiny",
        lambda *, weights: model,
    )

    loaded_model, loaded_transform = load_frozen_extractor("convnext_tiny")

    assert loaded_model is model
    assert loaded_transform is transform


def test_preferred_device_orders_cuda_before_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert preferred_device().type == "cuda"


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


def test_illumination_transform_applies_named_brightness_shift() -> None:
    image = Image.new("RGB", (256, 256), (128, 128, 128))

    native = illumination_transform("native")(image)
    dark = illumination_transform("dark_60pct")(image)
    bright = illumination_transform("bright_140pct")(image)

    assert dark.mean() < native.mean() < bright.mean()


def test_illumination_transform_orders_gamma_underexposure() -> None:
    image = Image.new("RGB", (256, 256), (128, 128, 128))

    moderate = illumination_transform("gamma_1p8")(image)
    severe = illumination_transform("gamma_2p2")(image)

    assert severe.mean() < moderate.mean()


def test_sensor_noise_stress_is_deterministic() -> None:
    image = Image.new("RGB", (256, 256), (128, 128, 128))
    transform = illumination_transform("gamma_2p2_sensor_noise")

    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)
    assert first.std() > illumination_transform("gamma_2p2")(image).std()


def test_illumination_transform_rejects_unknown_condition() -> None:
    try:
        illumination_transform("unknown")
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown illumination condition was accepted")


def test_cosine_similarity_rows_scores_identical_and_orthogonal_vectors() -> None:
    first = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    second = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    scores = cosine_similarity_rows(first, second)

    assert np.allclose(scores, [1.0, 0.0])
