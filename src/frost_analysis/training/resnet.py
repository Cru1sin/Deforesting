"""Shared ResNet50 image model and input pipeline."""

from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


class ImageRows(Dataset):
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose) -> None:
        self.rows, self.transform = rows.reset_index(drop=True), transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        with Image.open(row["absolute_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["target"])


class BinaryResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(network.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images).flatten(1)
        return self.classifier(features), features


def image_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return the training and deterministic evaluation transforms."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluation
