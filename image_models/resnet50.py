"""Paper ResNet50 classifier and one fixed-epoch LOEO fold.

This module is imported only for ``resnet50_end_to_end`` so sklearn and dry-run
usage do not require torch or torchvision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from image_models.evaluation import classification_metrics


class ImageRows(Dataset):  # type: ignore[misc]
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        with Image.open(row["absolute_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["target"])


class ResNet50Classifier(nn.Module):  # type: ignore[misc]
    """ResNet50 features plus the existing 2048→1000→64→classes paper MLP."""

    def __init__(
        self,
        num_classes: int = 2,
        weights: models.ResNet50_Weights | None = models.ResNet50_Weights.IMAGENET1K_V2,
    ) -> None:
        super().__init__()
        network = models.resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(network.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images).flatten(1)
        return self.classifier(features), features


def image_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(
    path: Path,
    *,
    weights: models.ResNet50_Weights | None = None,
    device: torch.device | None = None,
) -> ResNet50Classifier:
    destination = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=destination, weights_only=True)
    model = ResNet50Classifier(
        num_classes=int(checkpoint.get("num_classes", 2)), weights=weights
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(destination)
    return model


def train_resnet_fold(
    rows: pd.DataFrame,
    *,
    heldout_experiment: str,
    task: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    save_path: Path | None = None,
    weights: models.ResNet50_Weights | None = models.ResNet50_Weights.IMAGENET1K_V2,
    device: torch.device | None = None,
    camera: str = "front",
    seed: int = 0,
) -> dict[str, Any]:
    """Train fixed epochs on non-heldout rows, then evaluate heldout once."""
    heldout = rows["experiment_id"].astype(str).eq(heldout_experiment)
    train, test = rows.loc[~heldout].copy(), rows.loc[heldout].copy()
    expected = set(range(2 if task == "binary" else 3))
    metrics = {
        "held_out_experiment": heldout_experiment,
        "image_feature": "resnet50_end_to_end",
        "classifier": "resnet_mlp",
        "camera": camera,
        "input_feature": "image_only",
        "train_images": len(train),
        "test_images": len(test),
        "status": "ok",
        "message": "",
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
    }
    present = set(train["target"].astype(int))
    if present != expected:
        metrics.update(
            status="invalid",
            message=f"train classes {sorted(present)}; expected {sorted(expected)}",
        )
        return {"metrics": metrics, "predictions": pd.DataFrame(), "model": None}

    destination = device or preferred_device()
    torch.manual_seed(seed)
    model = ResNet50Classifier(num_classes=len(expected), weights=weights).to(destination)
    train_transform, evaluation_transform = image_transforms()
    loader = DataLoader(
        ImageRows(train, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    counts = train["target"].value_counts().reindex(sorted(expected)).to_numpy()
    class_weights = torch.tensor(
        len(train) / (len(expected) * counts), dtype=torch.float32, device=destination
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for images, targets in loader:
            optimizer.zero_grad()
            logits, _ = model(images.to(destination))
            loss = loss_function(logits, targets.to(destination))
            loss.backward()
            optimizer.step()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": len(expected),
                "task": task,
            },
            save_path,
        )

    probabilities: list[np.ndarray] = []
    model.eval()
    evaluation_loader = DataLoader(
        ImageRows(test, evaluation_transform), batch_size=batch_size, num_workers=0
    )
    with torch.no_grad():
        for images, _ in evaluation_loader:
            logits, _ = model(images.to(destination))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    scores = np.concatenate(probabilities)
    prediction = scores.argmax(axis=1)
    columns = [
        column
        for column in (
            "experiment_id",
            "cycle_name",
            "camera_role",
            "image_path",
            "file_name",
            "image_time",
            "relative_regret",
            "target",
        )
        if column in test
    ]
    predictions = test[columns].reset_index(drop=True).copy()
    predictions["held_out_experiment"] = heldout_experiment
    predictions["camera"] = camera
    predictions["image_feature"] = "resnet50_end_to_end"
    predictions["classifier"] = "resnet_mlp"
    predictions["input_feature"] = "image_only"
    predictions["prediction"] = prediction
    predictions["decision_score"] = scores[:, 1] if len(expected) == 2 else scores.max(axis=1)
    if len(expected) > 2:
        for class_index in sorted(expected):
            predictions[f"decision_score_{class_index}"] = scores[:, class_index]
    target = test["target"].to_numpy(dtype=int)
    metrics.update(classification_metrics(target, prediction, task))
    return {"metrics": metrics, "predictions": predictions, "model": None}
