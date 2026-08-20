#!/usr/bin/env python3
# ruff: noqa: E501
"""Run five local RGB model baselines on cycle-safe empirical-cost labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from frost_analysis.rgb_smoke import even_sample_groups, image_feature_matrix, selected_names

CAMERA_GROUPS = {
    "top": ("top",),
    "top_close": ("top_close",),
    "left": ("left",),
    "left_close": ("left_close",),
    "front": ("front",),
    "extreme": ("extreme",),
    "top_pair": ("top", "top_close"),
    "left_pair": ("left", "left_close"),
    "all": ("top", "top_close", "left", "left_close", "front", "extreme"),
}
ROLE_ORDER = ("top", "top_close", "left", "left_close", "front", "extreme")
MODEL_NAMES = (
    "color_logistic",
    "color_random_forest",
    "color_rbf_svm",
    "small_cnn",
    "resnet18_linear_probe",
)


class ImageRows(Dataset):
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


class SmallCNN(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images).flatten(1))


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    try:
        auc = (
            roc_auc_score(y_true, probabilities[:, 1])
            if probabilities.shape[1] == 2
            else roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "macro_f1": f1_score(y_true, predicted, average="macro", zero_division=0),
        "macro_auroc": auc,
    }


def _record_predictions(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    model_name: str,
    class_names: list[str],
) -> pd.DataFrame:
    result = rows[["cycle_name", "camera_role", "image_time", "split", "cost_state"]].copy()
    result["model"] = model_name
    result["predicted_state"] = [class_names[index] for index in probabilities.argmax(axis=1)]
    for index, name in enumerate(class_names):
        result[f"probability_{name}"] = probabilities[:, index]
    return result


def _torch_probabilities(
    model: nn.Module,
    rows: pd.DataFrame,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(ImageRows(rows, transform), batch_size=64, shuffle=False)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            probabilities.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def _train_torch(
    model: nn.Module,
    train: pd.DataFrame,
    transform: transforms.Compose,
    *,
    classes: int,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    counts = train["target"].value_counts().reindex(range(classes), fill_value=1).to_numpy()
    weights = torch.tensor(len(train) / (classes * counts), dtype=torch.float32, device=device)
    loader = DataLoader(ImageRows(train, transform), batch_size=64, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(images)
        print(f"[train] epoch={epoch + 1}/{epochs} loss={total / len(train):.4f}", flush=True)
    return model


def run(  # noqa: C901
    dataset_root: Path,
    labels_path: Path,
    output_root: Path,
    *,
    task: str,
    camera_group: str,
    maximum_per_group: int,
    model: str = "all",
) -> None:
    states = ["pre_optimal", "near_optimal", "post_optimal"]
    if task == "binary":
        states = ["pre_optimal", "post_optimal"]
    labels = pd.read_parquet(labels_path)
    labels = labels.loc[
        labels["local_available"]
        & labels["relative_regret"].notna()
        & labels["camera_role"].isin(CAMERA_GROUPS[camera_group])
    ].copy()
    labels["cost_state"] = labels["cost_state_01pct"]
    labels = labels.loc[labels["cost_state"].isin(states)]
    labels["target"] = labels["cost_state"].map({name: index for index, name in enumerate(states)})
    labels["absolute_path"] = labels["image_path"].map(lambda path: str(dataset_root / path))
    sampled = even_sample_groups(
        labels,
        ["split", "cycle_name", "cost_state", "camera_role"],
        maximum_per_group=maximum_per_group,
    )
    print(f"[data] sampled={len(sampled)}", flush=True)
    features, good_positions, excluded = image_feature_matrix(sampled, ROLE_ORDER)
    sampled = sampled.iloc[good_positions].reset_index(drop=True)
    train_mask = sampled["split"].eq("train").to_numpy()
    validation_mask = sampled["split"].eq("validation").to_numpy()
    test_mask = sampled["split"].eq("test").to_numpy()
    train = sampled.loc[train_mask].reset_index(drop=True)
    validation = sampled.loc[validation_mask].reset_index(drop=True)
    test = sampled.loc[test_mask].reset_index(drop=True)
    x_train = features[train_mask]
    x_validation = features[validation_mask]
    x_test = features[test_mask]
    print(
        f"[data] task={task} camera={camera_group} train={len(train)} validation={len(validation)} test={len(test)} excluded={len(excluded)}",
        flush=True,
    )
    y_train = train["target"].to_numpy()

    estimators = {
        "color_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
        ),
        "color_random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        ),
        "color_rbf_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(C=2.0, class_weight="balanced", random_state=0),
                method="sigmoid",
                cv=3,
            ),
        ),
    }
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    chosen_models = selected_names(model, MODEL_NAMES)
    for name in chosen_models:
        if name not in estimators:
            continue
        estimator = estimators[name]
        print(f"[model] {name}", flush=True)
        estimator.fit(x_train, y_train)
        for split, rows, values in (
            ("validation", validation, x_validation),
            ("test", test, x_test),
        ):
            probabilities = estimator.predict_proba(values)
            metric_rows.append(
                {
                    "model": name,
                    "split": split,
                    **_metrics(rows["target"].to_numpy(), probabilities),
                    "image_count": len(rows),
                    "cycle_count": rows["cycle_name"].nunique(),
                }
            )
            prediction_frames.append(
                _record_predictions(rows, probabilities, model_name=name, class_names=states)
            )
            matrix = confusion_matrix(
                rows["target"], probabilities.argmax(axis=1), labels=range(len(states))
            )
            for true_index, true_name in enumerate(states):
                for predicted_index, predicted_name in enumerate(states):
                    confusion_rows.append(
                        {
                            "model": name,
                            "split": split,
                            "true_state": true_name,
                            "predicted_state": predicted_name,
                            "count": int(matrix[true_index, predicted_index]),
                        }
                    )

    neural_models = []
    if any(name in chosen_models for name in ("small_cnn", "resnet18_linear_probe")):
        torch.manual_seed(0)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if "small_cnn" in chosen_models:
            neural_models.append(
                (
                    "small_cnn",
                    SmallCNN(len(states)),
                    transforms.Compose(
                        [transforms.Resize((64, 64)), transforms.ToTensor(), normalize]
                    ),
                    3,
                )
            )
        if "resnet18_linear_probe" in chosen_models:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
            for parameter in resnet.parameters():
                parameter.requires_grad = False
            resnet.fc = nn.Linear(resnet.fc.in_features, len(states))
            neural_models.append(
                (
                    "resnet18_linear_probe",
                    resnet,
                    transforms.Compose(
                        [transforms.Resize((112, 112)), transforms.ToTensor(), normalize]
                    ),
                    2,
                )
            )
    for name, model, transform, epochs in neural_models:
        print(f"[model] {name} device={device}", flush=True)
        model = _train_torch(
            model,
            train,
            transform,
            classes=len(states),
            epochs=epochs,
            device=device,
        )
        for split, rows in (("validation", validation), ("test", test)):
            probabilities = _torch_probabilities(model, rows, transform, device)
            metric_rows.append(
                {
                    "model": name,
                    "split": split,
                    **_metrics(rows["target"].to_numpy(), probabilities),
                    "image_count": len(rows),
                    "cycle_count": rows["cycle_name"].nunique(),
                }
            )
            prediction_frames.append(
                _record_predictions(rows, probabilities, model_name=name, class_names=states)
            )
            matrix = confusion_matrix(
                rows["target"], probabilities.argmax(axis=1), labels=range(len(states))
            )
            for true_index, true_name in enumerate(states):
                for predicted_index, predicted_name in enumerate(states):
                    confusion_rows.append(
                        {
                            "model": name,
                            "split": split,
                            "true_state": true_name,
                            "predicted_state": predicted_name,
                            "count": int(matrix[true_index, predicted_index]),
                        }
                    )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(output_root / "excluded_images.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_root / "metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_root / "confusion_matrices.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output_root / "predictions.parquet", index=False
    )
    sampled.drop(columns="absolute_path").to_parquet(
        output_root / "sample_manifest.parquet", index=False
    )
    (output_root / "README.md").write_text(
        f"""# Local RGB model smoke test

- Task: {task}; classes: {", ".join(states)}.
- Camera group: {camera_group}; roles: {", ".join(CAMERA_GROUPS[camera_group])}.
- Label: 1% pointwise empirical-cost regret.
- Sampling: at most {maximum_per_group} evenly spaced frames per split × cycle × state × camera role.
- Decode QA: {len(excluded)} unreadable sampled images were excluded from every model and recorded in `excluded_images.csv`; source files were not modified or deleted.
- Split: fixed experiment-level split from `report/03_RGB标签与模型/成本标签/cycle_splits.csv`.
- Models: {", ".join(chosen_models)}.
- Scope: local-image engineering smoke test only. No hyperparameter search, repeated seeds, confidence intervals or cloud-cycle completion; do not use these metrics as publication evidence.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("report/03_RGB标签与模型/成本标签/image_cost_labels.parquet"),
    )
    parser.add_argument("--task", choices=("binary", "three"), default="three")
    parser.add_argument("--camera-group", choices=tuple(CAMERA_GROUPS), default="all")
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--maximum-per-group", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("outputs/RGB模型冒烟测试") / f"{args.task}_{args.camera_group}"
    run(
        args.dataset,
        args.labels,
        output,
        task=args.task,
        camera_group=args.camera_group,
        maximum_per_group=args.maximum_per_group,
        model=args.model,
    )


if __name__ == "__main__":
    main()
