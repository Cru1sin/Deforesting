#!/usr/bin/env python3
# ruff: noqa: E501
"""Train the paper-style binary ResNet50 on the fixed experiment split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from frost_analysis.rgb_cost_labels import assign_image_cost_states
from frost_analysis.rgb_deep_features import preferred_device
from frost_analysis.rgb_evaluation import CAMERA_GROUPS

DEFAULT_CANDIDATES = Path("report/02_经济除霜窗口/经验经济窗口/源数据/candidate_cost_curves.parquet")
DEFAULT_LABELS = Path("report/03_RGB标签与模型/成本标签/image_cost_labels.parquet")


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
            nn.Linear(2048, 1000), nn.ReLU(), nn.Linear(1000, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(images).flatten(1)
        return self.classifier(features), features


def _keep_frozen_batch_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()


def _set_stage(model: BinaryResNet50, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if stage in ("finetune", "adapt"):
        for parameter in model.feature_extractor[7].parameters():  # ResNet layer4
            parameter.requires_grad = True


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    class_f1 = f1_score(targets, predicted, labels=[0, 1], average=None, zero_division=0)
    try:
        auroc = roc_auc_score(targets, probabilities[:, 1])
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": accuracy_score(targets, predicted),
        "balanced_accuracy": balanced_accuracy_score(targets, predicted),
        "macro_f1": f1_score(
            targets, predicted, labels=[0, 1], average="macro", zero_division=0
        ),
        "class0_pre_f1": class_f1[0],
        "class1_post_f1": class_f1[1],
        "auroc": auroc,
    }


def _evaluate(model, rows, transform, device, batch_size, workers):  # type: ignore[no-untyped-def]
    loader = DataLoader(ImageRows(rows, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
    probabilities = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            logits, _ = model(images.to(device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    values = np.concatenate(probabilities)
    return _metrics(rows["target"].to_numpy(), values), values


def _select_camera_group(rows: pd.DataFrame, camera_group: str) -> pd.DataFrame:
    return rows.loc[rows["camera_role"].isin(CAMERA_GROUPS[camera_group])].copy()


def _boundary_sample_weights(rows: pd.DataFrame) -> torch.Tensor:
    counts = rows.assign(near_1pct=rows["relative_regret"].le(0.01)).groupby(
        ["near_1pct", "target"]
    )["target"].transform("size")
    return torch.tensor((1.0 / counts).to_numpy(), dtype=torch.double)


def _select_stage(stage_metrics: list[dict[str, object]]) -> dict[str, object]:
    validation = {
        str(row["stage"]): float(row["macro_f1"])
        for row in stage_metrics if row["split"] == "validation"
    }
    near = {
        str(row["stage"]): float(row["macro_f1"])
        for row in stage_metrics if row["split"] == "near_1pct_validation"
    }
    best_full = max(validation.values())
    stage = max(
        (name for name, score in validation.items() if score >= best_full - 0.01),
        key=near.__getitem__,
    )
    return {
        "stage": stage,
        "checkpoint": f"best_{stage}.pt",
        "validation_macro_f1": validation[stage],
        "near_1pct_validation_macro_f1": near[stage],
    }


def _limit_rows(rows: pd.DataFrame, limit_per_split: int) -> pd.DataFrame:
    stratified = rows.assign(near_1pct=rows["relative_regret"].le(0.01))
    limited = pd.concat(
        [
            group.sample(min(len(group), max(1, limit_per_split // 4)), random_state=0)
            for _, group in stratified.groupby(["split", "target", "near_1pct"], sort=True)
        ],
        ignore_index=True,
    )
    return limited.drop(columns="near_1pct")


def _load_rows(dataset_root, labels_path, candidates_path, heat_basis, limit_per_split, camera_group="all"):  # type: ignore[no-untyped-def]
    labels = pd.read_parquet(labels_path)
    curves = pd.read_parquet(candidates_path)
    selected_regret = f"relative_regret_{heat_basis}"
    curves = curves.assign(
        optimization_eligible=(
            curves["optimization_eligible"].fillna(False)
            & curves["relative_regret_water"].notna()
            & curves["relative_regret_unit"].notna()
        ),
        relative_regret=curves[selected_regret],
    )
    curve_groups = {str(name): group for name, group in curves.groupby("cycle_name", sort=False)}
    labeled = []
    for cycle_name, images in labels.groupby("cycle_name", sort=False):
        curve = curve_groups.get(str(cycle_name))
        if curve is None:
            continue
        states = assign_image_cost_states(images["image_time"], curve, regret_threshold=0.01)
        if not states["relative_regret"].notna().any():
            continue
        eligible = curve.loc[curve["optimization_eligible"]]
        result = images.reset_index(drop=True).copy()
        result["relative_regret"] = states["relative_regret"]
        result["t_star"] = pd.to_datetime(
            eligible.loc[eligible["relative_regret"].idxmin(), "candidate_time"]
        )
        labeled.append(result.loc[result["relative_regret"].notna()])
    labels = _select_camera_group(pd.concat(labeled, ignore_index=True), camera_group)
    authoritative_count = len(labels)
    labels["absolute_path"] = labels["image_path"].map(lambda value: dataset_root / value)
    labels = labels.loc[labels["absolute_path"].map(Path.is_file)].copy()
    labels["image_time"] = pd.to_datetime(labels["image_time"])
    labels["t_star"] = pd.to_datetime(labels["t_star"])
    labels["target"] = labels["image_time"].ge(labels["t_star"]).astype("int64")
    local = labels.copy()
    if limit_per_split:
        labels = _limit_rows(labels, limit_per_split)
    experiments = {split: set(group["experiment_id"].astype(str)) for split, group in labels.groupby("split")}
    assert set(experiments) == {"train", "validation", "test"}
    assert not (experiments["train"] & experiments["validation"] or experiments["train"] & experiments["test"] or experiments["validation"] & experiments["test"]), "experiment leakage"
    assert labels.groupby("split")["target"].nunique().eq(2).all(), "both classes required"
    def split_counts(frame):  # type: ignore[no-untyped-def]
        return {
            str(split): {
                "experiments": int(group["experiment_id"].nunique()),
                "cycles": int(group["cycle_name"].nunique()),
                "images": len(group),
                "class_0": int(group["target"].eq(0).sum()),
                "class_1": int(group["target"].eq(1).sum()),
            }
            for split, group in frame.groupby("split", sort=True)
        }

    counts = {
        "available_by_split": split_counts(local),
        "selected_by_split": split_counts(labels),
        "local_cycles": int(local["cycle_name"].nunique()),
        "authoritative_interpolated_images": authoritative_count,
        "local_image_coverage": len(local),
        "local_image_coverage_fraction": len(local) / authoritative_count,
        "selected_images": len(labels),
    }
    return labels.reset_index(drop=True), counts


def _transforms():  # type: ignore[no-untyped-def]
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = transforms.Compose([
        transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomRotation(5), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(), normalize,
    ])
    evaluation = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize
    ])
    return train, evaluation


def _stage_plan(args):  # type: ignore[no-untyped-def]
    if args.init_checkpoint:
        return [("adapt", args.adapt_epochs, args.adapt_lr)]
    stages = [("head", args.head_epochs, args.lr), ("finetune", args.finetune_epochs, args.finetune_lr)]
    if args.boundary_epochs:
        stages.append(("boundary", args.boundary_epochs, args.finetune_lr))
    return stages


def run(args: argparse.Namespace) -> Path:  # noqa: C901
    started = time.monotonic()
    run_dir = args.output / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"isolated run already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, data_counts = _load_rows(
        args.dataset, args.labels, args.candidates, args.heat_basis, args.limit_per_split,
        args.camera_group,
    )
    train_rows = rows.loc[rows["split"].eq("train")].reset_index(drop=True)
    validation_rows = rows.loc[rows["split"].eq("validation")].reset_index(drop=True)
    test_rows = rows.loc[rows["split"].eq("test")].reset_index(drop=True)
    device = preferred_device()
    train_transform, evaluation_transform = _transforms()
    torch.manual_seed(0)
    model = BinaryResNet50().to(device)
    if args.init_checkpoint:
        model.load_state_dict(
            torch.load(args.init_checkpoint, map_location=device)["model_state_dict"]
        )
    model.eval()
    with torch.no_grad():
        probe_logits, probe_features = model(torch.zeros(2, 3, 224, 224, device=device))
    assert probe_logits.shape == (2, 2) and probe_features.shape == (2, 2048)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device), "mps_available": torch.backends.mps.is_available(),
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "head": "Linear(2048,1000)-ReLU-Linear(1000,64)-ReLU-Linear(64,2)",
        "label": "image_time < t_star => 0; image_time >= t_star => 1",
        "test_features": "not exported (minimal path; predictions only)", "data_counts": data_counts,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows.drop(columns="absolute_path").to_parquet(run_dir / "manifest.parquet", index=False)
    class_counts = train_rows["target"].value_counts().reindex([0, 1], fill_value=1).to_numpy()
    class_weights = torch.tensor(len(train_rows) / (2 * class_counts), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    history, stage_metrics, predictions = [], [], []
    stages = _stage_plan(args)
    validation_near_mask = validation_rows["relative_regret"].le(0.01).to_numpy()
    for stage, epochs, learning_rate in stages:
        if stage == "boundary":
            general_stage = max(
                (row for row in stage_metrics if row["split"] == "validation"),
                key=lambda row: row["macro_f1"],
            )["stage"]
            model.load_state_dict(
                torch.load(run_dir / f"best_{general_stage}.pt", map_location=device)["model_state_dict"]
            )
        _set_stage(model, stage)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        stage_loss_function = nn.CrossEntropyLoss() if stage == "boundary" else loss_function
        sampler = None
        if stage == "boundary":
            sampler = WeightedRandomSampler(
                _boundary_sample_weights(train_rows),
                num_samples=len(train_rows),
                replacement=True,
                generator=torch.Generator().manual_seed(0),
            )
        loader = DataLoader(
            ImageRows(train_rows, train_transform), batch_size=args.batch_size,
            shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
        )
        best_f1, checkpoint = -1.0, run_dir / f"best_{stage}.pt"
        best_near_f1, stage_start_f1 = -1.0, -1.0
        if stage in ("boundary", "adapt"):
            stage_start, stage_start_probabilities = _evaluate(
                model, validation_rows, evaluation_transform, device, args.batch_size, args.workers
            )
            stage_start_near = _metrics(
                validation_rows.loc[validation_near_mask, "target"].to_numpy(),
                stage_start_probabilities[validation_near_mask],
            )
            stage_start_f1 = stage_start["macro_f1"]
            best_near_f1 = stage_start_near["macro_f1"]
            torch.save({"stage": stage, "model_state_dict": model.state_dict()}, checkpoint)
        for epoch in range(1, epochs + 1):
            model.train()
            _keep_frozen_batch_norm_eval(model)
            loss_sum = 0.0
            for batch, (images, targets) in enumerate(loader, start=1):
                optimizer.zero_grad()
                logits, _ = model(images.to(device))
                loss = stage_loss_function(logits, targets.to(device))
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(images)
                if batch % 100 == 0 or batch == len(loader):
                    print(f"[{stage}] epoch={epoch}/{epochs} batch={batch}/{len(loader)}", flush=True)
            validation, validation_probabilities = _evaluate(model, validation_rows, evaluation_transform, device, args.batch_size, args.workers)
            validation_near = _metrics(
                validation_rows.loc[validation_near_mask, "target"].to_numpy(),
                validation_probabilities[validation_near_mask],
            )
            history.append({
                "stage": stage, "epoch": epoch, "train_loss": loss_sum / len(train_rows),
                **{f"validation_{k}": v for k, v in validation.items()},
                "validation_near_1pct_macro_f1": validation_near["macro_f1"],
            })
            print(f"[{stage}] epoch={epoch}/{epochs} loss={loss_sum / len(train_rows):.4f} val_macro_f1={validation['macro_f1']:.4f}", flush=True)
            if stage in ("boundary", "adapt"):
                should_save = (
                    validation["macro_f1"] >= stage_start_f1 - 0.01
                    and validation_near["macro_f1"] > best_near_f1
                )
                if should_save:
                    best_near_f1 = validation_near["macro_f1"]
            else:
                should_save = validation["macro_f1"] > best_f1
                if should_save:
                    best_f1 = validation["macro_f1"]
            if should_save:
                torch.save({"stage": stage, "model_state_dict": model.state_dict()}, checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
        validation, validation_probabilities = _evaluate(model, validation_rows, evaluation_transform, device, args.batch_size, args.workers)
        validation_near = _metrics(
            validation_rows.loc[validation_near_mask, "target"].to_numpy(),
            validation_probabilities[validation_near_mask],
        )
        test, probabilities = _evaluate(model, test_rows, evaluation_transform, device, args.batch_size, args.workers)
        near = test_rows["relative_regret"].le(0.01).to_numpy()
        near_test = _metrics(test_rows.loc[near, "target"].to_numpy(), probabilities[near])
        stage_metrics.extend([
            {"stage": stage, "split": "validation", **validation},
            {"stage": stage, "split": "near_1pct_validation", **validation_near},
            {"stage": stage, "split": "test", **test},
            {"stage": stage, "split": "near_1pct_test", **near_test},
        ])
        frame = test_rows[["image_path", "cycle_name", "experiment_id", "camera_role", "image_time", "relative_regret", "target"]].rename(columns={"cycle_name": "cycle", "experiment_id": "experiment", "camera_role": "camera", "image_time": "time"})
        frame["prediction"], frame["p0"], frame["p1"], frame["stage"] = probabilities.argmax(axis=1), probabilities[:, 0], probabilities[:, 1], stage
        predictions.append(frame)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(stage_metrics).to_csv(run_dir / "stage_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(run_dir / "test_predictions.parquet", index=False)
    (run_dir / "selected_stage.json").write_text(
        json.dumps(_select_stage(stage_metrics), indent=2) + "\n", encoding="utf-8"
    )
    config["elapsed_seconds"] = time.monotonic() - started
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] {run_dir} elapsed_seconds={config['elapsed_seconds']:.1f}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--camera-group", choices=tuple(CAMERA_GROUPS), default="all")
    parser.add_argument("--heat-basis", choices=("water", "unit"), default="water")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--boundary-epochs", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--adapt-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--adapt-lr", type=float, default=1e-5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/resnet50_binary"))
    parser.add_argument("--limit-per-split", type=int, default=0, help=argparse.SUPPRESS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
