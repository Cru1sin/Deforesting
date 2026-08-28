#!/usr/bin/env python3
# ruff: noqa: E501
"""Evaluate cached ResNet50 latents with strictly paired causal sensor inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from frost_analysis.training.features import preferred_device
from frost_analysis.training.resnet import BinaryResNet50, ImageRows, image_transforms

CURRENT_SENSORS = (
    "ambient_temperature",  # T4
    "environment_relative_humidity",
    "water_in_temperature",
    "water_out_temperature",
    "water_temperature_setpoint",
    "water_flow",
    "evaporating_pressure",  # Pe
    "coil_temperature",  # T3
    "suction_temperature",
    "superheat",
    "condensing_pressure",
    "condensing_temperature",
    "discharge_temperature",
    "plate_heat_exchanger_inlet_temperature",
    "plate_heat_exchanger_outlet_temperature",
    "compressor_frequency",
    "compressor_frequency_setpoint",
    "compressor_current",
    "compressor_power",
    "fan_speed",
    "fan_current",
    "exv_opening",
    "heating_capacity",
    "power_total",
    "cop",
    "evaporator_capacity",
    "pressure_ratio",
    "water_delta_temperature",
)
SLOPE_SENSORS = (
    "evaporating_pressure",
    "coil_temperature",
    "fan_current",
    "compressor_frequency",
    "exv_opening",
    "compressor_power",
    "power_total",
    "heating_capacity",
    "cop",
    "evaporator_capacity",
    "water_out_temperature",
    "water_delta_temperature",
    "suction_temperature",
    "superheat",
    "discharge_temperature",
    "pressure_ratio",
)
KEYS = ["cycle_name", "camera_role", "image_time", "image_path"]


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1000), nn.ReLU(), nn.Linear(1000, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def export_latents(run_dir: Path, dataset: Path, batch_size: int, workers: int) -> Path:
    """Create the one reusable 2048D cache, or validate and reuse it."""
    cache = run_dir / "finetune_latents.parquet"
    manifest = pd.read_parquet(run_dir / "manifest.parquet")
    if cache.is_file():
        cached = pd.read_parquet(cache, columns=KEYS)
        if not cached.equals(manifest[KEYS].reset_index(drop=True)):
            raise ValueError("latent cache keys do not match this run manifest")
        return cache
    rows = manifest.copy()
    rows["absolute_path"] = rows["image_path"].map(lambda value: dataset / value)
    if not rows["absolute_path"].map(Path.is_file).all():
        raise FileNotFoundError("one or more manifest images are no longer local")
    device = preferred_device()
    model = BinaryResNet50().to(device)
    checkpoint = torch.load(run_dir / "best_finetune.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    evaluation_transform = image_transforms()[1]
    loader = DataLoader(
        ImageRows(rows, evaluation_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    chunks = []
    with torch.no_grad():
        for batch, (images, _) in enumerate(loader, start=1):
            _, features = model(images.to(device))
            chunks.append(features.cpu().numpy())
            if batch % 100 == 0 or batch == len(loader):
                print(f"[latent] batch={batch}/{len(loader)}", flush=True)
    values = np.concatenate(chunks)
    if values.shape != (len(rows), 2048):
        raise ValueError(f"unexpected latent shape: {values.shape}")
    latent_columns = [f"z_{index:04d}" for index in range(2048)]
    pd.concat(
        [manifest.reset_index(drop=True), pd.DataFrame(values, columns=latent_columns)], axis=1
    ).to_parquet(cache, index=False, compression="zstd")
    return cache


def _sensor_frame(path: Path) -> pd.DataFrame:
    required = {*CURRENT_SENSORS, *SLOPE_SENSORS}
    columns = ["cycle_name", "timestamp", *sorted(required)]
    imputed = [f"{name}__imputed" for name in sorted(required)]
    frame = pd.read_parquet(path, columns=[*columns, *imputed])
    frame["sensor_timestamp"] = pd.to_datetime(frame.pop("timestamp"), format="mixed")
    for name in required:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").mask(
            frame[f"{name}__imputed"].fillna(True)
        )
    frame = frame.drop(columns=imputed).sort_values("sensor_timestamp")
    past = frame[["sensor_timestamp", *SLOPE_SENSORS]].rename(
        columns={"sensor_timestamp": "past_timestamp", **{name: f"past_{name}" for name in SLOPE_SENSORS}}
    )
    paired = pd.merge_asof(
        frame.assign(slope_target_time=frame["sensor_timestamp"] - pd.Timedelta(minutes=5)),
        past,
        left_on="slope_target_time",
        right_on="past_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=15),
    )
    elapsed_minutes = (paired["sensor_timestamp"] - paired["past_timestamp"]).dt.total_seconds() / 60
    for name in SLOPE_SENSORS:
        paired[f"{name}__slope_5min"] = (paired[name] - paired[f"past_{name}"]) / elapsed_minutes
    return paired.drop(columns=["slope_target_time", "past_timestamp", *[f"past_{name}" for name in SLOPE_SENSORS]])


def align_sensors(latents: pd.DataFrame, sensor_dir: Path) -> tuple[pd.DataFrame, dict[str, int | float]]:
    paths = [sensor_dir / f"{cycle}.parquet" for cycle in latents["cycle_name"].unique()]
    missing = [path.stem for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sensor cycles: {', '.join(missing)}")
    sensors = pd.concat((_sensor_frame(path) for path in paths), ignore_index=True)
    paired = pd.merge_asof(
        latents.assign(image_time=pd.to_datetime(latents["image_time"], format="mixed")).sort_values("image_time"),
        sensors.sort_values("sensor_timestamp"),
        left_on="image_time",
        right_on="sensor_timestamp",
        by="cycle_name",
        direction="backward",
        tolerance=pd.Timedelta(seconds=15),
    )
    paired = paired.loc[paired["sensor_timestamp"].notna()].reset_index(drop=True)
    experiments = {split: set(rows["experiment_id"]) for split, rows in paired.groupby("split")}
    if set(experiments) != {"train", "validation", "test"} or any(
        experiments[a] & experiments[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("experiment-level split leakage")
    audit = {
        "manifest_images": len(latents),
        "strictly_paired_images": len(paired),
        "pair_rate": len(paired) / len(latents),
        "paired_cycles": int(paired["cycle_name"].nunique()),
    }
    return paired, audit


def _metrics(target, probability):  # type: ignore[no-untyped-def]
    prediction = probability >= 0.5
    return {
        "accuracy": accuracy_score(target, prediction),
        "balanced_accuracy": balanced_accuracy_score(target, prediction),
        "macro_f1": f1_score(target, prediction, labels=[0, 1], average="macro", zero_division=0),
        "class0_pre_f1": f1_score(target, prediction, labels=[0], average="macro", zero_division=0),
        "class1_post_f1": f1_score(target, prediction, labels=[1], average="macro", zero_division=0),
        "auroc": roc_auc_score(target, probability) if pd.Series(target).nunique() == 2 else float("nan"),
    }


def _probabilities(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size)
    result = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            result.append(model(batch.to(device)).softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def _fit_mlp(
    name: str,
    values: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[FusionMLP, list[dict[str, object]]]:
    torch.manual_seed(0)
    model = FusionMLP(values.shape[1]).to(device)
    counts = np.bincount(targets[train], minlength=2)
    weights = torch.tensor(train.sum() / (2 * counts), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values[train]), torch.from_numpy(targets[train])),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    checkpoint = output / f"best_{name}.pt"
    best_f1, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_values, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_values.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_values)
        probability = _probabilities(model, values[validation], device, batch_size)
        validation_metrics = _metrics(targets[validation], probability)
        history.append({
            "input": name,
            "epoch": epoch,
            "train_loss": loss_sum / train.sum(),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
        print(f"[{name}] epoch={epoch}/{epochs} val_macro_f1={validation_metrics['macro_f1']:.4f}", flush=True)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            torch.save({"model_state_dict": model.state_dict(), "input_dim": values.shape[1]}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"])
    return model, history


def evaluate(
    run_dir: Path,
    dataset: Path,
    output: Path,
    batch_size: int,
    workers: int,
    epochs: int,
    learning_rate: float,
) -> None:
    latent_path = export_latents(run_dir, dataset, batch_size, workers)
    paired, audit = align_sensors(pd.read_parquet(latent_path), dataset / "cycles")
    z = [column for column in paired if column.startswith("z_")]
    slopes = [f"{name}__slope_5min" for name in SLOPE_SENSORS]
    output.mkdir(parents=True, exist_ok=False)
    train = paired["split"].eq("train").to_numpy()
    validation = paired["split"].eq("validation").to_numpy()
    test = paired["split"].eq("test").to_numpy()
    sensor_columns = [*CURRENT_SENSORS, *slopes]
    sensor_values = paired[sensor_columns].apply(pd.to_numeric, errors="coerce")
    medians = sensor_values.loc[train].median()
    if medians.isna().any():
        raise ValueError(f"train has all-missing sensors: {medians.index[medians.isna()].tolist()}")
    scaler = StandardScaler().fit(sensor_values.loc[train].fillna(medians))
    scaled_sensors = scaler.transform(sensor_values.fillna(medians)).astype("float32")
    latent_values = paired[z].to_numpy(dtype="float32")
    current_count = len(CURRENT_SENSORS)
    inputs = {
        "rgb_z": latent_values,
        "z_current": np.concatenate([latent_values, scaled_sensors[:, :current_count]], axis=1),
        "z_current_slope": np.concatenate([latent_values, scaled_sensors], axis=1),
    }
    targets = paired["target"].to_numpy(dtype="int64")
    device = preferred_device()
    predictions, metrics, history = [], [], []
    for name, values in inputs.items():
        model, model_history = _fit_mlp(
            name, values, targets, train, validation, output, device, epochs, batch_size, learning_rate
        )
        history.extend(model_history)
        probability = _probabilities(model, values[test], device, batch_size)
        frame = paired.loc[test, [*KEYS, "experiment_id", "split", "relative_regret", "target"]].copy()
        frame["input"] = name
        frame["probability"] = probability
        frame["prediction"] = probability >= 0.5
        predictions.append(frame)
        near = frame["relative_regret"].le(0.01).to_numpy()
        metrics.extend([
            {"input": name, "split": "test", "images": len(frame), **_metrics(frame["target"], probability)},
            {"input": name, "split": "near_1pct_test", "images": int(near.sum()), **_metrics(frame.loc[near, "target"], probability[near])},
        ])
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "config.json").write_text(
        json.dumps({"run_dir": str(run_dir), "current_sensors": CURRENT_SENSORS, "slope_sensors": SLOPE_SENSORS, "alignment_tolerance_seconds": 15, "slope_minutes": 5, "sensor_preprocessing": "train median then train StandardScaler; RGB z unscaled", "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "device": str(device), "audit": audit}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def structure_check(run_dir: Path, dataset: Path) -> None:
    manifest = pd.read_parquet(run_dir / "manifest.parquet").head(2).copy()
    manifest["absolute_path"] = manifest["image_path"].map(lambda value: dataset / value)
    model = BinaryResNet50()
    model.load_state_dict(torch.load(run_dir / "best_finetune.pt", map_location="cpu")["model_state_dict"])
    images, _ = next(
        iter(DataLoader(ImageRows(manifest, image_transforms()[1]), batch_size=2))
    )
    with torch.no_grad():
        logits, features = model(images)
    sensors = _sensor_frame(dataset / "cycles" / f"{manifest.iloc[0]['cycle_name']}.parquet")
    fusion_dim = 2048 + len(CURRENT_SENSORS) + len(SLOPE_SENSORS)
    fusion_logits = FusionMLP(fusion_dim)(torch.zeros(2, fusion_dim))
    assert logits.shape == (2, 2) and features.shape == (2, 2048)
    assert fusion_logits.shape == (2, 2)
    assert set(CURRENT_SENSORS).issubset(sensors) and all(f"{name}__slope_5min" in sensors for name in SLOPE_SENSORS)
    print("[check] logits=(2,2) features=(2,2048) sensors_and_slopes=ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--check-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_only:
        structure_check(args.run_dir, args.dataset)
    else:
        evaluate(
            args.run_dir,
            args.dataset,
            args.output or args.run_dir / "sensor_fusion",
            args.batch_size,
            args.workers,
            args.epochs,
            args.lr,
        )


if __name__ == "__main__":
    main()
