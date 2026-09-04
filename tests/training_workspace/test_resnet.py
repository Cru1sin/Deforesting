from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402
from torchvision import transforms  # noqa: E402

import train_image_models  # noqa: E402
from image_models import resnet50 as resnet  # noqa: E402


def test_resnet50_classifier_weights_none_forward() -> None:
    network = resnet.ResNet50Classifier(weights=None).eval()

    with torch.no_grad():
        logits, features = network(torch.zeros(1, 3, 64, 64))

    assert logits.shape == (1, 2)
    assert features.shape == (1, 2048)


def test_one_resnet_fold_trains_for_fixed_epochs_without_validation_selection(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    class TinyNetwork(nn.Module):
        def __init__(self, num_classes: int = 2, weights: object = None) -> None:
            super().__init__()
            self.classifier = nn.Linear(3 * 8 * 8, num_classes)

        def forward(self, images):  # type: ignore[no-untyped-def]
            flat = images.flatten(1)
            return self.classifier(flat), flat

    monkeypatch.setattr(resnet, "ResNet50Classifier", TinyNetwork)
    conversion = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    monkeypatch.setattr(resnet, "image_transforms", lambda: (conversion, conversion))
    rows: list[dict[str, object]] = []
    for experiment in ("a", "b", "c"):
        for target in (0, 1):
            path = tmp_path / f"{experiment}-{target}.png"
            Image.new("RGB", (8, 8), (20 + target * 200,) * 3).save(path)
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": f"cycle_{experiment}",
                    "camera_role": "front",
                    "image_path": path.name,
                    "absolute_path": str(path),
                    "image_time": pd.Timestamp("2026-01-01"),
                    "relative_regret": 0.2,
                    "target": target,
                }
            )
    checkpoint = tmp_path / "fold.pt"
    manual_seeds: list[int] = []
    generator_seeds: list[int] = []
    original_manual_seed = torch.manual_seed
    original_data_loader = resnet.DataLoader

    def record_manual_seed(seed: int):  # type: ignore[no-untyped-def]
        manual_seeds.append(seed)
        return original_manual_seed(seed)

    def record_data_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        generator = kwargs.get("generator")
        if generator is not None:
            generator_seeds.append(generator.initial_seed())
        return original_data_loader(*args, **kwargs)

    monkeypatch.setattr(resnet.torch, "manual_seed", record_manual_seed)
    monkeypatch.setattr(resnet, "DataLoader", record_data_loader)

    result = resnet.train_resnet_fold(
        pd.DataFrame(rows),
        heldout_experiment="c",
        task="binary",
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        save_path=checkpoint,
        weights=None,
        device=torch.device("cpu"),
        seed=19,
    )

    assert result["metrics"]["status"] == "ok"
    assert len(result["predictions"]) == 2
    assert result["predictions"]["camera"].eq("front").all()
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["num_classes"] == 2
    assert saved["task"] == "binary"
    assert manual_seeds == [19]
    assert generator_seeds == [19]


def test_main_dispatches_every_resnet_fold_with_lazy_training_import(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    labels = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b", "b"],
            "cycle_name": ["ca", "ca", "cb", "cb"],
            "camera_role": ["front"] * 4,
            "file_name": ["0.png", "1.png", "0.png", "1.png"],
            "image_path": ["a/0.png", "a/1.png", "b/0.png", "b/1.png"],
            "image_time": pd.date_range("2026-01-01", periods=4, freq="min"),
            "stable_heating_start": [pd.Timestamp("2025-12-31 23:50:00")] * 4,
            "relative_regret": [0.2] * 4,
            "timing_state_01pct": [
                "before_reference",
                "after_reference",
                "before_reference",
                "after_reference",
            ],
            "binary_target_01pct": [
                "before_reference",
                "after_reference",
                "before_reference",
                "after_reference",
            ],
        }
    )
    labels_path = tmp_path / "image_labels.parquet"
    labels.to_parquet(labels_path, index=False)
    calls: list[tuple[str, Path | None]] = []

    def fake_fold(rows: pd.DataFrame, **kwargs):  # type: ignore[no-untyped-def]
        heldout = kwargs["heldout_experiment"]
        assert rows["absolute_path"].map(lambda value: Path(value).is_absolute()).all()
        assert kwargs["seed"] == 29
        calls.append((heldout, kwargs["save_path"]))
        test = rows.loc[rows["experiment_id"].eq(heldout)].copy()
        test["held_out_experiment"] = heldout
        test["image_feature"] = "resnet50_end_to_end"
        test["classifier"] = "resnet_mlp"
        test["input_feature"] = "image_only"
        test["prediction"] = test["target"]
        test["decision_score"] = 1.0
        return {
            "metrics": {
                "held_out_experiment": heldout,
                "image_feature": "resnet50_end_to_end",
                "classifier": "resnet_mlp",
                "camera": "front",
                "input_feature": "image_only",
                "train_images": 2,
                "test_images": 2,
                "status": "ok",
                "message": "",
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
            },
            "predictions": test,
            "model": None,
        }

    monkeypatch.setattr(resnet, "train_resnet_fold", fake_fold)
    output = tmp_path / "run"
    args = train_image_models.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset"),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
            "--image-features",
            "resnet50_end_to_end",
            "--classifiers",
            "resnet_mlp",
            "--workers",
            "1",
            "--epochs",
            "1",
            "--save-models",
            "--seed",
            "29",
        ]
    )

    assert train_image_models.run(args) == 0

    assert [heldout for heldout, _ in calls] == ["a", "b"]
    assert all(path is not None and path.suffix == ".pt" for _, path in calls)
    assert len(pd.read_csv(output / "fold_metrics.csv")) == 2
    assert len(pd.read_parquet(output / "predictions.parquet")) == 4


def test_resnet_fold_evaluates_missing_test_class_with_full_label_metrics(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    class AlwaysZero(nn.Module):
        def __init__(self, num_classes: int = 2, weights: object = None) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, images):  # type: ignore[no-untyped-def]
            batch = len(images)
            logits = torch.stack(
                [torch.ones(batch) + self.bias * 0, torch.zeros(batch) + self.bias * 0],
                dim=1,
            )
            return logits, images.flatten(1)

    monkeypatch.setattr(resnet, "ResNet50Classifier", AlwaysZero)
    conversion = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    monkeypatch.setattr(resnet, "image_transforms", lambda: (conversion, conversion))
    rows = []
    for experiment, targets in (("a", (0, 1)), ("b", (0, 1)), ("c", (0,))):
        for target in targets:
            path = tmp_path / f"{experiment}-{target}.png"
            Image.new("RGB", (8, 8), (target * 255,) * 3).save(path)
            rows.append(
                {
                    "experiment_id": experiment,
                    "cycle_name": f"cycle_{experiment}",
                    "camera_role": "front",
                    "image_path": path.name,
                    "absolute_path": str(path),
                    "image_time": pd.Timestamp("2026-01-01"),
                    "relative_regret": 0.2,
                    "target": target,
                }
            )

    result = resnet.train_resnet_fold(
        pd.DataFrame(rows),
        heldout_experiment="c",
        task="binary",
        batch_size=2,
        epochs=1,
        learning_rate=0.0,
        weights=None,
        device=torch.device("cpu"),
    )

    assert result["metrics"]["status"] == "ok"
    assert result["metrics"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result["metrics"]["macro_f1"] == pytest.approx(0.5)
    assert len(result["predictions"]) == 1


def test_resnet_fold_still_rejects_missing_training_class() -> None:
    rows = pd.DataFrame(
        {
            "experiment_id": ["a", "b", "c", "c"],
            "target": [0, 0, 0, 1],
        }
    )

    result = resnet.train_resnet_fold(
        rows,
        heldout_experiment="c",
        task="binary",
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weights=None,
        device=torch.device("cpu"),
    )

    assert result["metrics"]["status"] == "invalid"
    assert "train classes" in result["metrics"]["message"]


def test_three_class_checkpoint_reloads_matching_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyNetwork(nn.Module):
        def __init__(self, num_classes: int = 2, weights: object = None) -> None:
            super().__init__()
            self.classifier = nn.Linear(4, num_classes)

        def forward(self, images):  # type: ignore[no-untyped-def]
            features = images.flatten(1)[:, :4]
            return self.classifier(features), features

    monkeypatch.setattr(resnet, "ResNet50Classifier", TinyNetwork)
    checkpoint = tmp_path / "three_class.pt"
    model = TinyNetwork(num_classes=3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 3,
            "task": "three",
        },
        checkpoint,
    )

    loaded = resnet.load_checkpoint(checkpoint, weights=None)

    assert loaded.classifier.out_features == 3


@pytest.mark.slow
def test_existing_best_head_checkpoint_loads_read_only_and_forwards() -> None:
    root = Path(os.environ.get("DEFROST_CHECKPOINT_ROOT", "output/test/model"))
    checkpoints = sorted(root.rglob("best_head.pt")) if root.is_dir() else []
    if not checkpoints:
        pytest.skip("no historical best_head.pt exists")

    network = resnet.load_checkpoint(checkpoints[0], weights=None).eval()
    with torch.no_grad():
        logits, features = network(torch.zeros(1, 3, 64, 64))

    assert logits.shape == (1, 2)
    assert features.shape == (1, 2048)
