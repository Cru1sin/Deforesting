from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch import nn  # noqa: E402
from torchvision import transforms  # noqa: E402

import main_train  # noqa: E402
from model import resnet  # noqa: E402


def test_binary_resnet50_weights_none_forward() -> None:
    network = resnet.BinaryResNet50(weights=None).eval()

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

    monkeypatch.setattr(resnet, "BinaryResNet50", TinyNetwork)
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
    )

    assert result["metrics"]["status"] == "ok"
    assert len(result["predictions"]) == 2
    assert checkpoint.is_file()
    assert "model_state_dict" in torch.load(checkpoint, map_location="cpu")


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
            "relative_regret": [0.2] * 4,
            "cost_state_01pct": [
                "pre_optimal",
                "post_optimal",
                "pre_optimal",
                "post_optimal",
            ],
        }
    )
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)
    calls: list[tuple[str, Path | None]] = []

    def fake_fold(rows: pd.DataFrame, **kwargs):  # type: ignore[no-untyped-def]
        heldout = kwargs["heldout_experiment"]
        assert rows["absolute_path"].map(lambda value: Path(value).is_absolute()).all()
        calls.append((heldout, kwargs["save_path"]))
        test = rows.loc[rows["experiment_id"].eq(heldout)].copy()
        test["held_out_experiment"] = heldout
        test["representation"] = "resnet50_finetune"
        test["head"] = "paper_mlp"
        test["modality"] = "rgb"
        test["prediction"] = test["target"]
        test["decision_score"] = 1.0
        return {
            "metrics": {
                "held_out_experiment": heldout,
                "representation": "resnet50_finetune",
                "head": "paper_mlp",
                "camera": "front",
                "modality": "rgb",
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
    args = main_train.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dataset"),
            "--labels",
            str(labels_path),
            "--output",
            str(output),
            "--representations",
            "resnet50_finetune",
            "--heads",
            "paper_mlp",
            "--jobs",
            "1",
            "--epochs",
            "1",
            "--save-models",
        ]
    )

    assert main_train.run(args) == 0

    assert [heldout for heldout, _ in calls] == ["a", "b"]
    assert all(path is not None and path.suffix == ".pt" for _, path in calls)
    assert len(pd.read_csv(output / "metrics.csv")) == 2
    assert len(pd.read_parquet(output / "predictions.parquet")) == 4


@pytest.mark.slow
def test_existing_best_head_checkpoint_loads_read_only_and_forwards() -> None:
    root = Path("/Users/cruisin/Documents/DeforestingSensor/output/test/model")
    checkpoints = sorted(root.rglob("best_head.pt")) if root.is_dir() else []
    if not checkpoints:
        pytest.skip("no historical best_head.pt exists")

    network = resnet.load_checkpoint(checkpoints[0], weights=None).eval()
    with torch.no_grad():
        logits, features = network(torch.zeros(1, 3, 64, 64))

    assert logits.shape == (1, 2)
    assert features.shape == (1, 2048)
