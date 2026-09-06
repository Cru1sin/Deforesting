"""One conditional boundary model: economic input × Pareto relation supervision.

The caller supplies fold-specific teachers/economics. This module never fits G,
changes the knee, predicts a future outcome, or chooses a control threshold.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ACCOUNTING = ["pre_defrost_heat_kwh", "pre_defrost_electricity_kwh"]
ECONOMIC = ["economic_c", "economic_h"]
RGB = [f"dinov2_{index:03d}" for index in range(384)]


class ParetoBoundaryModel(nn.Module):
    """Sensor Sin60/60/32 → unit-normal conditional visual hyperplane."""

    def __init__(
        self, sensor_width: int, *, use_economic_context: bool = False,
        nonvisual: bool = False,
    ) -> None:
        super().__init__()
        self.sensor_width = sensor_width
        self.use_economic_context = use_economic_context
        self.nonvisual = nonvisual
        self.sensor_first = nn.Linear(sensor_width, 60)
        self.sensor_second = nn.Linear(60, 60)
        self.sensor_output = nn.Linear(60, 32)
        self.dropout = nn.Dropout(.2)
        self.visual = nn.Linear(384, 32)
        self.normal = nn.Linear(36, 32)
        self.bias = nn.Linear(36, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = self.sensor_width
        sensor = torch.sin(self.sensor_first(x[:, :width]))
        sensor = self.sensor_output(self.dropout(torch.sin(self.sensor_second(sensor))))
        economic = x[:, width + 2:width + 4]
        if not self.use_economic_context:
            economic = torch.zeros_like(economic)
        context = torch.cat([sensor, x[:, width:width + 2], economic], dim=1)
        visual = self.visual(x[:, width + 4:])
        if self.nonvisual:
            visual = torch.zeros_like(visual)
        normal = F.normalize(self.normal(context), dim=1)
        return (normal * visual).sum(dim=1) - self.bias(context).squeeze(1)


def relation_pairs(rows: pd.DataFrame) -> np.ndarray:
    """Adjacent distinct K levels, one deterministic representative per tied level.

Pairs are (closer, farther), never across a cycle, strict side, selector branch,
or support run. Off-front NaN scores and the knee itself supply no rank pairs.
"""
    candidates = rows.reset_index(drop=True).copy()
    candidates["_position"] = np.arange(len(candidates))
    valid = np.isfinite(candidates["pareto_selection_score"]) & candidates["target"].isin([0, 1])
    keys = ["experiment_id", "cycle_name", "target", "relation_branch", "relation_support_run"]
    pairs = []
    for _, group in candidates.loc[valid].groupby(keys, sort=True):
        ordered = group.sort_values(
            ["pareto_selection_score", "row_id"], ascending=[False, True], kind="stable"
        ).drop_duplicates("pareto_selection_score")
        positions = ordered["_position"].to_numpy(dtype=np.int64)
        pairs.extend(zip(positions[:-1], positions[1:], strict=True))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _batch(rows: pd.DataFrame, columns: list[str], preprocessor: Any) -> tuple:
    x = torch.tensor(preprocessor.transform(rows[columns]), dtype=torch.float32)
    y = torch.tensor(rows["target"].to_numpy(), dtype=torch.float32)
    mask = torch.tensor((rows["is_frame"] | rows["is_knee"]).to_numpy(dtype=bool))
    pairs = torch.tensor(relation_pairs(rows), dtype=torch.long)
    return x, y, mask, pairs


def _losses(logits: torch.Tensor, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    _, target, mask, pairs = batch
    side = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
    rank = (
        F.softplus(logits[pairs[:, 0]].abs() - logits[pairs[:, 1]].abs()).mean()
        if len(pairs) else logits.sum() * 0
    )
    return side, rank


def _fit(
    train: pd.DataFrame, validation: pd.DataFrame | None, columns: list[str],
    config: dict[str, Any], epochs: int, patience: int,
) -> tuple[ParetoBoundaryModel, Any, pd.DataFrame, int]:
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    ).fit(train[columns])
    training = _batch(train, columns, preprocessor)
    validating = _batch(validation, columns, preprocessor) if validation is not None else None
    torch.manual_seed(config["seed"])
    model = ParetoBoundaryModel(
        config["sensor_width"], use_economic_context=config["use_economic_context"],
        nonvisual=config["nonvisual"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history = []
    best_loss, best_epoch, stale = np.inf, 1, 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        side, rank = _losses(model(training[0]), training)
        objective = side + rank if config["use_pareto_relation"] else side
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_side, train_rank = _losses(model(training[0]), training)
            val_side = (
                float(_losses(model(validating[0]), validating)[0])
                if validating is not None else np.nan
            )
        history.extend(
            {"epoch": epoch, "split": name, "loss": value}
            for name, value in (
                ("train_side", float(train_side)), ("val_side", val_side),
                ("train_rank", float(train_rank)),
            )
        )
        if validating is not None:
            if val_side < best_loss:
                best_loss, best_epoch, stale = val_side, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, preprocessor, pd.DataFrame(history), best_epoch if validating else epochs


def _predict_all(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    config = checkpoint["config"]
    model = ParetoBoundaryModel(
        config["sensor_width"], use_economic_context=config["use_economic_context"],
        nonvisual=config["nonvisual"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x = checkpoint["preprocessor"].transform(rows[checkpoint["feature_columns"]])
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def predict_pareto_rows(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    """Replay the saved model on native frames; retain fractional knee targets."""
    native = rows.loc[rows["is_frame"]].copy()
    columns = [
        name for name in native if not name.startswith(("stat_", "dinov2_"))
    ]
    result = native[columns].reset_index(drop=True)
    logits = _predict_all(native, checkpoint)
    result["logit"] = logits
    result["decision_score"] = torch.sigmoid(torch.tensor(logits)).numpy()
    result["prediction"] = (logits >= 0).astype(int)
    return result


def _pair_summary(rows: pd.DataFrame, logits: np.ndarray, split: str) -> dict:
    pairs = relation_pairs(rows)
    return {
        "split": split, "rows": len(rows), "pairs": len(pairs),
        "finite_strict_side_rows": int(
            (np.isfinite(rows["pareto_selection_score"]) & rows["target"].isin([0, 1])).sum()
        ),
        "paired_rows": len(np.unique(pairs)),
        "pair_order_accuracy": (
            float(np.mean(np.abs(logits[pairs[:, 0]]) < np.abs(logits[pairs[:, 1]])))
            if len(pairs) else np.nan
        ),
    }


def train_pareto_fold(
    inner_train: pd.DataFrame, inner_validation: pd.DataFrame,
    outer_train: pd.DataFrame, outer_test: pd.DataFrame, *,
    use_economic_context: bool = False, use_pareto_relation: bool = False,
    nonvisual: bool = False, seed: int = 0, maximum_epochs: int = 200, patience: int = 25,
) -> dict[str, Any]:
    """Choose epoch with grouped inner BCE; refit all outer-training rows once.

    G/teacher for inner tables must exclude both outer-test and inner-validation
    experiments. Outer tables use G fitted without the outer-test experiment.
    """
    sensor = sorted(name for name in outer_train if name.startswith("stat_"))
    columns = [*sensor, *ACCOUNTING, *ECONOMIC, *RGB]
    config = {
        "sensor_width": len(sensor), "use_economic_context": use_economic_context,
        "use_pareto_relation": use_pareto_relation, "nonvisual": nonvisual, "seed": seed,
        "maximum_epochs": maximum_epochs, "patience": patience,
        "relation_rule": "adjacent_distinct_K_same_cycle_side_branch_support",
    }
    inner_model, inner_processor, losses, selected_epoch = _fit(
        inner_train, inner_validation, columns, config, maximum_epochs, patience
    )
    model, processor, _, _ = _fit(
        outer_train, None, columns, config, selected_epoch, patience
    )
    checkpoint = {
        "model_state_dict": model.state_dict(), "preprocessor": processor,
        "feature_columns": columns, "config": config, "selected_epoch": selected_epoch,
        "training_experiments": sorted(outer_train["experiment_id"].astype(str).unique()),
    }
    inner_checkpoint = {
        **checkpoint, "model_state_dict": inner_model.state_dict(),
        "preprocessor": inner_processor,
    }
    pair_metrics = pd.DataFrame([
        _pair_summary(rows, _predict_all(rows, artifact), split)
        for split, rows, artifact in (
            ("inner_train", inner_train, inner_checkpoint),
            ("inner_validation", inner_validation, inner_checkpoint),
            ("outer_train", outer_train, checkpoint),
            ("outer_test", outer_test, checkpoint),
        )
    ])
    return {
        "predictions": predict_pareto_rows(outer_test, checkpoint), "losses": losses,
        "pair_metrics": pair_metrics, "checkpoint": checkpoint,
    }
