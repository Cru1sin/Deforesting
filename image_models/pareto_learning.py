"""One conditional boundary model: economic input × Pareto relation supervision.

The caller supplies fold-specific teachers/economics. This module never fits G,
changes the knee, predicts a future outcome, or chooses a control threshold.
"""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

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
TRAJECTORY_SUFFIXES = (
    "current", "mean", "std", "delta", "slope", "valid_count", "age_seconds",
    "current_missing", "mean_missing", "std_missing", "delta_missing", "slope_missing",
)
STOP_OUTPUT = {
    "row_id", "cycle_name", "experiment_id", "candidate_defrost_time", "image_time",
    "heating_start", "stable_heating_start", "teacher_time", "file_name", "target",
    "is_frame", "is_knee", "economic_c", "economic_h", "rgb_missing", "rgb_age_seconds",
    "cycle_cop_eligible_without_extrapolation",
    "cycle_heating_rate_kw_eligible_without_extrapolation",
}


class MethodName(NamedTuple):
    name: str
    label: str


class ProbeRecipe(NamedTuple):
    use_latent: bool
    use_raw: bool
    hidden_width: int | None


PROBE_RECIPES = {
    "latent_ridge_ch_linear": ProbeRecipe(True, False, None),
    "latent_ridge_ch_mlp": ProbeRecipe(True, False, 16),
    "latent_raw_ridge_ch_mlp": ProbeRecipe(True, True, 16),
    "raw_ridge_ch_mlp": ProbeRecipe(False, True, 16),
}
STATE_TRANSFER_RECIPES = {
    "d32_ridge_ch_mlp": ProbeRecipe(True, False, 16),
    "t32_ridge_ch_mlp": ProbeRecipe(True, False, 16),
    "raw_ridge_ch_mlp": PROBE_RECIPES["raw_ridge_ch_mlp"],
}


METHOD_NAMES = {
    "s0": MethodName("multimodal_latent", "RGB+sensor latent"),
    "s1": MethodName("multimodal_ridge_ch", "RGB+sensor + Ridge C/H"),
    "s2": MethodName("multimodal_ridge_ch_history", "RGB+sensor + Ridge C/H history"),
    "s3": MethodName(
        "multimodal_ridge_cho_history", "RGB+sensor + Ridge C/H/O history"
    ),
    "s4": MethodName(
        "multimodal_selected_relation", "RGB+sensor + selected inputs + relation"
    ),
    "n2": MethodName("sensor_ridge_ch_history", "Sensor-only + Ridge C/H history"),
    "s1_neural": MethodName("multimodal_neural_ch", "RGB+sensor + Neural C/H"),
    "ridge_head_ridge_ch": MethodName(
        "ridge_head_ridge_ch", "Ridge head · Ridge C/H"
    ),
    "ridge_head_neural_ch": MethodName(
        "ridge_head_neural_ch", "Ridge head · Neural C/H"
    ),
    "neural_head_ridge_ch": MethodName(
        "neural_head_ridge_ch", "Neural head · Ridge C/H"
    ),
    "neural_head_neural_ch": MethodName(
        "neural_head_neural_ch", "Neural head · Neural C/H"
    ),
    "latent_ridge_ch_linear": MethodName(
        "latent_ridge_ch_linear", "Latent + Ridge C/H · linear"
    ),
    "latent_ridge_ch_mlp": MethodName(
        "latent_ridge_ch_mlp", "Latent + Ridge C/H · MLP"
    ),
    "latent_raw_ridge_ch_mlp": MethodName(
        "latent_raw_ridge_ch_mlp", "Latent + pre-compression features + Ridge C/H"
    ),
    "raw_ridge_ch_mlp": MethodName(
        "raw_ridge_ch_mlp", "Pre-compression features + Ridge C/H"
    ),
    "d32_ridge_ch_mlp": MethodName(
        "d32_ridge_ch_mlp", "D32 state + Ridge C/H"
    ),
    "t32_ridge_ch_mlp": MethodName(
        "t32_ridge_ch_mlp", "T32 state + Ridge C/H"
    ),
}


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
    rank = signed_relation_loss(logits, target, pairs)
    return side, rank


def signed_relation_loss(
    logits: torch.Tensor, target: torch.Tensor, pairs: torch.Tensor
) -> torch.Tensor:
    """Make same-side teacher-nearer points have the smaller correct-side margin."""
    if not len(pairs):
        return logits.sum() * 0
    margin = (2 * target - 1) * logits
    return F.softplus(margin[pairs[:, 0]] - margin[pairs[:, 1]]).mean()


class StopHead(nn.Module):
    """One deliberately small readout from a frozen fold representation."""

    def __init__(self, width: int, *, hidden_width: int | None = None) -> None:
        super().__init__()
        self.output = (
            nn.Linear(width, 1)
            if hidden_width is None
            else nn.Sequential(
                nn.Linear(width, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, 1),
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(values).squeeze(1)


def stop_feature_columns(rows: pd.DataFrame, method: str) -> list[str]:
    """Compose S0-S3/N2 as explicit column groups, without positional slicing."""
    latent_prefix = "nz_" if method == "n2" else "z_"
    columns = sorted(name for name in rows if name.startswith(latent_prefix))
    if method == "s0":
        return columns
    if method == "s1":
        suffixes = ("current", "current_missing", "valid_count", "age_seconds")
        objectives = ("c", "h")
    elif method in {"s2", "s4", "n2"}:
        suffixes, objectives = TRAJECTORY_SUFFIXES, ("c", "h")
    elif method == "s3":
        suffixes, objectives = TRAJECTORY_SUFFIXES, ("c", "h", "o")
    else:
        raise ValueError(f"unknown stopping method: {method}")
    columns.extend(
        f"online_{objective}_{suffix}"
        for objective in objectives
        for suffix in suffixes
        if f"online_{objective}_{suffix}" in rows
    )
    return columns


def probe_feature_columns(
    rows: pd.DataFrame, method: str, raw_columns: list[str]
) -> list[str]:
    """Compose the four matched representation probes from explicit groups."""
    recipe = {**PROBE_RECIPES, **STATE_TRANSFER_RECIPES}[method]
    latent = sorted(name for name in rows if name.startswith("z_"))
    economic = [
        name for name in stop_feature_columns(rows, "s1") if not name.startswith("z_")
    ]
    return [
        *(latent if recipe.use_latent else []),
        *([name for name in raw_columns if name in rows] if recipe.use_raw else []),
        *economic,
    ]


def _fit_stop(
    train: pd.DataFrame, validation: pd.DataFrame | None, columns: list[str], *,
    use_relation: bool, epochs: int, patience: int, seed: int,
    hidden_width: int | None = None,
) -> tuple[StopHead, Any, pd.DataFrame, int, float]:
    supervised = train["is_frame"] | train["is_knee"]
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    ).fit(train.loc[supervised, columns])
    relation = np.isfinite(train["pareto_selection_score"]) & train["target"].isin([0, 1])
    training_rows = train.loc[supervised | relation] if use_relation else train.loc[supervised]
    training = _batch(training_rows, columns, preprocessor)
    if validation is not None:
        validation = validation.loc[validation["is_frame"] | validation["is_knee"]]
    validating = _batch(validation, columns, preprocessor) if validation is not None else None
    torch.manual_seed(seed)
    model = StopHead(len(columns), hidden_width=hidden_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history, best_loss, best_epoch, stale, best_state = [], np.inf, 1, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        side, rank = _losses(model(training[0]), training)
        objective = side + rank if use_relation else side
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
            {"epoch": epoch, "split": split, "loss": value}
            for split, value in (
                ("train_side", float(train_side)), ("val_side", val_side),
                ("train_rank", float(train_rank)),
            )
        )
        if validation is not None:
            if val_side < best_loss:
                best_loss, best_epoch, stale = val_side, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, preprocessor, pd.DataFrame(history), best_epoch, float(best_loss)


def train_stop_fold(
    inner_train: pd.DataFrame, inner_validation: pd.DataFrame,
    outer_train: pd.DataFrame, outer_test: pd.DataFrame, *, method: str,
    base_method: str | None = None, seed: int = 0, maximum_epochs: int = 200,
    patience: int = 25, feature_columns: list[str] | None = None,
    hidden_width: int | None = None,
) -> dict[str, Any]:
    """Train one stopping probe; S4 reuses the inner-selected base feature group."""
    feature_method = base_method if method == "s4" else method
    if feature_method is None:
        raise ValueError("S4 requires the inner-selected base method")
    columns = feature_columns or stop_feature_columns(outer_train, feature_method)
    use_relation = method == "s4"
    inner = _fit_stop(
        inner_train, inner_validation, columns, use_relation=use_relation,
        epochs=maximum_epochs, patience=patience, seed=seed, hidden_width=hidden_width,
    )
    outer = _fit_stop(
        outer_train, None, columns, use_relation=use_relation,
        epochs=inner[3], patience=patience, seed=seed, hidden_width=hidden_width,
    )
    checkpoint = {
        "model_state_dict": outer[0].state_dict(), "preprocessor": outer[1],
        "feature_columns": columns, "selected_epoch": inner[3], "method": method,
        "base_method": feature_method,
        "hidden_width": hidden_width,
    }
    inner_checkpoint = {
        **checkpoint, "model_state_dict": inner[0].state_dict(), "preprocessor": inner[1],
    }
    pair_metrics = pd.DataFrame([
        _pair_summary(rows, _predict_stop_all(rows, artifact), split)
        for split, rows, artifact in (
            ("inner_train", inner_train, inner_checkpoint),
            ("inner_validation", inner_validation, inner_checkpoint),
            ("outer_train", outer_train, checkpoint),
            ("outer_test", outer_test, checkpoint),
        )
    ])
    return {
        "checkpoint": checkpoint, "losses": inner[2], "validation_loss": inner[4],
        "predictions": predict_stop_rows(outer_test, checkpoint), "pair_metrics": pair_metrics,
    }


def train_stop_fixed(
    train: pd.DataFrame, test: pd.DataFrame, *, method: str, epochs: int, seed: int = 0,
    feature_columns: list[str] | None = None, hidden_width: int | None = None,
) -> dict[str, Any]:
    """Fit one frozen-representation stopping head for a preselected epoch budget."""
    columns = feature_columns or stop_feature_columns(train, method)
    fitted = _fit_stop(
        train, None, columns, use_relation=False, epochs=epochs, patience=epochs,
        seed=seed, hidden_width=hidden_width,
    )
    checkpoint = {
        "model_state_dict": fitted[0].state_dict(), "preprocessor": fitted[1],
        "feature_columns": columns, "selected_epoch": epochs, "method": method,
        "base_method": method,
        "hidden_width": hidden_width,
    }
    return {
        "checkpoint": checkpoint, "losses": fitted[2],
        "predictions": predict_stop_rows(test, checkpoint),
    }


def _predict_stop_all(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> np.ndarray:
    columns = checkpoint["feature_columns"]
    model = StopHead(len(columns), hidden_width=checkpoint.get("hidden_width"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    values = checkpoint["preprocessor"].transform(rows[columns])
    with torch.no_grad():
        return model(torch.tensor(values, dtype=torch.float32)).numpy()


def predict_stop_rows(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    """Replay one frozen head on native frames only."""
    native = rows.loc[rows["is_frame"]].copy()
    logits = _predict_stop_all(native, checkpoint)
    metadata = [
        name for name in native
        if name in STOP_OUTPUT
        or name.startswith("online_")
        or name.endswith(("_in_training_domain", "_measurement_reconstructed"))
    ]
    result = native[metadata].reset_index(drop=True)
    result["logit"] = logits
    result["decision_score"] = torch.sigmoid(torch.tensor(logits)).numpy()
    result["prediction"] = (logits >= 0).astype(int)
    return result


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
