"""Independent one-target Ridge fitting and replay for V2.6.8."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.1, 1.0, 10.0, 100.0)
STATIC_5 = (
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
)
PHYSICAL_STATIC_6 = (
    "ambient_temperature",
    "mean_water_temperature",
    "setpoint_outlet_difference",
    "ambient_coil_difference",
    "evaporating_pressure",
    "compressor_frequency",
)
DYNAMIC_8 = (*PHYSICAL_STATIC_6, "heating_elapsed_minutes", "evaporating_pressure_slope_5m")
MODEL_FEATURES = {
    "ticket_ridge_static5": STATIC_5,
    "ticket_ridge_physical6": PHYSICAL_STATIC_6,
    "ticket_ridge_dynamic8": DYNAMIC_8,
}


def experiment_weights(groups: pd.Series) -> np.ndarray:
    values = groups.astype(str)
    counts = values.map(values.value_counts()).to_numpy(dtype=float)
    return cast(np.ndarray, np.asarray(len(values) / (values.nunique() * counts), dtype=float))


@dataclass
class RidgeOutcomeModel:
    features: tuple[str, ...]
    target: str
    imputer: SimpleImputer
    scaler: StandardScaler
    ridge: Ridge
    alpha: float
    sample_weight_sum: float
    support_threshold: float
    training_z: np.ndarray
    training_groups: np.ndarray
    training_event_ids: np.ndarray
    target_mean: float

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(values[list(self.features)])
        return cast(np.ndarray, np.asarray(self.scaler.transform(imputed), dtype=float))

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, np.asarray(self.ridge.predict(self.transform(values)), dtype=float))

    def support_distance(self, values: pd.DataFrame) -> np.ndarray:
        z = self.transform(values)
        distances = np.sqrt(np.square(z[:, None, :] - self.training_z[None, :, :]).sum(axis=2))
        return cast(np.ndarray, distances.min(axis=1))


def _cross_experiment_support_threshold(z: np.ndarray, groups: np.ndarray) -> float:
    distances = np.sqrt(np.square(z[:, None, :] - z[None, :, :]).sum(axis=2))
    nearest = np.where(groups[:, None] != groups[None, :], distances, np.inf).min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.quantile(finite, 0.95)) if finite.size else float("inf")


def fit_weighted_ridge(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    *,
    alpha: float,
) -> RidgeOutcomeModel:
    """Fit one median-imputed, weighted-standardized target."""
    selected = frame.loc[frame[target].notna()].copy()
    if selected.empty:
        raise ValueError(f"no complete training targets for {target}")
    weights = experiment_weights(selected["experiment_id"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(selected[list(features)])
    scaler = StandardScaler().fit(imputed, sample_weight=weights)
    z = np.asarray(scaler.transform(imputed), dtype=float)
    y = selected[target].to_numpy(dtype=float)
    ridge = Ridge(alpha=alpha).fit(z, y, sample_weight=weights)
    groups = selected["experiment_id"].astype(str).to_numpy()
    event_ids = (
        selected.get("event_id", pd.Series(selected.index.astype(str), index=selected.index))
        .astype(str)
        .to_numpy()
    )
    return RidgeOutcomeModel(
        features=features,
        target=target,
        imputer=imputer,
        scaler=scaler,
        ridge=ridge,
        alpha=float(alpha),
        sample_weight_sum=float(weights.sum()),
        support_threshold=_cross_experiment_support_threshold(z, groups),
        training_z=z,
        training_groups=groups,
        training_event_ids=event_ids,
        target_mean=float(np.average(y, weights=weights)),
    )


def _inner_macro_mse(
    frame: pd.DataFrame, features: tuple[str, ...], target: str, alpha: float
) -> float:
    losses: list[float] = []
    for heldout in frame["experiment_id"].dropna().astype(str).unique():
        train = frame.loc[~frame["experiment_id"].astype(str).eq(heldout)]
        test = frame.loc[frame["experiment_id"].astype(str).eq(heldout) & frame[target].notna()]
        if train["experiment_id"].nunique() < 2 or test.empty:
            continue
        model = fit_weighted_ridge(train, features, target, alpha=alpha)
        residual = test[target].to_numpy(dtype=float) - model.predict(test)
        losses.append(float(np.mean(np.square(residual))))
    if not losses:
        raise ValueError("no evaluable inner LOEO folds")
    return float(np.mean(losses))


def fit_outcome_fold(
    events: pd.DataFrame,
    heldout_experiment: str,
    features: tuple[str, ...],
    target: str,
) -> RidgeOutcomeModel:
    """Select alpha by nested experiment LOEO, then fit one outer fold."""
    train = events.loc[
        ~events["experiment_id"].astype(str).eq(str(heldout_experiment)) & events[target].notna()
    ].copy()
    scores = {alpha: _inner_macro_mse(train, features, target, alpha) for alpha in ALPHAS}
    alpha = min(ALPHAS, key=lambda candidate: (scores[candidate], candidate))
    return fit_weighted_ridge(train, features, target, alpha=alpha)


def _raw_formula(model: RidgeOutcomeModel) -> str:
    terms = [
        f"({coefficient!r})*((x[{name!r}]-{mean!r})/{scale!r})"
        for name, coefficient, mean, scale in zip(
            model.features,
            np.asarray(model.ridge.coef_, dtype=float),
            np.asarray(model.scaler.mean_, dtype=float),
            np.asarray(model.scaler.scale_, dtype=float),
            strict=True,
        )
    ]
    return f"{float(model.ridge.intercept_)!r}+" + "+".join(terms)


def model_to_artifact(model: RidgeOutcomeModel) -> dict[str, object]:
    return {
        "feature_order": list(model.features),
        "target": model.target,
        "imputer_median": np.asarray(model.imputer.statistics_, dtype=float).tolist(),
        "scaler_mean": np.asarray(model.scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(model.scaler.scale_, dtype=float).tolist(),
        "alpha": model.alpha,
        "coefficients": np.asarray(model.ridge.coef_, dtype=float).tolist(),
        "intercept": float(model.ridge.intercept_),
        "raw_formula": _raw_formula(model),
        "support_threshold": model.support_threshold,
        "training_standardized_references": model.training_z.tolist(),
        "training_experiment_ids": model.training_groups.astype(str).tolist(),
        "training_event_ids": model.training_event_ids.astype(str).tolist(),
        "training_event_count": int(len(model.training_z)),
        "training_experiment_count": int(len(set(model.training_groups.astype(str)))),
        "sample_weight_sum": model.sample_weight_sum,
        "mean_baseline": model.target_mean,
    }


def fit_full_outcome(
    events: pd.DataFrame, features: tuple[str, ...], target: str
) -> RidgeOutcomeModel:
    scores = {alpha: _inner_macro_mse(events, features, target, alpha) for alpha in ALPHAS}
    alpha = min(ALPHAS, key=lambda candidate: (scores[candidate], candidate))
    return fit_weighted_ridge(events, features, target, alpha=alpha)


def assemble_target_artifact(
    target: str,
    features: tuple[str, ...],
    folds: dict[str, RidgeOutcomeModel],
    full_model: RidgeOutcomeModel,
) -> dict[str, object]:
    return {
        "artifact_version": "v2.6.8",
        "target": target,
        "feature_order": list(features),
        "support_policy": "95th_percentile_nearest_cross_experiment_training_distance",
        "folds": {name: model_to_artifact(model) for name, model in folds.items()},
        "full_data_model": model_to_artifact(full_model),
    }


def mean_outcome_artifact(events: pd.DataFrame, target: str) -> dict[str, object]:
    """Build one experiment-balanced mean target model for the supplied training rows."""
    selected = events.loc[events[target].notna()]
    if selected.empty:
        raise ValueError(f"no complete training targets for {target}")
    weights = experiment_weights(selected["experiment_id"])
    mean = float(np.average(selected[target].to_numpy(dtype=float), weights=weights))
    return {
        "feature_order": [],
        "target": target,
        "imputer_median": [],
        "scaler_mean": [],
        "scaler_scale": [],
        "alpha": 0.0,
        "coefficients": [],
        "intercept": mean,
        "raw_formula": repr(mean),
        "support_threshold": 0.0,
        "training_standardized_references": [[] for _ in range(len(selected))],
        "training_experiment_ids": selected["experiment_id"].astype(str).tolist(),
        "training_event_ids": selected["event_id"].astype(str).tolist(),
        "training_event_count": len(selected),
        "training_experiment_count": selected["experiment_id"].nunique(),
        "sample_weight_sum": float(weights.sum()),
        "mean_baseline": mean,
    }


def load_artifacts(path: Path | None = None) -> dict[str, Any]:
    source = path or Path(__file__).with_name("params") / "ticket_ridge_models.json"
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"V2.6.8 parameter artifact is missing: {source}") from exc
    if not isinstance(value, dict) or value.get("artifact_version") != "v2.6.8":
        raise ValueError("invalid V2.6.8 parameter artifact")
    return value


def predict_from_artifact(
    artifact: dict[str, Any], values: pd.DataFrame, experiment_id: str
) -> pd.DataFrame:
    fold = artifact.get("folds", {}).get(str(experiment_id))
    if not isinstance(fold, dict):
        raise ValueError(f"no retrospective fold for experiment {experiment_id}")
    features = [str(name) for name in fold["feature_order"]]
    x = values.reindex(columns=features).to_numpy(dtype=float)
    medians = np.asarray(fold["imputer_median"], dtype=float)
    x = np.where(np.isnan(x), medians, x)
    mean = np.asarray(fold["scaler_mean"], dtype=float)
    scale = np.asarray(fold["scaler_scale"], dtype=float)
    z = (x - mean) / scale
    coefficients = np.asarray(fold["coefficients"], dtype=float)
    prediction = z @ coefficients + float(fold["intercept"])
    references = np.asarray(fold["training_standardized_references"], dtype=float)
    distance = np.sqrt(np.square(z[:, None, :] - references[None, :, :]).sum(axis=2)).min(axis=1)
    return pd.DataFrame({"prediction": prediction, "support_distance": distance})


def predict_independent_targets(
    energy_artifact: dict[str, Any],
    heat_artifact: dict[str, Any],
    values: pd.DataFrame,
    experiment_id: str,
) -> pd.DataFrame:
    energy = predict_from_artifact(energy_artifact, values, experiment_id)
    heat = predict_from_artifact(heat_artifact, values, experiment_id)
    e_fold = energy_artifact["folds"][experiment_id]
    q_fold = heat_artifact["folds"][experiment_id]
    e_supported = energy["support_distance"].le(float(e_fold["support_threshold"]))
    q_supported = heat["support_distance"].le(float(q_fold["support_threshold"]))
    e_evaluable = np.isfinite(energy["prediction"])
    q_evaluable = np.isfinite(heat["prediction"])
    return pd.DataFrame(
        {
            "transition_energy_kwh": energy["prediction"].to_numpy(),
            "transition_heat_kwh": heat["prediction"].to_numpy(),
            "E_support_distance": energy["support_distance"].to_numpy(),
            "Q_support_distance": heat["support_distance"].to_numpy(),
            "ET_evaluable": e_evaluable,
            "QT_evaluable": q_evaluable,
            "ET_in_support": e_supported.to_numpy(),
            "QT_in_support": q_supported.to_numpy(),
        }
    )
