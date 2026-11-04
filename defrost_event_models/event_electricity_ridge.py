"""Experiment-isolated Ridge fitting and replay for preparation/defrost electricity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_MODEL_FILE = Path(__file__).resolve().parents[1] / Path(
    "dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑/candidate_model_parameters.json"
)
THREE_INPUTS = ("compressor_frequency", "ambient_temperature", "coil_temperature")


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


def fit_defrost_event_ridge_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    *,
    alpha: float,
) -> RidgeOutcomeModel:
    """Fit one median-imputed, weighted-standardized target."""
    selected = frame.loc[frame[target].notna()]
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
        model = fit_defrost_event_ridge_model(train, features, target, alpha=alpha)
        residual = test[target].to_numpy(dtype=float) - model.predict(test)
        losses.append(float(np.mean(np.square(residual))))
    if not losses:
        raise ValueError("no evaluable inner LOEO folds")
    return float(np.mean(losses))


def fit_model_for_heldout_experiment(
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
    return fit_defrost_event_ridge_model(train, features, target, alpha=alpha)


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


def model_to_parameters(model: RidgeOutcomeModel) -> dict[str, object]:
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


def fit_model_on_all_experiments(
    events: pd.DataFrame, features: tuple[str, ...], target: str
) -> RidgeOutcomeModel:
    scores = {alpha: _inner_macro_mse(events, features, target, alpha) for alpha in ALPHAS}
    alpha = min(ALPHAS, key=lambda candidate: (scores[candidate], candidate))
    return fit_defrost_event_ridge_model(events, features, target, alpha=alpha)


def assemble_target_model(
    target: str,
    features: tuple[str, ...],
    folds: dict[str, RidgeOutcomeModel],
    full_model: RidgeOutcomeModel,
) -> dict[str, object]:
    return {
        "model_format_version": "v2.6.8",
        "target": target,
        "feature_order": list(features),
        "support_rule": "95th_percentile_nearest_cross_experiment_training_distance",
        "folds": {name: model_to_parameters(model) for name, model in folds.items()},
        "full_data_model": model_to_parameters(full_model),
    }


def load_defrost_event_models(path: Path | None = None) -> dict[str, Any]:
    source = path or DEFAULT_MODEL_FILE
    try:
        value: Any = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"defrost-event model file is missing: {source}") from exc
    if not isinstance(value, dict) or value.get("model_format_version") != "1":
        raise ValueError("invalid defrost-event model file")
    return value


def predict_with_model_parameters(parameters: dict[str, Any], values: pd.DataFrame) -> pd.DataFrame:
    """Apply one fitted parameter set and report its training-domain distance."""
    features = [str(name) for name in parameters["feature_order"]]
    x = values.reindex(columns=features).to_numpy(dtype=float)
    medians = np.asarray(parameters["imputer_median"], dtype=float)
    x = np.where(np.isnan(x), medians, x)
    mean = np.asarray(parameters["scaler_mean"], dtype=float)
    scale = np.asarray(parameters["scaler_scale"], dtype=float)
    z = (x - mean) / scale
    coefficients = np.asarray(parameters["coefficients"], dtype=float)
    prediction = z @ coefficients + float(parameters["intercept"])
    references = np.asarray(parameters["training_standardized_references"], dtype=float)
    distance = np.sqrt(np.square(z[:, None, :] - references[None, :, :]).sum(axis=2)).min(axis=1)
    return pd.DataFrame(
        {
            "prediction": prediction,
            "support_distance": distance,
            "support_threshold": float(parameters["support_threshold"]),
        }
    )


def predict_with_event_model(
    model: dict[str, Any],
    values: pd.DataFrame,
    experiment_id: str,
    *,
    prediction_mode: str,
) -> pd.DataFrame:
    """Use a held-out fold for retrospective replay or the full model for new data."""
    if prediction_mode == "cross-fitted":
        parameters = model.get("folds", {}).get(str(experiment_id))
        if not isinstance(parameters, dict):
            raise ValueError(f"no retrospective fold for experiment {experiment_id}")
    elif prediction_mode == "full-model":
        parameters = model.get("full_data_model")
        if not isinstance(parameters, dict):
            raise ValueError("event model has no full-data parameters")
    else:
        raise ValueError(f"unknown prediction mode: {prediction_mode}")
    return predict_with_model_parameters(parameters, values)


def prepare_training_events(events):
    """Mark valid three-input events and construct the measured electricity target."""
    events = events.copy()
    events["selected_for_training"] = (
        events.E_prep_valid.eq(True)
        & events.E_D_valid.eq(True)
        & events[list(THREE_INPUTS)].notna().all(axis=1)
        & events[[name + "_valid_second_count" for name in THREE_INPUTS]].ge(48).all(axis=1)
    )
    events["E_prep_D_kwh"] = events.E_prep_kwh + events.E_D_kwh
    if not events.selected_for_training.any():
        raise ValueError("no valid preparation/defrost events for the three-input recipe")
    return events


def fit_preparation_defrost_model(loader, events, run, workers, figures):
    """Fit admitted events through the shared Ridge fit and save validation results."""
    features = THREE_INPUTS
    training = events.loc[events.selected_for_training].copy()
    if training.empty:
        raise ValueError("no valid preparation/defrost events for the three-input recipe")
    run.mkdir(parents=True, exist_ok=True)
    experiments = sorted(training.experiment_id.astype(str).unique())
    # Parallelize outer experiments only; each nested Ridge fit stays single-threaded.
    with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
        fitted = Parallel(verbose=5)(
            delayed(fit_model_for_heldout_experiment)(training, heldout, features, "E_prep_D_kwh")
            for heldout in experiments
        )
    model = assemble_target_model(
        "E_prep_D_kwh",
        features,
        dict(zip(experiments, fitted, strict=True)),
        fit_model_on_all_experiments(training, features, "E_prep_D_kwh"),
    )
    phases = {"full_data_model": {"E_prep_D_kwh": model["full_data_model"]}, "folds": {}}
    for experiment in loader.list_cycles().experiment_id.unique():
        parameters = model["folds"].get(experiment, model["full_data_model"])
        assert experiment not in parameters["training_experiment_ids"]
        phases["folds"][experiment] = {"E_prep_D_kwh": parameters}
    bundle = dict(
        model_format_version="1",
        cop_definition="refrigerant_effective_heat",
        measurement_processing=loader.registry.get("measurement_processing", "dataset_causal_60s"),
        recovery_settings={"recovery_rule": "dataset"},
        dataset_resample_interval_seconds=loader.registry["resample_interval_seconds"],
        dataset_recovery_rule=loader.registry["recovery_edit"]["mode"],
        electricity_accounting="measured_current_recovery_plus_heating",
        phase_models=phases,
        training_source=str(run),
        training_events=len(training),
        coil_statistic="processed",
        scope="Observed preparation and defrost electricity; initiation code is not an admission criterion",
        limitation="Retrospective Dataset boundaries; early-trigger interventions are not observed",
    )
    from plots.event_calibration import event_calibration_metrics, render_calibration_figures

    validation = []
    for experiment, rows in training.groupby("experiment_id", sort=False):
        prediction = predict_with_event_model(
            model, rows, experiment, prediction_mode="cross-fitted"
        )
        validation.append(
            rows.assign(
                predicted_E_prep_D_kwh=prediction.prediction.to_numpy(),
                event_electricity_in_training_domain=prediction.support_distance.le(
                    prediction.support_threshold
                ).to_numpy(),
                representation="ridge_joint_three_input",
                panel="preparation_defrost",
                seed=0,
            )
        )
    validation = pd.concat(validation, ignore_index=True)
    events.to_csv(run / "event_admission.csv", index=False)
    training.to_csv(run / "defrost_events.csv", index=False)
    validation.to_csv(run / "model_validation.csv", index=False)
    event_calibration_metrics(validation, ["E_prep_D_kwh"]).to_csv(
        run / "validation_summary.csv", index=False
    )
    (run / "candidate_model_parameters.json").write_text(json.dumps(bundle, ensure_ascii=False))
    if figures:
        baseline = validation.copy()
        baseline["predicted_E_prep_D_kwh"] = baseline.experiment_id.map(
            {
                experiment: parameters["mean_baseline"]
                for experiment, parameters in model["folds"].items()
            }
        )
        baseline["representation"] = "experiment_balanced_mean"
        render_calibration_figures(
            pd.concat([validation, baseline], ignore_index=True),
            None,
            run / "diagnostics",
            targets=["E_prep_D_kwh"],
            bootstrap_replicates=0,
            focus_model="ridge_joint_three_input",
            labels={"E_prep_D_kwh": "Preparation + defrost electricity [kWh]"},
        )
    print(
        f"Refitted joint three-input Ridge: {len(training)} events, {training.experiment_id.nunique()} experiments",
        flush=True,
    )
