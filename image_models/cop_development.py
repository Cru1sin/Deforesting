"""Grouped development CV for the conditional near-optimal COP classifier."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from dataset_tools import DatasetLoader
from defrost_decision.baselines.electricity import integrate_heating_curve
from defrost_decision.candidate_quantities import measure_candidate_quantities
from image_models.relative_cop import (
    RGB,
    SENSORS,
    STATISTICS,
    apply_reference,
    build_fold_rows,
    cycle_weights,
    near_optimal_labels,
)
from image_models.sensor_features import build_past_only_sensor_statistics
from plots.image_models import two_of_three_trigger

CHECKPOINT_EPOCHS = (1, 3, 5, 10, 20, 30, 50)
THRESHOLDS = tuple(np.arange(5, 100, 5) / 100)


def development_feature_columns():
    numeric = [
        *[f"stat_{sensor}_{statistic}" for sensor in SENSORS for statistic in STATISTICS],
        "causal_elapsed_minutes",
        "causal_total_electricity_kwh",
    ]
    return numeric, RGB.copy()


class StaticNearOptimalClassifier(nn.Module):
    def __init__(self, numeric_width, visual_width=384):
        super().__init__()
        self.numeric = nn.Sequential(
            nn.Linear(numeric_width, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
        )
        self.visual = nn.Sequential(
            nn.Linear(visual_width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
        ) if visual_width else None
        self.classifier = nn.Sequential(
            nn.Linear(64 if self.visual is not None else 32, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, numeric, visual=None):
        state = self.numeric(numeric)
        if self.visual is None:
            return self.classifier(state).squeeze(1)
        image = (
            self.visual(visual)
            if visual is not None
            else torch.zeros((len(numeric), 32), dtype=state.dtype, device=state.device)
        )
        return self.classifier(torch.cat([state, image], dim=1)).squeeze(1)


def candidate_clock(boundary):
    start = pd.Timestamp(boundary.heating_start)
    recovery = pd.Timestamp(boundary.stable_heating_start)
    end = pd.to_datetime(boundary.observed_defrost_preparation_start, errors="coerce")
    if pd.isna(end):
        end = pd.Timestamp(boundary.observation_end)
    times = pd.date_range(start, end, freq="30s")
    times = times[times < end]
    return pd.DataFrame({
        "cycle_name": boundary.cycle_name,
        "experiment_id": str(boundary.experiment_id),
        "candidate_defrost_time": times,
        "heating_accounting_start": recovery,
        "causal_elapsed_minutes": (times - start).total_seconds() / 60,
    })


def cycle_equal_weights(groups):
    """Mean-one form of the shared cycle-equal weights."""
    return cycle_weights(groups) * len(groups)


def label_complete_curves(rows, epsilon=.01, reference_max_by_cycle=None):
    result = rows.copy()
    labels = pd.Series(np.nan, index=result.index)
    for name, cycle in result.groupby("cycle_name", sort=False):
        supported = cycle.cycle_cop_eligible.fillna(False) & np.isfinite(cycle.cycle_cop)
        peak = cycle.loc[supported, "cycle_cop"].max()
        if reference_max_by_cycle is not None:
            candidates = [peak, reference_max_by_cycle.get(name, np.nan)]
            finite = [value for value in candidates if pd.notna(value)]
            peak = max(finite) if finite else np.nan
        if pd.notna(peak) and peak > 0:
            relative = cycle.loc[supported, "cycle_cop"] / peak
            labels.loc[relative.index] = near_optimal_labels(relative, epsilon)
    result["binary_target"] = labels
    return result


def _present_class_scores(target, predicted):
    labels = sorted(target.unique())
    ba = recall_score(target, predicted, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(target, predicted, labels=labels, average="macro", zero_division=0)
    return float(ba), float(f1), len(labels) == 2


def _proposal(cycle, threshold, strategy):
    ordered = cycle.sort_values("candidate_defrost_time")
    valid = ordered.probability.notna()
    if strategy == "first_positive":
        times = ordered.loc[valid & ordered.probability.ge(threshold), "candidate_defrost_time"]
        trigger = times.iloc[0] if len(times) else pd.NaT
    else:
        trigger, _ = two_of_three_trigger(
            ordered.candidate_defrost_time,
            ordered.probability.ge(threshold).where(valid).astype(float).fillna(0),
            .5,
        )
    point = ordered.loc[ordered.candidate_defrost_time.eq(trigger)]
    if pd.isna(trigger):
        return trigger, "no_proposal", np.nan
    label = point.binary_target.iloc[0] if len(point) else np.nan
    if pd.isna(label):
        return trigger, "unsupported_proposal", np.nan
    return trigger, ("supported_near_hit" if bool(label) else "supported_miss"), float(label)


def _training_targets(hard_targets, loss):
    if loss == "bce":
        return hard_targets
    if loss == "label-smoothing":
        return .9 * hard_targets + .05
    raise ValueError(f"unknown development loss: {loss}")


def _probability_metrics(target, probability, weights):
    if not len(target):
        return np.nan, np.nan, np.nan
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    brier = np.average((probability - target) ** 2, weights=weights)
    nll = np.average(
        -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)),
        weights=weights,
    )
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.
    for index in range(10):
        selected = bins == index
        if selected.any():
            bin_weight = weights[selected].sum()
            ece += bin_weight * abs(
                np.average(probability[selected], weights=weights[selected])
                - np.average(target[selected], weights=weights[selected])
            )
    ece /= weights.sum()
    return float(brier), float(ece), float(nll)


def score_classifier(rows, threshold):
    valid = rows.loc[rows.binary_target.notna() & rows.probability.notna()].copy()
    weights = cycle_equal_weights(valid.cycle_name) if len(valid) else np.array([])
    cycle_rows = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        labeled = cycle.loc[cycle.binary_target.notna() & cycle.probability.notna()]
        available_frames = int(cycle.probability.notna().sum())
        record = {
            "cycle_name": name, "available_frames": available_frames,
            "status": (
                "evaluated" if len(labeled) else
                "no_supported_labels" if available_frames else "no_prediction"
            ),
        }
        if len(labeled):
            predicted = labeled.probability.ge(threshold).astype(int)
            ba, macro_f1, both = _present_class_scores(
                labeled.binary_target.astype(int), predicted
            )
            record.update(balanced_accuracy=ba, macro_f1=macro_f1,
                          both_classes=both)
        else:
            record.update(balanced_accuracy=np.nan, macro_f1=np.nan,
                          both_classes=False)
        for strategy in ("first_positive", "two_of_three"):
            trigger, status, hit = _proposal(cycle, threshold, strategy)
            record[f"{strategy}_time"] = trigger
            record[f"{strategy}_status"] = status
            record[f"{strategy}_supported_near_hit"] = hit
        cycle_rows.append(record)
    cycles = pd.DataFrame(cycle_rows)
    if valid.empty or valid.binary_target.nunique() < 2:
        ap = np.nan
    else:
        ap = average_precision_score(valid.binary_target, valid.probability,
                                     sample_weight=weights)
    predicted = valid.probability.ge(threshold) if len(valid) else pd.Series(dtype=bool)
    positive = valid.binary_target.eq(1) if len(valid) else pd.Series(dtype=bool)
    tp = float(weights[positive & predicted].sum()) if len(valid) else 0.
    fp = float(weights[~positive & predicted].sum()) if len(valid) else 0.
    fn = float(weights[positive & ~predicted].sum()) if len(valid) else 0.
    tn = float(weights[~positive & ~predicted].sum()) if len(valid) else 0.
    brier, ece, nll = _probability_metrics(
        valid.binary_target, valid.probability, weights
    )
    classified_cycle_count = int(cycles.balanced_accuracy.notna().sum())
    metrics = {
        "cycle_weighted_average_precision": float(ap),
        "cycle_weighted_brier_score": brier,
        "cycle_weighted_expected_calibration_error": ece,
        "cycle_weighted_negative_log_likelihood": nll,
        "balanced_accuracy": float(cycles.balanced_accuracy.mean()),
        "macro_f1": float(cycles.macro_f1.mean()),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "classified_frames": len(valid),
        "validation_cycles": len(cycles),
        "classified_cycle_count": classified_cycle_count,
        "both_class_cycles": int(cycles.both_classes.sum()),
        "single_class_cycles": int(classified_cycle_count - cycles.both_classes.sum()),
        "no_prediction_cycles": int(cycles.status.eq("no_prediction").sum()),
        "no_supported_labels_cycles": int(
            cycles.status.eq("no_supported_labels").sum()
        ),
    }
    return metrics, cycles


def select_epoch_threshold(grid):
    required = ["balanced_accuracy", "macro_f1", "validation_bce"]
    selectable = grid.loc[np.isfinite(grid[required]).all(axis=1)]
    if selectable.empty:
        raise ValueError("no selectable epoch/threshold configuration")
    return selectable.sort_values(
        ["balanced_accuracy", "macro_f1", "validation_bce", "epoch", "threshold"],
        ascending=[False, False, True, True, False], na_position="last",
        kind="stable",
    ).iloc[0]


def _front_images(metadata, cache_path, visual):
    images = metadata.loc[
        metadata.camera_role.eq("front"), ["image_time", "file_name"]
    ].copy()
    images["image_time"] = pd.to_datetime(images.image_time)
    if cache_path.exists():
        features = pd.read_parquet(cache_path)
        features = features.loc[
            features.camera_role.eq("front"), ["file_name", *visual]
        ]
        return images.merge(features, on="file_name", how="left", validate="one_to_one")
    missing = pd.DataFrame(np.nan, index=images.index, columns=visual)
    return pd.concat([images, missing], axis=1)


def _add_visual_history_rows(reference, images, visual):
    available = images.loc[images[visual].notna().all(axis=1)].sort_values("image_time")
    past_columns = [f"past_{column}" for column in visual]
    history = available.rename(columns={
        "image_time": "past_image_time", "file_name": "past_file_name",
        **dict(zip(visual, past_columns)),
    })[["past_image_time", "past_file_name", *past_columns]]
    result = reference.sort_values("candidate_defrost_time").copy()
    result["history_match_deadline"] = (
        pd.to_datetime(result.candidate_defrost_time) - pd.Timedelta(seconds=150)
    )
    result = pd.merge_asof(
        result, history.sort_values("past_image_time"),
        left_on="history_match_deadline", right_on="past_image_time",
        direction="backward", allow_exact_matches=True,
        tolerance=pd.Timedelta(seconds=45),
    ).drop(columns="history_match_deadline")
    deltas = pd.DataFrame(
        result[visual].to_numpy() - result[past_columns].to_numpy(),
        index=result.index, columns=[f"delta_{column}" for column in visual],
    )
    result = pd.concat([result, deltas], axis=1)
    result["visual_history_available"] = result[past_columns].notna().all(axis=1)
    result["development_input_available"] = (
        result.joint_input_available.fillna(False) & result.visual_history_available
    )
    return result


def _add_visual_history(args, reference, cycle_name):
    _, visual = development_feature_columns()
    loader = DatasetLoader(args.dataset)
    images = _front_images(
        loader.load_image_metadata(cycle_name),
        args.rgb_cache / f"{cycle_name}.parquet",
        visual,
    )
    return _add_visual_history_rows(reference, images, visual)


def _input_available(rows):
    column = (
        "development_input_available"
        if "development_input_available" in rows else "joint_input_available"
    )
    return rows[column].fillna(False)


def _model_visual_columns(mechanism):
    _, visual = development_feature_columns()
    return (
        [*visual, *[f"delta_{column}" for column in visual]]
        if mechanism == "delta" else visual
    )


def _prepare_cycle(args, boundary):
    path = args.output / "base" / f"{boundary.cycle_name}.parquet"
    if path.exists():
        if (
            getattr(args, "require_visual_history", False)
            or args.development_mechanism == "delta"
        ):
            reference = pd.read_parquet(path)
            if "development_input_available" not in reference:
                _add_visual_history(args, reference, boundary.cycle_name).to_parquet(
                    path, index=False
                )
        return
    loader = DatasetLoader(args.dataset)
    clock = candidate_clock(boundary)
    raw = loader.load_cycle_original(boundary.cycle_name)
    reference = measure_candidate_quantities(
        raw,
        clock.drop(columns="causal_elapsed_minutes"),
        pd.Timestamp(boundary.heating_start),
        heat_column="heating_capacity",
    )
    causal = integrate_heating_curve(
        raw.timestamp,
        raw.power_total,
        clock.candidate_defrost_time,
        pd.Timestamp(boundary.heating_start),
        "strict_causal",
        causal_candidates=True,
    )
    reference["causal_elapsed_minutes"] = clock.causal_elapsed_minutes
    reference["causal_total_electricity_kwh"] = causal.strict_energy_kwh.where(
        causal.strict_coverage.gt(0)
    ).to_numpy()
    reference["causal_total_electricity_coverage"] = causal.strict_coverage.to_numpy()
    reference["cycle_name"] = boundary.cycle_name
    reference["experiment_id"] = str(boundary.experiment_id)
    reference["heating_start"] = pd.Timestamp(boundary.heating_start)
    reference["stable_heating_start"] = pd.Timestamp(boundary.stable_heating_start)
    reference["t_RB"] = pd.to_datetime(boundary.t_RB)

    processed = loader.load_cycle(boundary.cycle_name)
    bucket = int(loader.registry["resample_interval_seconds"])
    stats = build_past_only_sensor_statistics(
        processed, current_sensors=SENSORS, bucket_seconds=bucket, include_current=True
    )
    columns, visual = development_feature_columns()
    stats = stats[["sensor_timestamp", *columns[:-2]]]
    reference = pd.merge_asof(
        reference.sort_values("candidate_defrost_time"),
        stats.sort_values("sensor_timestamp"),
        left_on="candidate_defrost_time", right_on="sensor_timestamp",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(seconds=15),
    )

    images = _front_images(
        loader.load_image_metadata(boundary.cycle_name),
        args.rgb_cache / f"{boundary.cycle_name}.parquet",
        visual,
    )
    images = images.loc[images[visual].notna().all(axis=1)].sort_values("image_time")
    reference = pd.merge_asof(
        reference, images, left_on="candidate_defrost_time", right_on="image_time",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(seconds=45),
    )
    reference["rgb_available"] = reference[visual].notna().all(axis=1)
    has_sensor_value = np.isfinite(reference[columns[:-2]]).any(axis=1)
    reference["joint_input_available"] = (
        reference.sensor_timestamp.notna() & has_sensor_value & reference.rgb_available
    )
    if (
        getattr(args, "require_visual_history", False)
        or args.development_mechanism == "delta"
    ):
        reference = _add_visual_history(args, reference, boundary.cycle_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(path, index=False)


def _tensors(
    rows, numeric_preprocessor, visual_preprocessor, rgb, numeric_columns=None,
    mechanism="static",
):
    numeric, _ = development_feature_columns()
    visual = _model_visual_columns(mechanism)
    numeric = numeric if numeric_columns is None else numeric_columns
    x_numeric = torch.tensor(numeric_preprocessor.transform(rows[numeric]), dtype=torch.float32)
    x_visual = (
        torch.tensor(visual_preprocessor.transform(rows[visual]), dtype=torch.float32)
        if rgb == "rgb" else None
    )
    return x_numeric, x_visual


def _fit_method(
    train, validation, method, fold_output, maximum_epochs, seed, batch_size,
    development_loss="bce", mechanism="static", thresholds=THRESHOLDS,
    compact_predictions=False,
):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    fold_output.mkdir(parents=True, exist_ok=True)
    saved_epochs = [epoch for epoch in CHECKPOINT_EPOCHS if epoch <= maximum_epochs]
    complete = (
        (fold_output / "threshold_grid.csv").exists()
        and (fold_output / "preprocessors.pkl").exists()
        and all((fold_output / f"checkpoint_epoch_{epoch}.pt").exists()
                and (fold_output / f"validation_epoch_{epoch}.parquet").exists()
                for epoch in saved_epochs)
    )
    if complete:
        return pd.read_csv(fold_output / "threshold_grid.csv")
    if train.empty:
        raise ValueError(f"{fold_output}: no labeled shared-input training rows")
    validation_labeled = (
        _input_available(validation) & validation.binary_target.notna()
    )
    if not validation_labeled.any():
        raise ValueError(f"{fold_output}: no labeled shared-input validation rows")
    all_numeric, _ = development_feature_columns()
    visual = _model_visual_columns(mechanism)
    numeric = [column for column in all_numeric if train[column].notna().any()]
    numeric_preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    visual_preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    numeric_preprocessor.fit(train[numeric])
    if method == "rgb":
        visual_preprocessor.fit(train[visual])
    x, image = _tensors(
        train, numeric_preprocessor, visual_preprocessor, method, numeric, mechanism
    )
    hard_y = torch.tensor(train.binary_target.to_numpy(), dtype=torch.float32)
    y = _training_targets(hard_y, development_loss)
    weights = torch.tensor(cycle_equal_weights(train.cycle_name), dtype=torch.float32)
    model = StaticNearOptimalClassifier(
        len(numeric), len(visual) if method == "rgb" else 0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    metric_rows = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(x[index], image[index] if image is not None else None)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, y[index], reduction="none"
            ) * weights[index]).mean()
            loss.backward()
            optimizer.step()
        if epoch not in CHECKPOINT_EPOCHS:
            continue
        model.eval()
        available = _input_available(validation)
        vx, vimage = _tensors(
            validation.loc[available], numeric_preprocessor, visual_preprocessor,
            method, numeric, mechanism,
        )
        with torch.no_grad():
            logits = model(vx, vimage)
        predicted = validation.assign(probability=np.nan)
        predicted.loc[available, "probability"] = logits.sigmoid().numpy()
        labeled = available & predicted.binary_target.notna()
        labeled_available = predicted.loc[labeled]
        validation_weights = torch.tensor(
            cycle_equal_weights(labeled_available.cycle_name), dtype=torch.float32
        )
        validation_probability = torch.tensor(
            labeled_available.probability.to_numpy(), dtype=torch.float32
        )
        validation_bce = float((nn.functional.binary_cross_entropy(
            validation_probability,
            torch.tensor(labeled_available.binary_target.to_numpy(), dtype=torch.float32),
            reduction="none",
        ) * validation_weights).mean()) if labeled.any() else np.nan
        saved = predicted
        if compact_predictions:
            columns = [
                "cycle_name", "experiment_id", "candidate_defrost_time",
                "binary_target", "joint_input_available", "probability",
            ]
            if "development_input_available" in predicted:
                columns.insert(-1, "development_input_available")
            saved = predicted[columns]
        saved.to_parquet(fold_output / f"validation_epoch_{epoch}.parquet", index=False)
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                   fold_output / f"checkpoint_epoch_{epoch}.pt")
        for threshold in thresholds:
            metrics, _ = score_classifier(predicted, threshold)
            metric_rows.append({"epoch": epoch, "threshold": threshold,
                                "validation_bce": validation_bce, **metrics})
    grid = pd.DataFrame(metric_rows)
    grid.to_csv(fold_output / "threshold_grid.csv", index=False)
    with (fold_output / "preprocessors.pkl").open("wb") as stream:
        pickle.dump({
            "numeric_columns": numeric,
            "visual_columns": visual if method == "rgb" else [],
            "numeric_preprocessor": numeric_preprocessor,
            "visual_preprocessor": visual_preprocessor if method == "rgb" else None,
        }, stream)
    return grid


def _fold_rows(args, cohort, events, validation_experiments, base_root=None):
    train_cohort = cohort.loc[~cohort.experiment_id.isin(validation_experiments)]
    validation_cohort = cohort.loc[cohort.experiment_id.isin(validation_experiments)]
    train, train_models = build_fold_rows(
        args, train_cohort, events, tuple(validation_experiments),
        include_history=False, base_root=base_root,
    )
    validation, validation_models = build_fold_rows(
        args, validation_cohort, events, tuple(validation_experiments),
        include_history=False, base_root=base_root,
    )
    for model in validation_models.values():
        assert not set(model["training_experiment_ids"]) & set(validation_experiments)
    train = label_complete_curves(
        train, args.near_optimal_epsilon,
        _old_grid_peaks(args, train_cohort, train_models, validation_experiments),
    )
    validation = label_complete_curves(
        validation, args.near_optimal_epsilon,
        _old_grid_peaks(
            args, validation_cohort, validation_models, validation_experiments
        ),
    )
    return train, validation, train_models, validation_models


def _old_grid_peaks(args, cohort, models, excluded=()):
    peaks = {}
    for experiment, cycles in cohort.groupby("experiment_id", sort=False):
        omitted = frozenset((*excluded, experiment))
        model = models["__".join(sorted(omitted))]
        for name in cycles.cycle_name:
            old = apply_reference(
                pd.read_parquet(args.data / "base" / f"{name}.parquet"),
                model, "off", include_history=False,
            )
            supported = old.cycle_cop_eligible.fillna(False) & np.isfinite(old.cycle_cop)
            peaks[name] = old.loc[supported, "cycle_cop"].max()
    return peaks


def _develop_fold(args, cohort, events, fold, validation_experiments, method):
    train, validation, _, _ = _fold_rows(args, cohort, events, validation_experiments)
    train = train.loc[_input_available(train) & train.binary_target.notna()].copy()
    _fit_method(
        train, validation, method, args.output / "folds" / fold / method,
        args.maximum_epochs, args.seed, args.batch_size, args.development_loss,
        args.development_mechanism,
    )
    return fold


def _pooled_threshold_grid(folders, maximum_epochs):
    pooled = []
    for epoch in (value for value in CHECKPOINT_EPOCHS if value <= maximum_epochs):
        predicted = pd.concat([
            pd.read_parquet(folder / f"validation_epoch_{epoch}.parquet").assign(
                fold=fold
            )
            for fold, folder in folders.items()
        ], ignore_index=True)
        labeled = predicted.loc[
            predicted.binary_target.notna() & predicted.probability.notna()
        ]
        weights = cycle_equal_weights(labeled.cycle_name) if len(labeled) else np.array([])
        validation_bce = float(np.average(
            -(labeled.binary_target * np.log(labeled.probability.clip(1e-7, 1 - 1e-7))
              + (1 - labeled.binary_target)
              * np.log((1 - labeled.probability).clip(1e-7, 1 - 1e-7))),
            weights=weights,
        )) if len(labeled) else np.nan
        for threshold in THRESHOLDS:
            metrics, _ = score_classifier(predicted, threshold)
            pooled.append({
                "epoch": epoch, "threshold": threshold,
                "validation_bce": validation_bce, **metrics,
            })
    return pd.DataFrame(pooled)


def _load_refit_recipe(reference_run):
    settings = json.loads((reference_run / "settings.json").read_text())
    selected = pd.read_csv(reference_run / "selected_configuration.csv").iloc[0]
    return {
        "audit_data": settings["audit_data"],
        "dataset": settings["dataset"],
        "rgb_cache": settings["rgb_cache"],
        "method": settings["method"],
        "mechanism": settings["mechanism"],
        "development_loss": settings["development_loss"],
        "require_visual_history": bool(settings.get("require_visual_history", False)),
        "epsilon": float(settings["epsilon"]),
        "batch_size": int(settings["batch_size"]),
        "seed": int(settings["seed"]),
        "selected_epoch": int(selected.epoch),
        "threshold": float(selected.threshold),
    }


def _predict_checkpoint(rows, checkpoint):
    numeric = checkpoint["numeric_columns"]
    visual_weight = checkpoint["model_state_dict"].get("visual.0.weight")
    visual_width = visual_weight.shape[1] if visual_weight is not None else 0
    model = StaticNearOptimalClassifier(len(numeric), visual_width)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    x, image = _tensors(
        rows, checkpoint["numeric_preprocessor"],
        checkpoint.get("visual_preprocessor"), checkpoint["method"], numeric,
        checkpoint["development_mechanism"],
    )
    with torch.no_grad():
        return model(x, image).sigmoid().numpy()


def refit_development(args):
    """Refit one frozen development recipe on all cross-fitted development labels."""
    reference_run = Path(args.reference_run)
    recipe = _load_refit_recipe(reference_run)
    args.data = Path(recipe["audit_data"])
    args.dataset = Path(recipe["dataset"])
    args.rgb_cache = Path(recipe["rgb_cache"])
    args.rgb = "on" if recipe["method"] == "rgb" else "off"
    args.require_rgb_input = False
    args.require_visual_history = recipe["require_visual_history"]
    args.development_mechanism = recipe["mechanism"]
    args.development_loss = recipe["development_loss"]
    args.near_optimal_epsilon = recipe["epsilon"]
    args.batch_size = recipe["batch_size"]
    args.seed = recipe["seed"]
    args.quality_filtered = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ridge").mkdir(exist_ok=True)

    cohort = pd.read_csv(reference_run / "cohort_manifest.csv")
    events = pd.read_csv(args.data / "defrost_events.csv")
    rows, models = build_fold_rows(
        args, cohort, events, (), include_history=False,
        base_root=reference_run / "base",
    )
    rows = label_complete_curves(
        rows, args.near_optimal_epsilon,
        _old_grid_peaks(args, cohort, models),
    )
    training = rows.loc[_input_available(rows) & rows.binary_target.notna()].copy()
    diagnostics = args.output / "training_diagnostics" / recipe["method"]
    _fit_method(
        training, training, recipe["method"], diagnostics,
        recipe["selected_epoch"], args.seed, args.batch_size,
        args.development_loss, args.development_mechanism,
        thresholds=(recipe["threshold"],),
        compact_predictions=True,
    )
    with (diagnostics / "preprocessors.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)  # noqa: S301 - run-owned artifact
    checkpoint.update({
        "method": recipe["method"],
        "development_mechanism": args.development_mechanism,
        "development_loss": args.development_loss,
        "selected_epoch": recipe["selected_epoch"],
        "threshold": recipe["threshold"],
        "model_state_dict": torch.load(
            diagnostics / f"checkpoint_epoch_{recipe['selected_epoch']}.pt",
            map_location="cpu", weights_only=True,
        )["model_state_dict"],
    })
    saved = pd.read_parquet(
        diagnostics / f"validation_epoch_{recipe['selected_epoch']}.parquet"
    )
    expected = saved.loc[_input_available(saved), "probability"].to_numpy()
    np.testing.assert_allclose(
        _predict_checkpoint(training, checkpoint), expected, rtol=1e-6, atol=1e-7
    )
    with (args.output / "selected.pkl").open("wb") as stream:
        pickle.dump(checkpoint, stream)

    counts = rows.assign(
        development_input=_input_available(rows),
        supported_label=rows.binary_target.notna(),
    ).groupby("cycle_name").agg(
        clock_row_count=("cycle_name", "size"),
        development_input_row_count=("development_input", "sum"),
        supported_label_row_count=("supported_label", "sum"),
    )
    trained = training.groupby("cycle_name").size().rename("training_row_count")
    training_cohort = cohort.merge(counts, on="cycle_name", validate="one_to_one")
    training_cohort = training_cohort.merge(
        trained, on="cycle_name", how="left", validate="one_to_one"
    )
    training_cohort["training_row_count"] = (
        training_cohort.training_row_count.fillna(0).astype(int)
    )
    training_cohort["trained"] = training_cohort.training_row_count.gt(0)
    training_cohort["training_status"] = np.select(
        [
            training_cohort.training_row_count.gt(0),
            training_cohort.development_input_row_count.eq(0),
        ],
        ["trained", "no_development_input"],
        default="no_supported_label",
    )
    training_cohort.to_csv(args.output / "training_cohort.csv", index=False)

    visual_weight = checkpoint["model_state_dict"].get("visual.0.weight")
    model = StaticNearOptimalClassifier(
        len(checkpoint["numeric_columns"]),
        visual_weight.shape[1] if visual_weight is not None else 0,
    )
    active_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if recipe["method"] == "rgb" or not name.startswith("visual.")
    )
    development_summary = pd.read_csv(reference_run / "development_summary.csv").iloc[0]
    metadata = {
        "action": "refit-development",
        "reference_run": str(reference_run.resolve()),
        **recipe,
        "output": str(args.output.resolve()),
        "label_reference": "own-experiment-cross-fitted Ridge; old10s union new30s peak",
        "selection": "frozen development epoch and threshold; no refit selection",
        "interpretation": "all-development-data training artifact; not generalization evidence",
        "threshold_use": "research replay only; no deployed controller gate",
        "controller_qualified": False,
        "development_unsupported_proposal_count": int(
            development_summary.unscoreable_2of3_count
        ),
        "development_proposal_count": int(development_summary.proposal_2of3_count),
        "training_cycle_count": int(training.cycle_name.nunique()),
        "training_row_count": len(training),
        "active_parameter_count": active_parameters,
        "reload_prediction_max_abs_difference": float(np.max(np.abs(
            _predict_checkpoint(training, checkpoint) - expected
        ))),
    }
    from train_pareto_boundary import save_settings
    save_settings(args.output / "settings.json", metadata)
    return checkpoint


def _frozen_recipe(reference_run):
    recipe = _load_refit_recipe(reference_run)
    settings = json.loads((reference_run / "settings.json").read_text())
    if (
        int(settings["maximum_epochs"]) != 50
        or list(settings["checkpoint_epochs"]) != list(CHECKPOINT_EPOCHS)
        or not np.allclose(settings["threshold_grid"], THRESHOLDS)
    ):
        raise ValueError(f"frozen development grid drift: {reference_run}")
    recipe_id = (
        "delta_rgb" if recipe["method"] == "rgb" and recipe["mechanism"] == "delta"
        else "history_sensor" if (
            recipe["method"] == "sensor" and recipe["mechanism"] == "static"
            and recipe["require_visual_history"]
        ) else None
    )
    if recipe_id is None or recipe["development_loss"] != "label-smoothing":
        raise ValueError(f"unsupported frozen retrospective recipe: {reference_run}")
    return recipe_id, recipe


def _recipe_args(args, reference_run, recipe, seed):
    configured = copy.copy(args)
    configured.data = Path(recipe["audit_data"])
    configured.dataset = Path(recipe["dataset"])
    configured.rgb_cache = Path(recipe["rgb_cache"])
    configured.reference_run = reference_run
    configured.rgb = "on" if recipe["method"] == "rgb" else "off"
    configured.require_rgb_input = False
    configured.require_visual_history = recipe["require_visual_history"]
    configured.development_mechanism = recipe["mechanism"]
    configured.development_loss = recipe["development_loss"]
    configured.near_optimal_epsilon = recipe["epsilon"]
    configured.batch_size = recipe["batch_size"]
    configured.seed = seed
    configured.quality_filtered = True
    return configured


def _checkpoint_from_fit(folder, recipe, epoch, threshold):
    with (folder / "preprocessors.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)  # noqa: S301 - run-owned artifact
    checkpoint.update({
        "method": recipe["method"],
        "development_mechanism": recipe["mechanism"],
        "development_loss": recipe["development_loss"],
        "selected_epoch": int(epoch), "threshold": float(threshold),
        "model_state_dict": torch.load(
            folder / f"checkpoint_epoch_{int(epoch)}.pt",
            map_location="cpu", weights_only=True,
        )["model_state_dict"],
    })
    return checkpoint


def _outer_evaluation_status(rows):
    supported = rows.loc[rows.binary_target.notna() & rows.probability.notna()]
    if supported.empty:
        return "unevaluable", supported
    if supported.binary_target.nunique() < 2:
        return "single_class", supported
    return "evaluated", supported


def _frozen_outer(args, reference_run, recipe_id, recipe, cohort, events, groups,
                  outer_experiment, seed):
    folder = (
        args.output / "fits" / recipe_id / f"seed_{seed}" / outer_experiment
    )
    completed = [
        folder / "outer_predictions.parquet", folder / "outer_fold_metrics.csv",
        folder / "cycle_metrics.csv", folder / "inner_selection.csv",
        folder / "selected.pkl",
    ]
    if all(path.exists() for path in completed):
        return tuple(
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            for path in completed[:4]
        )
    configured = _recipe_args(args, reference_run, recipe, seed)
    outer_cohort = cohort.loc[cohort.experiment_id.ne(outer_experiment)]
    inner_folders = {}
    for fold, validation_group in groups.items():
        excluded = tuple(dict.fromkeys([outer_experiment, *validation_group]))
        train, validation, train_models, validation_models = _fold_rows(
            configured, outer_cohort, events, excluded,
            base_root=reference_run / "base",
        )
        for model in (*train_models.values(), *validation_models.values()):
            assert outer_experiment not in model["training_experiment_ids"]
            assert not set(validation_group) & set(model["training_experiment_ids"])
        train = train.loc[_input_available(train) & train.binary_target.notna()].copy()
        inner_folder = folder / "inner" / fold
        _fit_method(
            train, validation, recipe["method"], inner_folder, 50, seed,
            recipe["batch_size"], recipe["development_loss"], recipe["mechanism"],
            compact_predictions=True,
        )
        inner_folders[fold] = inner_folder
    inner_grid = _pooled_threshold_grid(inner_folders, 50)
    selected = select_epoch_threshold(inner_grid)

    train, outer, train_models, outer_models = _fold_rows(
        configured, cohort, events, (outer_experiment,),
        base_root=reference_run / "base",
    )
    for model in (*train_models.values(), *outer_models.values()):
        assert outer_experiment not in model["training_experiment_ids"]
    train = train.loc[_input_available(train) & train.binary_target.notna()].copy()
    final_folder = folder / "outer_refit"
    _fit_method(
        train, train, recipe["method"], final_folder, int(selected.epoch), seed,
        recipe["batch_size"], recipe["development_loss"], recipe["mechanism"],
        thresholds=(float(selected.threshold),), compact_predictions=True,
    )
    checkpoint = _checkpoint_from_fit(
        final_folder, recipe, int(selected.epoch), float(selected.threshold)
    )
    available = _input_available(outer)
    probability = np.full(len(outer), np.nan)
    if available.any():
        probability[available.to_numpy()] = _predict_checkpoint(
            outer.loc[available], checkpoint
        )
    prediction_columns = [
        "cycle_name", "experiment_id", "candidate_defrost_time", "binary_target",
        "joint_input_available",
    ]
    for column in ("development_input_available", "past_image_time"):
        if column in outer:
            prediction_columns.append(column)
    predictions = outer[prediction_columns].copy()
    predictions["probability"] = probability
    predictions.insert(0, "outer_experiment", outer_experiment)
    predictions.insert(0, "seed", seed)
    predictions.insert(0, "method", recipe["method"])
    predictions.insert(0, "recipe_id", recipe_id)

    metrics, cycles = score_classifier(predictions, float(selected.threshold))
    status, supported = _outer_evaluation_status(predictions)
    fold_metrics = pd.DataFrame([{
        "recipe_id": recipe_id, "method": recipe["method"], "seed": seed,
        "outer_experiment": outer_experiment, "evaluation_status": status,
        "outer_row_count": len(predictions),
        "available_input_row_count": int(available.sum()),
        "supported_label_row_count": int(predictions.binary_target.notna().sum()),
        "predicted_supported_row_count": len(supported),
        "selected_epoch": int(selected.epoch),
        "threshold": float(selected.threshold), **metrics,
    }])
    cycles.insert(0, "experiment_id", outer_experiment)
    cycles.insert(0, "outer_experiment", outer_experiment)
    cycles.insert(0, "seed", seed)
    cycles.insert(0, "method", recipe["method"])
    cycles.insert(0, "recipe_id", recipe_id)
    inner_selection = pd.DataFrame([{
        "recipe_id": recipe_id, "method": recipe["method"], "seed": seed,
        "outer_experiment": outer_experiment,
        **selected.to_dict(),
    }])
    folder.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(completed[0], index=False)
    fold_metrics.to_csv(completed[1], index=False)
    cycles.to_csv(completed[2], index=False)
    inner_selection.to_csv(completed[3], index=False)
    with completed[4].open("wb") as stream:
        pickle.dump(checkpoint, stream)
    return predictions, fold_metrics, cycles, inner_selection


def evaluate_frozen_development(args):
    """Nested retrospective evaluation of two frozen development recipes."""
    if not args.runs or len(args.runs) != 2:
        raise ValueError("evaluate-frozen-development requires two --runs")
    references = [Path(path) for path in args.runs]
    frozen = [(*_frozen_recipe(path), path) for path in references]
    if {item[0] for item in frozen} != {"delta_rgb", "history_sensor"}:
        raise ValueError("frozen runs must be Delta RGB and matched History Sensor")
    first_recipe = frozen[0][1]
    if len({item[1]["audit_data"] for item in frozen}) != 1:
        raise ValueError("frozen development runs use different audit data")
    if len({item[1]["dataset"] for item in frozen}) != 1:
        raise ValueError("frozen development runs use different datasets")
    audit = Path(first_recipe["audit_data"])
    cohort = pd.read_csv(references[0] / "cohort_manifest.csv")
    pd.testing.assert_frame_equal(
        cohort, pd.read_csv(references[1] / "cohort_manifest.csv")
    )
    events = pd.read_csv(audit / "defrost_events.csv")
    groups = json.loads((audit / "folds.json").read_text())
    for reference in references:
        if json.loads((reference / "settings.json").read_text())["folds"] != groups:
            raise ValueError("frozen development run groups differ from audit groups")
    experiments = sorted(cohort.experiment_id.astype(str).unique())
    grouped_experiments = [
        str(experiment) for group in groups.values() for experiment in group
    ]
    if len(grouped_experiments) != len(set(grouped_experiments)) or set(
        grouped_experiments
    ) != set(experiments):
        raise ValueError("development groups must partition the frozen cohort experiments")
    seeds = [args.seed] if args.heldout_experiment else [0, 1, 2]
    if args.heldout_experiment:
        experiments = [args.heldout_experiment]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ridge").mkdir(exist_ok=True)
    tasks = [
        (reference, recipe_id, recipe, outer, seed)
        for recipe_id, recipe, reference in frozen
        for seed in seeds for outer in experiments
    ]
    settings = {
        "action": "evaluate-frozen-development",
        "recipes": {recipe_id: str(reference.resolve())
                    for recipe_id, _, reference in frozen},
        "seeds": seeds, "outer_experiments": experiments,
        "total_outer_fold_count": len(experiments),
        "inner_groups": groups, "n_jobs": args.n_jobs,
        "selection": "outer-specific pooled inner OOF BA, macro-F1, BCE, epoch, threshold",
        "ridge_isolation": "outer excluded permanently; inner group excluded from inner fit",
        "interpretation": "freeze-only retrospective evaluation; cannot tune recipes",
        "independence": (
            "retrospective and not prospective: recipe selection used these same "
            f"{len(experiments)} frozen cohort experiments"
        ),
        "cop_gain_gate": "not_evaluated",
    }
    from train_pareto_boundary import save_settings
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel()(delayed(_frozen_outer)(
            args, reference, recipe_id, recipe, cohort, events, groups, outer, seed
        ) for reference, recipe_id, recipe, outer, seed in tasks)
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    fold_metrics = pd.concat([result[1] for result in results], ignore_index=True)
    cycle_metrics = pd.concat([result[2] for result in results], ignore_index=True)
    inner_selection = pd.concat([result[3] for result in results], ignore_index=True)

    match_columns = [
        "cycle_name", "candidate_defrost_time", "binary_target",
        "development_input_available",
    ]
    for (seed, outer), matched in predictions.groupby(
        ["seed", "outer_experiment"], sort=False
    ):
        by_recipe = [
            rows[match_columns].reset_index(drop=True)
            for _, rows in matched.groupby("recipe_id", sort=True)
        ]
        if len(by_recipe) != 2:
            raise AssertionError(f"{seed}/{outer}: expected both frozen recipes")
        pd.testing.assert_frame_equal(by_recipe[0], by_recipe[1])
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    fold_metrics.to_csv(args.output / "outer_fold_metrics.csv", index=False)
    cycle_metrics.to_csv(args.output / "cycle_metrics.csv", index=False)
    inner_selection.to_csv(args.output / "inner_selection.csv", index=False)
    metric_names = [
        "balanced_accuracy", "macro_f1", "cycle_weighted_average_precision",
        "cycle_weighted_brier_score", "cycle_weighted_expected_calibration_error",
        "cycle_weighted_negative_log_likelihood",
    ]
    aggregate_rows = []
    for (recipe_id, method, seed), selected_folds in fold_metrics.groupby(
        ["recipe_id", "method", "seed"], sort=True
    ):
        selected_predictions = predictions.loc[
            predictions.recipe_id.eq(recipe_id) & predictions.seed.eq(seed)
        ]
        probability_metrics, _ = score_classifier(selected_predictions, .5)
        selected_cycles = cycle_metrics.loc[
            cycle_metrics.recipe_id.eq(recipe_id) & cycle_metrics.seed.eq(seed)
        ]
        row = {
            "recipe_id": recipe_id, "method": method, "seed": seed,
            "total_outer_fold_count": len(selected_folds),
            "evaluable_outer_fold_count": int(
                selected_folds.evaluation_status.eq("evaluated").sum()
            ),
            "unevaluable_outer_fold_count": int(
                selected_folds.evaluation_status.ne("evaluated").sum()
            ),
            "balanced_accuracy": float(selected_cycles.balanced_accuracy.mean()),
            "macro_f1": float(selected_cycles.macro_f1.mean()),
            **{column: probability_metrics[column] for column in metric_names[2:]},
            **{
                f"experiment_mean_{column}": float(selected_folds[column].mean())
                for column in metric_names
            },
        }
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output / "aggregate_summary.csv", index=False)
    if not args.heldout_experiment:
        from plots.pareto_learning import render_development_retrospective
        render_development_retrospective(cycle_metrics, args.output / "figures")
    return fold_metrics


def develop(args):
    """Run fixed grouped three-fold development CV; no COP-gain optimization."""
    if args.development_mechanism == "delta":
        args.require_visual_history = True
        if args.rgb != "on":
            raise ValueError("delta development requires --rgb on")
    audit = Path(args.data)
    audit_settings = json.loads((audit / "settings.json").read_text())
    if not args.dataset.exists():
        args.dataset = Path(audit_settings["dataset"])
    if not args.rgb_cache.exists():
        args.rgb_cache = args.dataset.resolve().parent / args.rgb_cache
    boundaries = pd.read_csv(audit / "recovery_boundaries.csv")
    events = pd.read_csv(audit / "defrost_events.csv")
    folds = json.loads((audit / "folds.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ridge").mkdir(exist_ok=True)
    args.quality_filtered = True
    method = "rgb" if args.rgb == "on" else "sensor"
    numeric, _ = development_feature_columns()
    visual = _model_visual_columns(args.development_mechanism)
    settings = {
        "task": "cop-classification", "action": "develop",
        "evaluation": "grouped_three_fold_development_cv_not_outer_test",
        "folds": folds, "fold_source": str((audit / "folds.json").resolve()),
        "dataset": str(args.dataset.resolve()), "audit_data": str(audit.resolve()),
        "rgb_cache": str(args.rgb_cache.resolve()),
        "mechanism": args.development_mechanism, "method": method,
        "require_visual_history": bool(args.require_visual_history),
        "visual_history": (
            "same-cycle front image at or before t-150s within 45s; no imputation"
            if args.require_visual_history else None
        ),
        "development_loss": args.development_loss,
        "label_smoothing_eta": .1 if args.development_loss == "label-smoothing" else None,
        "label_smoothing_target": ".9 * hard_label + .05",
        "loss_hypothesis": "reduce confident errors; not evidence of predictive uncertainty",
        "training_class_weights": "none; cycle-equal sample weights retained",
        "evaluation_labels": "original hard supported labels",
        "probability_metrics": "cycle-weighted Brier, 10-bin equal-width ECE, and NLL clipped to [1e-7, 1-1e-7]",
        "epsilon": args.near_optimal_epsilon, "seed": args.seed,
        "maximum_epochs": args.maximum_epochs, "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3,
                      "weight_decay": 1e-4},
        "candidate_clock": "30s_from_verified_heating_start_strictly_before_preparation",
        "reference_accounting_start": "stable_heating_start_frozen_effective_cop",
        "reference_peak": "same_fold_supported_old10s_union_new30s_before_input_filter",
        "input_electricity_start": "heating_start_strict_causal",
        "numeric_feature_contract": numeric,
        "visual_feature_contract": visual if method == "rgb" else [],
        "availability": (
            "same_sensor_current_front_and_past_front_rows_for_all_history_controls"
            if args.require_visual_history
            else "same_sensor_and_front_rows_for_sensor_and_rgb"
        ),
        "model": "matched_silu_numeric_and_visual_branches_no_dropout",
        "mechanism_hypothesis": (
            "physically motivated independent temporal-difference hypothesis; "
            "static error association was inconclusive and does not establish causality"
            if args.development_mechanism == "delta" else None
        ),
        "threshold_grid": list(THRESHOLDS),
        "selection": "pooled_development_cycle_mean_BA_then_macroF1_then_BCE_then_epoch_then_threshold",
        "cop_gain_gate": "not_evaluated",
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
    }
    from train_pareto_boundary import save_settings
    save_settings(args.output / "settings.json", settings)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(_prepare_cycle)(args, boundary)
                   for boundary in boundaries.itertuples(index=False))
    manifest = boundaries[["cycle_name", "experiment_id"]].copy()
    base_rows = {
        name: pd.read_parquet(args.output / "base" / f"{name}.parquet")
        for name in manifest.cycle_name
    }
    manifest["joint_input_eligible"] = manifest.cycle_name.map({
        name: bool(rows.joint_input_available.any()) for name, rows in base_rows.items()
    })
    manifest["development_input_eligible"] = manifest.cycle_name.map({
        name: bool(_input_available(rows).any()) for name, rows in base_rows.items()
    })
    manifest["status"] = np.where(
        manifest.development_input_eligible, "eligible",
        "no_shared_sensor_front_history_input" if args.require_visual_history
        else "no_shared_sensor_front_input",
    )
    manifest.to_csv(args.output / "cohort_manifest.csv", index=False)
    cohort = manifest
    chosen_folds = (
        {args.heldout_experiment: folds[args.heldout_experiment]}
        if args.heldout_experiment else folds
    )
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(_develop_fold)(
            args, cohort, events, fold, validation_experiments, method
        ) for fold, validation_experiments in chosen_folds.items())

    fold_folders = {
        fold: args.output / "folds" / fold / method for fold in chosen_folds
    }
    pooled_grid = _pooled_threshold_grid(fold_folders, args.maximum_epochs)
    pooled_grid.to_csv(args.output / "threshold_grid.csv", index=False)
    selected = select_epoch_threshold(pooled_grid)

    fold_metrics, cycle_metrics, predictions = [], [], []
    for fold, validation_experiments in chosen_folds.items():
        folder = args.output / "folds" / fold / method
        predicted = pd.read_parquet(
            folder / f"validation_epoch_{int(selected.epoch)}.parquet"
        )
        metrics, cycles = score_classifier(predicted, selected.threshold)
        fold_grid = pd.read_csv(folder / "threshold_grid.csv")
        selected_fold = fold_grid.loc[
            fold_grid.epoch.eq(selected.epoch)
            & np.isclose(fold_grid.threshold, selected.threshold)
        ].iloc[0]
        fold_metrics.append({"method": method, "fold": fold,
                             "epoch": int(selected.epoch),
                             "threshold": float(selected.threshold),
                             "validation_bce": float(selected_fold.validation_bce), **metrics})
        cycles.insert(0, "fold", fold)
        cycles.insert(0, "method", method)
        cycle_metrics.append(cycles)
        predictions.append(predicted.assign(method=method, development_fold=fold))
        with (folder / "preprocessors.pkl").open("rb") as stream:
            checkpoint = pickle.load(stream)  # noqa: S301 - run-owned artifact
        checkpoint.update({
            "method": method, "selected_epoch": int(selected.epoch),
            "threshold": float(selected.threshold),
            "development_loss": args.development_loss,
            "development_mechanism": args.development_mechanism,
            "model_state_dict": torch.load(
                folder / f"checkpoint_epoch_{int(selected.epoch)}.pt",
                map_location="cpu", weights_only=True,
            )["model_state_dict"],
        })
        with (folder / "selected.pkl").open("wb") as stream:
            pickle.dump(checkpoint, stream)
    fold_table = pd.DataFrame(fold_metrics)
    cycle_table = pd.concat(cycle_metrics, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    fold_table.to_csv(args.output / "fold_metrics.csv", index=False)
    cycle_table.to_csv(args.output / "cycle_metrics.csv", index=False)
    prediction_table.to_parquet(args.output / "validation_predictions.parquet", index=False)
    pooled_metrics, _ = score_classifier(prediction_table, selected.threshold)
    summaries = []
    for method, rows in fold_table.groupby("method", sort=False):
        cycles = cycle_table.loc[cycle_table.method.eq(method)]
        cohort_cycles = len(cycles)
        proposed = cycles.two_of_three_status.ne("no_proposal")
        hit = cycles.two_of_three_status.eq("supported_near_hit")
        unscoreable = cycles.two_of_three_status.eq("unsupported_proposal")
        summaries.append({
            "method": method, "selected_fold_count": len(rows),
            **{column: pooled_metrics[column] for column in (
                "cycle_weighted_average_precision", "balanced_accuracy", "macro_f1",
                "cycle_weighted_brier_score",
                "cycle_weighted_expected_calibration_error",
                "cycle_weighted_negative_log_likelihood",
                "recall", "fpr", "classified_cycle_count", "both_class_cycles",
                "single_class_cycles", "no_prediction_cycles",
                "no_supported_labels_cycles")},
            "proposal_2of3_count": int(proposed.sum()),
            "near_hit_2of3_count": int(hit.sum()),
            "unscoreable_2of3_count": int(unscoreable.sum()),
            "scored_cycle_count": int(cycles.two_of_three_status.isin(
                ["supported_near_hit", "supported_miss"]).sum()),
            "cohort_cycle_count": cohort_cycles,
            "proposal_coverage": float(proposed.mean()),
            "near_hit_coverage": float(hit.mean()),
            "conditional_near_hit": float(hit.sum() / max(1, proposed.sum() - unscoreable.sum())),
        })
    summary = pd.DataFrame(summaries)
    pd.DataFrame([selected.to_dict()]).to_csv(
        args.output / "selected_configuration.csv", index=False
    )
    summary.to_csv(args.output / "development_summary.csv", index=False)
    try:
        from plots.pareto_learning import render_development_classification
        render_development_classification(summary, args.figure_output or args.output / "figures")
    except ImportError:
        pass
    return summary


DEVELOPMENT_COMPARISON_RECIPES = {
    "cop_development_static_sensor": "Static Sensor · BCE",
    "cop_development_static_rgb": "Static RGB · BCE",
    "cop_development_smooth_sensor": "Static Sensor · smoothed",
    "cop_development_smooth_rgb": "Static RGB · smoothed",
    "cop_development_history_sensor": "History Sensor · smoothed",
    "cop_development_history_rgb": "History RGB · smoothed",
    "cop_development_delta_rgb": "Delta RGB · smoothed",
}

DEVELOPMENT_SENSOR_RGB_PAIRS = {
    "static_rgb_minus_sensor": (
        "cop_development_static_sensor", "cop_development_static_rgb"
    ),
    "smooth_rgb_minus_sensor": (
        "cop_development_smooth_sensor", "cop_development_smooth_rgb"
    ),
    "history_rgb_minus_sensor": (
        "cop_development_history_sensor", "cop_development_history_rgb"
    ),
    "delta_rgb_minus_history_sensor": (
        "cop_development_history_sensor", "cop_development_delta_rgb"
    ),
    "delta_rgb_minus_history_rgb": (
        "cop_development_history_rgb", "cop_development_delta_rgb"
    ),
}


def _paired_ba_interval(sensor, rgb, experiments, replicates, seed):
    paired = sensor[["cycle_name", "balanced_accuracy"]].merge(
        rgb[["cycle_name", "balanced_accuracy"]], on="cycle_name",
        validate="one_to_one", suffixes=("_sensor", "_rgb"),
    ).dropna()
    paired["difference"] = (
        paired.balanced_accuracy_rgb - paired.balanced_accuracy_sensor
    )
    paired["experiment_id"] = paired.cycle_name.map(experiments)
    clusters = paired.groupby("experiment_id").difference.agg(["sum", "count"])
    if clusters.empty:
        return np.nan, np.nan, np.nan, 0, 0
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    draws = (
        clusters["sum"].to_numpy()[sampled].sum(axis=1)
        / clusters["count"].to_numpy()[sampled].sum(axis=1)
    )
    return (
        float(paired.difference.mean()), float(np.quantile(draws, .025)),
        float(np.quantile(draws, .975)), len(paired), len(clusters),
    )


def compare_development(runs, output, bootstrap_replicates=2000, seed=0):
    """Compare seven frozen classifiers on one common development clock."""
    runs = [Path(run) for run in runs]
    by_recipe = {run.name: run for run in runs}
    expected = set(DEVELOPMENT_COMPARISON_RECIPES)
    if set(by_recipe) != expected or len(runs) != len(expected):
        raise ValueError(
            "development comparison requires exactly: "
            + ", ".join(DEVELOPMENT_COMPARISON_RECIPES)
        )
    keys = [
        "cycle_name", "experiment_id", "candidate_defrost_time",
        "development_fold",
    ]
    tables, configurations, settings = {}, {}, {}
    reference = None
    for recipe, run in by_recipe.items():
        table = pd.read_parquet(
            run / "validation_predictions.parquet",
            columns=[*keys, "binary_target", "probability"],
        ).sort_values(keys).reset_index(drop=True)
        if table.duplicated(keys).any():
            raise ValueError(f"duplicate prediction keys in {run}")
        shared = table[[*keys, "binary_target"]]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, shared)
        reference = shared
        tables[recipe] = table
        configurations[recipe] = pd.read_csv(
            run / "selected_configuration.csv"
        ).iloc[0]
        settings[recipe] = json.loads((run / "settings.json").read_text())

    common = np.logical_and.reduce([
        table.probability.notna().to_numpy() for table in tables.values()
    ])
    common_labeled = common & reference.binary_target.notna().to_numpy()
    experiments = reference.groupby("cycle_name").experiment_id.first()
    cycle_tables, records = {}, []
    for recipe in DEVELOPMENT_COMPARISON_RECIPES:
        table = tables[recipe].copy()
        table["probability"] = table.probability.where(common)
        selected = configurations[recipe]
        metrics, cycles = score_classifier(table, float(selected.threshold))
        cycle_tables[recipe] = cycles
        proposed = cycles.two_of_three_status.ne("no_proposal")
        hit = cycles.two_of_three_status.eq("supported_near_hit")
        unscoreable = cycles.two_of_three_status.eq("unsupported_proposal")
        records.append({
            "recipe_id": recipe,
            "recipe_name": DEVELOPMENT_COMPARISON_RECIPES[recipe],
            "run_path": str(by_recipe[recipe]),
            "method": settings[recipe]["method"],
            "mechanism": settings[recipe]["mechanism"],
            "development_loss": settings[recipe].get("development_loss", "bce"),
            "require_visual_history": bool(
                settings[recipe].get("require_visual_history", False)
            ),
            "epoch": int(selected.epoch),
            "threshold": float(selected.threshold),
            "validation_bce": metrics["cycle_weighted_negative_log_likelihood"],
            **metrics,
            "full_clock_rows": len(reference),
            "full_clock_cycles": reference.cycle_name.nunique(),
            "common_prediction_rows": int(common.sum()),
            "common_prediction_cycles": reference.loc[common, "cycle_name"].nunique(),
            "common_labeled_rows": int(common_labeled.sum()),
            "common_labeled_cycles": reference.loc[
                common_labeled, "cycle_name"
            ].nunique(),
            "proposal_2of3_count": int(proposed.sum()),
            "near_hit_2of3_count": int(hit.sum()),
            "unscoreable_2of3_count": int(unscoreable.sum()),
            "scored_cycle_count": int(cycles.two_of_three_status.isin(
                ["supported_near_hit", "supported_miss"]
            ).sum()),
        })
    summary = pd.DataFrame(records)
    order = summary.sort_values(
        ["balanced_accuracy", "macro_f1", "validation_bce", "epoch", "threshold"],
        ascending=[False, False, True, True, False], kind="stable",
    ).index
    summary["selection_rank"] = pd.Series(
        np.arange(1, len(summary) + 1), index=order
    ).sort_index().to_numpy()
    summary["selected"] = summary.selection_rank.eq(1)

    intervals = []
    for comparison, (sensor_recipe, rgb_recipe) in DEVELOPMENT_SENSOR_RGB_PAIRS.items():
        mean, low, high, paired_cycles, experiment_count = _paired_ba_interval(
            cycle_tables[sensor_recipe], cycle_tables[rgb_recipe], experiments,
            bootstrap_replicates, seed,
        )
        intervals.append({
            "comparison": comparison,
            "sensor_recipe": sensor_recipe,
            "rgb_recipe": rgb_recipe,
            "metric": "balanced_accuracy_rgb_minus_sensor",
            "mean_difference": mean,
            "ci_low": low,
            "ci_high": high,
            "paired_cycles": paired_cycles,
            "experiments": experiment_count,
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "scope": "descriptive_post_selection_development_uncertainty",
        })
    bootstrap = pd.DataFrame(intervals)
    from plots.pareto_learning import render_development_common_comparison

    render_development_common_comparison(summary, bootstrap, Path(output))
    return summary, bootstrap
