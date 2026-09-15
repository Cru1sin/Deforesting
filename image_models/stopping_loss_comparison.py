"""Shared cycle-level losses and controller replay for paired stopping experiments."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

ARCHITECTURES = (
    "r_sensor", "r_sensor_rgb", "chen_rgb", "new_sensor", "new_rgb_difference",
)
LOSSES = ("after_optimum", "cop_stopping")
THRESHOLDS = tuple(np.arange(5, 100, 5) / 100)
SOURCE_RUNS = {
    "r_sensor": Path("output/image_models/cop_after_optimum_sensor"),
    "r_sensor_rgb": Path("output/image_models/cop_after_optimum_rgb"),
    "chen_rgb": Path("output/image_models/dinov2_binary_tref"),
    "new_sensor": Path("output/test/cop_development_history_sensor"),
    "new_rgb_difference": Path("output/test/cop_development_delta_rgb"),
}


def architecture_contract(architecture: str) -> tuple[list[str], list[str]]:
    """Return numeric and visual columns in the existing architecture's order."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown stopping architecture: {architecture}")
    from image_models.cop_development import (
        _model_visual_columns,
        development_feature_columns,
    )
    from image_models.relative_cop import RGB, feature_columns

    if architecture == "r_sensor":
        return feature_columns("off"), []
    if architecture == "r_sensor_rgb":
        return feature_columns("off"), RGB.copy()
    if architecture == "chen_rgb":
        return [], RGB.copy()
    numeric, _ = development_feature_columns()
    return (
        numeric,
        _model_visual_columns("delta") if architecture == "new_rgb_difference" else [],
    )


class _NewPolicyModel(nn.Module):
    def __init__(self, numeric_width: int, visual_width: int):
        super().__init__()
        from image_models.cop_development import StaticNearOptimalClassifier

        self.numeric_width = numeric_width
        self.network = StaticNearOptimalClassifier(numeric_width, visual_width)

    def forward(self, values):
        visual = values[:, self.numeric_width:] if values.shape[1] > self.numeric_width else None
        return self.network(values[:, :self.numeric_width], visual)


def build_model(
    architecture: str, *, numeric_width: int, visual_width: int, seed: int
) -> nn.Module:
    """Build one of the five frozen architectures from a paired random seed."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown stopping architecture: {architecture}")
    from image_models.relative_cop import RelativeCOP

    torch.manual_seed(seed)
    if architecture in ("r_sensor", "r_sensor_rgb"):
        return RelativeCOP(numeric_width + visual_width)
    if architecture == "chen_rgb":
        from image_models.relative_cop import regression_model

        return regression_model([str(index) for index in range(visual_width)], "dinov2-binary")
    return _NewPolicyModel(numeric_width, visual_width)


def positive_logit(output: torch.Tensor) -> torch.Tensor:
    """Return log-odds for class one; sigmoid then equals Chen's softmax P(y=1)."""
    return output[:, 1] - output[:, 0] if output.ndim == 2 else output


def cycle_batches(rows: pd.DataFrame, cycles_per_batch: int, seed: int):
    """Yield positional indices containing whole cycles only."""
    if cycles_per_batch < 1:
        raise ValueError("cycles per batch must be positive")
    groups = {
        name: np.flatnonzero(rows.cycle_name.to_numpy() == name)
        for name in rows.cycle_name.unique()
    }
    names = np.asarray(list(groups), dtype=object)
    np.random.default_rng(seed).shuffle(names)
    for start in range(0, len(names), cycles_per_batch):
        yield np.concatenate([groups[name] for name in names[start:start + cycles_per_batch]])


def action_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep exactly the rows where the model may act and trusted COP is defined."""
    required = {
        "cycle_name", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible",
        "model_input_available", "physically_allowed",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"action rows lack columns: {sorted(missing)}")
    allowed = (
        rows.cycle_cop_eligible.fillna(False)
        & rows.model_input_available.fillna(False)
        & np.isfinite(rows.cycle_cop)
    )
    allowed &= rows.physically_allowed.fillna(False)
    result = rows.loc[allowed].sort_values(
        ["cycle_name", "candidate_defrost_time"], kind="stable"
    ).copy()
    if result.empty:
        return result.assign(
            optimal_time=pd.Series(dtype="datetime64[ns]"),
            optimal_cop=pd.Series(dtype=float),
            after_optimum_target=pd.Series(dtype=float),
        )
    peak = result.groupby("cycle_name").cycle_cop.transform("max")
    optimum = (
        result.loc[result.cycle_cop.eq(peak)]
        .groupby("cycle_name").candidate_defrost_time.min()
    )
    result["optimal_time"] = result.cycle_name.map(optimum)
    result["optimal_cop"] = peak
    result["after_optimum_target"] = result.candidate_defrost_time.ge(
        result.optimal_time
    ).astype(float)
    return result.reset_index(drop=True)


def stopping_distribution(logits: torch.Tensor) -> torch.Tensor:
    """First-stop mass with the final legal action forced to absorb survival mass."""
    if logits.ndim != 1 or not len(logits):
        raise ValueError("stopping logits must be one non-empty cycle")
    probability = logits.sigmoid()
    survival = torch.cumprod(
        torch.cat([torch.ones(1, device=logits.device), 1 - probability[:-1]]), dim=0
    )
    first = probability * survival
    return torch.cat([first[:-1], survival[-1:]])


def cop_stopping_loss(logits: torch.Tensor, cop: torch.Tensor) -> torch.Tensor:
    """Regret from the best legal COP under a soft first-stop policy."""
    if logits.shape != cop.shape:
        raise ValueError("logits and COP must have the same cycle shape")
    return cop.max() - (stopping_distribution(logits) * cop).sum()


def after_optimum_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cycle BCE with equal weight for each class that exists in the legal action set."""
    if logits.shape != target.shape:
        raise ValueError("logits and targets must have the same cycle shape")
    terms = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    present = [terms[target.eq(value)].mean() for value in (0, 1) if target.eq(value).any()]
    if not present:
        raise ValueError("cycle has no binary targets")
    return torch.stack(present).mean()


def replay_cycles(rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply the deployed 1/1 rule, forcing the final legal action when none fires."""
    records = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        ordered = cycle.sort_values("candidate_defrost_time", kind="stable")
        positive = ordered.probability.notna() & ordered.probability.ge(threshold)
        selected = ordered.loc[positive].iloc[0] if positive.any() else ordered.iloc[-1]
        optimum = pd.Timestamp(ordered.optimal_time.iloc[0])
        records.append({
            "cycle_name": name,
            "experiment_id": ordered.experiment_id.iloc[0]
            if "experiment_id" in ordered else pd.NA,
            "selected_time": pd.Timestamp(selected.candidate_defrost_time),
            "selected_cop": float(selected.cycle_cop),
            "optimal_time": optimum,
            "optimal_cop": float(ordered.cycle_cop.max()),
            "forced_final": not bool(positive.any()),
            "early_trigger": pd.Timestamp(selected.candidate_defrost_time) < optimum,
        })
    return pd.DataFrame(records)


def select_cop_threshold(rows: pd.DataFrame, thresholds=THRESHOLDS):
    """Select the inner-validation threshold by mean replayed cycle COP."""
    scored = []
    for threshold in thresholds:
        replay = replay_cycles(rows, threshold)
        scored.append({
            "threshold": float(threshold),
            "mean_cycle_cop": float(replay.selected_cop.mean()),
            "early_trigger_rate": float(replay.early_trigger.mean()),
            "forced_final_rate": float(replay.forced_final.mean()),
        })
    grid = pd.DataFrame(scored)
    selected = grid.sort_values(
        ["mean_cycle_cop", "threshold"], ascending=[False, False], kind="stable"
    ).iloc[0]
    return float(selected.threshold), grid


def policy_cycle_metrics(
    predicted: pd.DataFrame,
    reference: pd.DataFrame,
    threshold: float,
    *,
    loss_name: str,
) -> pd.DataFrame:
    """Score hard 1/1 decisions against supported RB and legal-action oracle COP."""
    metrics = replay_cycles(predicted, threshold)
    rb = []
    soft = {}
    for name, cycle in reference.groupby("cycle_name", sort=False):
        time = pd.to_datetime(cycle.t_RB.iloc[0], errors="coerce")
        supported = (
            cycle.cycle_cop_eligible.fillna(False)
            & np.isfinite(cycle.cycle_cop)
            & pd.to_datetime(cycle.candidate_defrost_time).eq(time)
        )
        rb.append({
            "cycle_name": name,
            "rb_cop": float(cycle.loc[supported, "cycle_cop"].iloc[0])
            if supported.any() else np.nan,
        })
    if loss_name == "cop_stopping":
        for name, cycle in predicted.groupby("cycle_name", sort=False):
            probability = torch.tensor(
                cycle.sort_values("candidate_defrost_time").probability.to_numpy(),
                dtype=torch.float64,
            ).clamp(1e-12, 1 - 1e-12)
            logits = torch.logit(probability)
            cop = torch.tensor(
                cycle.sort_values("candidate_defrost_time").cycle_cop.to_numpy(),
                dtype=torch.float64,
            )
            soft[name] = float((stopping_distribution(logits) * cop).sum())
    metrics = metrics.merge(pd.DataFrame(rb), on="cycle_name", how="left")
    metrics["gain_vs_rb_pct"] = 100 * (
        metrics.selected_cop - metrics.rb_cop
    ) / metrics.rb_cop
    denominator = metrics.optimal_cop - metrics.rb_cop
    metrics["headroom_captured_pct"] = 100 * (
        metrics.selected_cop - metrics.rb_cop
    ) / denominator
    metrics.loc[~np.isfinite(metrics.rb_cop) | metrics.rb_cop.le(0) | denominator.eq(0), [
        "gain_vs_rb_pct", "headroom_captured_pct",
    ]] = np.nan
    metrics["headroom_remaining_pct"] = 100 - metrics.headroom_captured_pct
    metrics["soft_cop"] = metrics.cycle_name.map(soft)
    metrics["soft_hard_gap"] = metrics.soft_cop - metrics.selected_cop
    return metrics


def summarize_metrics(rows: pd.DataFrame) -> dict:
    """Aggregate the three primary metrics from paired mean COP values."""
    policy = rows.loc[rows.selected_cop.notna()]
    paired = rows.loc[
        rows.selected_cop.notna() & rows.rb_cop.notna() & rows.optimal_cop.notna()
    ]
    model_mean = policy.selected_cop.mean()
    rb_mean = paired.rb_cop.mean()
    paired_model_mean = paired.selected_cop.mean()
    oracle_mean = paired.optimal_cop.mean()
    gain = (
        100 * (paired_model_mean - rb_mean) / rb_mean
        if len(paired) and rb_mean > 0 else np.nan
    )
    headroom = (
        100 * (paired_model_mean - rb_mean) / (oracle_mean - rb_mean)
        if len(paired) and oracle_mean != rb_mean else np.nan
    )
    return {
        "cohort_cycles": rows.cycle_name.nunique(), "policy_cycles": len(policy),
        "rb_paired_cycles": len(paired), "mean_cycle_cop": model_mean,
        "gain_vs_rb_pct": gain, "headroom_captured_pct": headroom,
        "headroom_remaining_pct": 100 - headroom,
        "early_trigger_rate": policy.early_trigger.mean(),
        "mean_soft_hard_gap": policy.soft_hard_gap.mean(),
    }


def _batch_loss(logits, rows, loss_name):
    losses = []
    for _, positions in rows.groupby("cycle_name", sort=False).indices.items():
        positions = np.asarray(positions)
        if loss_name == "after_optimum":
            target = torch.tensor(
                rows.iloc[positions].after_optimum_target.to_numpy(),
                dtype=logits.dtype, device=logits.device,
            )
            losses.append(after_optimum_loss(logits[positions], target))
        elif loss_name == "cop_stopping":
            cop = torch.tensor(
                rows.iloc[positions].cycle_cop.to_numpy(),
                dtype=logits.dtype, device=logits.device,
            )
            losses.append(cop_stopping_loss(logits[positions], cop))
        else:
            raise ValueError(f"unknown stopping loss: {loss_name}")
    return torch.stack(losses).mean()


def _model_values(rows, columns, preprocessor):
    return torch.tensor(preprocessor.transform(rows[columns]), dtype=torch.float32)


def _ordered_columns(architecture, numeric, visual):
    return [*visual, *numeric] if architecture == "r_sensor_rgb" else [*numeric, *visual]


def predict_policy(rows: pd.DataFrame, checkpoint: dict) -> pd.DataFrame:
    """Predict action probabilities without applying an extra availability gate."""
    rng_state = torch.random.get_rng_state()
    try:
        model = build_model(
            checkpoint["architecture"],
            numeric_width=len(checkpoint["numeric_columns"]),
            visual_width=len(checkpoint["visual_columns"]),
            seed=checkpoint["seed"],
        )
    finally:
        torch.random.set_rng_state(rng_state)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    columns = _ordered_columns(
        checkpoint["architecture"], checkpoint["numeric_columns"],
        checkpoint["visual_columns"],
    )
    values = _model_values(rows, columns, checkpoint["preprocessor"])
    with torch.no_grad():
        probability = positive_logit(model(values)).sigmoid().numpy()
    return rows.assign(probability=probability)


def fit_policy(
    train: pd.DataFrame,
    validation: pd.DataFrame | None,
    *,
    architecture: str,
    loss_name: str,
    numeric_columns: list[str],
    visual_columns: list[str],
    seed: int,
    maximum_epochs: int,
    patience: int,
    cycles_per_batch: int,
    thresholds=THRESHOLDS,
):
    """Fit whole-cycle batches; inner replayed COP selects epoch and threshold."""
    if train.empty or (validation is not None and validation.empty):
        raise ValueError("stopping policy requires non-empty train and validation cycles")
    columns = _ordered_columns(architecture, numeric_columns, visual_columns)
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    preprocessor.fit(train[columns])
    x = _model_values(train, columns, preprocessor)
    model = build_model(
        architecture, numeric_width=len(numeric_columns),
        visual_width=len(visual_columns), seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, best_threshold = -np.inf, maximum_epochs, .5
    best_state, stale, history, grids = None, 0, [], []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for index in cycle_batches(train, cycles_per_batch, seed + epoch):
            optimizer.zero_grad()
            batch = train.iloc[index].reset_index(drop=True)
            loss = _batch_loss(positive_logit(model(x[index])), batch, loss_name)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        record = {
            "epoch": epoch, "training_loss": total / steps, "optimizer_steps": steps,
        }
        if validation is not None:
            current = {
                "architecture": architecture, "loss_name": loss_name,
                "numeric_columns": numeric_columns, "visual_columns": visual_columns,
                "seed": seed, "preprocessor": preprocessor,
                "model_state_dict": model.state_dict(),
            }
            predicted = predict_policy(validation, current)
            threshold, grid = select_cop_threshold(predicted, thresholds)
            grid.insert(0, "epoch", epoch)
            grids.append(grid)
            score = float(grid.loc[grid.threshold.eq(threshold), "mean_cycle_cop"].iloc[0])
            record.update(validation_mean_cycle_cop=score, threshold=threshold)
            if score > best_score:
                best_score, best_epoch, best_threshold = score, epoch, threshold
                best_state, stale = copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "architecture": architecture, "loss_name": loss_name,
        "numeric_columns": list(numeric_columns), "visual_columns": list(visual_columns),
        "seed": seed, "preprocessor": preprocessor,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "selected_epoch": best_epoch, "threshold": best_threshold,
        "losses": pd.DataFrame(history),
        "threshold_grid": pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
    }


def architecture_action_rows(rows: pd.DataFrame, architecture: str) -> pd.DataFrame:
    """Apply one existing architecture's input clock before defining its optimum."""
    from image_models.cop_development import _input_available
    from image_models.relative_cop import processing_rows

    if architecture in ("r_sensor", "r_sensor_rgb"):
        available = rows.sensor_timestamp.notna()
        if architecture == "r_sensor_rgb":
            available &= rows.rgb_available.fillna(False)
    elif architecture == "chen_rgb":
        available = rows.rgb_available.fillna(False) & processing_rows(rows)
    elif architecture == "new_sensor":
        numeric, _ = architecture_contract(architecture)
        present = [column for column in numeric if column in rows]
        available = rows.sensor_timestamp.notna() & rows[present].notna().any(axis=1)
    elif architecture == "new_rgb_difference":
        available = _input_available(rows)
    else:
        raise ValueError(f"unknown stopping architecture: {architecture}")
    time = pd.to_datetime(rows.candidate_defrost_time)
    start = pd.to_datetime(rows.stable_heating_start, errors="coerce")
    end = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns]")
    if "observed_defrost_preparation_start" in rows:
        end = pd.to_datetime(rows.observed_defrost_preparation_start, errors="coerce")
    if "observation_end" in rows:
        end = end.fillna(pd.to_datetime(rows.observation_end, errors="coerce"))
    physically_allowed = time.ge(start) & end.notna() & time.lt(end)
    return action_rows(rows.assign(
        model_input_available=available, physically_allowed=physically_allowed,
    ))


def _fold_reference(args, cohort, events, excluded, base_root):
    from image_models.relative_cop import build_fold_rows

    configured = SimpleNamespace(**vars(args))
    configured.output = args.output
    configured.reference_run = args.data
    configured.rgb = "off"
    configured.require_rgb_input = False
    configured.quality_filtered = True
    rows = build_fold_rows(
        configured, cohort, events, tuple(excluded), include_history=True,
        base_root=base_root,
    )[0]
    boundaries = [column for column in (
        "stable_heating_start", "observed_defrost_preparation_start",
        "observation_end", "t_RB",
    ) if column in cohort]
    return rows.drop(columns=boundaries, errors="ignore").merge(
        cohort[["cycle_name", *boundaries]].drop_duplicates("cycle_name"),
        on="cycle_name", how="left", validate="many_to_one",
    )


def _fit_fold(args, architecture, loss_name, seed, test, inner, cohort, events, base_root):
    folder = args.output / architecture / loss_name / f"seed_{seed}" / "folds"
    path = folder / f"{test}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)  # noqa: S301 - run-owned resumable artifact
    nested_cohort = cohort.loc[~cohort.experiment_id.eq(test)]
    inner_train = _fold_reference(
        args, nested_cohort.loc[~nested_cohort.experiment_id.eq(inner)], events,
        (test, inner), base_root,
    )
    inner_validation = _fold_reference(
        args, nested_cohort.loc[nested_cohort.experiment_id.eq(inner)], events,
        (test, inner), base_root,
    )
    outer_train_reference = _fold_reference(
        args, cohort.loc[~cohort.experiment_id.eq(test)], events, (test,), base_root,
    )
    outer_test_reference = _fold_reference(
        args, cohort.loc[cohort.experiment_id.eq(test)], events, (test,), base_root,
    )
    train = architecture_action_rows(inner_train, architecture)
    validation = architecture_action_rows(inner_validation, architecture)
    outer_train = architecture_action_rows(outer_train_reference, architecture)
    outer_test = architecture_action_rows(outer_test_reference, architecture)
    status = "evaluated"
    if any(rows.empty for rows in (train, validation, outer_train, outer_test)):
        status = "no_legal_actions"
        result = {
            "status": status, "heldout_experiment": test, "inner_experiment": inner,
            "predictions": pd.DataFrame(), "cycle_metrics": pd.DataFrame(),
            "losses": pd.DataFrame(), "threshold_grid": pd.DataFrame(),
        }
    else:
        numeric, visual = architecture_contract(architecture)
        numeric = [column for column in numeric if train[column].notna().any()]
        nested = fit_policy(
            train, validation, architecture=architecture, loss_name=loss_name,
            numeric_columns=numeric, visual_columns=visual, seed=seed,
            maximum_epochs=args.maximum_epochs, patience=args.patience,
            cycles_per_batch=args.batch_size,
        )
        checkpoint = fit_policy(
            outer_train, None, architecture=architecture, loss_name=loss_name,
            numeric_columns=numeric, visual_columns=visual, seed=seed,
            maximum_epochs=nested["selected_epoch"], patience=args.patience,
            cycles_per_batch=args.batch_size,
        )
        checkpoint["threshold"] = nested["threshold"]
        predicted = predict_policy(outer_test, checkpoint)
        metrics = policy_cycle_metrics(
            predicted, outer_test_reference, nested["threshold"], loss_name=loss_name,
        )
        metrics["evaluation_status"] = np.where(
            metrics.rb_cop.notna(), "evaluated", "rb_outside_trusted_support"
        )
        missing = sorted(set(outer_test_reference.cycle_name) - set(metrics.cycle_name))
        if missing:
            metrics = pd.concat([metrics, pd.DataFrame({
                "cycle_name": missing, "evaluation_status": "no_legal_actions",
            })], ignore_index=True)
        result = {
            "status": status, "heldout_experiment": test, "inner_experiment": inner,
            "checkpoint": checkpoint,
            "predictions": predicted.assign(
                architecture=architecture, loss_name=loss_name, seed=seed,
                heldout_experiment=test,
            ),
            "cycle_metrics": metrics.assign(
                architecture=architecture, loss_name=loss_name, seed=seed,
                heldout_experiment=test,
            ),
            "losses": pd.concat([
                nested["losses"].assign(stage="inner"),
                checkpoint["losses"].assign(stage="outer"),
            ], ignore_index=True),
            "threshold_grid": nested["threshold_grid"],
        }
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream)
    temporary.replace(path)
    return result


def _source_runs(args):
    if args.runs:
        if len(args.runs) != len(ARCHITECTURES):
            raise ValueError("--runs must provide source runs in the five architecture order")
        return dict(zip(ARCHITECTURES, map(Path, args.runs), strict=True))
    return SOURCE_RUNS


def _saved_fold_results(folder: Path):
    results = []
    for path in sorted(folder.glob("*.pkl")):
        with path.open("rb") as stream:
            results.append(pickle.load(stream))  # noqa: S301 - run-owned artifacts
    return results


def run(args):
    """Run resumable LOEO fits for every requested architecture, loss, and seed."""
    from train_pareto_boundary import fold_exclusions, save_settings

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ridge").mkdir(exist_ok=True)
    boundaries = pd.read_csv(args.data / "recovery_boundaries.csv")
    boundaries["experiment_id"] = boundaries.experiment_id.astype(str)
    events = pd.read_csv(args.data / "defrost_events.csv")
    sources = _source_runs(args)
    missing = [str(path / "base") for path in sources.values() if not (path / "base").exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen architecture base directories: {missing}")
    settings = {
        "schema": "paired_stopping_loss_v1",
        "task": "cop-classification", "action": "compare-stopping-losses",
        "architectures": args.stopping_architectures,
        "losses": args.stopping_losses, "seeds": args.seeds,
        "source_runs": {key: str(value.resolve()) for key, value in sources.items()},
        "data": str(args.data.resolve()), "maximum_epochs": args.maximum_epochs,
        "patience": args.patience, "cycles_per_batch": args.batch_size,
        "outer_cv": "leave_one_experiment_out",
        "inner_validation": "next_experiment_by_sorted_rotation",
        "action_set": (
            "finite_trusted_COP_and_architecture_input_available_between_"
            "stable_heating_start_and_preparation_or_observation_end"
        ),
        "binary_loss": "cycle_mean_of_present_class_means_no_label_smoothing",
        "cop_loss": "cycle_optimal_COP_minus_soft_first_stop_expected_COP",
        "controller": "first_probability_at_or_above_threshold_else_forced_final",
        "threshold_selection": "inner_validation_mean_replayed_cycle_COP",
        "threshold_grid": list(np.arange(5, 100, 5) / 100),
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4},
    }
    save_settings(args.output / "settings.json", settings)
    for architecture in args.stopping_architectures:
        base_root = sources[architecture] / "base"
        source_cohort = boundaries.loc[boundaries.cycle_name.map(
            lambda name, root=base_root: (root / f"{name}.parquet").exists()
        )].copy()
        inners = fold_exclusions(sorted(source_cohort.experiment_id.unique()))
        for loss_name in args.stopping_losses:
            for seed in args.seeds:
                with parallel_config(
                    backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1
                ):
                    tests = (
                        [args.heldout_experiment]
                        if args.heldout_experiment else sorted(source_cohort.experiment_id.unique())
                    )
                    unknown = set(tests) - set(inners)
                    if unknown:
                        raise ValueError(
                            f"held-out experiments absent from {architecture}: {sorted(unknown)}"
                        )
                    Parallel()(delayed(_fit_fold)(
                        args, architecture, loss_name, seed, test, inners[test],
                        source_cohort, events, base_root,
                    ) for test in tests)
                output = args.output / architecture / loss_name / f"seed_{seed}"
                output.mkdir(parents=True, exist_ok=True)
                results = _saved_fold_results(output / "folds")
                for key, suffix in (
                    ("predictions", "parquet"), ("cycle_metrics", "csv"),
                    ("losses", "csv"), ("threshold_grid", "csv"),
                ):
                    tables = [result[key] for result in results if not result[key].empty]
                    if not tables:
                        continue
                    table = pd.concat(tables, ignore_index=True)
                    getattr(table, f"to_{suffix}")(
                        output / f"{key}.{suffix}", index=False
                    )
    summaries = []
    for path in args.output.glob("*/*/seed_*/cycle_metrics.csv"):
        rows = pd.read_csv(path)
        metrics = summarize_metrics(rows)
        summaries.append({
            "architecture": path.parents[2].name, "loss_name": path.parents[1].name,
            "seed": int(path.parent.name.removeprefix("seed_")),
            **metrics,
        })
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
