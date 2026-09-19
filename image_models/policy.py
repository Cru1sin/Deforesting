"""Cycle-balanced Binary fitting and frozen trigger/COP evaluation."""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from image_models.core import model

THRESHOLDS = (0.5,)


def build_model(architecture, *, numeric_width, visual_width, seed):
    return model(architecture, numeric_width, visual_width, seed)


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
        yield np.concatenate([groups[name] for name in names[start : start + cycles_per_batch]])


def after_optimum_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cycle BCE with equal weight for each class that exists in the legal action set."""
    if logits.shape != target.shape:
        raise ValueError("logits and targets must have the same cycle shape")
    terms = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    present = [terms[target.eq(value)].mean() for value in (0, 1) if target.eq(value).any()]
    if not present:
        raise ValueError("cycle has no binary targets")
    return torch.stack(present).mean()


def replay_cycles(
    rows: pd.DataFrame,
    threshold: float,
    *,
    strategy="first_positive",
    reference=None,
) -> pd.DataFrame:
    """Replay one controller, preserving missing slots for two-of-three confirmation."""

    records = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        ordered = cycle.sort_values("candidate_defrost_time", kind="stable")
        clock = ordered
        if strategy == "two_of_three" and reference is not None:
            full = reference.loc[reference.cycle_name.eq(name)].sort_values(
                "candidate_defrost_time", kind="stable"
            )
            start, end = ordered.candidate_defrost_time.iloc[[0, -1]]
            full = full.loc[full.candidate_defrost_time.between(start, end, inclusive="both")]
            clock = full[["candidate_defrost_time"]].merge(
                ordered[["candidate_defrost_time", "probability"]],
                on="candidate_defrost_time",
                how="left",
            )
        if strategy == "first_positive":
            positive = clock.probability.notna() & clock.probability.ge(threshold)
            trigger = (
                pd.Timestamp(clock.loc[positive, "candidate_defrost_time"].iloc[0])
                if positive.any()
                else pd.NaT
            )
        else:
            trigger, _ = two_of_three_trigger(
                clock.candidate_defrost_time, clock.probability.fillna(-np.inf), threshold
            )
        forced = pd.isna(trigger)
        selected = (
            ordered.iloc[-1]
            if forced
            else ordered.loc[ordered.candidate_defrost_time.eq(trigger)].iloc[0]
        )
        optimum = pd.Timestamp(ordered.optimal_time.iloc[0])
        records.append(
            {
                "cycle_name": name,
                "experiment_id": ordered.experiment_id.iloc[0]
                if "experiment_id" in ordered
                else pd.NA,
                "selected_time": pd.Timestamp(selected.candidate_defrost_time),
                "selected_cop": float(selected.cycle_cop),
                "optimal_time": optimum,
                "optimal_cop": float(ordered.optimal_cop.iloc[0])
                if "optimal_cop" in ordered
                else float(ordered.cycle_cop.max()),
                "forced_final": forced,
                "early_trigger": pd.Timestamp(selected.candidate_defrost_time) < optimum,
                "strategy": strategy,
            }
        )
    return pd.DataFrame(records)


def select_cop_threshold(
    rows: pd.DataFrame,
    thresholds=THRESHOLDS,
    *,
    strategy="first_positive",
    reference=None,
):
    """Select the inner-validation threshold by mean replayed cycle COP."""
    scored = []
    for threshold in thresholds:
        replay = replay_cycles(rows, threshold, strategy=strategy, reference=reference)
        scored.append(
            {
                "threshold": float(threshold),
                "mean_cycle_cop": float(replay.selected_cop.mean()),
                "early_trigger_rate": float(replay.early_trigger.mean()),
                "forced_final_rate": float(replay.forced_final.mean()),
            }
        )
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
    strategy: str = "first_positive",
    controller_reference=None,
) -> pd.DataFrame:
    """Score hard 1/1 decisions against supported RB and legal-action oracle COP."""
    metrics = replay_cycles(
        predicted,
        threshold,
        strategy=strategy,
        reference=reference if controller_reference is None else controller_reference,
    )
    rb = []
    soft = {}
    for name, cycle in reference.groupby("cycle_name", sort=False):
        time = pd.to_datetime(cycle.t_RB.iloc[0], errors="coerce")
        supported = np.isfinite(cycle.cycle_cop) & pd.to_datetime(cycle.candidate_defrost_time).eq(
            time
        )
        if controller_reference is not None:
            supported &= cycle.cycle_cop_eligible.fillna(False)
        rb.append(
            {
                "cycle_name": name,
                "rb_cop": float(cycle.loc[supported, "cycle_cop"].iloc[0])
                if supported.any()
                else np.nan,
            }
        )
    metrics = metrics.merge(pd.DataFrame(rb), on="cycle_name", how="left")
    metrics["gain_vs_rb_pct"] = 100 * (metrics.selected_cop - metrics.rb_cop) / metrics.rb_cop
    denominator = metrics.optimal_cop - metrics.rb_cop
    metrics["headroom_captured_pct"] = 100 * (metrics.selected_cop - metrics.rb_cop) / denominator
    metrics.loc[
        ~np.isfinite(metrics.rb_cop) | metrics.rb_cop.le(0) | denominator.eq(0),
        [
            "gain_vs_rb_pct",
            "headroom_captured_pct",
        ],
    ] = np.nan
    metrics["headroom_remaining_pct"] = 100 - metrics.headroom_captured_pct
    metrics["soft_cop"] = metrics.cycle_name.map(soft)
    metrics["soft_hard_gap"] = metrics.soft_cop - metrics.selected_cop
    return metrics


def summarize_metrics(rows: pd.DataFrame) -> dict:
    """Aggregate the three primary metrics from paired mean COP values."""
    policy = rows.loc[rows.selected_cop.notna()]
    paired = rows.loc[rows.selected_cop.notna() & rows.rb_cop.notna() & rows.optimal_cop.notna()]
    model_mean = policy.selected_cop.mean()
    rb_mean = paired.rb_cop.mean()
    paired_model_mean = paired.selected_cop.mean()
    oracle_mean = paired.optimal_cop.mean()
    gain = 100 * (paired_model_mean - rb_mean) / rb_mean if len(paired) and rb_mean > 0 else np.nan
    headroom = (
        100 * (paired_model_mean - rb_mean) / (oracle_mean - rb_mean)
        if len(paired) and oracle_mean != rb_mean
        else np.nan
    )
    return {
        "cohort_cycles": rows.cycle_name.nunique(),
        "policy_cycles": len(policy),
        "rb_paired_cycles": len(paired),
        "mean_cycle_cop": model_mean,
        "gain_vs_rb_pct": gain,
        "headroom_captured_pct": headroom,
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
                dtype=logits.dtype,
                device=logits.device,
            )
            losses.append(after_optimum_loss(logits[positions], target))
        else:
            raise ValueError(f"unknown stopping loss: {loss_name}")
    return torch.stack(losses).mean()


def _model_values(rows, columns, preprocessor):
    return torch.tensor(preprocessor.transform(rows[columns]), dtype=torch.float32)


def _ordered_columns(architecture, numeric, visual):
    return [*numeric, *visual]


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
        checkpoint["architecture"],
        checkpoint["numeric_columns"],
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
    controller_reference=None,
):
    """Fit whole-cycle batches; inner replayed COP selects epoch and threshold."""
    if train.empty or (validation is not None and validation.empty):
        raise ValueError("stopping policy requires non-empty train and validation cycles")
    columns = _ordered_columns(architecture, numeric_columns, visual_columns)
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    preprocessor.fit(train[columns])
    x = _model_values(train, columns, preprocessor)
    model = build_model(
        architecture,
        numeric_width=len(numeric_columns),
        visual_width=len(visual_columns),
        seed=seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, best_threshold = -np.inf, maximum_epochs, 0.5
    best_state, stale, history = None, 0, []
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
            "epoch": epoch,
            "training_loss": total / steps,
            "optimizer_steps": steps,
        }
        if validation is not None:
            current = {
                "architecture": architecture,
                "loss_name": loss_name,
                "numeric_columns": numeric_columns,
                "visual_columns": visual_columns,
                "seed": seed,
                "preprocessor": preprocessor,
                "model_state_dict": model.state_dict(),
            }
            predicted = predict_policy(validation, current)
            threshold, grid = select_cop_threshold(
                predicted, thresholds, strategy="two_of_three", reference=controller_reference
            )
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
    result = {
        "architecture": architecture,
        "loss_name": loss_name,
        "numeric_columns": list(numeric_columns),
        "visual_columns": list(visual_columns),
        "seed": seed,
        "preprocessor": preprocessor,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "selected_epoch": best_epoch,
        "threshold": best_threshold,
        "losses": pd.DataFrame(history),
        "threshold_grid": pd.DataFrame(),
    }
    return result


def two_of_three_trigger(
    times: pd.Series | pd.DatetimeIndex,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> tuple[pd.Timestamp, pd.Series]:
    """Return the first frame where at least two of the latest three are positive."""
    positive = probabilities.ge(threshold)
    rolling = positive.rolling(3, min_periods=2).sum().ge(2)
    positions = np.flatnonzero(rolling.to_numpy())
    trigger = pd.NaT
    if len(positions):
        trigger = pd.Timestamp(
            times.iloc[positions[0]] if hasattr(times, "iloc") else times[positions[0]]
        )
    return trigger, rolling
