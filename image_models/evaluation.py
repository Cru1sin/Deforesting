"""Frozen prediction, trigger scoring, and external experiment evaluation."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config

from image_models.checkpoints import checkpoint_camera, checkpoint_files
from image_models.policy import (
    first_trigger_log_probabilities,
    near_set_timing,
    policy_cycle_metrics,
)
from output_layout import experiment_root


def score(predicted, reference, *, strategy="first_positive_terminal"):
    metrics = policy_cycle_metrics(
        predicted,
        reference,
        0.5,
        loss_name="cdf_ramp",
        strategy=strategy,
        controller_reference=reference,
    )
    metrics["regret_pct"] = 100 * (1 - metrics.selected_cop / metrics.optimal_cop)
    metrics["near_optimal_hit"] = metrics.selected_cop.ge(0.99 * metrics.optimal_cop).where(
        metrics.selected_cop.notna()
    )
    metrics["fallback_rb"] = False
    metrics["evaluable"] = metrics.selected_cop.notna()
    metrics["unsupported_trigger"] = metrics.selected_time.notna() & ~metrics.evaluable
    return metrics


def _score_external_checkpoint(path, rows, reference, conditions):
    """Use the ordinary frozen prediction, CDF and terminal-fallback scoring path."""
    import numpy as np
    import torch

    from image_models.policy import predict_policy

    torch.set_num_threads(1)
    saved = pickle.load(path.open("rb"))
    model = saved["checkpoint"]
    # Cycle numbers restart in a new Dataset; experiment IDs define the split.
    if set(rows.experiment_id) & set(model["training_experiments"]):
        raise ValueError("External evaluation overlaps model training data")
    if model["threshold"] != 0.5:
        raise ValueError("The shared stopping scorer uses the frozen 0.5 threshold")
    from image_models.inputs import prepare_inputs

    rows = prepare_inputs(rows, model["numeric_columns"])
    predicted = rows.copy().assign(split="test", probability=np.nan, decision_margin=np.nan)
    valid = predicted.input_valid
    if valid.any():
        predicted.loc[valid, ["probability", "decision_margin"]] = predict_policy(
            predicted.loc[valid], model
        )[["probability", "decision_margin"]].to_numpy()
    metrics = score_stopping_predictions(
        predicted,
        reference,
        saved["cycle_metrics"].version.iloc[0],
        model["seed"],
        None,
        "first_positive_terminal",
        model["trigger_rule"],
    )
    metrics["operating_condition"] = metrics.cycle_name.map(conditions)
    return metrics.assign(
        checkpoint=path.stem,
        source_checkpoint=str(path),
        reset_history=model.get("reset_history", False),
    )


def evaluate_external(
    run, dataset, experiments, output, *, n_jobs=6, checkpoint=None, camera_role=None
):
    """Evaluate saved folds or one deployment weight without fitting on the new batch.

    Fold scores are repeated predictions of the same new cycles, not independent
    samples; report their distribution and retain each checkpoint's actual count.
    """
    from dataset_tools import DatasetLoader
    from dataset_tools.cycle_metadata import cycle_operating_condition
    from image_models.cycle_selection import screen_published_cycles
    from image_models.policy import summarize_metrics
    from output_layout import analysis_output

    run = experiment_root(run) if run is not None else None
    output = analysis_output(output, replay=True)
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {output}")
    paths = [Path(checkpoint)] if checkpoint is not None else checkpoint_files(run)
    if not paths:
        raise ValueError(f"No frozen model checkpoints: {run}")
    cameras = {checkpoint_camera(path) for path in paths}
    if len(cameras) != 1 or (camera_role is not None and cameras != {camera_role}):
        raise ValueError(
            f"Checkpoint camera {sorted(cameras)} does not match requested camera {camera_role}"
        )
    camera_role = cameras.pop()
    print(f"Camera: {camera_role}\nExperiments: {', '.join(experiments)}", flush=True)
    for path in paths:
        print(f"Checkpoint: {path.resolve()}", flush=True)
    loader = DatasetLoader(dataset)
    screening = screen_published_cycles(loader.dataset_root, camera_role)
    rows, reference = loader.load_training_data(objective="cdf_ramp", camera_role=camera_role)
    admitted = set(
        screening.loc[screening.selected & screening.experiment_id.isin(experiments), "cycle_name"]
    )
    if not admitted:
        raise ValueError("No newly admitted cycles for external evaluation")
    rows = rows.loc[rows.cycle_name.isin(admitted)].copy()
    reference = reference.loc[reference.cycle_name.isin(admitted)].copy()
    conditions = {
        name: cycle_operating_condition(loader.load_cycle(name)) for name in sorted(admitted)
    }
    with parallel_config(backend="loky", n_jobs=min(n_jobs, len(paths)), inner_max_num_threads=1):
        metrics = pd.concat(
            Parallel(verbose=5)(
                delayed(_score_external_checkpoint)(path, rows, reference, conditions)
                for path in paths
            ),
            ignore_index=True,
        )
    output.mkdir(parents=True)
    screening.loc[screening.experiment_id.isin(experiments)].to_csv(
        output / "cycle_screening.csv", index=False
    )
    metrics.to_csv(output / "cycle_metrics.csv", index=False)
    summaries = []
    for (checkpoint_name, version, seed), g in metrics.groupby(["checkpoint", "version", "seed"]):
        for condition, part in [("all", g), *g.groupby("operating_condition")]:
            summaries.append(
                dict(
                    checkpoint=checkpoint_name,
                    version=version,
                    seed=seed,
                    operating_condition=condition,
                    **summarize_metrics(part),
                )
            )
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(output / "checkpoint_summary.csv", index=False)
    scores = [
        "cohort_cycles",
        "policy_cycles",
        "rb_paired_cycles",
        "mean_cycle_cop",
        "hit_rate_all_pct",
        "gain_vs_rb_pct",
        "headroom_captured_pct",
    ]
    summary = summaries.groupby(["version", "operating_condition"])[scores].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output / "summary.csv")
    (output / "settings.json").write_text(
        json.dumps(
            dict(
                source_run=str(run) if run is not None else None,
                dataset=str(Path(dataset).resolve()),
                camera_role=camera_role,
                experiments=experiments,
                checkpoints=len(paths),
                cycles=len(admitted),
                weights="frozen",
                interpretation=(
                    "Frozen deployment replay on experiments excluded from training"
                    if checkpoint is not None
                    else "Distribution across saved historical fold models; new cycles are not independent repeats"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        summaries[
            ["checkpoint", "operating_condition", "cohort_cycles", "gain_vs_rb_pct"]
        ].to_string(index=False),
        flush=True,
    )
    print("Cycle results (first 10; complete table: cycle_metrics.csv):", flush=True)
    print(
        metrics[["cycle_name", "selected_time", "selected_cop", "rb_cop", "near_optimal_hit"]]
        .head(10)
        .to_string(index=False),
        flush=True,
    )
    print(output, flush=True)
    return metrics


def score_stopping_predictions(predicted, reference, version, seed, fold, strategy, rule):
    """One scoring path for new fits and explicit frozen-weight trigger replay."""
    from image_models.policy import apply_trigger_rule

    scored = predicted if rule == "auto" else apply_trigger_rule(predicted, rule)
    metrics = score(scored, reference, strategy=strategy).assign(version=version, seed=seed)
    metrics = metrics.merge(
        predicted[["cycle_name", "split"]].drop_duplicates(), validate="one_to_one"
    )
    if fold is not None:
        metrics["outer_experiment"] = fold
    timing, _ = near_set_timing(predicted, metrics)
    metrics = metrics.merge(timing, on="cycle_name", validate="one_to_one")
    metrics = metrics.merge(event_probabilities(predicted), on="cycle_name", validate="one_to_one")
    metrics.loc[metrics.available_actions.eq(0), "trigger_status"] = "no_input"
    if strategy == "first_positive_terminal":
        metrics["fallback_terminal"] = metrics.forced_final
        metrics["input_missing"] = metrics.available_actions.eq(0)
    return metrics


def event_probabilities(rows):
    """Stochastic event masses; unavailable slots force continuation, as in replay."""
    records = []
    for name, cycle in rows.groupby("cycle_name"):
        allowed = cycle.loc[cycle.near_optimal, "candidate_defrost_time"]
        left, right = allowed.min(), allowed.max()
        cycle = cycle.loc[cycle.input_valid].sort_values("candidate_defrost_time")
        before, gap, after, max_hazard = 0.0, 0.0, 0.0, np.nan
        if cycle.empty:
            mass, remaining = 0.0, 1.0
        else:
            logits = torch.tensor(cycle.decision_margin.to_numpy(), dtype=torch.float64)
            logq, log_remaining = first_trigger_log_probabilities(logits)
            q = logq.exp().numpy()
            mass = float(q[cycle.near_optimal.to_numpy(bool)].sum())
            remaining = float(log_remaining.exp())
            before = float(q[cycle.candidate_defrost_time.lt(left)].sum())
            after = float(q[cycle.candidate_defrost_time.gt(right)].sum())
            gap = float(
                q[cycle.candidate_defrost_time.between(left, right) & ~cycle.near_optimal].sum()
            )
            max_hazard = float(logits.sigmoid().max())
        records.append(
            dict(
                cycle_name=name,
                available_actions=len(cycle),
                probability_first_in_set=mass,
                probability_no_event=remaining,
                probability_before_set=before,
                probability_in_gap=gap,
                probability_after_set=after,
                max_hazard=max_hazard,
            )
        )
    return pd.DataFrame(records)
