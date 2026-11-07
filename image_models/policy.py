"""Cycle-balanced Binary fitting and frozen trigger/COP evaluation."""

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from image_models.model import build_model

THRESHOLDS = (0.5,)


def observable_stopping_cycles(rows):
    """Training-only: a set-labelled cycle needs at least one observed candidate."""
    return rows.loc[rows.groupby("cycle_name").near_optimal.transform("any")]


def positive_logit(output: torch.Tensor) -> torch.Tensor:
    """Return log-odds for class one; sigmoid then equals Chen1000x64's softmax P(y=1)."""
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


def replay_cycles(
    rows: pd.DataFrame,
    threshold: float,
    *,
    strategy="first_positive_terminal",
    reference=None,
) -> pd.DataFrame:
    """Trigger at the first threshold crossing; otherwise use the recorded preparation time.

    Without a crossing or a recorded preparation time, retain a no-trigger result.
    """

    if strategy != "first_positive_terminal":
        raise ValueError(f"Unknown replay strategy: {strategy}")
    records = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        ordered = cycle.sort_values("candidate_defrost_time", kind="stable")
        positive = ordered.probability.ge(threshold)
        forced = not positive.any()
        selected = ordered.loc[positive].iloc[0] if not forced else ordered.iloc[-1]
        terminal = ordered.terminal_time.iloc[0]
        if forced and pd.notna(terminal):
            selected = reference.loc[
                reference.cycle_name.eq(name) & reference.candidate_defrost_time.eq(terminal)
            ].iloc[0]
        no_trigger = forced and pd.isna(terminal)
        selected_cop = float(selected.cycle_cop)
        if no_trigger or not bool(selected.cycle_cop_eligible):
            selected_cop = np.nan
        selected_time = pd.NaT if no_trigger else pd.Timestamp(selected.candidate_defrost_time)
        optimum = pd.Timestamp(ordered.optimal_time.iloc[0])
        records.append(
            dict(
                cycle_name=name,
                experiment_id=ordered.experiment_id.iloc[0]
                if "experiment_id" in ordered
                else pd.NA,
                selected_time=selected_time,
                selected_cop=selected_cop,
                optimal_time=optimum,
                optimal_cop=float(ordered.optimal_cop.iloc[0]),
                forced_final=forced and not no_trigger,
                no_trigger=no_trigger,
                early_trigger=selected_time < optimum if not no_trigger else np.nan,
                strategy=strategy,
            )
        )
    return pd.DataFrame(records)


def near_set_timing(rows, metrics):
    """Locate already-selected stops against the exact near set, preserving holes.

    Normalized time is descriptive: [0, 1] spans the set's envelope, not its holes.
    A singleton has no normalized position; exact membership still defines a hit.
    """
    records, bands = [], []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        cycle = cycle.sort_values("candidate_defrost_time")
        near = cycle.near_optimal.fillna(False)
        allowed = cycle.loc[near, "candidate_defrost_time"]
        left, right = allowed.min(), allowed.max()
        width = (right - left).total_seconds() if len(allowed) else np.nan
        runs = near.ne(near.shift()).cumsum()
        for _, segment in cycle.loc[near].groupby(runs[near]):
            bands.append(
                dict(
                    cycle_name=name,
                    start=segment.candidate_defrost_time.iloc[0],
                    end=segment.candidate_defrost_time.iloc[-1],
                    points=len(segment),
                )
            )
        decision = metrics.loc[metrics.cycle_name.eq(name)].iloc[0]
        t = decision.selected_time
        if pd.isna(t):
            status = "no_trigger"
        elif decision.unsupported_trigger:
            status = "unsupported"
        elif allowed.empty:
            status = "no_near_set"
        elif allowed.eq(t).any():
            status = "hit"
        else:
            status = "early" if t < left else "late" if t > right else "gap"
        records.append(
            dict(
                cycle_name=name,
                trigger_status=status,
                near_left=left,
                near_right=right,
                near_components=int((near & ~near.shift(fill_value=False)).sum()),
                near_points=int(near.sum()),
                near_set_position=(t - left).total_seconds() / width
                if pd.notna(t) and width > 0
                else np.nan,
            )
        )
    return pd.DataFrame(records), pd.DataFrame(
        bands, columns=["cycle_name", "start", "end", "points"]
    )


def select_cop_threshold(
    rows: pd.DataFrame,
    thresholds=THRESHOLDS,
    *,
    strategy="first_positive_terminal",
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
    strategy: str = "first_positive_terminal",
    controller_reference=None,
) -> pd.DataFrame:
    """Score hard 1/1 decisions against supported RB and legal-action oracle COP."""
    metrics = replay_cycles(
        predicted,
        threshold,
        strategy=strategy,
        reference=reference if controller_reference is None else controller_reference,
    )
    return finalize_policy_metrics(
        metrics, reference, require_supported_rb=controller_reference is not None
    )


def predicted_defrost_electricity_metrics(metrics, reference):
    """Evaluate the same frozen preparation-to-defrost-end predictor at two fixed decisions."""
    result = metrics.copy()
    reference = reference[
        [
            "cycle_name",
            "candidate_defrost_time",
            "cycle_cop",
            "defrost_event_electricity_kwh",
            "defrost_event_electricity_prediction_available",
        ]
    ].copy()
    reference["candidate_defrost_time"] = pd.to_datetime(
        reference.candidate_defrost_time, format="mixed"
    )
    for prefix, time_column, cop_column in [
        ("selected", "selected_time", "selected_cop"),
        ("rb", "t_RB", "rb_cop"),
    ]:
        requests = result[["cycle_name", time_column]].copy()
        requests[time_column] = pd.to_datetime(requests[time_column], format="mixed")
        paired = requests.merge(
            reference,
            left_on=["cycle_name", time_column],
            right_on=["cycle_name", "candidate_defrost_time"],
            how="left",
            validate="many_to_one",
        )
        if not paired.defrost_event_electricity_prediction_available.eq(
            True
        ).all() or not np.allclose(paired.cycle_cop, result[cop_column], rtol=1e-9, atol=1e-10):
            raise ValueError("Exact selected/RB predictions must match the frozen COP evaluation")
        result[f"{prefix}_defrost_electricity_kwh"] = (
            paired.defrost_event_electricity_kwh.to_numpy()
        )
    energy = result[["selected_defrost_electricity_kwh", "rb_defrost_electricity_kwh"]]
    if (
        not np.isfinite(energy).all().all()
        or not energy.ge(0).all().all()
        or not result.rb_defrost_electricity_kwh.gt(0).all()
    ):
        raise ValueError(
            "Relative event-energy change requires finite nonnegative predictions and positive RB energy"
        )
    result["defrost_electricity_change_kwh"] = (
        result.selected_defrost_electricity_kwh - result.rb_defrost_electricity_kwh
    )
    result["defrost_electricity_change_pct"] = (
        100 * result.defrost_electricity_change_kwh / result.rb_defrost_electricity_kwh
    )
    return result


def fixed_duration_defrost_power_metrics(metrics, events):
    """Normalize both decisions by the same observed preparation/defrost duration, not a predicted duration."""
    duration = events[
        ["cycle_name", "defrost_preparation_start", "defrost_end", "duration_event_valid"]
    ].copy()
    duration["observed_defrost_duration_minutes"] = (
        pd.to_datetime(duration.defrost_end, format="mixed")
        - pd.to_datetime(duration.defrost_preparation_start, format="mixed")
    ).dt.total_seconds() / 60
    duration["power_duration_available"] = (
        duration.duration_event_valid.eq(True)
        & np.isfinite(duration.observed_defrost_duration_minutes)
        & duration.observed_defrost_duration_minutes.gt(0)
    )
    result = metrics.merge(
        duration[["cycle_name", "observed_defrost_duration_minutes", "power_duration_available"]],
        on="cycle_name",
        how="left",
        validate="many_to_one",
    )
    result["power_duration_available"] = result.power_duration_available.eq(True)
    minutes = result.observed_defrost_duration_minutes.where(result.power_duration_available)
    # E is in kWh and t in minutes: W = 60,000 E / t.
    result["selected_defrost_mean_power_w"] = (
        60000 * result.selected_defrost_electricity_kwh / minutes
    )
    result["rb_defrost_mean_power_w"] = 60000 * result.rb_defrost_electricity_kwh / minutes
    result["defrost_mean_power_change_w"] = (
        result.selected_defrost_mean_power_w - result.rb_defrost_mean_power_w
    )
    result["defrost_mean_power_change_pct"] = (
        100 * result.defrost_mean_power_change_w / result.rb_defrost_mean_power_w
    )
    return result


def policy_timing_metrics(metrics, reference):
    """Paired elapsed times, including known stops whose COP is unsupported."""
    result = metrics.copy()
    clock = reference.groupby("cycle_name", sort=False).first()
    for column in ["t_RB", "cycle_start", "heating_accounting_start"]:
        result[column] = (
            pd.to_datetime(result.cycle_name.map(clock[column]), format="mixed")
            if column in clock
            else pd.NaT
        )
    selected = pd.to_datetime(result.selected_time, format="mixed")
    result["delay_vs_rb_minutes"] = (selected - result.t_RB).dt.total_seconds() / 60
    for prefix, start in [("cycle", "cycle_start"), ("heating", "heating_accounting_start")]:
        result[f"{prefix}_duration_minutes"] = (selected - result[start]).dt.total_seconds() / 60
        result[f"rb_{prefix}_duration_minutes"] = (
            result.t_RB - result[start]
        ).dt.total_seconds() / 60
    # Use actual supported COP points, not filled label gaps or recording length.
    result["near_end_time"] = pd.NaT
    if "optimal_cop" in result and {"cycle_cop", "cycle_cop_eligible"} <= set(reference):
        peaks = result.set_index("cycle_name").optimal_cop
        near = reference.cycle_cop_eligible & reference.cycle_cop.ge(
            0.99 * reference.cycle_name.map(peaks)
        )
        last = reference.loc[near].groupby("cycle_name").candidate_defrost_time.max()
        result["near_end_time"] = pd.to_datetime(result.cycle_name.map(last), format="mixed")
        # Match training CDF bounds on the legal grid; preserve raw-COP hit metrics.
        legal_near = near & reference.get(
            "on_action_grid", pd.Series(True, index=reference.index)
        ).fillna(False)
        bounds = (
            reference.loc[legal_near]
            .groupby("cycle_name")
            .candidate_defrost_time.agg(["min", "max"])
        )
        result["filled_near_start_time"] = pd.to_datetime(
            result.cycle_name.map(bounds["min"]), format="mixed"
        )
        result["filled_near_end_time"] = pd.to_datetime(
            result.cycle_name.map(bounds["max"]), format="mixed"
        )
        result["filled_near_hit"] = selected.ge(result.filled_near_start_time) & selected.le(
            result.filled_near_end_time
        )
        if "no_trigger" in result:
            result["filled_near_hit"] &= ~result.no_trigger.fillna(False)
    result["delay_space_minutes"] = (result.near_end_time - result.t_RB).dt.total_seconds() / 60
    result["delay_space_used_pct"] = (
        100
        * result.delay_vs_rb_minutes
        / result.delay_space_minutes.where(result.delay_space_minutes.gt(0))
    )
    return result


def summarize_timing(rows):
    """Equal-cycle means on known, paired decision times; COP is not a gate."""
    if "delay_vs_rb_minutes" not in rows:
        return {}
    paired = rows.loc[rows.delay_vs_rb_minutes.notna()]
    values = dict(
        timing_paired_cycles=len(paired),
        timing_missing_cycles=len(rows) - len(paired),
        mean_delay_vs_rb_minutes=paired.delay_vs_rb_minutes.mean(),
        timing_fallback_cycles=int(paired.forced_final.sum()) if "forced_final" in paired else 0,
    )
    for prefix in ["cycle", "heating"]:
        valid = paired.loc[
            paired[f"{prefix}_duration_minutes"].notna()
            & paired[f"rb_{prefix}_duration_minutes"].notna()
        ]
        values.update(
            {
                f"{prefix}_timing_cycles": len(valid),
                f"mean_{prefix}_duration_minutes": valid[f"{prefix}_duration_minutes"].mean(),
                f"mean_rb_{prefix}_duration_minutes": valid[f"rb_{prefix}_duration_minutes"].mean(),
            }
        )
    if "delay_space_minutes" in paired:
        space = paired.loc[paired.delay_space_minutes.gt(0)]
        values.update(
            delay_space_cycles=len(space),
            no_positive_delay_space_cycles=int(paired.delay_space_minutes.le(0).sum()),
            unknown_delay_space_cycles=int(paired.delay_space_minutes.isna().sum()),
            mean_delay_space_minutes=space.delay_space_minutes.mean(),
            delay_space_used_pct=100
            * space.delay_vs_rb_minutes.sum()
            / space.delay_space_minutes.sum()
            if len(space)
            else np.nan,
        )
    return values


def finalize_policy_metrics(metrics, reference, *, require_supported_rb=True):
    """Apply the same RB comparison to fresh or saved fixed-time decisions."""
    rb = []
    soft = {}
    for name, cycle in reference.groupby("cycle_name", sort=False):
        time = pd.to_datetime(cycle.t_RB.iloc[0], errors="coerce")
        supported = np.isfinite(cycle.cycle_cop) & pd.to_datetime(cycle.candidate_defrost_time).eq(
            time
        )
        if require_supported_rb:
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
    return policy_timing_metrics(metrics, reference)


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
        "mean_paired_model_cop": paired_model_mean,
        "mean_rb_cop": rb_mean,
        "mean_optimal_cop": oracle_mean,
        "delta_cop_vs_rb": paired_model_mean - rb_mean,
        "hit_cycles": int(policy.selected_cop.ge(0.99 * policy.optimal_cop).sum()),
        "hit_rate_all_pct": 100 * rows.selected_cop.ge(0.99 * rows.optimal_cop).mean(),
        "hit_rate_evaluable_pct": 100 * policy.selected_cop.ge(0.99 * policy.optimal_cop).mean(),
        "hit_cycles_exact": int(rows.selected_cop.ge(0.99 * rows.optimal_cop).sum()),
        "hit_rate_exact_all_pct": 100 * rows.selected_cop.ge(0.99 * rows.optimal_cop).mean(),
        **(
            dict(
                hit_cycles_filled=int(rows.filled_near_hit.sum()),
                hit_rate_filled_all_pct=100 * rows.filled_near_hit.mean(),
                filled_only_hit_cycles=int(
                    (rows.filled_near_hit & ~rows.selected_cop.ge(0.99 * rows.optimal_cop)).sum()
                ),
            )
            if "filled_near_hit" in rows
            else {}
        ),
        "gain_vs_rb_pct": gain,
        "headroom_captured_pct": headroom,
        "headroom_remaining_pct": 100 - headroom,
        "early_trigger_rate": policy.early_trigger.mean(),
        "mean_soft_hard_gap": policy.soft_hard_gap.mean(),
        **summarize_timing(rows),
    }


def first_trigger_log_probabilities(logits):
    """Discrete hazard on available action slots; no event renormalization."""
    log_continue = -nn.functional.softplus(logits)
    log_survival = torch.cat([logits.new_zeros(1), log_continue.cumsum(0)[:-1]])
    return -nn.functional.softplus(-logits) + log_survival, log_continue.sum()


def _batch_loss(logits, rows, loss_name):
    losses = []
    for _, positions in rows.groupby("cycle_name", sort=False).indices.items():
        positions = np.asarray(positions)
        if loss_name == "cdf_ramp":
            positions = positions[
                np.argsort(rows.candidate_defrost_time.iloc[positions].to_numpy(), kind="stable")
            ]
            cycle = rows.iloc[positions]
            target = torch.as_tensor(
                cycle.stopping_cdf_target.to_numpy(), dtype=logits.dtype, device=logits.device
            )
            if "stopping_cdf_instantaneous" in rows and bool(
                cycle.stopping_cdf_instantaneous.iloc[0]
            ):
                losses.append(
                    nn.functional.binary_cross_entropy_with_logits(logits[positions], target)
                )
                continue
            log_survival = nn.functional.logsigmoid(-logits[positions]).cumsum(0)
            # log(1-exp(s)) remains stable for both tiny and large event mass.
            log_cdf = torch.log(
                -torch.expm1(log_survival).clamp(max=-torch.finfo(logits.dtype).tiny)
            )
            losses.append(-(target * log_cdf + (1 - target) * log_survival).mean())
        elif loss_name == "set_stop":
            positions = positions[
                np.argsort(rows.candidate_defrost_time.iloc[positions].to_numpy(), kind="stable")
            ]
            cycle = rows.iloc[positions]
            near = torch.as_tensor(
                cycle.get("stopping_candidate", cycle.near_optimal).to_numpy(bool),
                device=logits.device,
            )
            if not near.any():
                raise ValueError("No observable candidate stopping time for set likelihood")
            logq, _ = first_trigger_log_probabilities(logits[positions])
            weights = (
                torch.as_tensor(
                    cycle.stopping_log_weight.to_numpy(), dtype=logits.dtype, device=logits.device
                )
                if "stopping_log_weight" in cycle
                else 0.0
            )
            if "stopping_distribution" in cycle and cycle.stopping_distribution.iloc[0]:
                # Linear preference defines a full target distribution, not a
                # utility whose maximizer places all mass at the right edge.
                log_weights = weights[near]
                normalizer = torch.logsumexp(log_weights, dim=0)
                if not torch.isfinite(normalizer):
                    raise ValueError("No observable positive-weight stopping candidate")
                target = (log_weights - normalizer).exp()
                losses.append(-(target * logq[near]).sum())
            else:
                losses.append(-torch.logsumexp((logq + weights)[near], dim=0))
        else:
            raise ValueError(f"unknown stopping loss: {loss_name}")
    return torch.stack(losses).mean()


def _model_values(rows, columns, preprocessor):
    return torch.tensor(preprocessor.transform(rows[columns]), dtype=torch.float32)


def _ordered_columns(architecture, numeric, visual):
    return [*numeric, *visual]


def policy_logits(model, values, rows):
    """Batch independent, time-ordered histories; padding cannot affect a prefix."""
    if not hasattr(model, "readout"):
        return positive_logit(model(values))
    positions = [
        np.asarray(pos)[np.argsort(rows.candidate_defrost_time.iloc[pos].to_numpy(), kind="stable")]
        for pos in rows.groupby("cycle_name", sort=False).indices.values()
    ]
    padded = nn.utils.rnn.pad_sequence([values[pos] for pos in positions], batch_first=True)
    output = model(
        padded, lengths=torch.as_tensor([len(pos) for pos in positions], device=values.device)
    )
    joined = torch.cat([output[i, : len(pos)] for i, pos in enumerate(positions)])
    return joined[torch.as_tensor(np.argsort(np.concatenate(positions)), device=values.device)]


def restore_policy(checkpoint):
    """Restore the same evaluation model for replay and frozen-input probes."""
    rng_state = torch.random.get_rng_state()
    try:
        model = build_model(
            checkpoint["architecture"],
            numeric_width=len(checkpoint["numeric_columns"]),
            visual_width=len(checkpoint["visual_columns"]),
            seed=checkpoint["seed"],
            dropout_config=checkpoint.get("dropout_config"),
            hidden_width=checkpoint.get("hidden_width", 32),
            recurrent_layers=checkpoint.get("recurrent_layers", 1),
            reset_history=checkpoint.get("reset_history", False),
            mlp_layers=checkpoint.get("mlp_layers", 2),
            mlp_widths=checkpoint.get("mlp_widths"),
            film=checkpoint.get("film"),
        )
    finally:
        torch.random.set_rng_state(rng_state)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_policy(rows: pd.DataFrame, checkpoint: dict, *, cached_values=None) -> pd.DataFrame:
    """Predict raw decision margins and probabilities without an availability gate."""
    model = restore_policy(checkpoint)
    columns = _ordered_columns(
        checkpoint["architecture"],
        checkpoint["numeric_columns"],
        checkpoint["visual_columns"],
    )
    values = (
        _model_values(rows, columns, checkpoint["preprocessor"])
        if cached_values is None
        else cached_values
    )
    with torch.no_grad():
        margin = policy_logits(model, values, rows)
        probability = margin.sigmoid().numpy()
    return rows.assign(probability=probability, decision_margin=margin.numpy())


def apply_trigger_rule(rows, rule):
    """Apply a causal event-CDF or leave conditional hazards unchanged."""
    rows = rows.sort_values(["cycle_name", "candidate_defrost_time"]).copy()
    if rule == "cdf":
        survival = 1.0 - rows.probability.where(rows.input_valid, 0.0).fillna(0.0)
        rows["probability"] = (1.0 - survival.groupby(rows.cycle_name).cumprod()).where(
            rows.input_valid
        )
    elif rule != "hazard":
        raise ValueError(f"Unknown trigger rule: {rule}")
    return rows


def trigger_rule_scores(rows, reference=None, rules=("hazard", "cdf")):
    records = []
    if reference is not None:
        reference = reference.loc[reference.cycle_name.isin(rows.cycle_name.unique())]
    for rule in rules:
        replay = replay_cycles(
            apply_trigger_rule(rows, rule),
            0.5,
            strategy="first_positive_terminal",
            reference=reference,
        )
        if reference is not None:
            replay = finalize_policy_metrics(replay, reference)
        else:
            replay["rb_cop"] = np.nan
        paired = replay.loc[
            replay.selected_cop.notna() & replay.rb_cop.notna() & replay.optimal_cop.notna()
        ]
        records.append(
            dict(
                rule=rule,
                validation_evaluable=int(replay.selected_cop.notna().sum()),
                validation_cop=replay.selected_cop.mean(),
                validation_cycles=len(replay),
                validation_paired=len(paired),
                model_sum=paired.selected_cop.sum(),
                rb_sum=paired.rb_cop.sum(),
                optimal_sum=paired.optimal_cop.sum(),
            )
        )
    return pd.DataFrame(records).sort_values(
        ["validation_evaluable", "validation_cop"],
        ascending=False,
        na_position="last",
        kind="stable",
    )  # Hazard wins exact ties.


def mix_stopping_hazards(members):
    """Equal mixture of first-event distributions, not an average of hazards."""
    keys = ["cycle_name", "candidate_defrost_time"]
    ordered = [g.sort_values(keys).reset_index(drop=True) for g in members]
    result = ordered[0].copy()
    masses, survivals = [], []
    for rows in ordered:
        pd.testing.assert_frame_equal(result[keys + ["input_valid"]], rows[keys + ["input_valid"]])
        hazard = rows.probability.where(rows.input_valid, 0.0).fillna(0.0)
        survival = (1.0 - hazard).groupby(rows.cycle_name).cumprod()
        previous = survival.groupby(rows.cycle_name).shift(fill_value=1.0)
        masses.append((previous * hazard).to_numpy())
        survivals.append(previous.to_numpy())
    mass = np.mean(masses, axis=0)
    previous = np.mean(survivals, axis=0)
    result["probability"] = np.divide(mass, previous, out=np.zeros_like(mass), where=previous > 0)
    result.loc[~result.input_valid, "probability"] = np.nan
    if "decision_margin" in result:
        with np.errstate(divide="ignore", invalid="ignore"):
            result["decision_margin"] = np.log(result.probability) - np.log1p(-result.probability)
    return result


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
    learning_rate=1e-3,
    weight_decay=1e-4,
    validation_strategy="first_positive_terminal",
    validation_metric="cop",
    validation_complete_first=False,
    initial_hazard=None,
    validation_tiebreak="earliest",
    joint_trigger_selection=False,
    dropout_config=None,
    trigger_rule="auto",
    hidden_width=32,
    recurrent_layers=1,
    reset_history=False,
    mlp_layers=2,
    mlp_widths=None,
    film=None,
):
    """Fit cycle-balanced batches; inner COP or BCE selects the training epoch."""
    if train.empty or (validation is not None and validation.empty):
        raise ValueError("stopping policy requires non-empty train and validation cycles")
    if joint_trigger_selection and (
        validation_strategy != "first_positive_terminal" or not validation_complete_first
    ):
        raise ValueError(
            "Joint trigger selection requires first-positive terminal and coverage-first validation"
        )
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
        dropout_config=dropout_config,
        hidden_width=hidden_width,
        recurrent_layers=recurrent_layers,
        reset_history=reset_history,
        mlp_layers=mlp_layers,
        mlp_widths=mlp_widths,
        film=film,
    )
    if initial_hazard is not None:
        readout = model.head if hasattr(model, "readout") else model
        output = next(
            layer for layer in reversed(list(readout.modules())) if isinstance(layer, nn.Linear)
        )
        with torch.no_grad():
            output.bias.fill_(np.log(initial_hazard / (1 - initial_hazard)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_score, best_epoch, best_threshold = -np.inf, maximum_epochs, 0.5
    if validation_complete_first:
        best_score = (-1, -np.inf)
    if validation_tiebreak == "set_loss":
        if loss_name not in ("set_stop", "cdf_ramp") or validation_metric != "cop":
            raise ValueError("Set-loss tie break requires set_stop training and COP validation")
        best_score = (
            (*best_score, -np.inf) if isinstance(best_score, tuple) else (best_score, -np.inf)
        )
    best_state, stale, history = None, 0, []
    validation_available = (
        validation.input_valid
        if validation is not None
        and validation_metric == "cop"
        and loss_name in ("set_stop", "cdf_ramp")
        and "input_valid" in validation
        else pd.Series(True, index=validation.index)
        if validation is not None
        else None
    )
    validation_values = (
        _model_values(validation.loc[validation_available], columns, preprocessor)
        if validation is not None and validation_available.any()
        else None
    )
    # Input tensors are cached above; losses and replay need no copies of DINO vectors.
    train = train.drop(columns=visual_columns)
    if validation is not None:
        validation = validation.drop(columns=visual_columns)
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for index in cycle_batches(train, cycles_per_batch, seed + epoch):
            optimizer.zero_grad()
            batch = train.iloc[index].reset_index(drop=True)
            loss = _batch_loss(policy_logits(model, x[index], batch), batch, loss_name)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        record = {
            "epoch": epoch,
            "training_loss": total / steps,
            "optimizer_steps": steps,
        }
        evaluation_state = model.state_dict()
        if validation is not None:
            current = {
                "training_cycles": sorted(train.cycle_name.unique()),
                "architecture": architecture,
                "film": film,
                "hidden_width": hidden_width,
                "recurrent_layers": recurrent_layers,
                "mlp_layers": mlp_layers,
                "mlp_widths": list(mlp_widths) if mlp_widths else None,
                "reset_history": reset_history,
                "loss_name": loss_name,
                "numeric_columns": numeric_columns,
                "visual_columns": visual_columns,
                "seed": seed,
                "preprocessor": preprocessor,
                "model_state_dict": evaluation_state,
                "dropout_config": dropout_config,
            }
            if validation_metric == "bce":
                model.eval()
                with torch.no_grad():
                    value = _batch_loss(
                        policy_logits(model, validation_values, validation), validation, loss_name
                    )
                score, threshold = -float(value), 0.5
                record.update(validation_bce=-score, threshold=threshold)
            elif validation_metric == "cop":
                predicted = validation.assign(probability=np.nan, decision_margin=np.nan)
                available = validation_available
                if available.any():
                    predicted.loc[available, ["probability", "decision_margin"]] = predict_policy(
                        validation.loc[available], current, cached_values=validation_values
                    )[["probability", "decision_margin"]].to_numpy()
                if loss_name == "cdf_ramp":
                    supported = predicted.loc[available].copy()
                    with torch.no_grad():
                        value = (
                            float(
                                _batch_loss(
                                    torch.as_tensor(supported.decision_margin.to_numpy()),
                                    supported,
                                    "cdf_ramp",
                                )
                            )
                            if len(supported)
                            else np.inf
                        )
                    record.update(
                        validation_cdf_loss=value,
                        validation_cdf_loss_cycles=supported.cycle_name.nunique(),
                    )
                if joint_trigger_selection or trigger_rule != "auto":
                    rules = ("hazard", "cdf") if trigger_rule == "auto" else (trigger_rule,)
                    rule_scores = trigger_rule_scores(predicted, controller_reference, rules)
                    chosen = rule_scores.iloc[0]
                    for candidate in rule_scores.itertuples():
                        record[f"validation_{candidate.rule}_cop"] = candidate.validation_cop
                        record[f"validation_{candidate.rule}_evaluable"] = (
                            candidate.validation_evaluable
                        )
                        for key in ["validation_paired", "model_sum", "rb_sum", "optimal_sum"]:
                            record[f"validation_{candidate.rule}_{key}"] = getattr(candidate, key)
                    threshold = 0.5
                    record.update(
                        validation_mean_cycle_cop=chosen.validation_cop,
                        threshold=threshold,
                        validation_evaluable=int(chosen.validation_evaluable),
                        validation_cycles=int(chosen.validation_cycles),
                        validation_trigger_rule=chosen.rule,
                    )
                    score = (
                        int(chosen.validation_evaluable),
                        float(chosen.validation_cop)
                        if np.isfinite(chosen.validation_cop)
                        else -np.inf,
                    )
                else:
                    threshold, grid = select_cop_threshold(
                        predicted,
                        thresholds,
                        strategy=validation_strategy,
                        reference=controller_reference,
                    )
                    score = float(grid.loc[grid.threshold.eq(threshold), "mean_cycle_cop"].iloc[0])
                    record.update(validation_mean_cycle_cop=score, threshold=threshold)
                    if validation_complete_first:
                        replay = replay_cycles(
                            predicted,
                            threshold,
                            strategy=validation_strategy,
                            reference=controller_reference,
                        )
                        count = int(replay.selected_cop.notna().sum())
                        record.update(validation_evaluable=count, validation_cycles=len(replay))
                        score = (count, score if np.isfinite(score) else -np.inf)
                if validation_tiebreak == "set_loss":
                    supported = predicted.loc[available].copy()
                    observable = supported.groupby("cycle_name").near_optimal.transform("any")
                    supported = supported.loc[observable]
                    with torch.no_grad():
                        value = (
                            float(
                                _batch_loss(
                                    torch.as_tensor(supported.decision_margin.to_numpy()),
                                    supported.drop(
                                        columns=[
                                            "stopping_candidate",
                                            "stopping_log_weight",
                                            "stopping_distribution",
                                        ],
                                        errors="ignore",
                                    ),
                                    "set_stop",
                                )
                            )
                            if len(supported)
                            else np.inf
                        )
                    record.update(
                        validation_set_loss=value,
                        validation_set_cycles=supported.cycle_name.nunique(),
                    )
                    score = (*score, -value) if isinstance(score, tuple) else (score, -value)
            else:
                raise ValueError(f"unknown validation metric: {validation_metric}")
            if score > best_score:
                best_score, best_epoch, best_threshold = score, epoch, threshold
                best_state, stale = copy.deepcopy(evaluation_state), 0
            else:
                stale += 1
        history.append(record)
        if validation is not None and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "training_cycles": sorted(train.cycle_name.unique()),
        "architecture": architecture,
        "film": film,
        "hidden_width": hidden_width,
        "recurrent_layers": recurrent_layers,
        "mlp_layers": mlp_layers,
        "mlp_widths": list(mlp_widths) if mlp_widths else None,
        "reset_history": reset_history,
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
        "trigger_rule": trigger_rule,
        "dropout_config": dropout_config,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
    if loss_name == "cdf_ramp":
        result["cdf_score"] = "bce"
        result["cdf_link"] = (
            "instantaneous"
            if "stopping_cdf_instantaneous" in train
            and bool(train.stopping_cdf_instantaneous.iloc[0])
            else "cumulative"
        )
    return result
