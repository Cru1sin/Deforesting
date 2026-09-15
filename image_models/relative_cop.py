"""Shared fold-isolated effective-COP training and online decision evaluation."""

from __future__ import annotations

import copy
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from dataset_tools import DatasetLoader
from dataset_tools.cycle_metadata import complete_peak_screen
from defrost_decision.candidate_quantities import measure_candidate_quantities
from defrost_decision.performance_objectives import calculate_cycle_cop
from defrost_event_models.ridge_models import (
    DYNAMIC_STATE_8,
    OUTCOME_TARGETS,
    fit_model_on_all_experiments,
    model_to_parameters,
    predict_with_model_parameters,
    select_valid_events_for_quantity,
)
from image_models.outcome_representation import OutcomeRepresentation
from image_models.sensor_features import CURRENT_SENSORS, build_past_only_sensor_statistics

SENSORS = tuple(c for c in CURRENT_SENSORS if c != "environment_relative_humidity")
STATISTICS = ("current", "mean", "std", "skew", "kurt", "slope", "entropy")
RGB = [f"dinov2_{i:03d}" for i in range(384)]
LEDGER = ["pre_defrost_heat_kwh", "pre_defrost_electricity_kwh"]
PHYSICAL_STATE = [
    f"stat_{name}_current"
    for name in (
        "ambient_temperature",
        "water_in_temperature",
        "water_out_temperature",
        "water_temperature_setpoint",
        "water_flow",
        "evaporating_pressure",
        "condensing_pressure",
        "coil_temperature",
        "condensing_temperature",
        "compressor_frequency",
        "superheat",
        "heating_capacity",
        "compressor_power",
        "power_total",
    )
] + LEDGER

def reliable_rgb_cohort(dataset, *, source=None):
    """Return the 87 cycles valid now, before COP filtering, and for RGB."""
    loader = dataset if isinstance(dataset, DatasetLoader) else DatasetLoader(dataset)
    cohort = loader.list_valid_cycles(require_rgb=True)[
        ["cycle_name", "experiment_id"]
    ].sort_values(
        ["experiment_id", "cycle_name"], kind="stable"
    ).reset_index(drop=True)
    if source is None:
        return cohort
    return cohort.merge(source, on=["cycle_name", "experiment_id"], how="inner")


def feature_columns(rgb="on"):
    return [
        *(RGB if rgb == "on" else []),
        *[f"stat_{c}_{s}" for c in (*SENSORS, "candidate_cop") for s in STATISTICS],
        *LEDGER,
        "elapsed_minutes",
    ]


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class RelativeCOP(nn.Module):
    """Exactly Solution_u's layers; only the physical input dimension changes."""

    def __init__(self, width, outputs=1, rgb_projection=0):
        super().__init__()
        self.rgb_projection = nn.Sequential(nn.Linear(384, rgb_projection), Sin()) if rgb_projection else None
        if rgb_projection:
            width = width - 384 + rgb_projection
        self.encoder = nn.Sequential(
            nn.Linear(width, 60),
            Sin(),
            nn.Linear(60, 60),
            Sin(),
            nn.Dropout(0.2),
            nn.Linear(60, 32),
        )
        self.predictor = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 32), Sin(), nn.Linear(32, outputs))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.rgb_projection is not None:
            x = torch.cat([self.rgb_projection(x[:, :384]), x[:, 384:]], dim=1)
        return self.predictor(self.encoder(x)).squeeze(1)


class D32COP(OutcomeRepresentation):
    """Original D32 encoder and time-linear head, with one relative-COP target."""

    def __init__(self, columns):
        sensors = [c for c in columns if c.startswith("stat_")]
        ledger = [c for c in LEDGER if c in columns]
        super().__init__(
            len(sensors), len(ledger), use_rgb=True, event_head="multimodal_time_linear"
        )
        self.positions = [columns.index(c) for c in [*sensors, *ledger, *RGB]]
        self.time_position = columns.index("elapsed_minutes")
        self.head = nn.Linear(33, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, values):
        return (
            super()
            .forward(
                values[:, self.positions], values[:, self.time_position : self.time_position + 1]
            )
            .squeeze(1)
        )


def binary_labels(rows):
    """Earliest full-support COP maximum; RGB coverage never changes the reference."""
    labels = pd.Series(np.nan, index=rows.index)
    for _, cycle in rows.groupby("cycle_name", sort=False):
        supported = cycle.loc[cycle.cycle_cop_eligible & np.isfinite(cycle.cycle_cop)]
        if supported.empty or supported.cycle_cop.max() <= 0:
            continue
        peak = supported.loc[
            supported.cycle_cop.eq(supported.cycle_cop.max()), "candidate_defrost_time"
        ].min()
        labels.loc[supported.index] = supported.candidate_defrost_time.ge(peak).astype(float)
    return labels


def near_optimal_labels(target, epsilon=0.01):
    return target.ge(1 - epsilon).astype(float).where(target.notna())


def grouped_audit_folds(cohort):
    """Balance sorted cycles across three experiment-grouped outer folds."""
    from sklearn.model_selection import GroupKFold

    ordered = cohort.sort_values(["experiment_id", "cycle_name"]).reset_index(drop=True)
    splitter = GroupKFold(n_splits=3)
    return {
        f"fold_{fold}": sorted(ordered.iloc[test].experiment_id.astype(str).unique())
        for fold, (_, test) in enumerate(splitter.split(ordered, groups=ordered.experiment_id))
    }


def audit_cycle(curve, epsilons=(.01, .02)):
    """Audit perfect-label compatibility on the complete 30-second input clock."""
    from plots.image_models import two_of_three_trigger

    curve = curve.sort_values("candidate_defrost_time").copy()
    supported = curve.cycle_cop_eligible.fillna(False) & np.isfinite(curve.cycle_cop)
    measurement = (
        curve.pre_defrost_electricity_measurement_valid.fillna(False)
        & curve.pre_defrost_heat_measurement_valid.fillna(False)
    )
    sensor_input = curve.sensor_timestamp.notna()
    front_input = curve.rgb_available.fillna(False)
    joint_input = sensor_input & front_input
    peak = curve.loc[supported, "cycle_cop"].max()
    peak = float(peak) if pd.notna(peak) and peak > 0 else np.nan
    times = pd.to_datetime(curve.candidate_defrost_time)
    clock_times = pd.date_range(times.min(), times.max(), freq="30s")
    on_grid = times.isin(clock_times)

    def supported_at(value):
        return bool((supported & times.eq(pd.to_datetime(value))).any()) if pd.notna(value) else False

    rb_time = pd.to_datetime(curve.t_RB.iloc[0], errors="coerce")
    actual = curve.get("observed_defrost_preparation_start")
    actual_time = pd.to_datetime(actual.iloc[0], errors="coerce") if actual is not None else pd.NaT
    rb_supported = supported_at(rb_time)
    rb_values = curve.loc[supported & times.eq(rb_time), "cycle_cop"]
    rb_cop = float(rb_values.iloc[0]) if not rb_values.empty else np.nan
    result = {
        "cycle_name": str(curve.cycle_name.iloc[0]),
        "experiment_id": str(curve.experiment_id.iloc[0]),
        "candidate_observation_count": len(curve),
        "processing_30s_count": len(clock_times),
        "processing_grid_complete": bool(clock_times.isin(times).all()),
        "trusted_reference_grid_count": int((supported & on_grid).sum()),
        "measurement_valid_count": int(measurement.sum()),
        "reference_available_count": int(supported.sum()),
        "sensor_input_available_count": int(sensor_input.sum()),
        "front_input_available_count": int(front_input.sum()),
        "joint_input_available_count": int(joint_input.sum()),
        "trusted_label_count": int(supported.sum()),
        "observed_defrost_preparation_start": actual_time,
        "rb_reference_supported": rb_supported,
        "actual_preparation_reference_supported": supported_at(actual_time),
        "reference_peak_cop": peak,
        "rb_cop": rb_cop,
        "rb_headroom": (peak - rb_cop) / rb_cop if rb_supported and rb_cop > 0 else np.nan,
    }
    reasons = []
    if not np.isfinite(peak): reasons.append("no_supported_reference")
    if not joint_input.any(): reasons.append("no_joint_sensor_front_input")
    if pd.isna(rb_time):
        reasons.append("no_rb_trigger")
    elif not rb_supported:
        reasons.append("rb_outside_reference_support")
    if pd.isna(actual_time):
        reasons.append("no_observed_preparation_boundary")
    elif not result["actual_preparation_reference_supported"]:
        reasons.append("actual_preparation_outside_reference_support")
    for epsilon in epsilons:
        label = f"{int(round(100 * epsilon))}pct"
        hits = {}
        for mode, available in (
            ("reference", pd.Series(True, index=curve.index)),
            ("sensor", sensor_input),
            ("shared", joint_input),
        ):
            positive = pd.Series(
                (supported & available & curve.cycle_cop.ge((1 - epsilon) * peak)).to_numpy(),
                index=times,
            ).reindex(clock_times, fill_value=False)
            trigger, _ = two_of_three_trigger(clock_times, positive.astype(float))
            at_trigger = supported & times.eq(trigger)
            hit = bool(
                pd.notna(trigger) and at_trigger.any()
                and curve.loc[at_trigger, "cycle_cop"].iloc[0] >= (1 - epsilon) * peak
            )
            result[f"oracle_{mode}_{label}_trigger_time"] = trigger
            result[f"oracle_{mode}_{label}_hit"] = hit
            hits[mode] = hit
        # Historical names remain the shared sensor+front oracle.
        result[f"oracle_{label}_trigger_time"] = result[f"oracle_shared_{label}_trigger_time"]
        result[f"oracle_{label}_hit"] = hits["shared"]
        if not hits["reference"]: reasons.append(f"oracle_reference_{label}_not_implementable")
    result["missing_data_reason"] = ";".join(reasons)
    return result


def summarize_audit(rows, epsilons=(.01, .02)):
    """Report whole-queue and trusted-reference denominators without imputation."""
    summaries = []
    grid = rows.get("trusted_reference_grid_count", pd.Series(1, index=rows.index)).gt(0)
    comparable = rows.reference_peak_cop.notna() & grid
    for mode in ("reference", "sensor", "shared"):
        for epsilon in epsilons:
            label = f"{int(round(100 * epsilon))}pct"
            column = f"oracle_{mode}_{label}_hit"
            hit = rows.get(column, rows[f"oracle_{label}_hit"]).fillna(False)
            summaries.append({
                "cohort": "valid", "oracle_mode": mode, "epsilon": epsilon,
                "cohort_denominator": len(rows),
                "trusted_reference_denominator": int(comparable.sum()),
                "trusted_reference_grid_denominator": int(comparable.sum()),
                "oracle_hit_count": int(hit.sum()),
                "oracle_hit_rate_full_cohort": float(hit.mean()) if len(rows) else np.nan,
                "oracle_hit_rate_trusted_reference": float(hit.loc[comparable].mean()) if comparable.any() else np.nan,
                "oracle_hit_rate_trusted_reference_grid": float(hit.loc[comparable].mean()) if comparable.any() else np.nan,
                "rb_reference_supported_count": int(rows.rb_reference_supported.fillna(False).sum()),
                "rb_reference_support_rate_full_cohort": float(rows.rb_reference_supported.fillna(False).mean()) if len(rows) else np.nan,
            })
    summary = pd.DataFrame(summaries)
    feasible = summary.loc[
        summary.oracle_mode.eq("reference")
        & summary.oracle_hit_rate_trusted_reference_grid.ge(.9)
    ]
    selected = float(feasible.epsilon.iloc[0]) if not feasible.empty else np.nan
    research = rows.get("rgb_valid_cohort", pd.Series(True, index=rows.index)).fillna(False)
    joint = rows.get("joint_input_available_count", pd.Series(0, index=rows.index)).gt(0)
    rb_comparable = (
        research & rows.reference_peak_cop.notna()
        & rows.rb_reference_supported.fillna(False) & joint
    )
    needs = not np.isfinite(selected)
    summary["selected_epsilon"] = selected
    summary["classifier_readiness"] = (
        "needs_label_development" if needs else "classifier_development_ready"
    )
    summary["cop_benefit_readiness"] = (
        "cop_benefit_comparison_available"
        if rb_comparable.any() else "needs_reference_support_development"
    )
    summary["gate"] = summary["classifier_readiness"]
    return summary


def support_decomposition_rows(oof, full, events):
    """Compare fixed-fold and full-development Ridge support at RB and actual preparation."""
    event_rows = events.drop_duplicates("cycle_name").set_index("cycle_name")

    def point_record(cycle, full_cycle, point, when, delta):
        missing = "no_rb_trigger" if point == "rb" else "no_observed_preparation_boundary"
        oof_point = cycle.loc[cycle.candidate_defrost_time.eq(when)] if pd.notna(when) else cycle.iloc[:0]
        full_point = full_cycle.loc[full_cycle.candidate_defrost_time.eq(when)] if pd.notna(when) else full_cycle.iloc[:0]
        row = oof_point.iloc[0] if len(oof_point) else None
        full_row = full_point.iloc[0] if len(full_point) else None

        def value(source, name, default=np.nan):
            return source.get(name, default) if source is not None else default

        measurement = bool(
            row is not None
            and value(row, "pre_defrost_electricity_measurement_valid", False)
            and value(row, "pre_defrost_heat_measurement_valid", False)
        )
        state_complete = bool(row is not None and row[list(DYNAMIC_STATE_8)].notna().all())

        def reason(source):
            if pd.isna(when): return missing
            if source is None: return "candidate_not_found"
            if not measurement: return "measurement_invalid"
            if not value(source, "defrost_event_electricity_prediction_available", False):
                return "prediction_unavailable"
            if not value(source, "defrost_event_electricity_in_training_domain", False):
                return "ridge_outside_support"
            return "supported" if value(source, "cycle_cop_eligible", False) else "reference_unavailable"

        sensor = bool(row is not None and pd.notna(value(row, "sensor_timestamp")))
        front = bool(row is not None and value(row, "rgb_available", False))
        name = str(cycle.cycle_name.iloc[0])
        event = event_rows.loc[name] if name in event_rows.index else None
        event_energy_valid = bool(
            event is not None and event.get("energy_event_valid", False)
        )
        own_event_available = bool(
            event_energy_valid
            and pd.notna(event.get("defrost_event_electricity_observed_kwh", np.nan))
        )
        record = {
            "cycle_name": name, "experiment_id": str(cycle.experiment_id.iloc[0]),
            "point": point, "candidate_defrost_time": when,
            "rb_vs_actual_prep_minutes": delta,
            "measurement_valid": measurement, "ridge_state_complete": state_complete,
            "sensor_input_available": sensor, "front_input_available": front,
            "shared_input_available": sensor and front,
            "rb_time_kind": "rule_based_replay",
            "event_energy_valid": event_energy_valid,
            "event_invalid_reason": event.get("event_invalid_reason", "") if event is not None else "no_observed_event",
            "own_event_outcome_available": own_event_available,
            "oof_root_reason": reason(row), "full_root_reason": reason(full_row),
        }
        for prefix, source in (("oof", row), ("full", full_row)):
            record[f"{prefix}_prediction_available"] = bool(value(source, "defrost_event_electricity_prediction_available", False))
            record[f"{prefix}_ridge_in_training_domain"] = bool(value(source, "defrost_event_electricity_in_training_domain", False))
            record[f"{prefix}_reference_supported"] = bool(value(source, "cycle_cop_eligible", False))
            record[f"{prefix}_event_electricity_prediction_kwh"] = value(source, "defrost_event_electricity_kwh")
            record[f"{prefix}_support_distance"] = value(source, "defrost_event_electricity_support_distance")
            record[f"{prefix}_support_threshold"] = value(source, "defrost_event_electricity_support_threshold")
        record["support_change"] = (
            "oof_supported" if record["oof_reference_supported"] else
            "full_development_rescue" if record["full_reference_supported"] else
            "outside_both"
        )
        return record

    records = []
    for name, cycle in oof.groupby("cycle_name", sort=True):
        full_cycle = full.loc[full.cycle_name.eq(name)]
        rb = pd.to_datetime(cycle.t_RB.iloc[0], errors="coerce")
        actual = pd.to_datetime(cycle.observed_defrost_preparation_start.iloc[0], errors="coerce")
        delta = (actual - rb).total_seconds() / 60 if pd.notna(rb) and pd.notna(actual) else np.nan
        records.extend([
            point_record(cycle, full_cycle, "rb", rb, delta),
            point_record(cycle, full_cycle, "actual_preparation", actual, delta),
        ])
    return pd.DataFrame(records)


def classification_regression_loss(outputs, target, epsilon=0.01, weight=1.0):
    positive = target.ge(1 - epsilon).to(target.dtype)
    if outputs.ndim == 1:
        return nn.functional.binary_cross_entropy_with_logits(outputs, positive)
    return nn.functional.binary_cross_entropy_with_logits(outputs[:, 0], positive) + weight * (
        outputs[:, 1] - target
    ).square().mean()


def processing_rows(rows):
    """Select the same fixed 30-second clock used for online replay."""
    first = rows.groupby("cycle_name").candidate_defrost_time.transform("min")
    return (rows.candidate_defrost_time - first).dt.total_seconds().mod(30).eq(0)


def regression_model(columns, architecture, rgb_projection=0):
    if rgb_projection and (columns[:384] != RGB or architecture not in ("r-cop32", "cop-classification", "cop-classification-regression")):
        raise ValueError("RGB projection requires complete leading RGB features and an R-COP32 model")
    if architecture == "dinov2-binary":
        return nn.Sequential(
            nn.Linear(len(columns), 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    return D32COP(columns) if architecture == "d32" else RelativeCOP(
        len(columns), outputs=2 if architecture == "cop-classification-regression" else 1, rgb_projection=rgb_projection
    )


def dynamic_network(width):
    """Official three-layer MLP; its biases retain nn.Linear initialization."""
    model = nn.Sequential(
        nn.Linear(width, 60),
        Sin(),
        nn.Linear(60, 60),
        Sin(),
        nn.Dropout(0.2),
        nn.Linear(60, 1),
    )
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
    return model


def pinn_forward(solution, dynamics, values):
    values = values.detach().requires_grad_(True)
    u = solution(values)
    derivative = torch.autograd.grad(u.sum(), values, create_graph=True)[0]
    # Derivatives are in standardized coordinates; time is the final input column.
    inputs = torch.cat((values, u[:, None], derivative), dim=1)
    return u, derivative[:, -1] - dynamics(inputs).squeeze(1)


def direction_loss(u1, u2, y1, y2):
    return torch.relu((u2 - u1) * (y1 - y2)).sum()


def state_transition_loss(dynamics, previous, current, u_previous, u_current, positions):
    inputs = torch.cat(
        (current[:, positions], previous[:, positions], current[:, -1:], u_previous[:, None]), dim=1
    )
    return (dynamics(inputs).squeeze(1) - u_current).square().mean()


def adjacent_pairs(rows):
    left = np.flatnonzero(rows.cycle_name.eq(rows.cycle_name.shift(-1)).to_numpy())
    return left, left + 1


def peak_pair_weights(rows, left, right):
    """Correct endpoint multiplicity so uniform pair sampling estimates the stated loss."""
    counts = np.bincount(np.r_[left, right], minlength=len(rows))
    raw = (1.0 + 4.0 * rows.target.ge(0.95)) * (counts > 0)
    total = raw.groupby(rows.cycle_name).transform("sum")
    cycles = rows.loc[counts > 0, "cycle_name"].nunique()
    weights = raw.div(total.replace(0, np.nan)).fillna(0).to_numpy() / cycles
    return weights * (2 * len(left)) / np.maximum(counts, 1)


def official_learning_rates(epochs):
    warmup = np.linspace(5e-4, 1e-3, min(10, epochs))
    decay = epochs - len(warmup)
    return np.r_[
        warmup, 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * np.arange(decay) / max(decay, 1)))
    ]


def cycle_weights(groups):
    return 1.0 / (groups.map(groups.value_counts()).to_numpy() * groups.nunique())


def normalize_curve(curve):
    result = curve.copy()
    supported = result.cycle_cop.where(result.cycle_cop_eligible & np.isfinite(result.cycle_cop))
    maximum = supported.max()
    result["reference_max"] = maximum if maximum > 0 else np.nan
    result["target"] = supported / result.reference_max
    result["relative_reference"] = result.cycle_cop / result.reference_max
    return result


def screen_curves(args):
    """Screen the frozen reference cohort without training or changing its labels."""
    from train_pareto_boundary import save_settings

    source = args.reference_run / "predictions.parquet"
    columns = ["cycle_name", "experiment_id", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible"]
    rows = pd.read_parquet(source, columns=columns)
    if not 0 < args.near_optimal_epsilon < 1 or args.screen_duration_seconds <= 0:
        raise ValueError("screening needs 0 < epsilon < 1 and positive duration")
    settings = dict(reference=str(source.resolve()), epsilon=args.near_optimal_epsilon,
                    minimum_seconds=args.screen_duration_seconds, maximum_gap_seconds=30,
                    rule="supported COP below (1-epsilon)*peak on both sides for a continuous interval",
                    interpretation="offline complete-peak cohort; not raw sensor quality")
    save_settings(args.output / "settings.json", settings)
    result = complete_peak_screen(rows, args.near_optimal_epsilon, args.screen_duration_seconds)
    result.to_csv(args.output / "cohort.csv", index=False)
    result.loc[result.selected].to_csv(args.output / "selected_cycles.csv", index=False)
    print(result.reason.value_counts().to_string(), flush=True)


def history_features(curve):
    signal = curve[["cycle_name", "candidate_defrost_time", "cycle_cop"]].rename(
        columns={"candidate_defrost_time": "timestamp", "cycle_cop": "candidate_cop"}
    )
    stats = build_past_only_sensor_statistics(
        signal, current_sensors=("candidate_cop",), bucket_seconds=0, include_current=True
    )
    return pd.merge_asof(
        curve[["candidate_defrost_time"]],
        stats.drop(columns="cycle_name"),
        left_on="candidate_defrost_time",
        right_on="sensor_timestamp",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(seconds=30),
    ).drop(columns=["candidate_defrost_time", "sensor_timestamp"])


def prepare_cycle(args, row):
    path = args.output / "base" / f"{row.cycle_name}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if "effective_heat_rule" not in cached:
            raise ValueError("Old heat definition in cached inputs; use a new --output directory")
        expected_end = getattr(row, "observation_end", None)
        if expected_end is not None and pd.to_datetime(cached.candidate_defrost_time).max() != pd.to_datetime(expected_end):
            raise ValueError(f"{row.cycle_name}: cached candidate boundary changed; use a new output directory")
        if hasattr(row, "observed_defrost_preparation_start"):
            expected_actual = pd.to_datetime(row.observed_defrost_preparation_start, errors="coerce")
            cached_actual = pd.to_datetime(
                cached.get("observed_defrost_preparation_start", pd.Series([pd.NaT])).iloc[0],
                errors="coerce",
            )
            if not (pd.isna(expected_actual) and pd.isna(cached_actual)) and expected_actual != cached_actual:
                raise ValueError(f"{row.cycle_name}: cached preparation boundary changed; use a new output directory")
        return
    source = args.reference_run / "base" / path.name
    if source.exists():
        cached = pd.read_parquet(source)
        if not cached.effective_heat_rule.eq("outlet_at_least_recovery_temperature").all():
            raise ValueError("reference inputs use a different effective heat definition")
        expected = {
            "heating_start": getattr(row, "heating_start", None),
            "stable_heating_start": getattr(row, "stable_heating_start", None),
            "t_RB": getattr(row, "t_RB", None),
        }
        matches = all(
            value is None
            or pd.to_datetime(cached[name].iloc[0], errors="coerce")
            == pd.to_datetime(value, errors="coerce")
            for name, value in expected.items() if name in cached
        )
        end = getattr(row, "observation_end", None)
        matches &= end is None or pd.to_datetime(cached.candidate_defrost_time).max() == pd.to_datetime(end)
        if matches:
            if args.rgb == "off" and not getattr(args, "require_rgb_input", False):
                cached = cached.drop(columns=[*RGB, "file_name", "image_time"], errors="ignore")
                cached["rgb_available"] = False
            if hasattr(row, "observed_defrost_preparation_start"):
                cached["observed_defrost_preparation_start"] = pd.to_datetime(
                    row.observed_defrost_preparation_start
                )
            cached.to_parquet(path, index=False)
            return
    loader = DatasetLoader(args.dataset)
    bounds = loader.get_cycle_record(row.cycle_name)["boundaries"]
    start = pd.Timestamp(row.heating_start)
    recovery = pd.Timestamp(row.stable_heating_start)
    end = pd.Timestamp(bounds.get("defrost_preparation_start") or row.observation_end)
    first = max(recovery + pd.Timedelta(seconds=1), start + pd.Timedelta(minutes=5))
    times = pd.date_range(first, end, freq="10s").union(pd.DatetimeIndex([end]))
    rb = pd.to_datetime(row.t_RB)
    if pd.notna(rb) and first <= rb <= end:
        times = times.union(pd.DatetimeIndex([rb]))
    measured = measure_candidate_quantities(
        loader.load_cycle_original(row.cycle_name),
        pd.DataFrame({"candidate_defrost_time": times, "heating_accounting_start": recovery}),
        start,
        heat_column="heating_capacity",
    )
    measured["cycle_name"] = row.cycle_name
    measured["experiment_id"] = row.experiment_id
    measured["heating_start"] = start
    measured["stable_heating_start"] = recovery
    measured["elapsed_minutes"] = (times - recovery).total_seconds() / 60
    measured["t_RB"] = rb
    measured["observed_defrost_preparation_start"] = pd.to_datetime(
        getattr(
            row, "observed_defrost_preparation_start", bounds.get("defrost_preparation_start")
        ),
        errors="coerce",
    )
    processed = loader.load_cycle(row.cycle_name)
    stats = build_past_only_sensor_statistics(
        processed,
        current_sensors=SENSORS,
        bucket_seconds=int(loader.registry["resample_interval_seconds"]),
        include_current=True,
    )
    stats = stats[["sensor_timestamp", *[f"stat_{c}_{s}" for c in SENSORS for s in STATISTICS]]]
    measured = pd.merge_asof(
        measured,
        stats.sort_values("sensor_timestamp"),
        left_on="candidate_defrost_time",
        right_on="sensor_timestamp",
        direction="backward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(seconds=15),
    )
    measured["rgb_available"] = False
    if args.rgb == "on" or getattr(args, "require_rgb_input", False):
        metadata = loader.load_image_metadata(row.cycle_name)
        metadata = metadata.loc[
            metadata.camera_role.eq("front"), ["image_time", "file_name"]
        ].copy()
        metadata["image_time"] = pd.to_datetime(metadata.image_time)
        metadata = metadata.loc[metadata.image_time.ge(start)].sort_values("image_time")
        cache = args.rgb_cache / f"{row.cycle_name}.parquet"
        if cache.exists():
            features = pd.read_parquet(cache)
            features = features.loc[features.camera_role.eq("front"), ["file_name", *RGB]]
            metadata = metadata.merge(features, on="file_name", how="left", validate="one_to_one")
        else:
            metadata = metadata.reindex(columns=[*metadata.columns, *RGB])
        measured = pd.merge_asof(
            measured,
            metadata,
            left_on="candidate_defrost_time",
            right_on="image_time",
            direction="backward",
            allow_exact_matches=False,
            tolerance=pd.Timedelta(seconds=45),
        )
        measured["rgb_available"] = measured[RGB].notna().all(axis=1)
    measured.to_parquet(path, index=False)
    print(
        f"[relative-cop base] {row.cycle_name}: {measured.rgb_available.sum()}/{len(measured)} RGB",
        flush=True,
    )


def complete_local_rgb(args, cohort):
    """Supplement missing frozen embeddings from already downloaded photos only."""
    from image_models.dinov2_features import _embed, _load_backbone

    model = None
    for name in cohort.cycle_name:
        path = args.output / "base" / f"{name}.parquet"
        frame = pd.read_parquet(path)
        missing = frame.loc[~frame.rgb_available & frame.file_name.notna(), "file_name"].unique()
        local = [(file, args.dataset / "images" / name / "front" / file) for file in missing]
        local = [(file, path) for file, path in local if path.is_file()]
        if not local:
            continue
        if model is None:
            torch.set_num_threads(1)
            model = _load_backbone(torch.device("cpu"))
        for start in range(0, len(local), 16):
            batch = local[start : start + 16]
            vectors = _embed([path.read_bytes() for _, path in batch], model, torch.device("cpu"))
            for (file, _), vector in zip(batch, vectors, strict=True):
                frame.loc[frame.file_name.eq(file), RGB] = vector
        frame["rgb_available"] = frame[RGB].notna().all(axis=1)
        frame.to_parquet(path, index=False)
        print(f"[relative-cop RGB] {name}: {len(local)} local embeddings added", flush=True)


def ridge_parameters(events, excluded):
    selected = select_valid_events_for_quantity(events, "event_electricity")
    selected = selected.loc[~selected.experiment_id.isin(excluded)]
    if selected.experiment_id.nunique() < 3:
        raise ValueError(f"insufficient Ridge experiments after exclusions: {sorted(excluded)}")
    return model_to_parameters(
        fit_model_on_all_experiments(
            selected, DYNAMIC_STATE_8, OUTCOME_TARGETS["event_electricity"]
        )
    )


def apply_reference(base, parameters, rgb="on", *, include_history=True):
    result = base.copy()
    prediction = predict_with_model_parameters(parameters, result)
    result["defrost_event_electricity_kwh"] = prediction.prediction.to_numpy()
    result["defrost_event_electricity_prediction_available"] = np.isfinite(prediction.prediction)
    result["defrost_event_electricity_in_training_domain"] = prediction.support_distance.le(
        prediction.support_threshold
    ).to_numpy()
    result["defrost_event_electricity_support_distance"] = prediction.support_distance.to_numpy()
    result["defrost_event_electricity_support_threshold"] = prediction.support_threshold.to_numpy()
    result["defrost_event_net_heat_kwh"] = 0.0
    result["defrost_event_net_heat_prediction_available"] = True
    result["defrost_event_net_heat_in_training_domain"] = True
    result = calculate_cycle_cop(result, effective=True)
    result = normalize_curve(result)
    # Unsupported predictions must not become apparently valid COP-history observations.
    if include_history:
        history = history_features(
            result.assign(cycle_cop=result.cycle_cop.where(result.cycle_cop_eligible))
        )
        result = pd.concat([result.reset_index(drop=True), history.reset_index(drop=True)], axis=1)
    result["input_available"] = result.sensor_timestamp.notna() & (
        result.rgb_available if rgb == "on" else True
    )
    return result


def _write_matching_csv(path, frame):
    text = frame.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"existing audit input changed; use a new output directory: {path.name}")
    path.write_text(text)


def _audit_fold(output, cohort, events, fold, excluded):
    parameters = ridge_parameters(events, frozenset(excluded))
    tables = []
    heldout = cohort.loc[cohort.experiment_id.astype(str).isin(excluded)]
    for row in heldout.itertuples():
        base = pd.read_parquet(output / "base" / f"{row.cycle_name}.parquet")
        table = apply_reference(base, parameters, "on")
        table["audit_fold"] = fold
        tables.append(table)
    return pd.concat(tables, ignore_index=True), parameters


def _full_reference_cycle(output, cycle_name, parameters):
    base = pd.read_parquet(output / "base" / f"{cycle_name}.parquet")
    return apply_reference(base, parameters, "on")


def audit(args):
    """Run the pre-training support and perfect-label compatibility audit."""
    from defrost_decision.baselines import rule_based
    from defrost_event_models.training_data import build_defrost_event_training_table
    from train_pareto_boundary import save_settings

    loader = DatasetLoader(args.dataset)
    event_settings_path = args.event_run / "run_settings.json"
    if not event_settings_path.is_file():
        raise ValueError(f"event recovery settings are missing: {event_settings_path}")
    event_settings = json.loads(event_settings_path.read_text())
    if event_settings.get("preparation_heat") != "zero":
        raise ValueError("COP classification audit requires zero preparation heat")
    loader.configure_recovery(event_settings["recovery_settings"])
    cohort = reliable_rgb_cohort(loader)
    cohort["catalog_valid"] = True
    cohort["rgb_valid"] = True
    cohort["rgb_valid_cohort"] = True
    boundaries = []
    for row in cohort.itertuples():
        record = loader.get_cycle_record(row.cycle_name)
        bounds = record.get("boundaries", record)
        rb = rule_based.calculate_cycle(loader, row.cycle_name)
        boundaries.append({
            "cycle_name": row.cycle_name, "experiment_id": str(row.experiment_id),
            "heating_start": bounds.get("heating_start"),
            "stable_heating_start": bounds.get("stable_heating_start"),
            "observed_defrost_preparation_start": bounds.get("defrost_preparation_start"),
            "observation_end": rb.get("t_observation_end"),
            "t_RB": rb.get("t_RB"), "rb_status": rb.get("rb_status"),
        })
    boundaries = pd.DataFrame(boundaries)
    cohort = cohort.drop(
        columns=["heating_start", "stable_heating_start", "t_RB", "rb_status"],
        errors="ignore",
    ).merge(boundaries, on=["cycle_name", "experiment_id"], validate="one_to_one")
    folds = grouped_audit_folds(cohort)
    events = build_defrost_event_training_table(
        loader, preparation_heat=event_settings["preparation_heat"]
    )
    events = events.loc[events.cycle_name.isin(cohort.cycle_name)].reset_index(drop=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "base").mkdir(exist_ok=True)
    (args.output / "ridge").mkdir(exist_ok=True)
    _write_matching_csv(args.output / "cohort.csv", cohort)
    _write_matching_csv(args.output / "defrost_events.csv", events)
    _write_matching_csv(args.output / "recovery_boundaries.csv", boundaries)
    save_settings(args.output / "folds.json", folds)
    settings = {
        "task": "cop-classification", "action": "audit",
        "dataset": str(args.dataset.resolve()),
        "reference_run": str(args.reference_run.resolve()),
        "decision_run": str(args.decision_run.resolve()),
        "cohort_rule": "current_valid_and_pre_COP_valid_and_RGB_valid",
        "catalog_valid_cycles": cohort.cycle_name.astype(str).tolist(),
        "rgb_valid_cycles": cohort.cycle_name.astype(str).tolist(),
        "ridge_cv": "sorted_GroupKFold_3",
        "ridge_events": "current_catalog_raw_quantity_valid_event_electricity",
        "event_recovery_settings": event_settings["recovery_settings"],
        "preparation_heat": event_settings["preparation_heat"],
        "processing_clock_seconds": 30,
        "oracle_confirmation": "two_of_three",
        "oracle_epsilons": [.01, .02],
        "epsilon_selection": "reference_only_oracle_all_valid_cycles",
        "causality": "conditional_on_offline_boundaries",
        "rb_time_kind": "rule_based_replay_not_recorded_defrost",
        "full_development_ridge": "coverage_diagnostic_only_may_include_own_event",
        "interpretation": "classifier_readiness_separate_from_cop_benefit_support",
    }
    save_settings(args.output / "settings.json", settings)

    audit_args = copy.copy(args)
    audit_args.rgb = "on"
    audit_args.require_rgb_input = True
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(prepare_cycle)(audit_args, row) for row in cohort.itertuples())
        fold_results = Parallel()(delayed(_audit_fold)(
            args.output, cohort, events, fold, excluded
        ) for fold, excluded in folds.items())
    for (fold, _), (_, parameters) in zip(folds.items(), fold_results, strict=True):
        save_settings(args.output / "ridge" / f"{fold}.json", parameters)
    reference = pd.concat([rows for rows, _ in fold_results], ignore_index=True)
    full_parameters = ridge_parameters(events, frozenset())
    save_settings(args.output / "ridge" / "full_development.json", full_parameters)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        full_reference = pd.concat(Parallel()(delayed(_full_reference_cycle)(
            args.output, name, full_parameters
        ) for name in cohort.cycle_name), ignore_index=True)
    reference_path = args.output / "reference_rows.parquet"
    if reference_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(reference_path), reference, check_like=False)
    else:
        reference.to_parquet(reference_path, index=False)
    cycle_audit = pd.DataFrame([audit_cycle(cycle) for _, cycle in reference.groupby("cycle_name")])
    rgb_valid = cohort.set_index("cycle_name").rgb_valid.eq(True)
    cycle_audit["rgb_valid_cohort"] = cycle_audit.cycle_name.map(rgb_valid).fillna(False)
    summary = summarize_audit(cycle_audit)
    _write_matching_csv(args.output / "cycle_audit.csv", cycle_audit)
    _write_matching_csv(args.output / "audit_summary.csv", summary)
    _write_matching_csv(
        args.output / "rb_support_decomposition.csv",
        support_decomposition_rows(reference, full_reference, events),
    )

    rb = pd.to_datetime(reference.t_RB, errors="coerce")
    at_rb = reference.candidate_defrost_time.eq(rb)
    measurement = reference.pre_defrost_electricity_measurement_valid.fillna(False) & reference.pre_defrost_heat_measurement_valid.fillna(False)
    interventions = reference.loc[
        at_rb & measurement & ~reference.defrost_event_electricity_in_training_domain.fillna(False),
        ["cycle_name", "experiment_id", "candidate_defrost_time", *DYNAMIC_STATE_8,
         "defrost_event_electricity_support_distance", "defrost_event_electricity_support_threshold"],
    ]
    _write_matching_csv(args.output / "intervention_candidates.csv", interventions)
    print(summary.to_string(index=False), flush=True)


def build_fold_rows(
    args, cohort, events, excluded, *, include_history=True, base_root=None,
):
    """Cross-fit every training experiment; excluded experiments never enter any Ridge."""
    tables, parameters = [], {}
    for experiment, cycles in cohort.groupby("experiment_id", sort=True):
        omit = frozenset((*excluded, experiment))
        key = "__".join(sorted(omit))
        model_path = args.output / "ridge" / f"{key or 'full'}.json"
        source = args.reference_run / "ridge" / model_path.name
        if not model_path.exists() and source.exists() and not getattr(args, "quality_filtered", False):
            shutil.copyfile(source, model_path)
        if model_path.exists():
            model = json.loads(model_path.read_text())
        else:
            model = ridge_parameters(events, omit)
            # Independent workers can produce the same deterministic fit; retain one file.
            temporary = model_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(model))
            temporary.replace(model_path)
        assert not set(model["training_experiment_ids"]) & omit
        parameters[key] = model
        for name in cycles.cycle_name:
            root = args.output / "base" if base_root is None else Path(base_root)
            base = pd.read_parquet(root / f"{name}.parquet")
            tables.append(apply_reference(
                base, model,
                "on" if getattr(args, "require_rgb_input", False) else args.rgb,
                include_history=include_history,
            ))
    return pd.concat(tables, ignore_index=True), parameters


def usable(rows):
    return rows.loc[rows.input_available & rows.target.notna()].copy()


def fit_regression(
    train,
    validation,
    *,
    epochs,
    patience,
    seed,
    architecture="r-cop32",
    rgb="on",
    batch_size=256,
    pinn_alpha=1.0,
    pinn_beta=1.0,
    peak_weighted=False,
    schedule_epochs=200,
    near_optimal_epsilon=0.01,
    regression_weight=1.0,
    classification_label="near-optimal",
    rgb_projection=0,
):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    joint = architecture in ("cop-classification", "cop-classification-regression")
    binary = architecture == "dinov2-binary"
    after_optimum = classification_label == "after-optimum"
    if after_optimum:
        if architecture != "cop-classification":
            raise ValueError("after-optimum labels require pure classification")
        # Find the reference before discarding missing RGB or sensor inputs.
        train = train.assign(binary_target=binary_labels(train))
        if validation is not None:
            validation = validation.assign(binary_target=binary_labels(validation))
    if binary:
        train = train.assign(
            binary_target=binary_labels(train), input_available=train.rgb_available
        )
        train = train.loc[processing_rows(train) & train.binary_target.notna()]
        if validation is not None:
            validation = validation.assign(
                binary_target=binary_labels(validation), input_available=validation.rgb_available
            )
            validation = validation.loc[
                processing_rows(validation) & validation.binary_target.notna()
            ]
    train = usable(train)
    if "candidate_defrost_time" in train:
        train = train.sort_values(["cycle_name", "candidate_defrost_time"])
    train = train.reset_index(drop=True)
    validation = usable(validation) if validation is not None else None
    if train.empty or (validation is not None and validation.empty):
        if not joint:
            raise ValueError("no usable training or validation samples")
        return dict(
            architecture=architecture, classification_label=classification_label,
            near_optimal_epsilon=near_optimal_epsilon, rgb_projection=rgb_projection,
            status="no_usable_training_samples" if train.empty else "no_usable_validation_samples",
            selected_epoch=0, model_state_dict=None, dynamic_state_dict=None,
            training_experiments=sorted(train.experiment_id.unique().tolist()),
            losses=pd.DataFrame([dict(epoch=0, validation_total_loss=np.nan)]),
        )
    left, right = adjacent_pairs(train)
    if not (binary or joint) and not len(left):
        raise ValueError("no within-cycle training pairs")
    columns = RGB if binary else [c for c in feature_columns(rgb) if train[c].notna().any()]
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(
        train[columns]
    )
    x = torch.tensor(preprocessor.transform(train[columns]), dtype=torch.float32)
    y = torch.tensor(
        train.binary_target.to_numpy() if binary or after_optimum else train.target.to_numpy(),
        dtype=torch.long if binary else torch.float32,
    )
    w = torch.tensor(
        cycle_weights(train.cycle_name) * len(train)
        if binary
        else peak_pair_weights(train, left, right)
        if peak_weighted
        else np.ones(len(train)),
        dtype=torch.float32,
    )
    if validation is not None:
        validation = validation.sort_values(
            ["cycle_name", "candidate_defrost_time"]
            if "candidate_defrost_time" in validation
            else ["cycle_name"]
        )
        vleft, _ = adjacent_pairs(validation)
        if binary or joint:
            vleft = np.arange(len(validation))
        if not len(vleft):
            raise ValueError("no within-cycle validation pairs")
        vx = torch.tensor(
            preprocessor.transform(validation.iloc[vleft][columns]), dtype=torch.float32
        )
        vy = torch.tensor(
            validation.binary_target.to_numpy()
            if binary or after_optimum
            else validation.target.iloc[vleft].to_numpy(),
            dtype=torch.long if binary else torch.float32,
        )
        if binary:
            vw = torch.tensor(cycle_weights(validation.cycle_name), dtype=torch.float32)
    model = regression_model(columns, architecture, rgb_projection)
    state_consistency = architecture == "state-consistency"
    dynamics = (
        dynamic_network(34)
        if state_consistency
        else dynamic_network(2 * len(columns) + 1)
        if architecture == "pinn4soh"
        else None
    )
    if state_consistency:
        positions = [columns.index(c) for c in PHYSICAL_STATE]
        intervals = (
            train.candidate_defrost_time.iloc[right]
            .reset_index(drop=True)
            .sub(train.candidate_defrost_time.iloc[left].reset_index(drop=True))
        )
        state_pairs = intervals.eq(pd.Timedelta(seconds=10)).to_numpy()
        if not state_pairs.any():
            raise ValueError("no observed ten-second state transitions")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    dynamic_optimizer = (
        torch.optim.Adam(dynamics.parameters(), lr=1e-3) if dynamics is not None else None
    )
    rates = official_learning_rates(schedule_epochs)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, stale, best_state = np.inf, epochs, 0, None
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        if dynamics is not None:
            dynamics.train()
        totals = np.zeros(3)
        batches = 0
        for indices in torch.randperm(
            len(train) if binary or joint else len(left), generator=generator
        ).split(batch_size):
            a, b = (
                (indices.numpy(), indices.numpy())
                if binary or joint
                else (left[indices.numpy()], right[indices.numpy()])
            )
            optimizer.zero_grad()
            if dynamic_optimizer is not None:
                dynamic_optimizer.zero_grad()
            if binary or joint:
                u1 = model(x[a])
                residual = direction = torch.zeros(())
            elif state_consistency:
                u1, u2 = model(x[a]), model(x[b])
                valid = state_pairs[indices.numpy()]
                residual = (
                    state_transition_loss(
                        dynamics, x[a][valid], x[b][valid], u1[valid], u2[valid], positions
                    )
                    if valid.any()
                    else torch.zeros(())
                )
                direction = torch.zeros(())
            elif dynamics is not None:
                u1, f1 = pinn_forward(model, dynamics, x[a])
                u2, f2 = pinn_forward(model, dynamics, x[b])
                residual = 0.5 * (f1.square().mean() + f2.square().mean())
                direction = direction_loss(u1, u2, y[a], y[b])
            else:
                u1, u2 = model(x[a]), model(x[b])
                residual = direction = torch.zeros(())
            data = (
                classification_regression_loss(u1, y[a], near_optimal_epsilon, regression_weight)
                if joint else
                (nn.functional.cross_entropy(u1, y[a], reduction="none") * w[a]).mean()
                if binary
                else 0.5
                * (((u1 - y[a]).square() * w[a]).mean() + ((u2 - y[b]).square() * w[b]).mean())
            )
            loss = data + pinn_alpha * residual + pinn_beta * direction
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            if dynamic_optimizer is not None:
                dynamic_optimizer.step()
            totals += [float(term.detach()) for term in (data, residual, direction)]
            batches += 1
        model.eval()
        with torch.no_grad():
            score = (
                (
                    float(classification_regression_loss(model(vx), vy, near_optimal_epsilon, regression_weight))
                    if joint else
                    float((nn.functional.cross_entropy(model(vx), vy, reduction="none") * vw).sum())
                    if binary
                    else float((model(vx) - vy).square().mean())
                )
                if validation is not None
                else np.nan
            )
        losses.append(
            {
                "epoch": epoch,
                "train_total_loss" if joint else "train_cross_entropy" if binary else "train_mse": totals[0] / batches,
                "dynamic_mse": totals[1] / batches,
                "direction_loss": totals[2] / batches,
                "validation_total_loss" if joint else "validation_cross_entropy" if binary else "validation_mse": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_steps": batches,
            }
        )
        # The official scheduler advances after the epoch, including its first warmup value.
        optimizer.param_groups[0]["lr"] = float(rates[epoch - 1])
        if validation is not None:
            if not np.isfinite(score):
                raise ValueError("nonfinite validation loss")
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                best_state = (
                    copy.deepcopy(model.state_dict()),
                    copy.deepcopy(dynamics.state_dict()) if dynamics is not None else None,
                )
            else:
                stale += 1
            if stale > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        if dynamics is not None:
            dynamics.load_state_dict(best_state[1])
    return {
        "architecture": architecture,
        "rgb_projection": rgb_projection,
        "classification_label": classification_label,
        "near_optimal_epsilon": near_optimal_epsilon,
        "regression_weight": regression_weight,
        "model_state_dict": model.state_dict(),
        "dynamic_state_dict": dynamics.state_dict() if dynamics is not None else None,
        "physical_state_columns": PHYSICAL_STATE if state_consistency else [],
        "state_transition_pairs": int(state_pairs.sum()) if state_consistency else 0,
        "state_transition_excluded_pairs": int((~state_pairs).sum()) if state_consistency else 0,
        "feature_columns": columns,
        "preprocessor": preprocessor,
        "selected_epoch": best_epoch,
        "training_experiments": sorted(train.experiment_id.unique().tolist()),
        "pair_coverage": train.assign(
            paired=True if joint else np.bincount(np.r_[left, right], minlength=len(train)) > 0
        )
        .groupby("cycle_name")
        .paired.agg(["size", "sum"]),
        "losses": pd.DataFrame(losses),
    }


def predict_regression(rows, checkpoint):
    result = rows.copy()
    result["prediction"] = np.nan
    joint = checkpoint.get("architecture") in ("cop-classification", "cop-classification-regression")
    if joint:
        result["binary_target"] = (binary_labels(result) if checkpoint.get("classification_label") == "after-optimum"
                                   else near_optimal_labels(result.target, checkpoint["near_optimal_epsilon"]))
        result["probability"] = np.nan
        result["classification_logit"] = np.nan
    binary = checkpoint.get("architecture") == "dinov2-binary"
    if binary:
        if "cycle_cop" in result:
            result["binary_target"] = binary_labels(result)
        result["input_available"] = result.rgb_available
    mask = result.input_available
    if mask.any() and checkpoint.get("model_state_dict") is not None:
        model = regression_model(
            checkpoint["feature_columns"], checkpoint.get("architecture", "r-cop32"), checkpoint.get("rgb_projection", 0)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        x = checkpoint["preprocessor"].transform(result.loc[mask, checkpoint["feature_columns"]])
        with torch.no_grad():
            values = model(torch.tensor(x, dtype=torch.float32))
            if joint:
                logits = values if values.ndim == 1 else values[:, 0]
                result.loc[mask, "classification_logit"] = logits.numpy()
                result.loc[mask, "probability"] = logits.sigmoid().numpy()
                values = torch.full_like(values, np.nan) if values.ndim == 1 else values[:, 1]
            result.loc[mask, "prediction"] = (values.softmax(1)[:, 1] if binary else values).numpy()
    result["trigger_positive"] = (
        result.prediction.ge(0.5 if binary else 0.99) & result.input_available
    )
    if joint:
        result = result.drop(columns="trigger_positive")
        for strategy, calibration in checkpoint.get("thresholds", {}).items():
            result[f"threshold_{strategy}"] = calibration["threshold"]
            result[f"calibration_{strategy}"] = calibration["status"]
    return result


def state_diagnostics(rows, checkpoint):
    """Compare learned transitions with persistence on held-out, fixed-interval pairs."""
    if checkpoint["architecture"] != "state-consistency":
        return {}
    rows = usable(rows).sort_values(["cycle_name", "candidate_defrost_time"]).reset_index(drop=True)
    left, right = adjacent_pairs(rows)
    intervals = (
        rows.candidate_defrost_time.iloc[right].to_numpy()
        - rows.candidate_defrost_time.iloc[left].to_numpy()
    )
    left, right = (
        left[intervals == np.timedelta64(10, "s")],
        right[intervals == np.timedelta64(10, "s")],
    )
    if not len(left):
        return {"pairs": 0}
    columns = checkpoint["feature_columns"]
    x = torch.tensor(checkpoint["preprocessor"].transform(rows[columns]), dtype=torch.float32)
    model = regression_model(columns, "state-consistency")
    model.load_state_dict(checkpoint["model_state_dict"])
    dynamics = dynamic_network(34)
    dynamics.load_state_dict(checkpoint["dynamic_state_dict"])
    model.eval()
    dynamics.eval()
    positions = [columns.index(c) for c in PHYSICAL_STATE]
    with torch.no_grad():
        u = model(x)
        forecast = (
            dynamics(
                torch.cat(
                    (x[right][:, positions], x[left][:, positions], x[right, -1:], u[left, None]),
                    dim=1,
                )
            )
            .squeeze(1)
            .numpy()
        )
    current, previous = u[right].numpy(), u[left].numpy()
    target = rows.target.iloc[right].to_numpy()
    errors = pd.DataFrame(
        {
            "cycle_name": rows.cycle_name.iloc[right].to_numpy(),
            "consistency_mse": (forecast - current) ** 2,
            "persistence_consistency_mse": (previous - current) ** 2,
            "state_teacher_mse": (forecast - target) ** 2,
            "persistence_teacher_mse": (previous - target) ** 2,
        }
    )
    return {"pairs": len(left), **errors.groupby("cycle_name").mean().mean().to_dict()}


def cycle_metrics(rows):
    metrics = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        cycle = cycle.sort_values("candidate_defrost_time")
        supported = cycle.loc[cycle.target.notna()]
        available = supported.loc[supported.prediction.notna()]
        record = {
            "cycle_name": name,
            "experiment_id": cycle.experiment_id.iloc[0],
            "candidate_count": len(cycle),
            "supported_count": len(supported),
            "rgb_count": int(cycle.rgb_available.sum()),
            "predicted_count": len(available),
        }
        if not supported.empty:
            best = supported.loc[supported.cycle_cop.idxmax()]
            record["reference_time"] = best.candidate_defrost_time
            record["reference_cop"] = float(best.cycle_cop)
        if supported.empty:
            record["status"] = "no_fold_supported_candidate"
        elif "probability" in supported and cycle.prediction.isna().all():
            record["status"] = "classification_only" if supported.probability.notna().any() else "no_supported_input"
        elif available.empty:
            record["status"] = "no_supported_input"
        else:
            best = supported.loc[supported.cycle_cop.idxmax()]
            chosen = available.loc[available.prediction.idxmax()]
            input_best = available.loc[available.cycle_cop.idxmax()]
            errors = available.prediction - available.target
            rb = supported.loc[supported.candidate_defrost_time.eq(cycle.t_RB.iloc[0])]
            record.update(
                status="evaluated",
                mse=float(errors.pow(2).mean()),
                rmse=float(np.sqrt(errors.pow(2).mean())),
                mae=float(errors.abs().mean()),
                reference_time=best.candidate_defrost_time,
                predicted_time=chosen.candidate_defrost_time,
                input_optimum_time=input_best.candidate_defrost_time,
                relative_cop_loss=float(1 - chosen.target),
                input_coverage_loss=float(1 - input_best.target),
                network_cop_loss=float(input_best.target - chosen.target),
                time_error_minutes=(
                    chosen.candidate_defrost_time - best.candidate_defrost_time
                ).total_seconds()
                / 60,
                reference_cop=float(best.cycle_cop),
                chosen_cop=float(chosen.cycle_cop),
                rb_cop=float(rb.cycle_cop.iloc[0]) if len(rb) else np.nan,
            )
            dt = record["time_error_minutes"]
            record["early_minutes"] = max(0.0, -dt)
            record["late_minutes"] = max(0.0, dt)
            record["absolute_time_error_minutes"] = abs(dt)
            record["chosen_vs_rb_pct"] = (
                100 * (record["chosen_cop"] / record["rb_cop"] - 1)
                if record["rb_cop"] > 0
                else np.nan
            )
            for percent in (1, 2, 5):
                record[f"within_{percent}pct"] = bool(1 - chosen.target <= percent / 100 + 1e-12)
                near = supported.loc[supported.target.ge(1 - percent / 100 - 1e-12)]
                offsets = (
                    chosen.candidate_defrost_time - near.candidate_defrost_time
                ).dt.total_seconds() / 60
                distance = float(offsets.loc[offsets.abs().idxmin()])
                record[f"near_{percent}pct_early_minutes"] = max(0.0, -distance)
                record[f"near_{percent}pct_late_minutes"] = max(0.0, distance)
        metrics.append(record)
    return pd.DataFrame(metrics)


def training_options(args):
    return dict(
        architecture=args.regression_architecture,
        rgb_projection=getattr(args, "rgb_projection", 0),
        rgb=args.rgb,
        batch_size=args.batch_size,
        pinn_alpha=args.pinn_alpha,
        pinn_beta=args.pinn_beta,
        peak_weighted=args.peak_weighted,
        schedule_epochs=args.maximum_epochs,
        near_optimal_epsilon=getattr(args, "near_optimal_epsilon", .01),
        regression_weight=getattr(args, "regression_weight", 1.),
        classification_label=getattr(args, "classification_label", "near-optimal"),
    )


def train_fold(args, cohort, events, test, inner):
    path = args.output / "folds" / f"{test}.pkl"
    if path.exists():
        return
    print(f"[relative-cop fold] {test}: inner validation {inner}", flush=True)
    nested_cohort = cohort.loc[~cohort.experiment_id.eq(test)]
    nested, inner_models = build_fold_rows(args, nested_cohort, events, {test, inner})
    fitted = fit_regression(
        nested.loc[~nested.experiment_id.eq(inner)],
        nested.loc[nested.experiment_id.eq(inner)],
        epochs=args.maximum_epochs,
        patience=args.patience,
        seed=args.seed,
        **training_options(args),
    )
    epoch = fitted["selected_epoch"]
    inner_losses = fitted["losses"].assign(stage="inner")
    inner_predictions = predict_regression(nested.loc[nested.experiment_id.eq(inner)], fitted)
    binary = args.regression_architecture == "dinov2-binary"
    metric_function = binary_cycle_metrics if binary else cycle_metrics
    inner_metrics = metric_function(inner_predictions)
    joint = args.regression_architecture in ("cop-classification", "cop-classification-regression")
    thresholds, calibration = calibrate_thresholds(inner_predictions, ("two_of_three",) if args.task == "cop-classification" else ("first_positive", "two_of_three")) if joint else ({}, pd.DataFrame())
    loss_name = "validation_total_loss" if joint else "validation_cross_entropy" if binary else "validation_mse"
    if fitted.get("status"):
        thresholds = {strategy: dict(threshold=np.nan, status=fitted["status"]) for strategy in thresholds}
    inner_metrics[loss_name] = fitted["losses"][loss_name].min()
    inner_dynamics = state_diagnostics(nested.loc[nested.experiment_id.eq(inner)], fitted)
    del nested
    outer, outer_models = build_fold_rows(args, cohort, events, {test})
    checkpoint = fitted if not epoch else fit_regression(
        outer.loc[~outer.experiment_id.eq(test)],
        None,
        epochs=epoch,
        patience=args.patience,
        seed=args.seed,
        **training_options(args),
    )
    checkpoint["thresholds"] = thresholds
    predictions = predict_regression(outer.loc[outer.experiment_id.eq(test)], checkpoint)
    output_columns = [
        "cycle_name",
        "experiment_id",
        "candidate_defrost_time",
        "elapsed_minutes",
        "cycle_cop",
        "cycle_cop_eligible",
        "reference_max",
        "relative_reference",
        "target",
        "prediction",
        "input_available",
        "rgb_available",
        "t_RB",
    ]
    if joint:
        output_columns += ["binary_target", "classification_logit", "probability"] + [
            f"{prefix}_{strategy}" for strategy in thresholds
            for prefix in ("threshold", "calibration")
        ]
    else:
        output_columns += ["trigger_positive"]
    if binary:
        output_columns += ["binary_target"]
    result = {
        "checkpoint": checkpoint,
        "calibration": calibration,
        "inner_predictions": inner_predictions if joint else None,
        "predictions": predictions[output_columns],
        "metrics": metric_function(predictions),
        "inner_experiment": inner,
        "inner_metrics": inner_metrics,
        "inner_dynamics": inner_dynamics,
        "outer_dynamics": state_diagnostics(outer.loc[outer.experiment_id.eq(test)], checkpoint),
        "ridge_exclusions": sorted(set(inner_models) | set(outer_models)),
        "losses": pd.concat([inner_losses, checkpoint["losses"].assign(stage="outer")]),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream)
    temporary.replace(path)
    print(f"[relative-cop fold] {test}: complete, epoch {epoch}", flush=True)


def run(args):
    from plots.pareto_learning import render_relative_cop
    from train_pareto_boundary import fold_exclusions, save_settings

    binary = args.regression_architecture == "dinov2-binary"
    current_classification = binary or args.regression_architecture in (
        "cop-classification", "cop-classification-regression",
    )
    metric_function = binary_cycle_metrics if binary else cycle_metrics
    source_settings = args.reference_run / "settings.json"
    if source_settings.exists():
        source = json.loads(source_settings.read_text())
        for key in ("dataset", "decision_run", "event_run"):
            if source.get(key) != str(getattr(args, key).resolve()):
                raise ValueError(f"reference cache has a different {key}")
    if args.batch_size < 1 or args.maximum_epochs < 1:
        raise ValueError("batch size and maximum epochs must be positive")
    if args.regression_architecture == "d32" and args.rgb == "off":
        raise ValueError("historical D32 requires --rgb on")

    cohort = pd.read_csv(args.decision_run / "cycle_comparison.csv")
    if current_classification:
        cohort = reliable_rgb_cohort(args.dataset, source=cohort)
        cohort = cohort.loc[cohort.cycle_status.eq("identified_curve")].copy()
        valid_names = set(cohort.cycle_name)
    else:
        catalog = DatasetLoader(args.dataset).list_cycles(statuses={"valid"})
        valid_names = set(catalog.cycle_name)
        if args.evaluation_cohort == "rgb-valid":
            catalog = catalog.loc[catalog.rgb_valid.eq(True)]
        cohort_names = set(catalog.cycle_name)
        cohort = cohort.loc[
            cohort.cycle_status.eq("identified_curve")
            & cohort.cycle_name.isin(cohort_names)
        ].copy()
    args.quality_filtered = True
    if not cohort.preparation_heat.eq("zero").all():
        raise ValueError("relative COP requires zero preparation heat")
    boundaries = pd.read_csv(args.decision_run / "recovery_boundaries.csv")
    if current_classification:
        boundaries = reliable_rgb_cohort(args.dataset, source=boundaries)
    cohort = cohort.merge(
        boundaries[["cycle_name", "heating_start"]], on="cycle_name", validate="one_to_one"
    )
    events = pd.read_csv(args.event_run / "defrost_events.csv")
    events = events.loc[events.cycle_name.isin(valid_names)].copy()
    settings = {
        "task": args.task,
        "model_name": (
            {
                "cop-classification": "R-COP32-BCE",
                "cop-classification-regression": "R-COP32-BCE-MSE",
                "dinov2-binary": "DINOv2-MLP-Binary",
                "d32": "D32-COP",
                "r-cop32": "R-COP32",
                "pinn4soh": "PINN4SOH-COP",
                "state-consistency": "PSDC-COP32",
            }[args.regression_architecture]
            + ("-RGB" if args.rgb == "on" else "-Sensor")
            + ("-Peak" if args.peak_weighted else "")
        ),
        "effective_heat_rule": "outlet_at_least_recovery_temperature",
        "cohort": cohort.cycle_name.tolist(),
        "dataset": str(args.dataset.resolve()),
        "decision_run": str(args.decision_run.resolve()),
        "event_run": str(args.event_run.resolve()),
        "rgb_cache": str(args.rgb_cache.resolve()),
        "maximum_epochs": args.maximum_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "features": RGB if binary else feature_columns(args.rgb),
        "architecture": args.regression_architecture,
        "training": "official_adam_minibatch",
        "loss": "cycle_equal_cross_entropy"
        if binary
        else "cycle_equal_peak_mse"
        if args.peak_weighted
        else "official_pair_mse",
        "rgb": args.rgb,
        "rgb_projection": args.rgb_projection,
        "require_rgb_input": args.require_rgb_input,
        "cohort_rule": args.evaluation_cohort,
        "batch_size": args.batch_size,
        "pinn_alpha": args.pinn_alpha,
        "pinn_beta": args.pinn_beta,
        "learning_rates": {"warmup": 5e-4, "base": 1e-3, "final": 1e-4, "F": 1e-3},
        "warmup_epochs": 10,
        "preparation_heat": "zero",
    }
    if current_classification:
        args.output.mkdir(parents=True, exist_ok=True)
        cohort[["cycle_name", "experiment_id"]].to_csv(
            args.output / "cohort.csv", index=False
        )
    if args.rgb_projection:
        settings["model_name"] += f"-Projection{args.rgb_projection}"
    joint = args.regression_architecture in ("cop-classification", "cop-classification-regression")
    if joint:
        settings.update(
            loss="sample_mean_bce" if args.task == "cop-classification" else "sample_mean_bce_plus_global_mse", near_optimal_epsilon=args.near_optimal_epsilon,
            regression_weight=args.regression_weight, time_coordinate="offline_aligned_elapsed_minutes",
            causality="conditional_on_offline_recovery_and_Tref",
            probability_threshold="inner_validation_constrained_selection_per_strategy",
        )
    if args.classification_label == "after-optimum":
        settings.update(classification_label="time_ge_earliest_full_supported_cop_maximum")
        settings["model_name"] += "-AfterOptimum"
    if args.regression_architecture == "state-consistency":
        if args.peak_weighted or args.pinn_beta != 0:
            raise ValueError("state consistency uses unweighted data loss and --pinn-beta 0")
        settings.update(
            physical_state_columns=PHYSICAL_STATE,
            transition_seconds=10,
            loss="official_pair_mse_plus_state_consistency",
            dynamics_inputs="p_current,p_previous,time_current,u_previous",
            time_coordinate="offline_aligned_elapsed_minutes",
            causality="conditional_on_offline_recovery_and_Tref",
        )
    if binary:
        settings.update(
            model_name="DINOv2-MLP-Binary",
            prediction_kind="positive_probability",
            trigger_threshold=0.5,
            confirmation="two_of_three",
            processing_seconds=30,
            cache_only=True,
            label="time_ge_earliest_full_supported_cop_maximum",
        )
    save_settings(args.output / "settings.json", settings)
    for folder in ("base", "ridge", "folds"):
        (args.output / folder).mkdir(exist_ok=True)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(prepare_cycle)(args, row) for row in cohort.itertuples())
    if (args.rgb == "on" or args.require_rgb_input) and not binary:
        complete_local_rgb(args, cohort)
    # All recipes use the same already-frozen experiment rotation, independent of RGB.
    fold_path = args.reference_run / "folds.json"
    folds = (
        json.loads(fold_path.read_text())
        if fold_path.exists()
        else fold_exclusions(cohort.experiment_id.tolist())
    )
    active = set(cohort.experiment_id)
    rotation = fold_exclusions(list(active))
    folds = {test: inner if inner in active else rotation[test] for test, inner in folds.items() if test in active}
    if set(folds) != set(cohort.experiment_id) or any(k == v for k, v in folds.items()):
        raise ValueError("reference folds do not match the fixed research cohort")
    save_settings(args.output / "folds.json", folds)
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        if args.heldout_experiment:
            folds = {args.heldout_experiment: folds[args.heldout_experiment]}
        Parallel()(
            delayed(train_fold)(args, cohort, events, test, inner) for test, inner in folds.items()
        )
    results = []
    for test in sorted(folds):
        path = args.output / "folds" / f"{test}.pkl"
        with path.open("rb") as stream:
            result = pickle.load(stream)  # noqa: S301 - locally generated training checkpoint
        result["metrics"] = metric_function(result["predictions"])
        result["losses"]["heldout_experiment"] = path.stem
        results.append(result)
    predictions = pd.concat([r["predictions"] for r in results], ignore_index=True)
    metrics = pd.concat([r["metrics"] for r in results], ignore_index=True)
    metadata = None
    if args.rgb == "on":
        metadata = DatasetLoader(args.dataset).load_image_metadata()
        metadata = metadata.loc[metadata.camera_role.eq("front")].copy()
        metadata["image_time"] = pd.to_datetime(metadata.image_time)
    metrics["reason"] = ""
    for index, record in metrics.loc[
        metrics.status.ne("scored" if binary else "evaluated")
    ].iterrows():
        reason = record.status
        if record.status == "no_supported_input":
            reason = "missing_sensor_history"
            if metadata is not None:
                images = metadata.loc[metadata.cycle_name.eq(record.cycle_name)]
                reason = (
                    "front_metadata_missing" if images.empty else "no_supported_rgb_sensor_input"
                )
        metrics.loc[index, "reason"] = reason
    pd.concat(
        [
            r["inner_metrics"].assign(outer_experiment=test)
            for test, r in zip(sorted(folds), results, strict=True)
        ]
    ).to_csv(args.output / "inner_metrics.csv", index=False)
    setpoints = {
        name: pd.read_parquet(
            args.output / "base" / f"{name}.parquet", columns=["water_temperature_setpoint"]
        ).water_temperature_setpoint.median()
        for name in predictions.cycle_name.unique()
    }
    predictions["water_temperature_setpoint"] = predictions.cycle_name.map(setpoints)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    pd.concat([r["losses"] for r in results]).to_csv(args.output / "losses.csv", index=False)
    if args.regression_architecture == "state-consistency":
        pd.DataFrame(
            [
                dict(heldout_experiment=test, stage=stage, **result[stage + "_dynamics"])
                for test, result in zip(sorted(folds), results, strict=True)
                for stage in ("inner", "outer")
            ]
        ).to_csv(args.output / "state_diagnostics.csv", index=False)
    if binary:
        from plots.pareto_learning import render_online_cop

        metrics["model"] = settings["model_name"]
        metrics.to_csv(args.output / "cycle_metrics.csv", index=False)
        trace = predictions.loc[processing_rows(predictions)].assign(
            model=settings["model_name"], score=lambda f: f.prediction, threshold=0.5
        )
        trace.to_parquet(args.output / "online_trace.parquet", index=False)
        summary = render_online_cop(metrics, args.figure_output, traces=trace)
        summary.to_csv(args.output / "metrics.csv", index=False)
        metrics.groupby(["experiment_id", "status"]).size().rename("cycles").to_csv(
            args.output / "coverage.csv"
        )
    elif args.task != "cop-classification":
        numeric = [
            "mse",
            "rmse",
            "mae",
            "relative_cop_loss",
            "input_coverage_loss",
            "network_cop_loss",
            "within_1pct",
            "within_2pct",
            "within_5pct",
            "time_error_minutes",
            "early_minutes",
            "late_minutes",
            "absolute_time_error_minutes",
            "chosen_vs_rb_pct",
            *[
                f"near_{p}pct_{direction}_minutes"
                for p in (1, 2, 5)
                for direction in ("early", "late")
            ],
        ]
        metrics = metrics.reindex(columns=list(dict.fromkeys([*metrics.columns, *numeric])))
        summary = metrics.groupby("experiment_id")[numeric].mean(numeric_only=False)
        summary.loc["all_cycles"] = metrics[numeric].mean()
        summary.to_csv(args.output / "metrics.csv")
        metrics.to_csv(args.output / "cycle_metrics.csv", index=False)
        pd.concat(
            [
                r["checkpoint"]["pair_coverage"].assign(heldout_experiment=test)
                for test, r in zip(sorted(folds), results, strict=True)
            ]
        ).to_csv(args.output / "pair_coverage.csv")
        metrics.groupby(["experiment_id", "status"]).size().rename("cycles").to_csv(
            args.output / "coverage.csv"
        )
        render_relative_cop(predictions, metrics, args.figure_output)
    if joint:
        from plots.pareto_learning import render_online_cop

        online = joint_trigger_metrics(predictions)
        online["model"] = settings["model_name"] + "/" + online.strategy
        online.to_csv(args.output / "online_cycle_metrics.csv", index=False)
        traces = []
        for strategy in online.strategy.unique():
            trace = predictions.loc[processing_rows(predictions)].assign(
                model=settings["model_name"] + "/" + strategy,
                score=lambda f: f.probability, threshold=lambda f: f[f"threshold_{strategy}"],
            )
            traces.append(trace)
        online_summary = render_online_cop(online, args.figure_output / "triggers", traces=pd.concat(traces))
        if args.task == "cop-classification":
            from plots.pareto_learning import render_cop_reference

            render_cop_reference(predictions, args.figure_output)
            metrics = online
            summary = online_summary
            metrics.to_csv(args.output / "cycle_metrics.csv", index=False)
            summary.to_csv(args.output / "metrics.csv", index=False)
        pd.concat([r["calibration"].assign(outer_experiment=test) for test, r in zip(sorted(folds), results, strict=True)]).to_csv(
            args.output / "threshold_calibration.csv", index=False
        )
        pd.concat([r["inner_predictions"].assign(outer_experiment=test) for test, r in zip(sorted(folds), results, strict=True)]).to_parquet(
            args.output / "inner_predictions.parquet", index=False
        )
    if not args.heldout_experiment:
        final_path = args.output / "checkpoint.pkl"
        if not final_path.exists():
            rows, models = build_fold_rows(args, cohort, events, set())
            epoch = int(np.ceil(np.median([r["checkpoint"]["selected_epoch"] for r in results if r["checkpoint"]["selected_epoch"] > 0])))
            final = fit_regression(
                rows,
                None,
                epochs=epoch,
                patience=args.patience,
                seed=args.seed,
                **training_options(args),
            )
            if joint:
                final["thresholds"] = {}
                for strategy in results[0]["checkpoint"]["thresholds"]:
                    values = [r["checkpoint"]["thresholds"][strategy]["threshold"] for r in results]
                    values = sorted(v for v in values if not np.isnan(v))
                    # Extended-real median preserves the explicit never-trigger option.
                    threshold = (values[len(values) // 2] if len(values) % 2 else
                                 (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2) if values else np.nan
                    final["thresholds"][strategy] = dict(threshold=threshold, status=(
                        "no_valid_fold_threshold" if not values else "never_trigger" if np.isinf(threshold) else "calibrated"
                    ))
            final["ridge_model"] = ridge_parameters(events, set())
            final["label_ridge_exclusions"] = sorted(models)
            with final_path.open("wb") as stream:
                pickle.dump(final, stream)
    if args.runs and not binary:
        from plots.pareto_learning import render_relative_cop_comparison

        render_relative_cop_comparison(args.runs, args.figure_output)
    print(metrics.status.value_counts().to_string(), flush=True)
    print((summary if binary or args.task == "cop-classification" else summary.loc["all_cycles"]).to_string(), flush=True)


def run_suite(args):
    """Named recipes use the same runner; selection reads internal validation only."""
    from plots.pareto_learning import render_relative_cop_comparison

    roots, validation = [], []
    for architecture, rgb in (
        ("r-cop32", "off"),
        ("pinn4soh", "off"),
        ("pinn4soh", "on"),
        ("r-cop32", "on"),
    ):
        recipe = copy.copy(args)
        recipe.regression_architecture, recipe.rgb = architecture, rgb
        recipe.peak_weighted, recipe.runs = False, None
        recipe.output = args.output / f"{architecture}_{rgb}"
        recipe.figure_output = args.figure_output / recipe.output.name
        run(recipe)
        roots.append(recipe.output)
        table = pd.read_csv(recipe.output / "inner_metrics.csv")
        validation.append(
            {
                "architecture": architecture,
                "rgb": rgb,
                "relative_cop_loss": table.relative_cop_loss.mean(),
                "within_2pct": table.within_2pct.mean(),
                "mse": table.validation_mse.mean(),
                "evaluated_cycles": int(table.status.eq("evaluated").sum()),
            }
        )
    ranking = pd.DataFrame(validation).sort_values(
        ["relative_cop_loss", "within_2pct", "mse", "rgb", "architecture"],
        ascending=[True, False, True, True, False],
    )
    ranking.to_csv(args.output / "internal_model_selection.csv", index=False)
    if not args.heldout_experiment:
        winner = ranking.iloc[0]
        recipe = copy.copy(args)
        recipe.regression_architecture, recipe.rgb = winner.architecture, winner.rgb
        recipe.peak_weighted, recipe.runs = True, None
        recipe.output = args.output / f"{winner.architecture}_{winner.rgb}_peak"
        recipe.figure_output = args.figure_output / recipe.output.name
        run(recipe)
        roots.append(recipe.output)
    render_relative_cop_comparison(roots, args.figure_output)


def reliable_cop_mask(curve):
    mask = np.isfinite(curve.cycle_cop)
    for column in (
        "cycle_cop_measurements_valid", "cycle_cop_physically_valid",
        "pre_defrost_feature_window_valid",
        "defrost_event_electricity_prediction_available",
        "defrost_event_net_heat_prediction_available",
    ):
        mask &= curve[column].fillna(False) if column in curve else False
    return mask


def online_trigger_metrics(curve, stream, threshold, strategy="two_of_three", *, allow_model_extrapolation=False):
    """Score executed confirmations, never the offline maximum or a future fallback."""
    from plots.image_models import two_of_three_trigger

    positive = stream.get("trigger_positive", stream.score.ge(threshold)).fillna(False)
    positive = positive & stream.score.notna()
    if strategy == "first_positive":
        trigger = stream.loc[positive, "candidate_defrost_time"].min()
    else:
        trigger, _ = two_of_three_trigger(stream.candidate_defrost_time, positive.astype(float))
    record = dict(
        trigger_time=trigger,
        status="no_trigger",
        relative_cop_loss=np.nan,
        trigger_cop=np.nan,
        reference_time=pd.NaT,
        reference_cop=np.nan,
        time_error_minutes=np.nan,
        absolute_time_error_minutes=np.nan,
        outside_reference_support=False,
        extrapolated_reference_gap=np.nan,
        processing_count=len(stream),
        available_count=int(stream.score.notna().sum()),
    )
    for percent in (1, 2, 5):
        record[f"within_{percent}pct"] = np.nan
    accounting = curve.get("heating_accounting_start", curve.get("stable_heating_start"))
    accounting = (
        pd.to_datetime(accounting, errors="coerce").dropna().min()
        if accounting is not None else pd.NaT
    )
    if pd.notna(trigger) and pd.notna(accounting) and trigger < accounting:
        record["status"] = "before_reference_accounting_start"
        return [record]
    supported = curve.loc[curve.cycle_cop_eligible & np.isfinite(curve.cycle_cop)]
    if supported.empty or supported.cycle_cop.max() <= 0:
        if pd.notna(trigger) and allow_model_extrapolation:
            point = curve.loc[
                reliable_cop_mask(curve) & curve.candidate_defrost_time.eq(trigger)
            ]
            if not point.empty:
                record.update(
                    status="scored", trigger_cop=float(point.cycle_cop.iloc[0]),
                    outside_reference_support=True,
                )
                return [record]
        record["status"] = "no_supported_reference"
        return [record]
    best = (
        supported.sort_values("candidate_defrost_time")
        .loc[lambda f: f.cycle_cop.eq(f.cycle_cop.max())]
        .iloc[0]
    )
    record.update(reference_time=best.candidate_defrost_time, reference_cop=float(best.cycle_cop))
    if not stream.score.notna().any():
        record["status"] = "no_available_input"
    if pd.notna(trigger):
        error = (trigger - best.candidate_defrost_time).total_seconds() / 60
        record.update(
            time_error_minutes=error,
            absolute_time_error_minutes=abs(error),
            status="trigger_outside_reference_support",
        )
        scoreable = curve.cycle_cop_eligible.copy()
        if allow_model_extrapolation:
            scoreable |= reliable_cop_mask(curve)
        point = curve.loc[scoreable & curve.candidate_defrost_time.eq(trigger)]
        record["outside_reference_support"] = not supported.candidate_defrost_time.eq(trigger).any()
        if allow_model_extrapolation and point.empty:
            record["status"] = "trigger_cop_unavailable"
        if not point.empty:
            cop = float(point.cycle_cop.iloc[0])
            regret = 1 - cop / float(best.cycle_cop)
            record.update(status="scored", trigger_cop=cop)
            if record["outside_reference_support"]:
                record["extrapolated_reference_gap"] = regret
            else:
                record["relative_cop_loss"] = regret
                for percent in (1, 2, 5):
                    record[f"within_{percent}pct"] = float(regret <= percent / 100 + 1e-12)
    return [record]


def calibrate_thresholds(rows, strategies=("first_positive", "two_of_three")):
    """Only inner-validation rows enter this selector; missing slots stay on the clock."""
    curves = list(rows.sort_values("candidate_defrost_time").groupby("cycle_name", sort=True))
    observable = rows.loc[processing_rows(rows) & rows.probability.notna(), "binary_target"].eq(1).any()
    selected, evidence = {}, []
    for strategy in strategies:
        if not observable:
            selected[strategy] = dict(threshold=np.nan, status="no_observable_positive")
            continue
        for threshold in [*np.arange(1, 101) / 100, np.inf]:
            records = []
            for _, curve in curves:
                stream = curve.loc[processing_rows(curve)].assign(score=lambda f: f.probability)
                records.extend(online_trigger_metrics(curve, stream, threshold, strategy))
            metrics = pd.DataFrame(records)
            early = metrics.time_error_minutes.lt(0) & metrics.relative_cop_loss.gt(.05 + 1e-12)
            evidence.append(dict(
                strategy=strategy, threshold=threshold, cohort=len(curves),
                severe_early_rate=float(early.mean()),
                hit_rate=float(metrics.within_1pct.fillna(0).mean()),
                mean_regret=float(metrics.relative_cop_loss.mean()),
            ))
        feasible = pd.DataFrame(evidence).loc[lambda f: f.strategy.eq(strategy) & f.severe_early_rate.le(.05)]
        best = feasible.sort_values(
            ["hit_rate", "severe_early_rate", "mean_regret", "threshold"],
            ascending=[False, True, True, False], na_position="last",
        ).iloc[0]
        selected[strategy] = dict(threshold=float(best.threshold), status="never_trigger" if np.isinf(best.threshold) else "calibrated")
    return selected, pd.DataFrame(evidence)


def joint_trigger_metrics(rows):
    records = []
    for name, curve in rows.groupby("cycle_name", sort=True):
        curve = curve.sort_values("candidate_defrost_time")
        stream = curve.loc[processing_rows(curve)].assign(score=lambda f: f.probability)
        rb = curve.loc[curve.candidate_defrost_time.eq(curve.t_RB) & curve.cycle_cop_eligible, "cycle_cop"]
        for strategy in (s.removeprefix("threshold_") for s in curve.columns if s.startswith("threshold_")):
            threshold = curve[f"threshold_{strategy}"].iloc[0]
            record = online_trigger_metrics(curve, stream, threshold, strategy)[0]
            if pd.isna(threshold):
                record["status"] = "threshold_uncalibrated"
            valid = curve.loc[curve.binary_target.notna() & curve.probability.notna()]
            error = valid.prediction - valid.target
            regression_count = int(error.notna().sum())
            if pd.isna(threshold):
                valid = valid.iloc[:0]
            positive = valid.binary_target.eq(1)
            predicted = valid.probability.ge(threshold)
            tp = int((positive & predicted).sum())
            fp = int((~positive & predicted).sum())
            fn = int((positive & ~predicted).sum())
            tn = int((~positive & ~predicted).sum())
            record.update(
                cycle_name=name, experiment_id=curve.experiment_id.iloc[0], strategy=strategy,
                threshold=threshold, calibration_status=curve[f"calibration_{strategy}"].iloc[0],
                baseline_rb_cop=float(rb.iloc[0]) if len(rb) else np.nan,
                tp=tp, fp=fp, fn=fn, tn=tn, regression_count=regression_count,
                balanced_accuracy=.5 * ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) if len(valid) else np.nan,
                macro_f1=.5 * ((2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0) + (2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0)) if len(valid) else np.nan,
                fnr=fn / (fn + tp) if fn + tp else np.nan,
                fpr=fp / (fp + tn) if fp + tn else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                mse=float((error ** 2).mean()),
                rmse=float(np.sqrt((error ** 2).mean())), mae=float(error.abs().mean()),
            )
            record["cop_gain_vs_rb_pct"] = (
                100 * (record["trigger_cop"] / record["baseline_rb_cop"] - 1)
                if record["baseline_rb_cop"] > 0 and record["status"] == "scored" else np.nan
            )
            records.append(record)
    return pd.DataFrame(records)


def binary_cycle_metrics(rows):
    """Classification diagnostics and the same causal trigger scorer as other models."""
    records = []
    for name, cycle in rows.groupby("cycle_name", sort=True):
        cycle = cycle.sort_values("candidate_defrost_time")
        stream = cycle.loc[processing_rows(cycle)].assign(score=lambda f: f.prediction)
        record = online_trigger_metrics(cycle, stream, 0.5)[0]
        valid = stream.loc[stream.binary_target.notna() & stream.prediction.notna()]
        record.update(
            cycle_name=name,
            experiment_id=cycle.experiment_id.iloc[0],
            balanced_accuracy=np.nan,
            macro_f1=np.nan,
            classified_count=len(valid),
        )
        if not valid.empty:
            target = valid.binary_target.astype(int)
            predicted = valid.prediction.ge(0.5).astype(int)
            record.update(
                balanced_accuracy=float(
                    recall_score(target, predicted, labels=[0, 1], average="macro", zero_division=0)
                ),
                macro_f1=float(
                    f1_score(target, predicted, labels=[0, 1], average="macro", zero_division=0)
                ),
            )
        records.append(record)
    return pd.DataFrame(records)


def replay_online_cycle(args, name, predictions, names):
    from defrost_decision.baselines import rule_based

    loader = DatasetLoader(args.dataset)
    reference_curve = predictions[0]
    rows, traces, rb_checks = [], [], []
    first, end = reference_curve.candidate_defrost_time.agg(["min", "max"])
    grid = pd.date_range(first, end, freq=f"{args.processing_seconds}s")
    base = pd.read_parquet(args.runs[0] / "base" / f"{name}.parquet")
    raw = loader.load_cycle_original(name, columns=list(rule_based.RAW_COLUMNS))
    trace = rule_based.first_trigger(
        raw, pd.Timestamp(base.heating_start.iloc[0]), end, return_trace=True
    )
    original = rule_based.first_trigger(raw, pd.Timestamp(base.heating_start.iloc[0]), end)
    recorded = pd.to_datetime(reference_curve.t_RB.iloc[0])
    rb_checks.append(dict(cycle_name=name, recorded_rb=recorded, recomputed_rb=original["t_RB"]))
    rb = trace.reindex(grid)
    streams = []
    for model_name, full in zip(names, predictions, strict=True):
        selected = (
            full.loc[full.cycle_name.eq(name)].set_index("candidate_defrost_time").reindex(grid)
        )
        score = selected.prediction.where(selected.input_available)
        threshold = 0.5 if "binary_target" in selected else args.trigger_threshold
        streams.append((model_name, score, threshold))
    streams.append(("RB", rb.triggered.astype(float), 0.5))
    for model_name, score, threshold in streams:
        stream = pd.DataFrame(
            dict(
                candidate_defrost_time=grid,
                score=score.to_numpy(),
                trigger_positive=score.ge(threshold).to_numpy(),
            )
        )
        metadata = dict(
            cycle_name=name,
            experiment_id=reference_curve.experiment_id.iloc[0],
            model=model_name,
        )
        traces.append(
            stream.assign(
                **metadata,
                threshold=threshold,
                target=reference_curve.set_index("candidate_defrost_time")
                .target.reindex(grid)
                .to_numpy(),
            )
        )
        rows.extend(
            dict(**metadata, **result)
            for result in online_trigger_metrics(reference_curve, stream, threshold)
        )
    return rows, traces, rb_checks


FROZEN_POLICY_NAMES = {
    "delta_rgb": "Delta RGB",
    "history_sensor": "History Sensor",
    "legacy_after_optimum_sensor": "Legacy Sensor After-Optimum",
    "legacy_after_optimum_rgb": "Legacy RGB After-Optimum",
    "chen_dinov2_binary": "Chen DINOv2 Binary",
    "rb": "Original RB",
}


def _common_policy_reference(current_base, legacy_base, parameters):
    """Apply one Ridge model to the old-10s and new-30s union clock."""
    current = pd.read_parquet(current_base)
    legacy = pd.read_parquet(legacy_base)
    for column in (
        "cycle_name", "experiment_id", "heating_start", "stable_heating_start",
        "heating_accounting_start", "t_RB",
    ):
        left = pd.to_datetime(current[column].iloc[0]) if column.endswith("start") or column == "t_RB" else str(current[column].iloc[0])
        right = pd.to_datetime(legacy[column].iloc[0]) if column.endswith("start") or column == "t_RB" else str(legacy[column].iloc[0])
        if not (pd.isna(left) and pd.isna(right)) and left != right:
            raise ValueError(f"common reference boundary differs: {column}")
    missing = ~legacy.candidate_defrost_time.isin(current.candidate_defrost_time)
    union = pd.concat([current, legacy.loc[missing]], ignore_index=True, sort=False)
    union = union.sort_values("candidate_defrost_time").reset_index(drop=True)
    return apply_reference(union, parameters, "off", include_history=False)


def _surface_peak(curve, mask):
    eligible = curve.loc[mask & np.isfinite(curve.cycle_cop)]
    return float(eligible.cycle_cop.max()) if len(eligible) else np.nan


def _percent_change(value, baseline):
    valid = np.isfinite(value) & np.isfinite(baseline) & baseline.gt(0)
    return (100 * (value / baseline.where(valid) - 1)).where(valid)


def compare_frozen_policies(args):
    """Replay five policies against one fixed RB, one peak, and two point scopes."""
    if not args.runs or len(args.runs) != 4:
        raise ValueError(
            "compare-frozen-policies requires --runs FROZEN_RETROSPECTIVE "
            "LEGACY_SENSOR LEGACY_RGB CHEN_BINARY"
        )
    frozen_root, *legacy_runs = map(Path, args.runs)
    frozen_settings = json.loads((frozen_root / "settings.json").read_text())
    if frozen_settings.get("action") != "evaluate-frozen-development":
        raise ValueError("first run must be the frozen development retrospective")
    recipe_root = Path(frozen_settings["recipes"]["delta_rgb"])
    recipe_settings = json.loads((recipe_root / "settings.json").read_text())
    dataset = Path(recipe_settings["dataset"])
    audit_base = Path(recipe_settings["audit_data"]) / "base"
    current = pd.read_parquet(frozen_root / "predictions.parquet")
    selected = pd.read_csv(frozen_root / "inner_selection.csv")[[
        "recipe_id", "seed", "outer_experiment", "threshold"
    ]]
    current = current.merge(
        selected, on=["recipe_id", "seed", "outer_experiment"],
        validate="many_to_one",
    )
    source_cycles = set(current.cycle_name.astype(str))
    catalog = DatasetLoader(dataset).list_cycles(statuses={"valid"})
    rgb_valid = set(catalog.loc[catalog.rgb_valid.eq(True), "cycle_name"].astype(str))
    cohort = sorted(source_cycles & rgb_valid)
    excluded = sorted(source_cycles - set(cohort))
    if not cohort:
        raise ValueError("frozen and RGB-valid cohorts do not overlap")

    expected_legacy = {
        "R-COP32-BCE-Sensor-AfterOptimum": "legacy_after_optimum_sensor",
        "R-COP32-BCE-RGB-AfterOptimum": "legacy_after_optimum_rgb",
        "DINOv2-MLP-Binary": "chen_dinov2_binary",
    }
    legacy = {}
    for run in legacy_runs:
        configuration = json.loads((run / "settings.json").read_text())
        model_name = configuration.get("model_name")
        if model_name not in expected_legacy:
            raise ValueError(f"unexpected legacy policy: {model_name}")
        policy_id = expected_legacy[model_name]
        table = pd.read_parquet(run / "predictions.parquet")
        legacy[policy_id] = (run, table.loc[table.cycle_name.isin(cohort)].copy())
    if set(legacy) != set(expected_legacy.values()):
        raise ValueError("legacy sensor, RGB, and Chen policies are all required")

    experiment_by_cycle = current.groupby("cycle_name").experiment_id.first()
    rows = []
    for name in cohort:
        experiment = str(experiment_by_cycle[name])
        parameter_path = frozen_root / "ridge" / f"{experiment}.json"
        parameters = json.loads(parameter_path.read_text())
        if experiment in parameters["training_experiment_ids"]:
            raise AssertionError(f"{name}: outer experiment entered the Ridge reference")
        reference = _common_policy_reference(
            recipe_root / "base" / f"{name}.parquet",
            audit_base / f"{name}.parquet",
            parameters,
        )
        formal_peak = _surface_peak(reference, reference.cycle_cop_eligible.fillna(False))
        # The optimum is frozen on supported candidates; stress scoring never reselects it OOD.
        sensitivity_peak = formal_peak

        streams = []
        for (recipe_id, seed), table in current.loc[
            current.cycle_name.eq(name)
        ].groupby(["recipe_id", "seed"], sort=False):
            table = table.sort_values("candidate_defrost_time")
            threshold = float(table.threshold.iloc[0])
            streams.append((
                recipe_id, "current_frozen", int(seed), threshold, "two_of_three",
                table[["candidate_defrost_time"]].assign(score=table.probability.to_numpy()),
            ))
        for policy_id, (_, table) in legacy.items():
            cycle = table.loc[table.cycle_name.eq(name)].sort_values("candidate_defrost_time")
            if cycle.empty:
                streams.append((
                    policy_id, "legacy_frozen", pd.NA, .5, "two_of_three",
                    reference[["candidate_defrost_time"]].iloc[:1].assign(score=np.nan),
                    "no_frozen_prediction",
                ))
                continue
            stream = cycle.loc[processing_rows(cycle)].copy()
            score = stream["probability"] if "probability" in stream else stream["prediction"]
            score = score.where(stream.input_available)
            streams.append((
                policy_id, "legacy_frozen", pd.NA, .5, "two_of_three",
                stream[["candidate_defrost_time"]].assign(score=score.to_numpy()),
                None,
            ))
        rb_time = pd.to_datetime(reference.t_RB.iloc[0], errors="coerce")
        rb_stream = pd.DataFrame({
            "candidate_defrost_time": [rb_time], "score": [1.],
        }).dropna(subset=["candidate_defrost_time"])
        streams.append(("rb", "baseline", pd.NA, .5, "recorded", rb_stream, None))

        # Current policies always have a frozen prediction for every cohort cycle.
        streams = [(*item, None) if len(item) == 6 else item for item in streams]
        for policy_id, family, seed, threshold, confirmation, stream, forced_status in streams:
            result = online_trigger_metrics(
                reference, stream, threshold,
                "first_positive" if confirmation == "recorded" else "two_of_three",
                allow_model_extrapolation=True,
            )[0]
            result.update(
                policy_id=policy_id, policy_name=FROZEN_POLICY_NAMES[policy_id],
                policy_family=family, seed=seed, cycle_name=name,
                experiment_id=experiment, threshold=threshold,
                confirmation=confirmation, formal_reference_cop=formal_peak,
                sensitivity_reference_cop=sensitivity_peak,
            )
            if forced_status:
                result.update(
                    status=forced_status, trigger_time=pd.NaT, trigger_cop=np.nan,
                    processing_count=0, available_count=0,
                )
            result["estimate_scope"] = (
                "unscored" if not np.isfinite(result["trigger_cop"])
                else "extrapolated" if result["outside_reference_support"]
                else "supported"
            )
            rows.append(result)

    metrics = pd.DataFrame(rows)
    rb = metrics.loc[metrics.policy_id.eq("rb"), [
        "cycle_name", "trigger_cop", "outside_reference_support"
    ]].rename(columns={
        "trigger_cop": "rb_cop", "outside_reference_support": "rb_outside_reference_support"
    })
    metrics = metrics.merge(rb, on="cycle_name", validate="many_to_one")
    finite_pair = (
        np.isfinite(metrics.trigger_cop) & np.isfinite(metrics.rb_cop)
        & metrics.rb_cop.gt(0)
    )
    formal_pair = finite_pair & ~metrics.outside_reference_support & ~metrics.rb_outside_reference_support
    metrics["cop_gain_vs_rb_pct"] = _percent_change(
        metrics.trigger_cop, metrics.rb_cop
    ).where(formal_pair)
    metrics["sensitivity_cop_gain_vs_rb_pct"] = _percent_change(
        metrics.trigger_cop, metrics.rb_cop
    ).where(finite_pair)
    formal_headroom = _percent_change(metrics.formal_reference_cop, metrics.rb_cop)
    sensitivity_headroom = _percent_change(
        metrics.sensitivity_reference_cop, metrics.rb_cop
    )
    metrics["formal_reference_headroom_vs_rb_pct"] = formal_headroom.where(
        np.isfinite(metrics.formal_reference_cop) & np.isfinite(metrics.rb_cop)
        & metrics.rb_cop.gt(0) & ~metrics.rb_outside_reference_support
    )
    metrics["sensitivity_reference_headroom_vs_rb_pct"] = sensitivity_headroom.where(
        np.isfinite(metrics.sensitivity_reference_cop) & np.isfinite(metrics.rb_cop)
        & metrics.rb_cop.gt(0)
    )
    metrics["remaining_to_formal_reference_vs_trigger_pct"] = (
        100 * (metrics.formal_reference_cop / metrics.trigger_cop - 1)
    ).where(
        np.isfinite(metrics.formal_reference_cop) & np.isfinite(metrics.trigger_cop)
        & metrics.trigger_cop.gt(0) & ~metrics.outside_reference_support
    )
    metrics["sensitivity_remaining_gap_pct_points"] = (
        metrics.sensitivity_reference_headroom_vs_rb_pct
        - metrics.sensitivity_cop_gain_vs_rb_pct
    ).where(metrics.sensitivity_reference_headroom_vs_rb_pct.gt(1.))

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "policy_cycle_metrics.csv", index=False)
    from plots.pareto_learning import render_frozen_policy_cop_comparison
    render_frozen_policy_cop_comparison(metrics, args.output)
    from train_pareto_boundary import save_settings
    save_settings(args.output / "settings.json", {
        "action": "compare-frozen-policies",
        "runs": [str(Path(run).resolve()) for run in args.runs],
        "dataset": str(dataset.resolve()), "cohort": cohort,
        "source_cycle_count": len(source_cycles), "comparison_cycle_count": len(cohort),
        "excluded_non_rgb_valid_cycles": excluded,
        "clock": "saved policy 30-second clocks; missing slots retained",
        "current_thresholds": "frozen outer-specific inner selections for seeds 0,1,2",
        "legacy_thresholds": .5, "baseline": "original recorded RB trigger",
        "reference": "one outer-experiment-excluded Ridge per cycle on old10s union new30s",
        "peak_reference": "one frozen Ridge-domain-supported candidate maximum",
        "formal_point_scope": "Ridge-domain-supported trigger and RB points",
        "sensitivity_point_scope": "reliable trigger and RB point gates with Ridge domain relaxed",
        "interpretation": "retrospective policy-package comparison; no controller qualification or causal architecture claim",
    })
    return metrics


def evaluate_online(args):
    if args.evaluation_cohort == "rgb-valid":
        from dataset_tools.cycle_metadata import update_rgb_validity

        source = args.reference_run / "predictions.parquet"
        quality = update_rgb_validity(args.dataset, pd.read_parquet(source), source=source.resolve())
        args.output.mkdir(parents=True, exist_ok=True)
        quality.to_csv(args.output / "rgb_validity.csv", index=False)
    catalog = (
        reliable_rgb_cohort(args.dataset)
        if args.task in (
            "effective-cop-binary", "cop-classification",
            "cop-classification-regression",
        )
        else DatasetLoader(args.dataset).list_cycles(statuses={"valid"})
    )
    if args.task not in (
        "effective-cop-binary", "cop-classification",
        "cop-classification-regression",
    ) and args.evaluation_cohort == "rgb-valid":
        catalog = catalog.loc[catalog.rgb_valid.eq(True)]
    valid_names = set(catalog.cycle_name)
    if args.task in ("cop-classification", "cop-classification-regression"):
        from plots.pareto_learning import render_online_cop

        tables, reference = [], None
        for run in args.runs:
            predictions = pd.read_parquet(run / "predictions.parquet")
            predictions = predictions.loc[predictions.cycle_name.isin(valid_names)].reset_index(drop=True)
            if args.allow_extrapolation:
                checked = []
                for name, curve in predictions.groupby("cycle_name", sort=False):
                    experiment = curve.experiment_id.iloc[0]
                    parameters = json.loads((run / "ridge" / f"{experiment}.json").read_text())
                    base = pd.read_parquet(run / "base" / f"{name}.parquet")
                    physical = apply_reference(base, parameters, "off")
                    pd.testing.assert_series_equal(curve.reset_index(drop=True).cycle_cop,
                        physical.reset_index(drop=True).cycle_cop, check_names=False)
                    fields = ["cycle_cop_measurements_valid", "cycle_cop_physically_valid",
                        "pre_defrost_feature_window_valid", "defrost_event_electricity_prediction_available",
                        "defrost_event_net_heat_prediction_available"]
                    checked.append(curve.merge(physical[["candidate_defrost_time", *fields]],
                        on="candidate_defrost_time", validate="one_to_one"))
                predictions = pd.concat(checked, ignore_index=True)
            columns = ["cycle_name", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible", "t_RB"]
            if reference is None:
                reference = predictions
            else:
                pd.testing.assert_frame_equal(reference[columns], predictions[columns])
            if args.evaluation_threshold is not None:
                records = []
                model_name = json.loads((run / "settings.json").read_text())["model_name"]
                for name, curve in predictions.groupby("cycle_name", sort=True):
                    curve = curve.sort_values("candidate_defrost_time")
                    stream = curve.loc[processing_rows(curve)].copy()
                    stream["score"] = stream.get("probability", stream.prediction).where(stream.input_available)
                    stream = stream.drop(columns="trigger_positive", errors="ignore")
                    record = online_trigger_metrics(curve, stream, args.evaluation_threshold,
                        allow_model_extrapolation=args.allow_extrapolation)[0]
                    record.update(cycle_name=name, experiment_id=curve.experiment_id.iloc[0],
                                  model=model_name, threshold=args.evaluation_threshold, strategy="two_of_three")
                    records.append(record)
                rows = pd.DataFrame(records)
            else:
                path = run / "online_cycle_metrics.csv"
                rows = pd.read_csv(path if path.exists() else run / "cycle_metrics.csv", parse_dates=["trigger_time", "reference_time"])
                rows = rows.loc[rows.cycle_name.isin(valid_names)].copy()
                if "model" not in rows:
                    rows["model"] = json.loads((run / "settings.json").read_text())["model_name"]
                if args.task == "cop-classification" and "strategy" in rows:
                    rows = rows.loc[rows.strategy.eq("two_of_three")].copy()
            tables.append(rows)
        metrics = pd.concat(tables, ignore_index=True)
        if args.task == "cop-classification":
            baselines = []
            for name, curve in reference.groupby("cycle_name", sort=True):
                # The recorded RB trigger is fixed; do not delay it by a synthetic confirmation.
                trigger = curve.t_RB.iloc[0]
                stream = pd.DataFrame({"candidate_defrost_time": pd.to_datetime([trigger]), "score": [1.]}).dropna(subset=["candidate_defrost_time"])
                record = online_trigger_metrics(curve, stream, .5, "first_positive", allow_model_extrapolation=args.allow_extrapolation)[0]
                if pd.isna(trigger):
                    record["status"] = "no_trigger"
                record.update(cycle_name=name, experiment_id=curve.experiment_id.iloc[0], model="RB", strategy="recorded")
                baselines.append(record)
            rb = pd.DataFrame(baselines)
            metrics = pd.concat([metrics, rb], ignore_index=True).drop(columns=["baseline_rb_cop", "cop_gain_vs_rb_pct"], errors="ignore")
            metrics = metrics.merge(rb[["cycle_name", "trigger_cop"]].rename(columns={"trigger_cop": "baseline_rb_cop"}), on="cycle_name", validate="many_to_one")
            metrics["cop_gain_vs_rb_pct"] = (100 * (metrics.trigger_cop / metrics.baseline_rb_cop - 1)).where(metrics.status.eq("scored") & metrics.baseline_rb_cop.gt(0))
        missing = []
        for model, group in metrics.groupby("model", sort=False):
            for name in sorted(valid_names - set(group.cycle_name)):
                missing.append(dict(model=model, cycle_name=name, status="no_frozen_prediction",
                    trigger_time=pd.NaT, reference_time=pd.NaT,
                    outside_reference_support=False))
        if missing:
            metrics = pd.concat([metrics, pd.DataFrame(missing)], ignore_index=True)
        from train_pareto_boundary import save_settings

        save_settings(args.output / "settings.json", dict(
            runs=[str(run.resolve()) for run in args.runs],
            dataset=str(args.dataset.resolve()), cohort=sorted(valid_names), cohort_rule=args.evaluation_cohort,
            evaluation="frozen predictions; catalog valid subset",
            evaluation_threshold=args.evaluation_threshold, allow_extrapolation=args.allow_extrapolation,
            baseline="original_recorded_rb_trigger"))
        render_online_cop(metrics, args.figure_output)
        return
    """Replay saved outer-fold predictions and unchanged RB conditions on one clock."""
    from plots.pareto_learning import render_online_cop
    from train_pareto_boundary import save_settings

    if not args.runs or args.processing_seconds <= 0 or args.processing_seconds % 10:
        raise ValueError("--runs is required; processing seconds must be a positive multiple of 10")
    settings = dict(
        runs=[str(p.resolve()) for p in args.runs],
        threshold=args.trigger_threshold,
        processing_seconds=args.processing_seconds,
        clock="first_candidate_then_fixed_interval",
        reference="full_supported_curve",
        missing_slots="invalid_not_positive",
        confirmation="two_of_three",
        boundaries="fixed_offline_recovery",
        baseline="original_recorded_rb_trigger",
    )
    save_settings(args.output / "settings.json", settings)
    predictions, names, reference = [], [], None
    for run in args.runs:
        configuration = json.loads((run / "settings.json").read_text())
        names.append(configuration["model_name"])
        curve = (
            pd.read_parquet(run / "predictions.parquet")
            .sort_values(["cycle_name", "candidate_defrost_time"])
            .reset_index(drop=True)
        )
        curve = curve.loc[curve.cycle_name.isin(valid_names)].reset_index(drop=True)
        shared = curve[
            ["cycle_name", "candidate_defrost_time", "target", "cycle_cop", "cycle_cop_eligible"]
        ]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, shared)
        reference = shared
        predictions.append(curve)
    grouped = [dict(tuple(frame.groupby("cycle_name", sort=True))) for frame in predictions]
    with parallel_config(backend="loky", n_jobs=args.n_jobs, inner_max_num_threads=1):
        results = Parallel()(
            delayed(replay_online_cycle)(args, name, [groups[name] for groups in grouped], names)
            for name in grouped[0]
        )
    rows = [row for result in results for row in result[0]]
    traces = [trace for result in results for trace in result[1]]
    rb_checks = [check for result in results for check in result[2]]
    metrics = pd.DataFrame(rows)
    baseline = (
        predictions[0].loc[predictions[0].candidate_defrost_time.eq(predictions[0].t_RB)].copy()
    )
    baseline["baseline_rb_cop"] = baseline.cycle_cop.where(baseline.cycle_cop_eligible)
    metrics = metrics.merge(
        baseline[["cycle_name", "baseline_rb_cop"]],
        on="cycle_name",
        how="left",
        validate="many_to_one",
    )
    metrics["cop_gain_vs_rb_pct"] = (
        100 * (metrics.trigger_cop / metrics.baseline_rb_cop - 1)
    ).where(metrics.status.eq("scored") & metrics.baseline_rb_cop.gt(0))
    metrics.to_csv(args.output / "online_cycle_metrics.csv", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(
        args.output / "online_trace.parquet", index=False
    )
    pd.DataFrame(rb_checks).to_csv(args.output / "rb_replay_check.csv", index=False)
    summary = render_online_cop(
        metrics,
        args.figure_output,
        args.processing_seconds,
        traces=pd.concat(traces, ignore_index=True),
    )
    summary.to_csv(args.output / "online_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
