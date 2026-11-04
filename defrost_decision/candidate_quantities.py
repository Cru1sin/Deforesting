"""Current-cycle COP: processed heating, measured electricity, predicted preparation/defrost."""

from __future__ import annotations

import numpy as np
import pandas as pd

from defrost_decision.performance_objectives import select_cycle_optimum
from defrost_event_models.training_data import (
    candidate_integral_table,
    extract_pre_defrost_features,
    timestamp,
)


def measure_candidate_quantities(
    frame,
    candidate_times,
    heating_start,
):
    """One prepared-measurement calculation for Dataset replay and cycle prefixes."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    candidates = [pd.Timestamp(value) for value in candidate_times["candidate_defrost_time"]]
    accounting_start = pd.Timestamp(candidate_times["heating_accounting_start"].iloc[0])

    valid_times = [value for value in candidates if value >= accounting_start]
    positions = [index for index, value in enumerate(candidates) if value >= accounting_start]

    if valid_times:
        heat = candidate_integral_table(
            frame, accounting_start, valid_times, "heating_capacity_processed"
        )
        heat.index = positions
        heat = heat.reindex(range(len(candidates)))
        heat["valid"] = heat["valid"].eq(True)
    else:
        heat = pd.DataFrame(
            {"energy": [float("nan")] * len(candidates), "valid": [False] * len(candidates)}
        )
    result = candidate_times.copy()
    result["effective_heat_rule"] = "without_trunc"
    result["reference_outlet_temperature"] = None
    result["pre_defrost_heat_kwh"] = heat["energy"].to_numpy()
    result["pre_defrost_heat_measurement_valid"] = heat["valid"].to_numpy()
    return pd.concat(
        [
            result.reset_index(drop=True),
            extract_pre_defrost_features(
                frame,
                candidates,
                heating_start,
            ),
        ],
        axis=1,
    )


def apply_current_cycle_cost(result, frame, heating_start, phase_models):
    """Measured current recovery/heating electricity plus predicted preparation/defrost."""
    from defrost_event_models.event_electricity_ridge import predict_with_model_parameters

    result = result.copy()
    measured = candidate_integral_table(
        frame, heating_start, result.candidate_defrost_time.tolist(), "power_total"
    )
    result["pre_defrost_electricity_kwh"] = measured.energy.to_numpy()
    result["pre_defrost_electricity_measurement_valid"] = measured.valid.to_numpy()
    result["measured_cycle_start"] = heating_start
    channels = sorted(
        {feature for model in phase_models.values() for feature in model["feature_order"]}
    )
    result["pre_defrost_feature_window_valid"] = result[channels].notna().all(axis=1) & result[
        [c + "_valid_second_count" for c in channels]
    ].ge(48).all(axis=1)
    result["defrost_event_electricity_kwh"] = 0.0
    result["defrost_event_electricity_in_training_domain"] = True
    nonnegative = pd.Series(True, index=result.index)
    for target, parameters in phase_models.items():
        predicted = predict_with_model_parameters(parameters, result)
        result["predicted_" + target] = predicted.prediction.to_numpy()
        result["defrost_event_electricity_kwh"] += predicted.prediction.to_numpy()
        result["defrost_event_electricity_in_training_domain"] &= predicted.support_distance.le(
            predicted.support_threshold
        ).to_numpy()
        nonnegative &= predicted.prediction.ge(0).to_numpy()
    result["defrost_event_electricity_prediction_available"] = (
        np.isfinite(result.defrost_event_electricity_kwh)
        & result.pre_defrost_feature_window_valid
        & nonnegative
    )
    result["defrost_event_scope"] = "preparation_defrost"
    result["defrost_event_breakdown"] = (
        "joint_preparation_defrost_ridge" if len(phase_models) == 1 else "two_phase_ridge"
    )
    result["electricity_accounting"] = "measured_current_recovery_plus_heating"
    return result


def effective_candidate_cop(
    frame,
    candidates,
    recovery_start,
    heating_start,
    models,
    experiment_id,
    *,
    prediction_mode="cross-fitted",
):
    """Estimate an action from strictly pre-action observations, without selecting it."""

    from .performance_objectives import calculate_cycle_cop

    if models.get("cop_definition") != "refrigerant_effective_heat":
        raise ValueError("effective COP requires refitted effective-heat event models")
    processing = frame.attrs.get("measurement_processing", "dataset_causal_60s")
    if models.get("measurement_processing") != processing:
        raise ValueError(
            "Refit the event model using the same Dataset measurement processing as the inputs"
        )
    if models.get("coil_statistic") != "processed":
        raise ValueError(
            "Refit with Dataset coil_temperature_processed; historical coil features differ"
        )
    candidates = pd.DatetimeIndex(candidates)
    times = pd.DataFrame(
        {"candidate_defrost_time": candidates, "heating_accounting_start": recovery_start}
    )
    result = measure_candidate_quantities(
        frame,
        times,
        heating_start,
    )
    if models.get("electricity_accounting") != "measured_current_recovery_plus_heating":
        raise ValueError(
            "Current COP requires the preparation/defrost model; use the historical comparison entry for complete-event models"
        )
    phases = models["phase_models"]
    if prediction_mode == "cross-fitted":
        parameters = phases["folds"][experiment_id]
    elif prediction_mode == "full-model":
        parameters = phases["full_data_model"]
    else:
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")
    result = apply_current_cycle_cost(result, frame, heating_start, parameters)
    result["defrost_event_net_heat_kwh"] = 0.0
    result["defrost_event_net_heat_prediction_available"] = True
    result["defrost_event_net_heat_in_training_domain"] = True
    result["measurement_processing"] = processing
    result["heat_column"] = "heating_capacity_processed"
    result["pressure_statistic"] = "processed"
    result["frequency_statistic"] = "current"
    result["coil_statistic"] = "processed"
    result["preparation_heat"] = "zero"
    result["prediction_mode"] = prediction_mode
    result["model_training_scope"] = (
        "held_out_experiment_excluded"
        if prediction_mode == "cross-fitted"
        else "all_available_training_experiments"
    )
    result["stable_heating_start"] = recovery_start
    return calculate_cycle_cop(result, effective=True)


def current_effective_cop(
    frame,
    models,
    experiment_id,
    *,
    now=None,
):
    """Evaluate a prepared heating-cycle prefix using full-data Ridge.

    The caller supplies observations beginning at heating restart. No future
    defrost time, next cycle, Dataset catalog, or offline optimum is required.
    """
    from dataset_tools.builder.detect_cycles import recovery_control_trace

    if "timestamp" not in frame:
        return {"cycle_cop": float("nan"), "status": "missing_channels:timestamp"}
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    now = pd.Timestamp(now) if now is not None else frame.timestamp.max()
    frame = frame.loc[frame.timestamp.le(now)]
    if frame.empty:
        return {
            "candidate_defrost_time": now,
            "cycle_cop": float("nan"),
            "status": "no_observations",
        }
    required = {"timestamp", "power_total", "compressor_frequency", "ambient_temperature"}
    required.update({"heating_capacity_processed", "coil_temperature_processed"})
    missing = required - set(frame)
    if missing:
        return {
            "candidate_defrost_time": now,
            "cycle_cop": float("nan"),
            "status": "missing_channels:" + ",".join(sorted(missing)),
        }
    active = frame.get("defrost_active")
    if active is not None and str(active.iloc[-1]).lower() in {"true", "1", "1.0"}:
        return {
            "candidate_defrost_time": now,
            "cycle_cop": float("nan"),
            "status": "defrost_active",
        }
    if models["recovery_settings"].get("recovery_rule") == "dataset":
        if "cycle_stage" not in frame:
            return {
                "candidate_defrost_time": now,
                "cycle_cop": float("nan"),
                "status": "missing_dataset_stage_boundary",
            }
        confirmed = frame.loc[frame.cycle_stage.eq("frost_development"), "timestamp"]
        recovery_status = "stable_heating_not_observed"
    else:
        trace = recovery_control_trace(frame, models["recovery_settings"])
        confirmed = trace.loc[trace.normal_heating, "timestamp"]
        recovery_status = trace.recovery_status.iloc[-1]
    if confirmed.empty:
        return {
            "candidate_defrost_time": now,
            "cycle_cop": float("nan"),
            "status": recovery_status,
        }
    row = (
        effective_candidate_cop(
            frame,
            [now],
            confirmed.iloc[0],
            frame.timestamp.min(),
            models,
            experiment_id,
            prediction_mode="full-model",
        )
        .iloc[0]
        .to_dict()
    )
    row["status"] = "supported" if row["cycle_cop_eligible"] else "unavailable_or_unsupported"
    if not row["cycle_cop_eligible"]:
        row["cycle_cop"] = float("nan")
    return row


def calculate_cycle(
    loader,
    cycle_name,
    models,
    *,
    rb,
    candidate_step_seconds=10,
    action_step_seconds=None,
    prediction_mode="cross-fitted",
    missing_preparation="exclude",
    initial_heating_start="recovery",
):
    """Calculate one COP curve using an explicitly supplied, frozen RB record."""
    record = loader.get_cycle_record(cycle_name)
    bounds = record["boundaries"]
    heating = timestamp(bounds.get("heating_start"))
    recovery = timestamp(bounds.get("stable_heating_start"))
    end = timestamp(bounds.get("defrost_preparation_start"))
    frame = loader.load_cycle_measurements(cycle_name)
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    observation_end = min(
        timestamp(bounds.get("end_time")) or frame.timestamp.max(), frame.timestamp.max()
    )
    open_heating = end is None and timestamp(bounds.get("defrost_start")) is None
    if open_heating:
        end = observation_end
    end_source = "recording_end" if open_heating else "actual_preparation"
    if end is None and missing_preparation == "defrost-start":
        # A missing unload command does not invalidate earlier heating observations.
        observed = frame.loc[
            frame.timestamp.lt(timestamp(bounds.get("defrost_start"))), "timestamp"
        ]
        if len(observed):
            end = min(observation_end, observed.max())
            end_source = "last_observation_before_defrost"
    accounting_source = "identified_recovery"
    if (
        recovery is None
        and initial_heating_start == "observed"
        and bounds.get("heating_origin") in {"initial_recording", "cold_start"}
    ):
        experiment = loader.list_cycles()
        experiment = experiment.loc[experiment.experiment_id.eq(record["experiment_id"])]
        if experiment.cycle_name.iloc[0] == cycle_name and heating is not None and end is not None:
            running = frame.loc[
                frame.timestamp.ge(heating)
                & frame.timestamp.lt(end)
                & frame.compressor_frequency_setpoint.gt(0),
                "timestamp",
            ]
            if len(running):
                recovery = running.min()
                accounting_source = "first_observed_running_first_cycle"
    # Cohort/whole-event exclusions do not determine pre-action data validity.
    reason = (
        str(record.get("recovery_status", "recovery_not_identified"))
        if recovery is None
        else "missing_preparation_boundary"
        if end is None
        else ""
    )
    if not reason:
        first = max(recovery + pd.Timedelta(seconds=1), heating + pd.Timedelta(minutes=5))
        times = list(pd.date_range(first, end, freq=pd.Timedelta(seconds=candidate_step_seconds)))
        if end >= first:
            times.append(end)
        rb_time = timestamp(rb.get("t_RB"))
        if rb_time is not None and first <= rb_time <= end:
            times.append(rb_time)
        times = sorted(set(times))
        if not times:
            reason = "insufficient_heating_history"
    if reason:
        result = pd.DataFrame(
            {
                "candidate_defrost_time": [end or frame.timestamp.max()],
                "cycle_cop": [np.nan],
                "cycle_cop_eligible": [False],
            }
        )
    else:
        result = effective_candidate_cop(
            frame,
            times,
            recovery,
            heating,
            models,
            str(record["experiment_id"]),
            prediction_mode=prediction_mode,
        )
    result["on_action_grid"] = (
        True
        if action_step_seconds is None
        else (
            (result.candidate_defrost_time - result.candidate_defrost_time.min())
            .dt.total_seconds()
            .mod(action_step_seconds)
            .eq(0)
        )
    )
    result = select_cycle_optimum(result)
    result["cycle_status"] = (
        "identified_curve" if result.is_selected.any() else reason or "no_supported_candidate"
    )
    result["cycle_name"] = cycle_name
    result["experiment_id"] = record["experiment_id"]
    result["cycle_start"] = (
        heating
        if bounds.get("heating_origin") == "cold_start"
        else timestamp(bounds.get("start_time")) or heating
    )
    result["observation_end"] = observation_end
    result["candidate_end_source"] = end_source
    result["catalog_status"] = record["status"]
    result["catalog_status_reason"] = record.get("status_reason")
    result["pipeline_status"] = record.get("pipeline_status", record["status"])
    result["pipeline_status_reason"] = record.get("pipeline_status_reason")
    result["stable_heating_start"] = recovery
    result["heating_accounting_start_source"] = (
        accounting_source if recovery is not None else "unavailable"
    )
    result["t_RB"] = rb.get("t_RB")
    result["rb_status"] = rb["rb_status"]
    result["algorithm"] = "effective_cop_without_trunc"
    result["measurement_processing"] = frame.attrs.get(
        "measurement_processing", "dataset_causal_60s"
    )
    result["heat_column"] = "heating_capacity_processed"
    result["effective_heat_rule"] = "without_trunc"
    result["preparation_heat"] = "zero"
    return result
