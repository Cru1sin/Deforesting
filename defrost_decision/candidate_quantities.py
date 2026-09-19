"""Build measured and predicted quantities for each candidate defrost time.

Paper notation: E_H/Q_H are pre-defrost electricity/heat and E_T/Q_T are the
complete defrost-event electricity/net heat, including preparation, defrost and recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from defrost_decision.baselines.electricity import MINIMUM_COVERAGE, integrate_heating_curve
from defrost_event_models.ridge_models import (
    load_defrost_event_models,
    predict_independent_targets,
)
from defrost_event_models.training_data import (
    RAW_COLUMNS,
    build_candidate_boundaries,
    candidate_integral_table,
    extract_pre_defrost_features,
    timestamp,
)

DEFAULT_OUTCOME_MODEL = "ridge_dynamic_state_8"


def build_measured_candidate_quantities(
    loader: Any,
    cycle_name: str,
    candidate_times: pd.DataFrame | None = None,
    *,
    candidate_step_seconds: int = 60,
    allow_measurement_reconstruction: bool = False,
    heat_column: str = "water_heat",
) -> pd.DataFrame:
    """Measure G-independent inputs at a grid or explicitly supplied timestamps."""
    record = loader.get_cycle_record(cycle_name)
    nested = record.get("boundaries")
    boundaries = nested if isinstance(nested, Mapping) else record
    heating_start = timestamp(boundaries.get("heating_start"))
    preparation_start = timestamp(boundaries.get("defrost_preparation_start"))
    if heating_start is None or preparation_start is None:
        raise ValueError(f"defrost boundaries are incomplete for {cycle_name}")

    candidate_times = (
        build_candidate_boundaries(
            cycle_name,
            str(record["experiment_id"]),
            heating_start,
            preparation_start,
            step_seconds=candidate_step_seconds,
        )
        if candidate_times is None
        else candidate_times.reset_index(drop=True)
    )
    columns = list(RAW_COLUMNS) + ([heat_column] if heat_column != "water_heat" else [])
    frame = loader.load_cycle_original(cycle_name, columns=columns).copy()
    return measure_candidate_quantities(
        frame,
        candidate_times,
        heating_start,
        heat_column=heat_column,
        allow_measurement_reconstruction=allow_measurement_reconstruction,
    )


def measure_candidate_quantities(
    frame,
    candidate_times,
    heating_start,
    *,
    heat_column="water_heat",
    allow_measurement_reconstruction=False,
):
    """One raw-frame calculation used by Dataset replay and a live cycle prefix."""
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    candidates = [pd.Timestamp(value) for value in candidate_times["candidate_defrost_time"]]
    accounting_start = pd.Timestamp(candidate_times["heating_accounting_start"].iloc[0])

    tref = None
    if heat_column == "heating_capacity":
        temperatures = frame.set_index("timestamp").get(
            "water_out_temperature", pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        )
        temperatures = pd.to_numeric(temperatures, errors="coerce").dropna()
        before = temperatures.loc[temperatures.index <= accounting_start].tail(1)
        after = temperatures.loc[temperatures.index >= accounting_start].head(1)
        tref = float("nan")
        if not before.empty and not after.empty:
            gap = (after.index[0] - before.index[0]).total_seconds()
            if gap <= 30:
                fraction = ((accounting_start - before.index[0]).total_seconds() / gap
                            if gap else 0.0)
                tref = float(before.iloc[0] + fraction * (after.iloc[0] - before.iloc[0]))

    valid_times = [value for value in candidates if value >= accounting_start]
    positions = [index for index, value in enumerate(candidates) if value >= accounting_start]

    def integral(column):
        if not valid_times:
            return pd.DataFrame(
                {
                    "energy": [float("nan")] * len(candidates),
                    "valid": [False] * len(candidates),
                    "uses_measurement_reconstruction": [False] * len(candidates),
                }
            )
        table = candidate_integral_table(
            frame, accounting_start, valid_times, column,
            minimum_outlet_temperature=tref if column == "heating_capacity" else None,
        )
        table["uses_measurement_reconstruction"] = False
        if allow_measurement_reconstruction and column != "heating_capacity":
            signal = (
                1.161
                * pd.to_numeric(frame["water_flow"], errors="coerce")
                * (
                    pd.to_numeric(frame["water_out_temperature"], errors="coerce")
                    - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
                )
                if column == "water_heat"
                else pd.to_numeric(frame[column], errors="coerce")
            )
            reconstructed = integrate_heating_curve(
                frame["timestamp"],
                signal,
                pd.Series(valid_times),
                accounting_start,
                "historical_reconstruction",
            )
            table["energy"] = reconstructed["energy_kwh"].to_numpy()
            table["valid"] = reconstructed["coverage"].ge(MINIMUM_COVERAGE).to_numpy()
            table["uses_measurement_reconstruction"] = (
                reconstructed["coverage"].gt(reconstructed["strict_coverage"]).to_numpy()
            )
        table.index = positions
        table = table.reindex(range(len(candidates)))
        if "energy" not in table:
            table["energy"] = float("nan")
            table["valid"] = False
        table["valid"] = table["valid"].eq(True)
        return table

    def online_integral(column):
        if not valid_times:
            return pd.DataFrame({
                "energy_kwh": [float("nan")] * len(candidates),
                "coverage": [0.] * len(candidates),
                "bridged_internal_gap": [False] * len(candidates),
                "valid": [False] * len(candidates),
            })
        if column == "water_heat":
            required = {"water_flow", "water_out_temperature", "water_in_temperature"}
            signal = (
                1.161
                * pd.to_numeric(frame["water_flow"], errors="coerce")
                * (
                    pd.to_numeric(frame["water_out_temperature"], errors="coerce")
                    - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
                )
                if required <= set(frame)
                else pd.Series(float("nan"), index=frame.index)
            )
        else:
            signal = (
                pd.to_numeric(frame[column], errors="coerce")
                if column in frame
                else pd.Series(float("nan"), index=frame.index)
            )
        table = integrate_heating_curve(
            frame["timestamp"], signal, pd.Series(valid_times), accounting_start,
            "historical_reconstruction", causal_candidates=True,
        )
        table.index = positions
        table = table.reindex(range(len(candidates)))
        table["valid"] = table["coverage"].ge(MINIMUM_COVERAGE)
        return table

    electricity = integral("power_total")
    heat = integral(heat_column)
    if heat_column == "heating_capacity":
        result = candidate_times.copy()
        result["effective_heat_rule"] = "outlet_at_least_recovery_temperature"
        result["reference_outlet_temperature"] = tref
        if not np.isfinite(tref):
            heat["valid"] = False
            heat["energy"] = np.nan
        for name, table in (("electricity", electricity), ("heat", heat)):
            result[f"pre_defrost_{name}_kwh"] = table["energy"].to_numpy()
            result[f"pre_defrost_{name}_measurement_valid"] = table["valid"].to_numpy()
        return pd.concat([result.reset_index(drop=True), extract_pre_defrost_features(
            frame, candidates, heating_start
        )], axis=1)
    compressor = integral("compressor_power")
    online_electricity = online_integral("power_total")
    online_heat = online_integral("water_heat")
    online_compressor = online_integral("compressor_power")
    features = extract_pre_defrost_features(frame, candidates, heating_start)

    result = pd.concat(
        [
            candidate_times,
            pd.DataFrame(
                {
                    "pre_defrost_electricity_kwh": electricity["energy"],
                    "pre_defrost_electricity_measurement_valid": electricity["valid"],
                    "pre_defrost_electricity_uses_measurement_reconstruction": electricity[
                        "uses_measurement_reconstruction"
                    ],
                    "pre_defrost_heat_kwh": heat["energy"],
                    "pre_defrost_heat_measurement_valid": heat["valid"],
                    "pre_defrost_heat_uses_measurement_reconstruction": heat[
                        "uses_measurement_reconstruction"
                    ],
                    "pre_defrost_compressor_electricity_kwh": compressor["energy"],
                    "pre_defrost_compressor_electricity_measurement_valid": compressor["valid"],
                    "pre_defrost_compressor_uses_measurement_reconstruction": compressor[
                        "uses_measurement_reconstruction"
                    ],
                    "online_pre_defrost_electricity_kwh": online_electricity["energy_kwh"],
                    "online_pre_defrost_electricity_measurement_valid": online_electricity[
                        "valid"
                    ],
                    "online_pre_defrost_electricity_uses_measurement_reconstruction": (
                        online_electricity["bridged_internal_gap"].eq(True)
                    ),
                    "online_pre_defrost_heat_kwh": online_heat["energy_kwh"],
                    "online_pre_defrost_heat_measurement_valid": online_heat["valid"],
                    "online_pre_defrost_heat_uses_measurement_reconstruction": online_heat[
                        "bridged_internal_gap"
                    ].eq(True),
                    "online_pre_defrost_compressor_electricity_kwh": online_compressor[
                        "energy_kwh"
                    ],
                    "online_pre_defrost_compressor_measurement_valid": online_compressor["valid"],
                }
            ),
            features,
        ],
        axis=1,
    )
    result["defrost_event_scope"] = "preparation_defrost_recovery"
    result["defrost_event_breakdown"] = "not_decomposed"
    return result


def build_candidate_quantities(
    loader: Any,
    cycle_name: str,
    models: Mapping[str, Any] | None = None,
    *,
    candidate_step_seconds: int = 60,
    defrost_event_electricity_model: str = DEFAULT_OUTCOME_MODEL,
    defrost_event_heat_model: str = DEFAULT_OUTCOME_MODEL,
    prediction_mode: str = "cross-fitted",
) -> pd.DataFrame:
    """Return neutral candidate quantities without selecting a defrost time."""
    measured = build_measured_candidate_quantities(
        loader, cycle_name, candidate_step_seconds=candidate_step_seconds
    )
    model_file = dict(load_defrost_event_models() if models is None else models)
    predicted = predict_independent_targets(
        model_file["models"][defrost_event_electricity_model]["event_electricity"],
        model_file["models"][defrost_event_heat_model]["event_net_heat"],
        measured,
        str(loader.get_cycle_record(cycle_name)["experiment_id"]),
        prediction_mode=prediction_mode,
    )
    result = pd.concat([measured, predicted], axis=1)
    result["prediction_mode"] = prediction_mode
    result["model_training_scope"] = (
        "held_out_experiment_excluded"
        if prediction_mode == "cross-fitted"
        else "all_available_training_experiments"
    )
    return result


def effective_candidate_cop(
    frame,
    candidates,
    recovery_start,
    heating_start,
    models,
    experiment_id,
    *,
    preparation_heat="zero",
    prediction_mode="cross-fitted",
):
    """Estimate an action from strictly pre-action observations, without selecting it."""
    import numpy as np

    from defrost_event_models.ridge_models import predict_with_event_model

    from .performance_objectives import calculate_cycle_cop

    if preparation_heat not in {"include", "zero"}:
        raise ValueError("preparation heat must be include or zero")
    if models.get("cop_definition") != "refrigerant_effective_heat":
        raise ValueError("effective COP requires refitted effective-heat event models")
    candidates = pd.DatetimeIndex(candidates)
    times = pd.DataFrame(
        {"candidate_defrost_time": candidates, "heating_accounting_start": recovery_start}
    )
    result = measure_candidate_quantities(
        frame, times, heating_start, heat_column="heating_capacity"
    )
    targets = models["models"][DEFAULT_OUTCOME_MODEL]
    for quantity, outcome in (("electricity", "event_electricity"), ("net_heat", "event_net_heat")):
        prefix = f"defrost_event_{quantity}"
        if quantity == "net_heat" and preparation_heat == "zero":
            result[prefix + "_kwh"] = 0.0
            result[prefix + "_prediction_available"] = True
            result[prefix + "_in_training_domain"] = True
        else:
            model = targets[outcome]
            if prediction_mode == "cross-fitted" and experiment_id not in model["folds"]:
                result[prefix + "_kwh"] = np.nan
                result[prefix + "_prediction_available"] = False
                result[prefix + "_in_training_domain"] = False
                continue
            predicted = predict_with_event_model(
                model, result, experiment_id, prediction_mode=prediction_mode
            )
            result[prefix + "_kwh"] = predicted.prediction.to_numpy()
            result[prefix + "_prediction_available"] = np.isfinite(predicted.prediction).to_numpy()
            result[prefix + "_in_training_domain"] = predicted.support_distance.le(
                predicted.support_threshold
            ).to_numpy()
    result["preparation_heat"] = preparation_heat
    result["prediction_mode"] = prediction_mode
    result["model_training_scope"] = (
        "held_out_experiment_excluded" if prediction_mode == "cross-fitted"
        else "all_available_training_experiments"
    )
    result["stable_heating_start"] = recovery_start
    return calculate_cycle_cop(result, effective=True)


def current_effective_cop(frame, models, experiment_id, *, preparation_heat="zero", now=None):
    """Evaluate 'defrost now' on a new heating-cycle prefix using full-data Ridge.

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
    required = set(RAW_COLUMNS) - {"water_flow", "compressor_power"}
    required.add("heating_capacity")
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
    trace = recovery_control_trace(frame, models["recovery_settings"])
    confirmed = trace.loc[trace.normal_heating, "timestamp"]
    if confirmed.empty:
        return {
            "candidate_defrost_time": now,
            "cycle_cop": float("nan"),
            "status": trace.recovery_status.iloc[-1],
        }
    row = (
        effective_candidate_cop(
            frame,
            [now],
            confirmed.iloc[0],
            frame.timestamp.min(),
            models,
            experiment_id,
            preparation_heat=preparation_heat,
            prediction_mode="full-model",
        )
        .iloc[0]
        .to_dict()
    )
    row["status"] = "supported" if row["cycle_cop_eligible"] else "unavailable_or_unsupported"
    if not row["cycle_cop_eligible"]:
        row["cycle_cop"] = float("nan")
    return row
