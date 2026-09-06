"""Build measured and predicted quantities for each candidate defrost time.

Paper notation: E_H/Q_H are pre-defrost electricity/heat and E_T/Q_T are the
complete defrost-event electricity/net heat, including preparation, defrost and recovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

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
    frame = loader.load_cycle_original(cycle_name, columns=list(RAW_COLUMNS)).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    candidates = [pd.Timestamp(value) for value in candidate_times["candidate_defrost_time"]]
    accounting_start = pd.Timestamp(candidate_times["heating_accounting_start"].iloc[0])

    valid_times = [value for value in candidates if value >= accounting_start]
    positions = [index for index, value in enumerate(candidates) if value >= accounting_start]

    def integral(column):
        table = candidate_integral_table(frame, accounting_start, valid_times, column)
        table.index = positions
        table = table.reindex(range(len(candidates)))
        if "energy" not in table:
            table["energy"] = float("nan")
            table["valid"] = False
        table["valid"] = table["valid"].eq(True)
        return table

    electricity = integral("power_total")
    heat = integral("water_heat")
    compressor = integral("compressor_power")
    features = extract_pre_defrost_features(frame, candidates, heating_start)

    result = pd.concat(
        [
            candidate_times,
            pd.DataFrame(
                {
                    "pre_defrost_electricity_kwh": electricity["energy"],
                    "pre_defrost_electricity_measurement_valid": electricity["valid"],
                    "pre_defrost_heat_kwh": heat["energy"],
                    "pre_defrost_heat_measurement_valid": heat["valid"],
                    "pre_defrost_compressor_electricity_kwh": compressor["energy"],
                    "pre_defrost_compressor_electricity_measurement_valid": compressor["valid"],
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
