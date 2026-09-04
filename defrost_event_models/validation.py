"""Build retrospective LOEO validation tables for defrost-event models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .ridge_models import (
    OUTCOME_TARGETS,
    OUTCOME_VALIDITY,
    TRAINING_COHORT_RULE,
    predict_with_heldout_event_model,
    select_events_complete_for_all_outcomes,
    select_valid_events_for_quantity,
)

_VALIDATION_COLUMNS = {
    "event_electricity": (
        "defrost_event_electricity_prediction_kwh",
        "defrost_event_electricity_training_distance",
    ),
    "event_net_heat": (
        "defrost_event_net_heat_prediction_kwh",
        "defrost_event_net_heat_training_distance",
    ),
    "event_compressor_electricity": (
        "defrost_event_compressor_electricity_prediction_kwh",
        "defrost_event_compressor_electricity_training_distance",
    ),
    "event_duration": (
        "defrost_event_duration_prediction_minutes",
        "defrost_event_duration_training_distance",
    ),
}


def build_validation_table(events: pd.DataFrame, models: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment, group in events.groupby("experiment_id", sort=False):
        for model_name, model_set in models["models"].items():
            event_rows: dict[str, dict[str, object]] = {}
            for name in OUTCOME_TARGETS:
                if name not in model_set:
                    continue
                selected = select_valid_events_for_quantity(group, name)
                if selected.empty:
                    continue
                prediction = predict_with_heldout_event_model(
                    model_set[name], selected, str(experiment)
                )
                for position, (_, event) in enumerate(selected.iterrows()):
                    event_id = str(event["event_id"])
                    row = event_rows.setdefault(
                        event_id,
                        {
                            "event_id": event_id,
                            "cycle_name": event["cycle_name"],
                            "experiment_id": str(experiment),
                            "model_name": model_name,
                            "event_valid": bool(event.get("event_valid", False)),
                            "event_invalid_reason": event.get("event_invalid_reason", ""),
                            **{
                                column: bool(event.get(column, event.get("event_valid", False)))
                                for column in OUTCOME_VALIDITY.values()
                            },
                        },
                    )
                    target = OUTCOME_TARGETS[name]
                    prediction_column, support_column = _VALIDATION_COLUMNS[name]
                    row[target] = event.get(target, np.nan)
                    row[prediction_column] = prediction.iloc[position]["prediction"]
                    row[support_column] = prediction.iloc[position]["support_distance"]
            rows.extend(event_rows.values())
    any_valid = pd.Series(False, index=events.index)
    for name, target in OUTCOME_TARGETS.items():
        if target in events:
            any_valid.loc[select_valid_events_for_quantity(events, name).index] = True
    for _, event in events.loc[~any_valid].iterrows():
        rows.append(
            {
                "event_id": event.get("event_id", event["cycle_name"]),
                "cycle_name": event["cycle_name"],
                "experiment_id": event["experiment_id"],
                "model_name": "excluded_event",
                "event_valid": False,
                "event_invalid_reason": event.get("event_invalid_reason", ""),
            }
        )
    result = pd.DataFrame(rows)
    result["training_cohort_rule"] = TRAINING_COHORT_RULE
    result["common_training_event_count"] = len(select_events_complete_for_all_outcomes(events))
    for outcome, target in OUTCOME_TARGETS.items():
        result[f"available_event_count_{outcome}"] = (
            len(select_valid_events_for_quantity(events, outcome)) if target in events else 0
        )
    return result
