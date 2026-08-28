"""Production v1/v2 cost functions with the selected models embedded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.dataset.loader import DatasetLoader

from .core import optimize_cycle_cop_cost

MINIMUM_INTEGRATION_COVERAGE = 0.95
FIXED_RECOVERY_ELECTRICITY_KWH = 0.279901897467
OFFLINE_OBSERVED_RULE_PROTOCOL = "offline_counterfactual_observed_rule_duration"
RECOVERY = {
    50.0: (9.0, 0.250930, 0.804970),
    55.0: (12.0, 0.395172, 1.146153),
}

# (intercept, Pe, Pe^2, training Pe minimum, training Pe maximum)
ED_BY_EXPERIMENT = {
    "exp_20260714": (0.109160437898849, -0.0524311159925975, -0.2089549607749103, 0.19, 0.4),
    "exp_20260715": (0.1063000255041619, -0.0367342364756182, -0.2285630939797699, 0.19, 0.4),
    "exp_20260717": (0.1111693628872409, -0.0648757625179157, -0.1912778509133092, 0.19, 0.4),
    "exp_20260720": (0.1091357037749996, -0.052600671492866, -0.2083264468366366, 0.19, 0.4),
    "exp_20260721": (0.1134541343406789, -0.0786373497007127, -0.1722215482960587, 0.19, 0.4),
    "exp_20260723": (0.0990348495914412, 0.0353126609917079, -0.3884353237352418, 0.19, 0.33),
    "exp_20260724": (0.1092026608902168, -0.0540005426575575, -0.204864977498866, 0.19, 0.4),
    "exp_20260727": (0.1088163168439893, -0.0540215395534654, -0.2021092010628559, 0.19, 0.4),
    "exp_20260728": (0.1108141826865397, -0.0680415907910462, -0.1806844358510142, 0.19, 0.4),
    "exp_20260729": (0.1103876576867418, -0.0620857198041468, -0.1929084977820067, 0.19, 0.4),
    "exp_20260730": (0.1090905229589749, -0.05365674785853, -0.2048965549016874, 0.19, 0.4),
    "exp_20260731": (0.1089030960750327, -0.0532014821767146, -0.2047046583133213, 0.19, 0.4),
    "exp_20260803": (0.1097438965348212, -0.0558531434949093, -0.2040535618137482, 0.19, 0.4),
    "exp_20260804": (0.1071474392323601, -0.03626176370601, -0.2291808986228213, 0.19, 0.4),
    "exp_20260805": (0.1099603596583693, -0.0589538822109364, -0.1977916147163611, 0.19, 0.4),
}

QD_COEFFICIENTS = {
    "intercept": 0.6268740230592904,
    "water_in_temperature": 0.0137588499217354,
    "water_out_temperature": -0.0159768197373373,
    "rule_defrost_duration_minutes": 0.0848404223153193,
    "coil_temperature": -0.0022281468360404,
    "evaporating_pressure": -0.3816007230713125,
    "water_in_temperature_squared": 0.0001937670259627,
    "water_out_temperature_squared": -0.00015920510852,
    "rule_defrost_duration_minutes_squared": 0.0052777716254524,
    "coil_temperature_squared": -0.0002357711327978,
    "evaporating_pressure_squared": -1.1330691119688487,
}

QD_SUPPORT = {
    "water_in_temperature": (40.1, 47.1),
    "water_out_temperature": (44.7, 54.75),
    "rule_defrost_duration_minutes": (3.6166666666666663, 5.833333333333333),
    "coil_temperature": (-20.9, -3.1),
    "evaporating_pressure": (0.19, 0.4),
}

STATE_COLUMNS = (
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "water_temperature_setpoint",
)
MODEL_COLUMNS = (
    "fold_intercept_kwh",
    "fold_linear_kwh_per_mpa",
    "fold_quadratic_kwh_per_mpa2",
    "fold_train_pe_min_mpa",
    "fold_train_pe_max_mpa",
)
COST_COLUMNS = (
    "defrost_electricity_kwh",
    "recovery_duration_minutes",
    "recovery_electricity_kwh",
    "defrost_absorbed_heat_kwh",
    "recovery_heat_kwh",
    "user_heating_kwh",
    "cycle_user_heating_kwh",
    "cycle_electricity_kwh",
    "inverse_cop",
    "cycle_cop",
    "relative_regret",
    "near_optimal_1pct",
    "near_optimal_5pct",
    "water_reference_inverse_cop",
    "water_reference_relative_regret",
    "observed_rule_defrost_duration_minutes",
    "qd_eligible",
    "qd_supported",
    "qd_outside_terms",
    "qd_max_normalized_extrapolation",
    "heat_balance_eligible",
    "model_protocol",
    "t_star",
    "minimum_location",
    "water_reference_t_star",
)


def _timestamp(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _candidate_states(
    loader: DatasetLoader, cycle_name: str, candidate_times: pd.Series
) -> pd.DataFrame:
    frame = loader.load_cycle_original(
        cycle_name, columns=["timestamp", *STATE_COLUMNS]
    )
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    rows = []
    for end in pd.to_datetime(candidate_times, errors="coerce"):
        window = frame.loc[
            timestamps.ge(end - pd.Timedelta(seconds=60)) & timestamps.lt(end)
        ]
        row = {"candidate_time": end}
        for column in STATE_COLUMNS:
            values = pd.to_numeric(window[column], errors="coerce").dropna()
            row[column] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _observed_rule_duration_minutes(loader: DatasetLoader, cycle_name: str) -> float:
    record = loader.get_cycle_record(cycle_name)
    boundaries = record.get("boundaries", {})
    start = _timestamp(record.get("defrost_start") or boundaries.get("defrost_start"))
    end = _timestamp(record.get("defrost_end") or boundaries.get("defrost_end"))
    if start is None or end is None or end <= start:
        raise ValueError(f"cannot reconstruct observed rule duration for {cycle_name}")
    raw = loader.load_cycle_original(
        cycle_name, columns=["timestamp", "coil_temperature"]
    ).copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce").dt.floor("s")
    raw["coil_temperature"] = pd.to_numeric(raw["coil_temperature"], errors="coerce")
    raw = raw.loc[raw["timestamp"].ge(start) & raw["timestamp"].lt(end)]
    raw = raw.dropna(subset=["timestamp"]).drop_duplicates("timestamp").set_index("timestamp")
    grid = pd.date_range(start.ceil("s"), end.ceil("s"), freq="s", inclusive="left")
    t3 = raw["coil_temperature"].reindex(raw.index.union(grid).sort_values())
    reached = np.flatnonzero(
        t3.interpolate(method="time", limit_area="inside").reindex(grid).ge(20).to_numpy()
    )
    if not len(reached):
        raise ValueError(f"cannot reconstruct observed rule duration for {cycle_name}")
    return min(int(reached[0]) + 40, 350) / 60


def _apply_ed(candidates: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    try:
        coefficients = ED_BY_EXPERIMENT[experiment_id]
    except KeyError as exc:
        raise ValueError(f"no selected ED model for {experiment_id}") from exc
    values = candidates.copy()
    for column, coefficient in zip(MODEL_COLUMNS, coefficients, strict=True):
        values[column] = coefficient
    pe = pd.to_numeric(values["evaporating_pressure_mpa"], errors="coerce")
    lower, upper = coefficients[-2:]
    values["support_status"] = np.select(
        [pe.isna(), pe.lt(lower), pe.gt(upper)],
        ["missing", "below", "above"],
        default="supported",
    )
    values["pe_extrapolation_distance_mpa_signed"] = np.select(
        [pe.lt(lower), pe.gt(upper)], [pe - lower, pe - upper], default=0.0
    )
    values.loc[pe.isna(), "pe_extrapolation_distance_mpa_signed"] = np.nan
    values["pe_extrapolation_distance_mpa_absolute"] = values[
        "pe_extrapolation_distance_mpa_signed"
    ].abs()
    values["pe_supported"] = values["support_status"].eq("supported")
    values["integration_eligible"] = values["integration_coverage"].ge(
        MINIMUM_INTEGRATION_COVERAGE
    )
    values["predicted_preparation_defrost_electricity_kwh"] = (
        coefficients[0] + coefficients[1] * pe + coefficients[2] * pe.pow(2)
    )
    values["fixed_recovery_electricity_kwh"] = FIXED_RECOVERY_ELECTRICITY_KWH
    values["dynamic_ticket_electricity_kwh"] = (
        values["predicted_preparation_defrost_electricity_kwh"]
        + FIXED_RECOVERY_ELECTRICITY_KWH
    )
    values["optimization_eligible"] = values["integration_eligible"] & pe.notna()
    return values


def _qd_states(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float
) -> dict[str, pd.Series]:
    return {
        "water_in_temperature": pd.to_numeric(candidates["water_in_temperature"], errors="coerce"),
        "water_out_temperature": pd.to_numeric(
            candidates["water_out_temperature"], errors="coerce"
        ),
        "rule_defrost_duration_minutes": pd.Series(
            observed_rule_duration_minutes, index=candidates.index
        ),
        "coil_temperature": pd.to_numeric(candidates["coil_temperature"], errors="coerce"),
        "evaporating_pressure": pd.to_numeric(
            candidates["evaporating_pressure_mpa"], errors="coerce"
        ),
    }


def _predict_qd(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float
) -> pd.Series:
    prediction = pd.Series(QD_COEFFICIENTS["intercept"], index=candidates.index)
    for term, values in _qd_states(candidates, observed_rule_duration_minutes).items():
        prediction += QD_COEFFICIENTS[term] * values
        prediction += QD_COEFFICIENTS[f"{term}_squared"] * values.pow(2)
    return prediction


def _audit_qd_support(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float
) -> pd.DataFrame:
    names = pd.Series([[] for _ in candidates.index], index=candidates.index)
    maximum = pd.Series(0.0, index=candidates.index)
    supported = pd.Series(True, index=candidates.index)
    missing_any = pd.Series(False, index=candidates.index)
    for term, values in _qd_states(candidates, observed_rule_duration_minutes).items():
        lower, upper = QD_SUPPORT[term]
        missing = values.isna()
        outside = values.lt(lower) | values.gt(upper)
        supported &= ~(missing | outside)
        missing_any |= missing
        names.loc[missing] = names.loc[missing].map(
            lambda items, name=term: [*items, f"{name}:missing"]
        )
        names.loc[outside] = names.loc[outside].map(
            lambda items, name=term: [*items, name]
        )
        distance = pd.concat(
            [(lower - values).clip(lower=0), (values - upper).clip(lower=0)], axis=1
        ).max(axis=1) / (upper - lower)
        maximum = pd.concat([maximum, distance], axis=1).max(axis=1)
    maximum.loc[missing_any] = np.nan
    return pd.DataFrame(
        {
            "qd_supported": supported,
            "qd_outside_terms": names.map(",".join),
            "qd_max_normalized_extrapolation": maximum,
        }
    )


def _apply_cost(
    candidates: pd.DataFrame,
    algorithm: str,
    *,
    setpoint: float | None = None,
    observed_rule_duration_minutes: float | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    values = candidates.copy()
    values["algorithm"] = algorithm
    values["defrost_electricity_kwh"] = pd.to_numeric(
        values["predicted_preparation_defrost_electricity_kwh"], errors="coerce"
    )
    if algorithm == "v1":
        values["recovery_duration_minutes"] = np.nan
        values["recovery_electricity_kwh"] = FIXED_RECOVERY_ELECTRICITY_KWH
        values["defrost_absorbed_heat_kwh"] = 0.0
        values["qd_supported"] = pd.Series(pd.NA, index=values.index, dtype="boolean")
        values["qd_outside_terms"] = ""
        values["qd_max_normalized_extrapolation"] = np.nan
        values["recovery_heat_kwh"] = 0.0
        values["user_heating_kwh"] = values["unit_heating_kwh"]
        values["model_protocol"] = "loeo_pe_quadratic_fixed_recovery"
        curve, optimum = optimize_cycle_cop_cost(
            values,
            defrost_recovery_electricity_kwh=(
                values["defrost_electricity_kwh"] + values["recovery_electricity_kwh"]
            ),
        )
        curve["water_reference_inverse_cop"] = (
            curve["cycle_electricity_kwh"] / curve["water_heating_kwh"]
        )
        eligible = curve["optimization_eligible"].fillna(False)
        minimum = curve["water_reference_inverse_cop"].where(eligible).min()
        curve["water_reference_relative_regret"] = (
            curve["water_reference_inverse_cop"] / minimum - 1.0
        ).where(eligible)
    else:
        if setpoint not in RECOVERY or observed_rule_duration_minutes is None:
            raise ValueError("v2 requires a 50 or 55 degC setpoint and rule duration")
        recovery_duration, recovery_electricity, recovery_heat = RECOVERY[setpoint]
        values["water_temperature_setpoint"] = setpoint
        values["observed_rule_defrost_duration_minutes"] = observed_rule_duration_minutes
        values["recovery_duration_minutes"] = recovery_duration
        values["recovery_electricity_kwh"] = recovery_electricity
        values["defrost_absorbed_heat_kwh"] = _predict_qd(
            values, observed_rule_duration_minutes
        )
        audit = _audit_qd_support(values, observed_rule_duration_minutes)
        values[list(audit)] = audit
        values["qd_eligible"] = (
            np.isfinite(values["defrost_absorbed_heat_kwh"])
            & values["defrost_absorbed_heat_kwh"].gt(0)
        )
        cycle_heat = (
            values["water_heating_kwh"]
            - values["defrost_absorbed_heat_kwh"]
            + recovery_heat
        )
        values["heat_balance_eligible"] = np.isfinite(cycle_heat) & cycle_heat.gt(0)
        values["optimization_eligible"] = (
            values["optimization_eligible"].fillna(False)
            & values["qd_eligible"]
            & values["heat_balance_eligible"]
        )
        values["recovery_heat_kwh"] = recovery_heat
        values["user_heating_kwh"] = (
            values["water_heating_kwh"] - values["defrost_absorbed_heat_kwh"]
        )
        values["model_protocol"] = OFFLINE_OBSERVED_RULE_PROTOCOL
        curve, optimum = optimize_cycle_cop_cost(
            values,
            defrost_recovery_electricity_kwh=(
                values["defrost_electricity_kwh"] + values["recovery_electricity_kwh"]
            ),
            defrost_recovery_heat_kwh=values["recovery_heat_kwh"],
        )
    minimum = float(optimum["inverse_cop"])
    eligible = curve["optimization_eligible"].fillna(False)
    curve["relative_regret"] = (curve["inverse_cop"] / minimum - 1.0).where(eligible)
    curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
    curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
    return curve, optimum


def build_cost_function_table(  # noqa: C901
    base: pd.DataFrame,
    points: pd.DataFrame,
    loader: DatasetLoader,
    algorithm: str,
) -> pd.DataFrame:
    """Build the auditable candidate table for selected algorithm v1 or v2."""
    if algorithm not in {"v1", "v2"}:
        raise ValueError("algorithm must be 'v1' or 'v2'")
    if points["cycle_name"].duplicated().any():
        raise ValueError("points must contain one row per cycle")
    metadata_columns = [
        "cycle_name",
        "experiment_id",
        "t_heating_stable",
        "t_actual_preparation",
        "t_RB",
        "rb_status",
        "trigger_type",
        "actual_minutes_from_stable",
    ]
    metadata = points[[column for column in metadata_columns if column in points]].rename(
        columns={"t_actual_preparation": "actual_preparation_time"}
    )
    experiment_by_cycle = metadata.set_index("cycle_name")["experiment_id"]
    values = base.copy()
    values["candidate_time"] = pd.to_datetime(values["candidate_time"], errors="coerce")
    tables = []
    for cycle_name, source in values.groupby("cycle_name", sort=False):
        candidates = source.copy()
        try:
            experiment_id = str(experiment_by_cycle.loc[cycle_name])
            candidates = _apply_ed(candidates, experiment_id)
            kwargs: dict[str, object] = {}
            if algorithm == "v2":
                states = _candidate_states(
                    loader, str(cycle_name), candidates["candidate_time"]
                ).set_index("candidate_time")
                for column in STATE_COLUMNS:
                    loaded = candidates["candidate_time"].map(states[column])
                    candidates[column] = (
                        candidates[column].where(candidates[column].notna(), loaded)
                        if column in candidates
                        else loaded
                    )
                setpoint = float(
                    pd.to_numeric(
                        candidates["water_temperature_setpoint"], errors="coerce"
                    ).median()
                )
                kwargs = {
                    "setpoint": setpoint,
                    "observed_rule_duration_minutes": _observed_rule_duration_minutes(
                        loader, str(cycle_name)
                    ),
                }
            curve, optimum = _apply_cost(candidates, algorithm, **kwargs)
            curve["valid"] = True
            curve["failure_reason"] = ""
            curve["t_star"] = optimum["candidate_time"]
            curve["minimum_location"] = optimum["minimum_location"]
            if algorithm == "v1":
                eligible = curve["optimization_eligible"].fillna(False)
                water_index = curve["water_reference_inverse_cop"].where(eligible).idxmin()
                curve["water_reference_t_star"] = curve.loc[water_index, "candidate_time"]
            tables.append(curve)
        except ValueError as exc:
            failed = candidates.copy()
            failed["algorithm"] = algorithm
            failed["valid"] = False
            failed["failure_reason"] = str(exc)
            for column in COST_COLUMNS:
                failed[column] = np.nan
            tables.append(failed)
    result = pd.concat(tables, ignore_index=True, sort=False) if tables else values
    result = result.drop(
        columns=[column for column in metadata if column != "cycle_name" and column in result],
        errors="ignore",
    ).merge(metadata, on="cycle_name", how="left", validate="many_to_one", sort=False)
    for column in (*STATE_COLUMNS, *COST_COLUMNS):
        if column not in result:
            result[column] = np.nan
    result = result.sort_values(
        ["cycle_name", "candidate_time"], kind="stable"
    ).reset_index(drop=True)
    if result.duplicated(["cycle_name", "candidate_time"]).any():
        raise ValueError("cost function table has duplicate cycle/candidate rows")
    return result


def write_cost_function_csv(
    table: pd.DataFrame, output_root: Path, algorithm: str
) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"cost_function_{algorithm}.csv"
    table.to_csv(path, index=False)
    return path
