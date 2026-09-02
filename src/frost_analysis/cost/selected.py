"""Production v1/v2 cost functions with the selected models embedded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataloader.loader import DatasetLoader

from .core import integrate_energy_kwh, optimize_cycle_cop_cost, water_side_heating_kw

MINIMUM_INTEGRATION_COVERAGE = 0.95
FIXED_RECOVERY_ELECTRICITY_KWH = 0.279901897467
OFFLINE_OBSERVED_RULE_PROTOCOL = "offline_counterfactual_observed_rule_duration"
RECOVERY = {
    50.0: (9.0, 0.250930, 0.804970),
    55.0: (12.0, 0.395172, 1.146153),
}
RECOVERY_FIXED_9_MINUTES = {
    50.0: (9.0, 0.25093046783625733, 0.80496951375),
    55.0: (9.0, 0.2515340107709751, 0.786563833239796),
}
FIXED_RECOVERY_BOUNDARY_MINUTES = 9.0
RECOVERY_UNIT_HEAT_KWH = {
    50.0: 1.057650730994152,
    55.0: 1.43915376984127,
}
PREPARATION_HEAT_COEFFICIENTS = {
    "intercept": -0.049851,
    "water_in_temperature": -0.001875,
    "water_out_temperature": 0.002498,
    "preparation_duration_minutes": 0.142823,
}
ROBUST_ELECTRICITY_MARGIN_KWH = 0.004859499035670531
ROBUST_HEAT_MARGIN_KWH = 0.03579830227974917
QPREP_SUPPORT = {
    "water_in_temperature": (40.1, 47.2),
    "water_out_temperature": (44.7, 55.0),
    "preparation_duration_minutes": (0.2166666666666666, 0.7166666666666667),
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

# LOEO StandardScaler + Ridge(alpha=1) for rule duration:
# intercept, T3 slope, T3 range, duration range.
DURATION_BY_EXPERIMENT = {
    "exp_20260714": (
        3.6857790568881903,
        -0.07108475653641054,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260715": (
        3.7831176586543163,
        -0.0657173446956769,
        -20.9,
        -3.1,
        3.8666666666666663,
        5.833333333333333,
    ),
    "exp_20260717": (
        3.728772611071597,
        -0.06780342725244977,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260720": (
        3.6888004034379236,
        -0.07113224911111446,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260721": (
        3.7481161568764545,
        -0.06632874252711553,
        -20.8,
        -3.1,
        3.6166666666666663,
        5.716666666666667,
    ),
    "exp_20260723": (
        3.4590410222766517,
        -0.08305816384200398,
        -20.9,
        -7.05,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260724": (
        3.6913893679552827,
        -0.07093483067892699,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260727": (
        3.6934170742993473,
        -0.07048150729241191,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260728": (
        3.6848387054548706,
        -0.07152877992318589,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260729": (
        3.65433234954208,
        -0.0743288242630281,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260730": (
        3.6581027739082943,
        -0.07413724654865594,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260731": (
        3.6697407267742457,
        -0.07367300677515999,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260803": (
        3.683979678287608,
        -0.0735125915739133,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260804": (
        3.6306239669658695,
        -0.0739187030788104,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
    "exp_20260805": (
        3.694295709460144,
        -0.07149729865556355,
        -20.9,
        -3.1,
        3.6166666666666663,
        5.833333333333333,
    ),
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
    "defrost_heat_kwh",
    "preparation_duration_minutes",
    "preparation_heat_kwh",
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
    "qprep_eligible",
    "qprep_supported",
    "qd_supported",
    "qd_outside_terms",
    "qd_max_normalized_extrapolation",
    "heat_balance_eligible",
    "model_protocol",
    "t_star",
    "minimum_location",
    "water_reference_t_star",
    "nominal_cycle_electricity_kwh",
    "nominal_cycle_user_heating_kwh",
    "nominal_inverse_cop",
    "robust_cycle_electricity_kwh",
    "robust_cycle_user_heating_kwh",
    "model_supported",
    "t_star_model_supported",
    "excess_electricity_kwh",
)


def _timestamp(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _candidate_states(
    loader: DatasetLoader, cycle_name: str, candidate_times: pd.Series
) -> pd.DataFrame:
    frame = loader.load_cycle_original(cycle_name, columns=["timestamp", *STATE_COLUMNS])
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    rows = []
    for end in pd.to_datetime(candidate_times, errors="coerce"):
        window = frame.loc[timestamps.ge(end - pd.Timedelta(seconds=60)) & timestamps.lt(end)]
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
    raw = loader.load_cycle_original(cycle_name, columns=["timestamp", "coil_temperature"]).copy()
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


def _observed_preparation_duration_minutes(loader: DatasetLoader, cycle_name: str) -> float:
    record = loader.get_cycle_record(cycle_name)
    boundaries = record.get("boundaries", {})
    start = _timestamp(
        record.get("defrost_preparation_start") or boundaries.get("defrost_preparation_start")
    )
    end = _timestamp(record.get("defrost_start") or boundaries.get("defrost_start"))
    if start is None or end is None or end <= start:
        raise ValueError(f"cannot reconstruct preparation duration for {cycle_name}")
    return (end - start).total_seconds() / 60


def _start_heating_after(
    candidates: pd.DataFrame,
    loader: DatasetLoader,
    cycle_name: str,
    minutes: float,
) -> pd.DataFrame:
    record = loader.get_cycle_record(cycle_name)
    boundaries = record.get("boundaries", {})
    heating_start = _timestamp(record.get("heating_start") or boundaries.get("heating_start"))
    stable_start = _timestamp(
        record.get("stable_heating_start") or boundaries.get("stable_heating_start")
    )
    if heating_start is None or stable_start is None:
        raise ValueError(f"cannot reconstruct fixed recovery boundary for {cycle_name}")
    boundary = heating_start + pd.Timedelta(minutes=minutes)
    values = candidates.copy()
    if boundary == stable_start:
        electricity = heat = 0.0
    else:
        start, end = sorted((boundary, stable_start))
        frame = loader.load_cycle_original(
            cycle_name,
            columns=[
                "timestamp",
                "power_total",
                "water_flow",
                "water_in_temperature",
                "water_out_temperature",
                "heating_capacity",
            ],
        ).copy()
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        interval = frame.loc[timestamps.between(start, end)]
        electricity, electricity_coverage = integrate_energy_kwh(
            interval["timestamp"], interval["power_total"], maximum_gap_seconds=np.inf
        )
        heat, heat_coverage = integrate_energy_kwh(
            interval["timestamp"],
            water_side_heating_kw(interval),
            maximum_gap_seconds=np.inf,
        )
        unit_heat, unit_heat_coverage = integrate_energy_kwh(
            interval["timestamp"],
            interval["heating_capacity"],
            maximum_gap_seconds=np.inf,
        )
        if (
            min(electricity_coverage, heat_coverage, unit_heat_coverage)
            < MINIMUM_INTEGRATION_COVERAGE
        ):
            raise ValueError(f"heating boundary is incomplete for {cycle_name}")
        sign = 1.0 if boundary < stable_start else -1.0
        electricity *= sign
        heat *= sign
        unit_heat *= sign
    if boundary == stable_start:
        unit_heat = 0.0
    values["heating_electricity_kwh"] += electricity
    values["water_heating_kwh"] += heat
    values["unit_heating_kwh"] += unit_heat
    values["heating_hours"] = (
        pd.to_datetime(values["candidate_time"], errors="coerce") - boundary
    ).dt.total_seconds() / 3600
    values["heating_boundary_time"] = boundary
    values["heating_boundary_minutes_after_start"] = minutes
    values["heating_boundary_electricity_adjustment_kwh"] = electricity
    values["heating_boundary_heat_adjustment_kwh"] = heat
    values["heating_boundary_unit_heat_adjustment_kwh"] = unit_heat
    return values


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
    values["integration_eligible"] = values["integration_coverage"].ge(MINIMUM_INTEGRATION_COVERAGE)
    values["predicted_preparation_defrost_electricity_kwh"] = (
        coefficients[0] + coefficients[1] * pe + coefficients[2] * pe.pow(2)
    )
    values["fixed_recovery_electricity_kwh"] = FIXED_RECOVERY_ELECTRICITY_KWH
    values["dynamic_ticket_electricity_kwh"] = (
        values["predicted_preparation_defrost_electricity_kwh"] + FIXED_RECOVERY_ELECTRICITY_KWH
    )
    values["optimization_eligible"] = values["integration_eligible"] & pe.notna()
    return values


def _qd_states(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float | pd.Series
) -> dict[str, pd.Series]:
    duration = (
        pd.to_numeric(observed_rule_duration_minutes, errors="coerce")
        if isinstance(observed_rule_duration_minutes, pd.Series)
        else pd.Series(observed_rule_duration_minutes, index=candidates.index)
    )
    return {
        "water_in_temperature": pd.to_numeric(candidates["water_in_temperature"], errors="coerce"),
        "water_out_temperature": pd.to_numeric(
            candidates["water_out_temperature"], errors="coerce"
        ),
        "rule_defrost_duration_minutes": duration,
        "coil_temperature": pd.to_numeric(candidates["coil_temperature"], errors="coerce"),
        "evaporating_pressure": pd.to_numeric(
            candidates["evaporating_pressure_mpa"], errors="coerce"
        ),
    }


def _predict_qd(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float | pd.Series
) -> pd.Series:
    prediction = pd.Series(QD_COEFFICIENTS["intercept"], index=candidates.index)
    for term, values in _qd_states(candidates, observed_rule_duration_minutes).items():
        prediction += QD_COEFFICIENTS[term] * values
        prediction += QD_COEFFICIENTS[f"{term}_squared"] * values.pow(2)
    return prediction


def _predict_preparation_heat(
    candidates: pd.DataFrame, preparation_duration_minutes: float
) -> pd.Series:
    return (
        PREPARATION_HEAT_COEFFICIENTS["intercept"]
        + PREPARATION_HEAT_COEFFICIENTS["water_in_temperature"]
        * pd.to_numeric(candidates["water_in_temperature"], errors="coerce")
        + PREPARATION_HEAT_COEFFICIENTS["water_out_temperature"]
        * pd.to_numeric(candidates["water_out_temperature"], errors="coerce")
        + PREPARATION_HEAT_COEFFICIENTS["preparation_duration_minutes"]
        * preparation_duration_minutes
    )


def _audit_qd_support(
    candidates: pd.DataFrame, observed_rule_duration_minutes: float | pd.Series
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
        names.loc[outside] = names.loc[outside].map(lambda items, name=term: [*items, name])
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


def _predict_candidate_duration(candidates: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    """Apply the frozen causal LOEO T3-duration model without clipping."""
    try:
        intercept, slope, t3_min, t3_max, duration_min, duration_max = DURATION_BY_EXPERIMENT[
            experiment_id
        ]
    except KeyError as exc:
        raise ValueError(f"no selected duration model for {experiment_id}") from exc
    values = candidates.copy()
    t3 = pd.to_numeric(values["coil_temperature"], errors="coerce")
    duration = intercept + slope * t3
    values["duration_fold_intercept_minutes"] = intercept
    values["duration_fold_slope_minutes_per_degC"] = slope
    values["duration_fold_train_t3_min"] = t3_min
    values["duration_fold_train_t3_max"] = t3_max
    values["duration_fold_train_min_minutes"] = duration_min
    values["duration_fold_train_max_minutes"] = duration_max
    values["predicted_rule_defrost_duration_minutes"] = duration
    values["candidate_duration_supported"] = t3.between(t3_min, t3_max) & duration.between(
        duration_min, duration_max
    )
    return values


def _replace_observed_duration_qd(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute the transition heat from candidate-time T3 only."""
    tables = []
    for experiment_id, source in table.groupby("experiment_id", sort=False):
        values = _predict_candidate_duration(source, str(experiment_id))
        duration = values["predicted_rule_defrost_duration_minutes"]
        values["rule_defrost_duration_minutes"] = duration
        values["defrost_absorbed_heat_kwh"] = _predict_qd(values, duration)
        values["defrost_heat_kwh"] = -values["defrost_absorbed_heat_kwh"]
        audit = _audit_qd_support(values, duration)
        values[list(audit)] = audit
        values["qd_eligible"] = np.isfinite(values["defrost_absorbed_heat_kwh"]) & values[
            "defrost_absorbed_heat_kwh"
        ].gt(0)
        values["transition_service_heat_kwh"] = (
            values["preparation_heat_kwh"]
            - values["defrost_absorbed_heat_kwh"]
            + values["projected_recovery_heat_kwh"]
        )
        values["user_heating_kwh"] = (
            values["stable_unit_heating_kwh"]
            + values["preparation_heat_kwh"]
            - values["defrost_absorbed_heat_kwh"]
        )
        cycle_heat = values["user_heating_kwh"] + values["projected_recovery_heat_kwh"]
        values["heat_balance_eligible"] = np.isfinite(cycle_heat) & cycle_heat.gt(0)
        values["optimization_eligible"] = (
            values["integration_eligible"].fillna(False)
            & pd.to_numeric(values["evaporating_pressure_mpa"], errors="coerce").notna()
            & values["qd_eligible"]
            & values["qprep_eligible"]
            & values["heat_balance_eligible"]
        )
        values["model_supported"] = (
            values["pe_supported"].fillna(False)
            & values["qd_supported"].fillna(False)
            & values["qprep_supported"].fillna(False)
            & values["candidate_duration_supported"].fillna(False)
        )
        tables.append(values)
    return pd.concat(tables, ignore_index=True, sort=False) if tables else table.copy()


def _latest_supported_in_optimal_basin(
    curve: pd.DataFrame, optimum_time: pd.Timestamp
) -> tuple[pd.Timestamp, str]:
    values = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True)
    optimum = values["candidate_time"].eq(pd.Timestamp(optimum_time))
    near = values["near_optimal_1pct"].fillna(False)
    if not optimum.any() or not near.loc[optimum].all():
        return pd.NaT, "abstain_invalid_optimum"
    segments = near.ne(near.shift(fill_value=False)).cumsum()
    basin = segments.loc[optimum].iloc[0]
    supported = near & segments.eq(basin) & values["model_supported"].fillna(False)
    if not supported.any():
        return pd.NaT, "abstain_model_support"
    return (
        pd.Timestamp(values.loc[supported, "candidate_time"].max()),
        "latest_supported_in_1pct_basin",
    )


def _apply_cost(  # noqa: C901
    candidates: pd.DataFrame,
    algorithm: str,
    *,
    setpoint: float | None = None,
    observed_rule_duration_minutes: float | None = None,
    preparation_duration_minutes: float | None = None,
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
        values["defrost_heat_kwh"] = 0.0
        values["preparation_duration_minutes"] = np.nan
        values["preparation_heat_kwh"] = 0.0
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
            raise ValueError("v2 variants require a 50 or 55 degC setpoint and rule duration")
        if (
            algorithm in {"v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}
            and preparation_duration_minutes is None
        ):
            raise ValueError("v2.1+ requires the observed preparation duration")
        if algorithm in {"v2.5", "v2.6", "v3"}:
            recovery_duration = recovery_electricity = recovery_heat = 0.0
        else:
            recovery_duration, recovery_electricity, recovery_heat = (
                RECOVERY_FIXED_9_MINUTES[setpoint]
                if algorithm in {"v2.3", "v2.4"}
                else RECOVERY[setpoint]
            )
        if algorithm == "v2.1":
            recovery_heat = RECOVERY_UNIT_HEAT_KWH[setpoint]
        values["water_temperature_setpoint"] = setpoint
        values["observed_rule_defrost_duration_minutes"] = observed_rule_duration_minutes
        values["recovery_duration_minutes"] = recovery_duration
        values["recovery_electricity_kwh"] = recovery_electricity
        values["defrost_absorbed_heat_kwh"] = _predict_qd(values, observed_rule_duration_minutes)
        values["defrost_heat_kwh"] = -values["defrost_absorbed_heat_kwh"]
        audit = _audit_qd_support(values, observed_rule_duration_minutes)
        values[list(audit)] = audit
        values["qd_eligible"] = np.isfinite(values["defrost_absorbed_heat_kwh"]) & values[
            "defrost_absorbed_heat_kwh"
        ].gt(0)
        if algorithm in {"v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
            values["preparation_duration_minutes"] = preparation_duration_minutes
            values["preparation_heat_kwh"] = _predict_preparation_heat(
                values, preparation_duration_minutes
            )
            values["qprep_eligible"] = np.isfinite(values["preparation_heat_kwh"]) & values[
                "preparation_heat_kwh"
            ].gt(0)
            values["qprep_supported"] = (
                pd.to_numeric(values["water_in_temperature"], errors="coerce").between(
                    *QPREP_SUPPORT["water_in_temperature"]
                )
                & pd.to_numeric(values["water_out_temperature"], errors="coerce").between(
                    *QPREP_SUPPORT["water_out_temperature"]
                )
                & (
                    QPREP_SUPPORT["preparation_duration_minutes"][0]
                    <= preparation_duration_minutes
                    <= QPREP_SUPPORT["preparation_duration_minutes"][1]
                )
            )
            heating = (
                values["unit_heating_kwh"]
                if algorithm in {"v2.1", "v2.6"}
                else values["water_heating_kwh"]
            )
            values["user_heating_kwh"] = (
                heating + values["preparation_heat_kwh"] + values["defrost_heat_kwh"]
            )
            if algorithm == "v3":
                transient_heat = values["preparation_heat_kwh"] + values["defrost_heat_kwh"]
                values["nominal_water_cycle_heat_kwh"] = (
                    values["water_heating_kwh"] + transient_heat
                )
                values["nominal_unit_cycle_heat_kwh"] = values["unit_heating_kwh"] + transient_heat
                values["nominal_cycle_user_heating_kwh"] = values[
                    ["nominal_water_cycle_heat_kwh", "nominal_unit_cycle_heat_kwh"]
                ].min(axis=1)
                values["user_heating_kwh"] = (
                    values["nominal_cycle_user_heating_kwh"] - ROBUST_HEAT_MARGIN_KWH
                )
        else:
            values["preparation_duration_minutes"] = np.nan
            values["preparation_heat_kwh"] = 0.0
            values["qprep_eligible"] = pd.Series(pd.NA, index=values.index, dtype="boolean")
            values["qprep_supported"] = pd.Series(pd.NA, index=values.index, dtype="boolean")
            values["user_heating_kwh"] = values["water_heating_kwh"] + values["defrost_heat_kwh"]
        cycle_heat = values["user_heating_kwh"] + recovery_heat
        values["heat_balance_eligible"] = np.isfinite(cycle_heat) & cycle_heat.gt(0)
        values["optimization_eligible"] = (
            values["optimization_eligible"].fillna(False)
            & values["qd_eligible"]
            & values["heat_balance_eligible"]
        )
        if algorithm in {"v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
            values["optimization_eligible"] &= values["qprep_eligible"]
        values["recovery_heat_kwh"] = recovery_heat
        values["model_protocol"] = {
            "v2": OFFLINE_OBSERVED_RULE_PROTOCOL,
            "v2.1": "offline_observed_preparation_and_rule_duration",
            "v2.2": "offline_observed_preparation_and_rule_duration_water_heat",
            "v2.3": "offline_observed_duration_water_heat_fixed_9min_recovery",
            "v2.4": "fixed_9min_to_fixed_9min_water_heat_cycle",
            "v2.5": "current_cycle_start_to_defrost_water_heat_cycle",
            "v2.6": "current_cycle_start_to_defrost_unit_heat_cycle",
            "v3": "robust_closed_cycle_lower_heat_loeo90",
        }[algorithm]
        curve, optimum = optimize_cycle_cop_cost(
            values,
            defrost_recovery_electricity_kwh=(
                values["defrost_electricity_kwh"]
                + values["recovery_electricity_kwh"]
                + (ROBUST_ELECTRICITY_MARGIN_KWH if algorithm == "v3" else 0.0)
            ),
            defrost_recovery_heat_kwh=values["recovery_heat_kwh"],
        )
        if algorithm in {"v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
            curve["model_supported"] = (
                curve["pe_supported"].fillna(False)
                & curve["qd_supported"].fillna(False)
                & curve["qprep_supported"].fillna(False)
            )
        if algorithm == "v3":
            curve["nominal_cycle_electricity_kwh"] = (
                curve["heating_electricity_kwh"] + curve["defrost_electricity_kwh"]
            )
            curve["nominal_inverse_cop"] = (
                curve["nominal_cycle_electricity_kwh"] / curve["nominal_cycle_user_heating_kwh"]
            )
            curve["robust_cycle_electricity_kwh"] = curve["cycle_electricity_kwh"]
            curve["robust_cycle_user_heating_kwh"] = curve["cycle_user_heating_kwh"]
    minimum = float(optimum["inverse_cop"])
    eligible = curve["optimization_eligible"].fillna(False)
    curve["relative_regret"] = (curve["inverse_cop"] / minimum - 1.0).where(eligible)
    curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
    curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
    if algorithm == "v3":
        curve["excess_electricity_kwh"] = (
            curve["robust_cycle_electricity_kwh"] - minimum * curve["robust_cycle_user_heating_kwh"]
        ).clip(lower=0)
    return curve, optimum


def build_cost_function_table(  # noqa: C901
    base: pd.DataFrame,
    points: pd.DataFrame,
    loader: DatasetLoader,
    algorithm: str,
) -> pd.DataFrame:
    """Build the auditable candidate table for selected v1/v2 test variants."""
    if "cycle_name" in points and points["cycle_name"].duplicated().any():
        raise ValueError("points must contain one row per cycle")
    if algorithm == "v2.6.7":
        from .ticket import build_v267_table

        result, artifacts = build_v267_table(points, loader)
        result.attrs.update(artifacts)
        return result
    if algorithm == "v2.6.6":
        from .identification import build_v266_table

        result, audit = build_v266_table(points, loader)
        result.attrs["cycle_audit"] = audit
        return result
    if algorithm == "v2.6.1":
        result = build_cost_function_table(base, points, loader, "v2.6")
        result["algorithm"] = algorithm
        return result
    if algorithm == "v2.6.2":
        from .iterations import close_stable_cycle

        result = build_cost_function_table(base, points, loader, "v2.6")
        return close_stable_cycle(result, RECOVERY)
    if algorithm == "v2.6.3":
        from .iterations import normalize_degradation

        result = build_cost_function_table(base, points, loader, "v2.6.2")
        return normalize_degradation(result)
    if algorithm == "v2.6.4":
        from .iterations import marginal_dinkelbach

        result = build_cost_function_table(base, points, loader, "v2.6.3")
        return marginal_dinkelbach(result)
    if algorithm == "v2.6.5":
        from .iterations import finalize_supported_basin, normalize_degradation

        result = build_cost_function_table(base, points, loader, "v2.6.2")
        result = normalize_degradation(_replace_observed_duration_qd(result))
        return finalize_supported_basin(result)
    if algorithm not in {"v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
        raise ValueError("unknown cost-function algorithm")
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
            if algorithm in {"v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
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
                if algorithm in {"v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v3"}:
                    kwargs["preparation_duration_minutes"] = _observed_preparation_duration_minutes(
                        loader, str(cycle_name)
                    )
                if algorithm == "v2.4":
                    candidates = _start_heating_after(
                        candidates,
                        loader,
                        str(cycle_name),
                        FIXED_RECOVERY_BOUNDARY_MINUTES,
                    )
                elif algorithm in {"v2.5", "v2.6", "v3"}:
                    candidates = _start_heating_after(candidates, loader, str(cycle_name), 0.0)
            curve, optimum = _apply_cost(candidates, algorithm, **kwargs)
            curve["valid"] = True
            curve["failure_reason"] = ""
            curve["t_star"] = optimum["candidate_time"]
            curve["minimum_location"] = optimum["minimum_location"]
            if "model_supported" in curve:
                curve["t_star_model_supported"] = bool(
                    curve.loc[
                        curve["candidate_time"].eq(optimum["candidate_time"]),
                        "model_supported",
                    ].iloc[0]
                )
            if algorithm == "v3":
                recommended, rule = _latest_supported_in_optimal_basin(
                    curve, pd.Timestamp(optimum["candidate_time"])
                )
                curve["recommended_time"] = recommended
                curve["recommended_rule"] = rule
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
    if algorithm == "v3":
        fallback = pd.to_datetime(result["recommended_time"], errors="coerce").isna() & result[
            "rb_status"
        ].eq("triggered")
        result.loc[fallback, "recommended_time"] = pd.to_datetime(
            result.loc[fallback, "t_RB"], errors="coerce"
        )
        result.loc[fallback, "recommended_rule"] = "rb_fallback"
    for column in (*STATE_COLUMNS, *COST_COLUMNS):
        if column not in result:
            result[column] = np.nan
    result = result.sort_values(["cycle_name", "candidate_time"], kind="stable").reset_index(
        drop=True
    )
    if result.duplicated(["cycle_name", "candidate_time"]).any():
        raise ValueError("cost function table has duplicate cycle/candidate rows")
    return result


def write_cost_function_csv(table: pd.DataFrame, output_root: Path, algorithm: str) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"cost_function_{algorithm}.csv"
    table.to_csv(path, index=False)
    return path
