"""V2.6.8 observational pre-action defrost-event outcome model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed

from .core import water_side_heating_kw
from .identification import _timestamp

ALPHAS = (0.1, 1.0, 10.0, 100.0)
STATIC_5 = (
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
)
PHYSICAL_STATIC_6 = (
    "ambient_temperature",
    "mean_water_temperature",
    "setpoint_outlet_difference",
    "ambient_coil_difference",
    "evaporating_pressure",
    "compressor_frequency",
)
DYNAMIC_8 = (*PHYSICAL_STATIC_6, "heating_elapsed_minutes", "evaporating_pressure_slope_5m")
MODEL_FEATURES = {
    "static_5": STATIC_5,
    "physical_static_6": PHYSICAL_STATIC_6,
    "dynamic_8": DYNAMIC_8,
}
RAW_COLUMNS = (
    "timestamp",
    "power_total",
    "compressor_power",
    "heating_capacity",
    "water_flow",
    "water_in_temperature",
    "water_out_temperature",
    "coil_temperature",
    "evaporating_pressure",
    "water_temperature_setpoint",
    "ambient_temperature",
    "compressor_frequency",
)
STAGES = ("prep", "D", "R")
QUALITY_COVERAGE = 0.95
MAXIMUM_GAP_SECONDS = 30.0
Q_MIN_KWH = 0.01
PHASE_INTERVAL_CONVENTION = "[start,end)"
INTEGRAL_SAMPLING_CONVENTION = (
    "raw_observations_in_[start,end);_trapezoids_between_adjacent_retained_samples;"
    "right_boundary_sample_excluded;last_left_observation_zero_order_hold_to_end;"
    "hold_limited_by_robust_observation_cadence"
)


def experiment_weights(groups: pd.Series) -> np.ndarray:
    """Give every experiment equal mass while preserving total sample weight."""
    values = groups.astype(str)
    counts = values.map(values.value_counts()).to_numpy(dtype=float)
    return len(values) / (values.nunique() * counts)


@dataclass
class RidgeOutcomeModel:
    features: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    ridge: Ridge
    alpha: float
    sample_weight_sum: float
    support_threshold: float
    training_z: np.ndarray
    training_groups: np.ndarray

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(values[list(self.features)])
        return np.asarray(self.scaler.transform(imputed), dtype=float)

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ridge.predict(self.transform(values)))

    def support_distance(self, values: pd.DataFrame) -> np.ndarray:
        z = self.transform(values)
        return np.sqrt(np.square(z[:, None, :] - self.training_z[None, :, :]).sum(axis=2)).min(
            axis=1
        )


def _cross_experiment_support_threshold(z: np.ndarray, groups: np.ndarray) -> float:
    distances = np.sqrt(np.square(z[:, None, :] - z[None, :, :]).sum(axis=2))
    different = groups[:, None] != groups[None, :]
    nearest = np.where(different, distances, np.inf).min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.quantile(finite, 0.95)) if finite.size else float("inf")


def fit_weighted_ridge(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    *,
    alpha: float,
) -> RidgeOutcomeModel:
    """Fit median-imputed, weighted-standardized Ridge without metadata routing."""
    selected = frame.loc[frame[target].notna()].copy()
    if selected.empty:
        raise ValueError("outcome model has no complete training targets")
    weights = experiment_weights(selected["experiment_id"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_imputed = imputer.fit_transform(selected[list(features)])
    scaler = StandardScaler().fit(x_imputed, sample_weight=weights)
    z = np.asarray(scaler.transform(x_imputed), dtype=float)
    y = selected[target].to_numpy(dtype=float)
    ridge = Ridge(alpha=alpha).fit(z, y, sample_weight=weights)
    groups = selected["experiment_id"].astype(str).to_numpy()
    return RidgeOutcomeModel(
        features=features,
        imputer=imputer,
        scaler=scaler,
        ridge=ridge,
        alpha=float(alpha),
        sample_weight_sum=float(weights.sum()),
        support_threshold=_cross_experiment_support_threshold(z, groups),
        training_z=z,
        training_groups=groups,
    )


def _inner_macro_mse(
    frame: pd.DataFrame, features: tuple[str, ...], target: str, alpha: float
) -> float:
    losses: list[float] = []
    for heldout in frame["experiment_id"].dropna().astype(str).unique():
        train = frame.loc[~frame["experiment_id"].astype(str).eq(heldout)]
        test = frame.loc[frame["experiment_id"].astype(str).eq(heldout)]
        if train["experiment_id"].nunique() < 2 or test.empty:
            continue
        model = fit_weighted_ridge(train, features, target, alpha=alpha)
        prediction = model.predict(test[list(features)])
        losses.append(float(np.mean(np.square(test[target].to_numpy(dtype=float) - prediction))))
    return float(np.mean(losses)) if losses else float("inf")


def fit_outcome_fold(
    events: pd.DataFrame,
    heldout_experiment: str,
    features: tuple[str, ...],
    target: str,
) -> RidgeOutcomeModel:
    """Select alpha and fit using only experiments other than the held-out group."""
    train = events.loc[
        ~events["experiment_id"].astype(str).eq(str(heldout_experiment)) & events[target].notna()
    ].copy()
    scores = {alpha: _inner_macro_mse(train, features, target, alpha) for alpha in ALPHAS}
    alpha = min(ALPHAS, key=lambda value: (scores[value], value))
    return fit_weighted_ridge(train, features, target, alpha=alpha)


def _sorted_time_slice(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    end_inclusive: bool = True,
) -> pd.DataFrame:
    """Slice a timestamp-sorted frame by binary search instead of scanning it."""
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    left = int(timestamps.searchsorted(start, side="left"))
    right = int(timestamps.searchsorted(end, side="right" if end_inclusive else "left"))
    return frame.iloc[left:right]


def _robust_observation_cadence_seconds(
    timestamps: pd.Series | pd.DatetimeIndex,
) -> float:
    """Estimate the sample cadence from positive, non-gap timestamp steps."""
    parsed = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce")).dropna()
    parsed = parsed.sort_values().drop_duplicates()
    if len(parsed) < 2:
        return float("nan")
    gaps = np.diff(parsed.asi8).astype(float) / 1_000_000_000
    valid = np.isfinite(gaps) & (gaps > 0) & (gaps <= MAXIMUM_GAP_SECONDS)
    return float(np.median(gaps[valid])) if valid.any() else float("nan")


def _window_audit(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str
) -> dict[str, float | bool]:
    # Select a half-open interval.  The exact right-boundary sample is not used
    # as a quadrature endpoint because it may already belong to the next phase.
    values = _sorted_time_slice(frame, start, end, end_inclusive=False).copy()
    values = values.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    signal = water_side_heating_kw(values) if column == "water_heat" else values[column]
    observed = pd.DataFrame(
        {"timestamp": pd.to_datetime(values["timestamp"], errors="coerce"), "value": signal}
    ).dropna()
    observed = observed.sort_values("timestamp", kind="stable").drop_duplicates("timestamp")
    gaps = observed["timestamp"].diff().dt.total_seconds().dropna()
    valid_gaps = gaps.gt(0) & gaps.le(MAXIMUM_GAP_SECONDS)
    increments = (
        (observed["value"] + observed["value"].shift()) / 2
        * gaps
        / 3600
    ).where(valid_gaps, 0.0)
    cadence = _robust_observation_cadence_seconds(observed["timestamp"])
    last = observed.iloc[-1] if not observed.empty else None
    trailing_seconds = (
        max((end - pd.Timestamp(last["timestamp"])).total_seconds(), 0.0)
        if last is not None
        else 0.0
    )
    hold_seconds = (
        min(trailing_seconds, cadence)
        if np.isfinite(cadence) and cadence > 0
        else 0.0
    )
    energy = (
        float(increments.sum() + last["value"] * hold_seconds / 3600)
        if last is not None
        else 0.0
    )
    valid_seconds = float(gaps.where(valid_gaps, 0.0).sum() + hold_seconds)
    duration = max((end - start).total_seconds(), 0.0)
    first = observed["timestamp"].min() if not observed.empty else pd.NaT
    last = observed["timestamp"].max() if not observed.empty else pd.NaT
    start_fresh = pd.notna(first) and abs((first - start).total_seconds()) <= MAXIMUM_GAP_SECONDS
    end_fresh = pd.notna(last) and abs((end - last).total_seconds()) <= MAXIMUM_GAP_SECONDS
    maximum_gap = (
        float(max(gaps.max(), trailing_seconds))
        if not gaps.empty
        else float(trailing_seconds) if pd.notna(last) else float("inf")
    )
    coverage = float(valid_seconds / duration) if duration > 0 else 0.0
    return {
        "energy": float(energy),
        "coverage": min(coverage, 1.0),
        "maximum_gap_seconds": maximum_gap,
        "integral_sampling_convention": INTEGRAL_SAMPLING_CONVENTION,
        "start_fresh": bool(start_fresh),
        "end_fresh": bool(end_fresh),
        "valid": bool(
            np.isfinite(energy)
            and coverage >= QUALITY_COVERAGE
            and maximum_gap <= MAXIMUM_GAP_SECONDS
            and start_fresh
            and end_fresh
        ),
    }


def _candidate_integral_table(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    candidates: pd.DatetimeIndex | list[pd.Timestamp],
    column: str,
) -> pd.DataFrame:
    """Vectorize the same raw, gap-aware integral audit over many end times."""
    ends = pd.DatetimeIndex(candidates)
    values = _sorted_time_slice(frame, start, ends.max(), end_inclusive=False).copy()
    signal = water_side_heating_kw(values) if column == "water_heat" else values[column]
    observed = (
        pd.DataFrame(
            {"timestamp": pd.to_datetime(values["timestamp"], errors="coerce"), "value": signal}
        )
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
    )
    if observed.empty:
        return pd.DataFrame(
            {
                "energy": 0.0,
                "coverage": 0.0,
                "maximum_gap_seconds": np.inf,
                "integral_sampling_convention": INTEGRAL_SAMPLING_CONVENTION,
                "start_fresh": False,
                "end_fresh": False,
                "valid": False,
            },
            index=range(len(ends)),
        )
    times = pd.DatetimeIndex(observed["timestamp"])
    dt = observed["timestamp"].diff().dt.total_seconds().to_numpy(dtype=float)
    powers = observed["value"].to_numpy(dtype=float)
    short = np.isfinite(dt) & (dt > 0) & (dt <= MAXIMUM_GAP_SECONDS)
    increments = np.where(short, (powers + np.r_[np.nan, powers[:-1]]) / 2 * dt / 3600, 0.0)
    energy = np.cumsum(increments)
    covered = np.cumsum(np.where(short, dt, 0.0))
    maximum_gap = np.r_[0.0, np.maximum.accumulate(dt[1:])]
    # Each candidate is the open right edge of its own interval.  ``left``
    # excludes an observation exactly at that candidate, even when the same
    # raw timestamp is retained for a later candidate in this vectorized pass.
    positions = times.searchsorted(ends, side="left") - 1
    safe = np.maximum(positions, 0)
    duration = (ends - start).total_seconds().to_numpy(dtype=float)
    trailing = np.where(
        positions >= 0,
        np.maximum((ends - times[safe]).total_seconds().to_numpy(dtype=float), 0.0),
        0.0,
    )
    # Estimate cadence only from observations available to each candidate.
    # Using the complete vector here would let a future timestamp establish a
    # hold for an early candidate that has only one left-side observation.
    cadence_by_sample = pd.Series(np.where(short, dt, np.nan)).expanding().median().to_numpy()
    cadence = np.where(positions >= 0, cadence_by_sample[safe], np.nan)
    hold = np.minimum(trailing, np.where(np.isfinite(cadence), cadence, 0.0))
    energy_at_end = np.where(positions >= 0, energy[safe] + powers[safe] * hold / 3600, 0.0)
    covered_at_end = np.where(positions >= 0, covered[safe] + hold, 0.0)
    coverage = np.divide(
        covered_at_end,
        duration,
        out=np.zeros_like(covered_at_end),
        where=duration > 0,
    )
    start_fresh = np.full(
        len(ends), abs((times[0] - start).total_seconds()) <= MAXIMUM_GAP_SECONDS
    ) & (positions >= 0)
    end_fresh = (positions >= 0) & (
        np.abs((ends - times[safe]).total_seconds().to_numpy(dtype=float)) <= MAXIMUM_GAP_SECONDS
    )
    gap_at_end = np.where(positions >= 0, np.maximum(maximum_gap[safe], trailing), np.inf)
    valid = (
        np.isfinite(energy_at_end)
        & (coverage >= QUALITY_COVERAGE)
        & (gap_at_end <= MAXIMUM_GAP_SECONDS)
        & start_fresh
        & end_fresh
    )
    return pd.DataFrame(
        {
            "energy": energy_at_end,
            "coverage": np.minimum(coverage, 1.0),
            "maximum_gap_seconds": gap_at_end,
            "integral_sampling_convention": INTEGRAL_SAMPLING_CONVENTION,
            "start_fresh": start_fresh,
            "end_fresh": end_fresh,
            "valid": valid,
        }
    )


def event_outcomes(
    current: pd.DataFrame,
    recovery: pd.DataFrame,
    *,
    preparation_start: pd.Timestamp,
    defrost_start: pd.Timestamp,
    defrost_end: pd.Timestamp,
    recovery_end: pd.Timestamp,
) -> dict[str, float | bool]:
    """Integrate preparation, defrost, and recovery electricity and signed water heat."""
    frames = {
        "prep": (current, preparation_start, defrost_start),
        "D": (current, defrost_start, defrost_end),
        "R": (recovery, defrost_end, recovery_end),
    }
    phase_partition_valid = bool(
        preparation_start < defrost_start < defrost_end < recovery_end
    )
    result: dict[str, float | bool] = {}
    for phase, (frame, start, end) in frames.items():
        electricity = _window_audit(frame, start, end, "power_total")
        compressor = _window_audit(frame, start, end, "compressor_power")
        heat = _window_audit(frame, start, end, "water_heat")
        result[f"E_{phase}_kwh"] = float(electricity["energy"])
        result[f"Q_{phase}_kwh"] = float(heat["energy"])
        result[f"E_{phase}_coverage"] = float(electricity["coverage"])
        result[f"Q_{phase}_coverage"] = float(heat["coverage"])
        result[f"E_{phase}_maximum_gap_seconds"] = float(electricity["maximum_gap_seconds"])
        result[f"Q_{phase}_maximum_gap_seconds"] = float(heat["maximum_gap_seconds"])
        result[f"E_{phase}_start_fresh"] = bool(electricity["start_fresh"])
        result[f"E_{phase}_end_fresh"] = bool(electricity["end_fresh"])
        result[f"Q_{phase}_start_fresh"] = bool(heat["start_fresh"])
        result[f"Q_{phase}_end_fresh"] = bool(heat["end_fresh"])
        result[f"E_{phase}_valid"] = bool(electricity["valid"])
        result[f"Q_{phase}_valid"] = bool(heat["valid"])
        result[f"E_comp_{phase}_kwh"] = float(compressor["energy"])
        result[f"E_comp_{phase}_coverage"] = float(compressor["coverage"])
        result[f"E_comp_{phase}_maximum_gap_seconds"] = float(
            compressor["maximum_gap_seconds"]
        )
        result[f"E_comp_{phase}_start_fresh"] = bool(compressor["start_fresh"])
        result[f"E_comp_{phase}_end_fresh"] = bool(compressor["end_fresh"])
        result[f"E_comp_{phase}_valid"] = bool(compressor["valid"])
    result["E_T_kwh"] = sum(result[f"E_{phase}_kwh"] for phase in frames)
    result["Q_T_kwh"] = sum(result[f"Q_{phase}_kwh"] for phase in frames)
    result["E_comp_T_kwh"] = sum(result[f"E_comp_{phase}_kwh"] for phase in frames)
    result["event_duration_minutes"] = (recovery_end - preparation_start).total_seconds() / 60
    result["phase_partition_valid"] = phase_partition_valid
    result["phase_interval_convention"] = PHASE_INTERVAL_CONVENTION
    result["integral_sampling_convention"] = INTEGRAL_SAMPLING_CONVENTION
    result["compressor_event_valid"] = all(
        bool(result[f"E_comp_{phase}_valid"]) for phase in frames
    )
    result["event_valid"] = all(
        bool(result[f"{quantity}_{phase}_valid"]) for quantity in ("E", "Q") for phase in frames
    ) and phase_partition_valid
    return result


def _catalog(loader: Any) -> pd.DataFrame:
    values = loader.list_cycles().copy()
    for column in (
        "start_time",
        "heating_start",
        "stable_heating_start",
        "defrost_preparation_start",
        "defrost_start",
        "defrost_end",
    ):
        if column in values:
            values[column] = pd.to_datetime(values[column], errors="coerce")
    return values.sort_values(["experiment_id", "start_time"], kind="stable").reset_index(drop=True)


def _load_frame(loader: Any, cycle_name: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cycle_name in cache:
        return cache[cycle_name]
    frame = loader.load_cycle_original(cycle_name, columns=list(RAW_COLUMNS)).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in frame.columns.drop("timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .drop_duplicates("timestamp")
    )
    cache[cycle_name] = frame
    return frame


def _pre_action_feature_table(
    frame: pd.DataFrame,
    candidates: pd.DatetimeIndex | list[pd.Timestamp],
    heating_start: pd.Timestamp,
) -> pd.DataFrame:
    """Extract all strictly pre-action candidate features from sorted arrays."""
    raw_names = (
        "water_flow",
        "power_total",
        "compressor_power",
        "heating_capacity",
        "water_in_temperature",
        "water_out_temperature",
        "coil_temperature",
        "evaporating_pressure",
        "water_temperature_setpoint",
        "ambient_temperature",
        "compressor_frequency",
    )
    timestamps = pd.DatetimeIndex(frame["timestamp"])
    timestamp_ns = timestamps.view("i8")
    timestamp_seconds = timestamp_ns // 1_000_000_000
    arrays = {
        name: pd.to_numeric(
            frame[name] if name in frame else pd.Series(np.nan, index=frame.index),
            errors="coerce",
        ).to_numpy(dtype=float)
        for name in raw_names
    }
    rows: list[dict[str, float | bool]] = []
    for candidate in pd.DatetimeIndex(candidates):
        candidate_ns = candidate.value
        right = int(np.searchsorted(timestamp_ns, candidate_ns, side="left"))
        left_60 = int(np.searchsorted(timestamp_ns, candidate_ns - 60_000_000_000, side="left"))
        left_5m = int(np.searchsorted(timestamp_ns, candidate_ns - 300_000_000_000, side="left"))
        result: dict[str, float | bool] = {}
        counts: dict[str, int] = {}
        for name in raw_names:
            values = arrays[name][left_60:right]
            valid = np.isfinite(values)
            result[name] = float(np.median(values[valid])) if valid.any() else np.nan
            counts[name] = int(np.unique(timestamp_seconds[left_60:right][valid]).size)
            result[f"{name}_valid_second_count"] = counts[name]
        tin = float(result["water_in_temperature"])
        tout = float(result["water_out_temperature"])
        ambient = float(result["ambient_temperature"])
        coil = float(result["coil_temperature"])
        setpoint = float(result["water_temperature_setpoint"])
        result["mean_water_temperature"] = (tin + tout) / 2
        result["setpoint_outlet_difference"] = setpoint - tout
        result["ambient_coil_difference"] = ambient - coil
        result["heating_elapsed_minutes"] = (candidate - heating_start).total_seconds() / 60

        pressure = arrays["evaporating_pressure"][left_5m:right]
        trend_valid = np.isfinite(pressure)
        trend_time = timestamp_ns[left_5m:right][trend_valid]
        trend_pressure = pressure[trend_valid]
        trend_seconds = int(np.unique(timestamp_seconds[left_5m:right][trend_valid]).size)
        slope = np.nan
        if trend_seconds >= 240 and trend_time.size >= 2:
            elapsed = (trend_time - trend_time[0]) / 60_000_000_000
            centered_time = elapsed - elapsed.mean()
            denominator = float(np.square(centered_time).sum())
            if denominator > 0:
                slope = float(
                    (centered_time * (trend_pressure - trend_pressure.mean())).sum() / denominator
                )
        result["evaporating_pressure_slope_5m"] = slope
        result["evaporating_pressure_slope_valid_second_count"] = trend_seconds
        result["static_5_complete"] = bool(np.isfinite([result[name] for name in STATIC_5]).all())
        result["physical_static_6_complete"] = bool(
            np.isfinite([result[name] for name in PHYSICAL_STATIC_6]).all()
        )
        result["dynamic_8_complete"] = bool(np.isfinite([result[name] for name in DYNAMIC_8]).all())
        result["pre_action_window_valid"] = bool(
            all(
                counts[name] >= 48
                for name in (
                    "water_in_temperature",
                    "water_out_temperature",
                    "coil_temperature",
                    "evaporating_pressure",
                    "water_temperature_setpoint",
                    "ambient_temperature",
                    "compressor_frequency",
                )
            )
            and trend_seconds >= 240
            and result["dynamic_8_complete"]
        )
        rows.append(result)
    return pd.DataFrame(rows)


def _boundary_state(frame: pd.DataFrame, boundary: pd.Timestamp) -> dict[str, float]:
    window = _sorted_time_slice(
        frame,
        boundary,
        boundary + pd.Timedelta(seconds=60),
        end_inclusive=False,
    ).copy()
    result = {
        name: float(pd.to_numeric(window[name], errors="coerce").median())
        for name in (
            "water_in_temperature",
            "water_out_temperature",
            "evaporating_pressure",
            "compressor_frequency",
        )
    }
    result["water_heat"] = float(water_side_heating_kw(window).median())
    return result


def _setpoint_duration(frame: pd.DataFrame, boundary: pd.Timestamp) -> float:
    values = _sorted_time_slice(
        frame,
        boundary,
        boundary + pd.Timedelta(seconds=60),
        end_inclusive=False,
    )["water_temperature_setpoint"]
    setpoint = pd.to_numeric(values, errors="coerce").median()
    return 13.0 if pd.notna(setpoint) and setpoint >= 52.5 else 9.0


def _reference_recovery_stable() -> dict[str, pd.Timestamp]:
    path = Path("output/test/成本函数/其他/经验经济窗口/证据/recovery_events.csv")
    if not path.exists():
        return {}
    values = pd.read_csv(path, usecols=["cycle_name", "recovery_stable"])
    values["recovery_stable"] = pd.to_datetime(values["recovery_stable"], errors="coerce")
    return (
        values.dropna(subset=["recovery_stable"])
        .set_index("cycle_name")["recovery_stable"]
        .to_dict()
    )


def build_event_table(loader: Any) -> pd.DataFrame:  # noqa: C901
    """Build observed complete defrost-event outcomes on fixed and sensitivity boundaries."""
    catalog = _catalog(loader)
    cache: dict[str, pd.DataFrame] = {}
    knee_reference = _reference_recovery_stable()
    next_cycle: dict[str, pd.Series] = {}
    for _, experiment in catalog.groupby("experiment_id", sort=False):
        ordered = experiment.sort_values("start_time", kind="stable")
        for index in range(len(ordered) - 1):
            next_cycle[str(ordered.iloc[index]["cycle_name"])] = ordered.iloc[index + 1]

    rows: list[dict[str, object]] = []
    actual_events = catalog.loc[catalog["defrost_preparation_start"].notna()]
    for _, record in actual_events.iterrows():
        name = str(record["cycle_name"])
        missing = [
            column
            for column in ("heating_start", "defrost_start", "defrost_end")
            if _timestamp(record.get(column)) is None
        ]
        if missing:
            rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": str(record["experiment_id"]),
                    "experiment_date": record.get("experiment_date"),
                    "defrost_preparation_start": _timestamp(
                        record.get("defrost_preparation_start")
                    ),
                    "event_valid": False,
                    "J_w_valid": False,
                    "event_invalid_reason": ";".join(f"missing_{column}" for column in missing),
                }
            )
            continue
        following = next_cycle.get(name)
        if following is None:
            rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": str(record["experiment_id"]),
                    "experiment_date": record.get("experiment_date"),
                    "event_valid": False,
                    "J_w_valid": False,
                    "event_invalid_reason": "missing_following_cycle",
                }
            )
            continue
        heating_start = _timestamp(record["heating_start"])
        preparation = _timestamp(record["defrost_preparation_start"])
        defrost = _timestamp(record["defrost_start"])
        defrost_end = _timestamp(record["defrost_end"])
        next_heating = _timestamp(following.get("heating_start"))
        if None in (heating_start, preparation, defrost, defrost_end, next_heating):
            rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": str(record["experiment_id"]),
                    "experiment_date": record.get("experiment_date"),
                    "event_valid": False,
                    "J_w_valid": False,
                    "event_invalid_reason": "missing_following_heating_start",
                }
            )
            continue
        assert heating_start is not None
        assert preparation is not None
        assert defrost is not None
        assert defrost_end is not None
        assert next_heating is not None
        adjacent_gap = abs((next_heating - defrost_end).total_seconds())
        if str(following["experiment_id"]) != str(record["experiment_id"]) or adjacent_gap > 60:
            reason = (
                "following_cycle_experiment_mismatch"
                if str(following["experiment_id"]) != str(record["experiment_id"])
                else "following_cycle_not_adjacent"
            )
            rows.append(
                {
                    "cycle_name": name,
                    "experiment_id": str(record["experiment_id"]),
                    "experiment_date": record.get("experiment_date"),
                    "event_valid": False,
                    "J_w_valid": False,
                    "event_invalid_reason": reason,
                }
            )
            continue
        current = _load_frame(loader, name, cache)
        recovery = _load_frame(loader, str(following["cycle_name"]), cache)
        fixed_end = defrost_end + pd.Timedelta(minutes=9)
        ts_minutes = _setpoint_duration(current, heating_start)
        ts_end = defrost_end + pd.Timedelta(minutes=ts_minutes)
        fixed = event_outcomes(
            current,
            recovery,
            preparation_start=preparation,
            defrost_start=defrost,
            defrost_end=defrost_end,
            recovery_end=fixed_end,
        )
        ts = event_outcomes(
            current,
            recovery,
            preparation_start=preparation,
            defrost_start=defrost,
            defrost_end=defrost_end,
            recovery_end=ts_end,
        )
        stable = heating_start + pd.Timedelta(minutes=9)
        heating_e = _window_audit(current, stable, preparation, "power_total")
        heating_q = _window_audit(current, stable, preparation, "water_heat")
        heating_unit = _window_audit(current, stable, preparation, "heating_capacity")
        features = _pre_action_feature_table(
            current, [preparation], heating_start
        ).iloc[0].to_dict()
        row: dict[str, object] = {
            "cycle_name": name,
            "next_cycle_name": str(following["cycle_name"]),
            "experiment_id": str(record["experiment_id"]),
            "experiment_date": record.get("experiment_date"),
            "heating_start": heating_start,
            "stable_start_fixed9": stable,
            "defrost_preparation_start": preparation,
            "defrost_start": defrost,
            "defrost_end": defrost_end,
            "recovery_end_fixed9": fixed_end,
            "recovery_duration_minutes": 9.0,
            "ts_recovery_duration_minutes": ts_minutes,
            "adjacent_gap_seconds": adjacent_gap,
            **features,
            **fixed,
            "E_T_observed_kwh": fixed["E_T_kwh"],
            "Q_T_observed_kwh": fixed["Q_T_kwh"],
            "E_T_ts_kwh": ts["E_T_kwh"],
            "Q_T_ts_kwh": ts["Q_T_kwh"],
            "ts_event_valid": ts["event_valid"],
            "E_PD_kwh": float(fixed["E_prep_kwh"]) + float(fixed["E_D_kwh"]),
            "Q_PD_kwh": float(fixed["Q_prep_kwh"]) + float(fixed["Q_D_kwh"]),
            "E_H_actual_kwh": heating_e["energy"],
            "Q_H_actual_kwh": heating_q["energy"],
            "Q_H_unit_actual_kwh": heating_unit["energy"],
            "heating_actual_valid": bool(heating_e["valid"] and heating_q["valid"]),
        }
        reference = knee_reference.get(name)
        row["fixed9_knee_error_minutes"] = (
            (fixed_end - reference).total_seconds() / 60 if reference is not None else np.nan
        )
        row["ts_knee_error_minutes"] = (
            (ts_end - reference).total_seconds() / 60 if reference is not None else np.nan
        )
        boundary_specs = {
            "fixed9": (stable, fixed_end),
            "ts": (
                heating_start + pd.Timedelta(minutes=ts_minutes),
                ts_end,
            ),
            "rr": (heating_start, next_heating),
        }
        for boundary_name, (left, right) in boundary_specs.items():
            left_state = _boundary_state(current, left)
            right_state = _boundary_state(recovery, right)
            for state_name in left_state:
                row[f"{boundary_name}_closure_{state_name}"] = (
                    right_state[state_name] - left_state[state_name]
                )
        event_valid = bool(fixed["event_valid"])
        row["event_valid"] = event_valid
        row["J_w_valid"] = bool(event_valid and row["heating_actual_valid"])
        invalid_reasons: list[str] = []
        if not bool(fixed["phase_partition_valid"]):
            invalid_reasons.append("phase_partition")
        if not event_valid:
            for quantity in ("E", "Q"):
                for phase in STAGES:
                    prefix = f"{quantity}_{phase}"
                    if float(row[f"{prefix}_coverage"]) < QUALITY_COVERAGE:
                        invalid_reasons.append(f"{prefix}_coverage")
                    if float(row[f"{prefix}_maximum_gap_seconds"]) > MAXIMUM_GAP_SECONDS:
                        invalid_reasons.append(f"{prefix}_continuous_gap")
                    if not bool(row[f"{prefix}_start_fresh"]):
                        invalid_reasons.append(f"{prefix}_start_boundary")
                    if not bool(row[f"{prefix}_end_fresh"]):
                        invalid_reasons.append(f"{prefix}_end_boundary")
        row["event_invalid_reason"] = ";".join(dict.fromkeys(invalid_reasons))
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        # Keep the interval and quadrature contract visible for valid and
        # rejected events alike; downstream CSV readers must not infer it from
        # the numeric columns.
        result["phase_interval_convention"] = PHASE_INTERVAL_CONVENTION
        result["integral_sampling_convention"] = INTEGRAL_SAMPLING_CONVENTION
    return result


def _prediction_support(
    model: RidgeOutcomeModel, values: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    complete = (
        values[list(model.features)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    distance = model.support_distance(values)
    return complete.to_numpy() & (distance <= model.support_threshold), distance


def _fit_fold_set(
    events: pd.DataFrame,
    features: tuple[str, ...],
    targets: tuple[str, ...],
    experiments: list[str] | None = None,
) -> dict[str, dict[str, RidgeOutcomeModel]]:
    result: dict[str, dict[str, RidgeOutcomeModel]] = {}
    heldouts = (
        events["experiment_id"].dropna().astype(str).unique().tolist()
        if experiments is None
        else experiments
    )
    for experiment in heldouts:
        result[experiment] = {
            target: fit_outcome_fold(events, experiment, features, target) for target in targets
        }
    return result


def _mean_prediction(events: pd.DataFrame, heldout: str, target: str) -> float:
    train = events.loc[~events["experiment_id"].astype(str).eq(heldout) & events[target].notna()]
    return float(train.groupby("experiment_id")[target].mean().mean())


def _model_provenance(model: RidgeOutcomeModel) -> str:
    return json.dumps(
        {
            "features": list(model.features),
            "alpha": model.alpha,
            "training_experiment_ids": sorted(set(model.training_groups.astype(str))),
            "training_event_count": len(model.training_groups),
            "support_threshold": model.support_threshold,
            "imputer_medians": np.asarray(model.imputer.statistics_, dtype=float).tolist(),
            "scaler_mean": np.asarray(model.scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(model.scaler.scale_, dtype=float).tolist(),
            "ridge_intercept": np.asarray(model.ridge.intercept_).tolist(),
            "ridge_coefficients": np.asarray(model.ridge.coef_).tolist(),
        },
        sort_keys=True,
    )


def build_validation_table(
    events: pd.DataFrame,
    fold_sets: dict[str, dict[str, dict[str, RidgeOutcomeModel]]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, RidgeOutcomeModel]]]]:
    """Return cross-fitted event predictions for the four prespecified model levels."""
    valid = events.loc[events["event_valid"].fillna(False)].copy()
    if fold_sets is None:
        fold_sets = {
            name: _fit_fold_set(valid, features, ("E_T_observed_kwh", "Q_T_observed_kwh"))
            for name, features in MODEL_FEATURES.items()
        }
    rows: list[dict[str, object]] = []
    for _, event in valid.iterrows():
        experiment = str(event["experiment_id"])
        for model_name in ("mean_baseline", *MODEL_FEATURES):
            row = event.to_dict()
            row["model_name"] = model_name
            if model_name == "mean_baseline":
                e_prediction = _mean_prediction(valid, experiment, "E_T_observed_kwh")
                q_prediction = _mean_prediction(valid, experiment, "Q_T_observed_kwh")
                training_ids = sorted(set(valid["experiment_id"].astype(str)) - {experiment})
                row.update(
                    {
                        "E_alpha": np.nan,
                        "Q_alpha": np.nan,
                        "E_support_distance": np.nan,
                        "Q_support_distance": np.nan,
                        "supported": True,
                        "model_training_experiment_ids": ",".join(training_ids),
                        "model_provenance": "experiment-balanced training mean",
                    }
                )
            else:
                fold = fold_sets[model_name][experiment]
                event_frame = pd.DataFrame([event])
                e_model = fold["E_T_observed_kwh"]
                q_model = fold["Q_T_observed_kwh"]
                e_prediction = float(e_model.predict(event_frame)[0])
                q_prediction = float(q_model.predict(event_frame)[0])
                e_supported, e_distance = _prediction_support(e_model, event_frame)
                q_supported, q_distance = _prediction_support(q_model, event_frame)
                training_ids = sorted(set(e_model.training_groups.astype(str)))
                row.update(
                    {
                        "E_alpha": e_model.alpha,
                        "Q_alpha": q_model.alpha,
                        "E_support_distance": float(e_distance[0]),
                        "Q_support_distance": float(q_distance[0]),
                        "supported": bool(e_supported[0] and q_supported[0]),
                        "model_training_experiment_ids": ",".join(training_ids),
                        "model_provenance": json.dumps(
                            {
                                "E_T": json.loads(_model_provenance(e_model)),
                                "Q_T": json.loads(_model_provenance(q_model)),
                            },
                            sort_keys=True,
                        ),
                    }
                )
            row["E_T_prediction_kwh"] = e_prediction
            row["Q_T_prediction_kwh"] = q_prediction
            row["E_T_residual_kwh"] = float(event["E_T_observed_kwh"]) - e_prediction
            row["Q_T_residual_kwh"] = float(event["Q_T_observed_kwh"]) - q_prediction
            observed_denominator = float(event["Q_H_actual_kwh"]) + float(event["Q_T_observed_kwh"])
            predicted_denominator = float(event["Q_H_actual_kwh"]) + q_prediction
            row["J_w_observed"] = (
                (float(event["E_H_actual_kwh"]) + float(event["E_T_observed_kwh"]))
                / observed_denominator
                if bool(event["J_w_valid"]) and observed_denominator > Q_MIN_KWH
                else np.nan
            )
            row["J_w_prediction"] = (
                (float(event["E_H_actual_kwh"]) + e_prediction) / predicted_denominator
                if predicted_denominator > Q_MIN_KWH
                else np.nan
            )
            row["J_w_residual"] = row["J_w_observed"] - row["J_w_prediction"]
            if model_name == "dynamic_8":
                train = valid.loc[~valid["experiment_id"].astype(str).eq(experiment)]
                event_frame = pd.DataFrame([event])
                e_model = fold_sets[model_name][experiment]["E_T_observed_kwh"]
                q_model = fold_sets[model_name][experiment]["Q_T_observed_kwh"]
                for quantity, model, phases in (
                    ("E", e_model, ("prep", "D", "R")),
                    ("Q", q_model, ("prep", "D", "R")),
                ):
                    for phase in phases:
                        target = f"{quantity}_{phase}_kwh"
                        stage_model = fit_weighted_ridge(
                            train,
                            DYNAMIC_8,
                            target,
                            alpha=model.alpha,
                        )
                        row[f"{quantity}_{phase}_prediction_kwh"] = float(
                            stage_model.predict(event_frame)[0]
                        )
            rows.append(row)
    validation = pd.DataFrame(rows)
    excluded = events.loc[~events["event_valid"].fillna(False)].copy()
    if not excluded.empty:
        excluded["model_name"] = "excluded_event"
        validation = pd.concat([validation, excluded], ignore_index=True, sort=False)
    return validation, fold_sets


def _raw_ratio(numerator: float, denominator: float) -> float:
    return (
        float(numerator / denominator)
        if np.isfinite([numerator, denominator]).all() and denominator != 0
        else np.nan
    )


def _copy_sensitivity_result(
    curve: pd.DataFrame, *, prefix: str, cost: str, supported: str, physical: str
) -> pd.DataFrame:
    view = curve.assign(
        J_model=curve[cost],
        supported=curve[supported],
        physical_valid=curve[physical],
    )
    finalized = finalize_v268_curve(view)
    result = curve.copy()
    for source, suffix in (
        ("diagnostic_minimum", "diagnostic_minimum"),
        ("optimization_eligible", "optimization_eligible"),
        ("basin_1pct_start", "basin_1pct_start"),
        ("basin_1pct_end", "basin_1pct_end"),
        ("basin_1pct_width_minutes", "basin_1pct_width_minutes"),
        ("basin_5pct_start", "basin_5pct_start"),
        ("basin_5pct_end", "basin_5pct_end"),
        ("basin_5pct_width_minutes", "basin_5pct_width_minutes"),
    ):
        result[f"{prefix}_{suffix}"] = finalized[source]
    return result


def _candidate_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    step_seconds: int = 60,
) -> list[pd.Timestamp]:
    if step_seconds <= 0:
        raise ValueError("candidate step must be positive")
    first = start + pd.Timedelta(minutes=1)
    if end < first:
        return []
    values = list(pd.date_range(first, end, freq=pd.Timedelta(seconds=step_seconds)))
    if not values or values[-1] != end:
        values.append(end)
    return values


def _candidate_model_predictions(
    states: pd.DataFrame,
    e_model: RidgeOutcomeModel,
    q_model: RidgeOutcomeModel,
) -> pd.DataFrame:
    e_prediction = e_model.predict(states)
    q_prediction = q_model.predict(states)
    e_supported, e_distance = _prediction_support(e_model, states)
    q_supported, q_distance = _prediction_support(q_model, states)
    return pd.DataFrame(
        {
            "E": e_prediction,
            "Q": q_prediction,
            "E_supported": e_supported,
            "Q_supported": q_supported,
            "supported": e_supported & q_supported,
            "E_distance": e_distance,
            "Q_distance": q_distance,
        }
    )


def _bootstrap_minima(  # noqa: C901
    table: pd.DataFrame,
    events: pd.DataFrame,
    folds: dict[str, dict[str, RidgeOutcomeModel]],
    *,
    replicates: int,
    seed: int = 268,
    n_jobs: int = 1,
    _replicate_specs: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    valid_events = events.loc[events["event_valid"].fillna(False)].copy()
    by_cycle = {name: values.copy() for name, values in table.groupby("cycle_name", sort=False)}
    specs = _replicate_specs or [
        (index, int(value))
        for index, value in enumerate(np.random.SeedSequence(seed).generate_state(replicates))
    ]
    workers = min(max(int(n_jobs), 1), len(specs)) if specs else 1
    if workers > 1:
        batches = [specs[index::workers] for index in range(workers)]
        with parallel_config(backend="loky", n_jobs=workers, inner_max_num_threads=1):
            pieces = Parallel()(delayed(_bootstrap_minima)(
                table,
                events,
                folds,
                replicates=len(batch),
                seed=seed,
                n_jobs=1,
                _replicate_specs=batch,
            ) for batch in batches)
        minima = {name: [] for name in by_cycle}
        for piece in pieces:
            for name, values in piece.attrs["minima"].items():
                minima[name].extend(values)
        return _bootstrap_minima_summary(by_cycle, minima, replicates)

    minima: dict[str, list[pd.Timestamp]] = {name: [] for name in by_cycle}
    for _, replicate_seed in specs:
        rng = np.random.default_rng(replicate_seed)
        for heldout, heldout_curves in table.groupby("experiment_id", sort=False):
            heldout = str(heldout)
            available = sorted(set(valid_events["experiment_id"].astype(str)) - {heldout})
            if len(available) < 2 or heldout not in folds:
                continue
            sampled = rng.choice(available, size=len(available), replace=True)
            parts = []
            for draw, experiment in enumerate(sampled):
                part = valid_events.loc[
                    valid_events["experiment_id"].astype(str).eq(str(experiment))
                ].copy()
                part["experiment_id"] = f"draw_{draw}"
                parts.append(part)
            training = pd.concat(parts, ignore_index=True)
            e_model = fit_weighted_ridge(
                training,
                DYNAMIC_8,
                "E_T_observed_kwh",
                alpha=folds[heldout]["E_T_observed_kwh"].alpha,
            )
            q_model = fit_weighted_ridge(
                training,
                DYNAMIC_8,
                "Q_T_observed_kwh",
                alpha=folds[heldout]["Q_T_observed_kwh"].alpha,
            )
            candidates = heldout_curves.copy()
            candidates["bootstrap_E_T"] = e_model.predict(candidates)
            candidates["bootstrap_Q_T"] = q_model.predict(candidates)
            candidates["bootstrap_J"] = (
                candidates["heating_electricity_kwh"] + candidates["bootstrap_E_T"]
            ) / (candidates["water_heating_kwh"] + candidates["bootstrap_Q_T"])
            candidates["bootstrap_physical"] = (
                candidates["heating_electricity_kwh"] + candidates["bootstrap_E_T"]
            ).gt(0) & (candidates["water_heating_kwh"] + candidates["bootstrap_Q_T"]).gt(Q_MIN_KWH)
            for cycle_name, curve in candidates.groupby("cycle_name", sort=False):
                base = (
                    curve["supported"].fillna(False)
                    & curve["pre_action_window_valid"].fillna(False)
                    & curve["measurement_eligible"].fillna(False)
                    & curve["bootstrap_physical"].fillna(False)
                    & curve["bootstrap_J"].notna()
                )
                eligible = _long_support_runs(curve["candidate_time"], base)
                if eligible.any():
                    position = curve.index[
                        eligible & curve["bootstrap_J"].eq(curve.loc[eligible, "bootstrap_J"].min())
                    ][0]
                    minima[str(cycle_name)].append(
                        pd.Timestamp(curve.loc[position, "candidate_time"])
                    )
    return _bootstrap_minima_summary(by_cycle, minima, replicates)


def _bootstrap_minima_summary(
    by_cycle: dict[str, pd.DataFrame],
    minima: dict[str, list[pd.Timestamp]],
    replicates: int,
) -> pd.DataFrame:
    rows = []
    for cycle_name, curve in by_cycle.items():
        values = minima[cycle_name]
        basin_start = pd.to_datetime(curve["basin_5pct_start"].iloc[0], errors="coerce")
        basin_end = pd.to_datetime(curve["basin_5pct_end"].iloc[0], errors="coerce")
        numeric = np.array([value.value for value in values], dtype=np.int64)
        rows.append(
            {
                "cycle_name": cycle_name,
                "experiment_id": curve["experiment_id"].iloc[0],
                "repeat_count": replicates,
                "valid_minimum_count": len(values),
                "valid_minimum_fraction": len(values) / replicates if replicates else np.nan,
                "argmin_median_time": (
                    pd.Timestamp(int(np.median(numeric))) if numeric.size else pd.NaT
                ),
                "argmin_q25_time": (
                    pd.Timestamp(int(np.quantile(numeric, 0.25))) if numeric.size else pd.NaT
                ),
                "argmin_q75_time": (
                    pd.Timestamp(int(np.quantile(numeric, 0.75))) if numeric.size else pd.NaT
                ),
                "argmin_in_original_5pct_basin_fraction": (
                    float(np.mean([(basin_start <= value <= basin_end) for value in values]))
                    if values and pd.notna(basin_start) and pd.notna(basin_end)
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["minima"] = minima
    return result


def build_v268_table(  # noqa: C901
    points: pd.DataFrame,
    loader: Any,
    *,
    bootstrap_replicates: int = 200,
    n_jobs: int = 1,
    candidate_step_seconds: int = 60,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build cross-fitted fixed-9-min water-side cost curves and audit tables."""
    events = build_event_table(loader)
    valid_events = events.loc[events["event_valid"].fillna(False)].copy()
    catalog = _catalog(loader)
    point_by_cycle = points.set_index("cycle_name", drop=False)
    experiments = sorted(
        set(catalog["experiment_id"].dropna().astype(str))
        | set(points["experiment_id"].dropna().astype(str))
    )
    fold_sets = {
        name: _fit_fold_set(
            valid_events,
            features,
            ("E_T_observed_kwh", "Q_T_observed_kwh"),
            experiments,
        )
        for name, features in MODEL_FEATURES.items()
    }
    validation, fold_sets = build_validation_table(events, fold_sets)
    ts_events = valid_events.loc[valid_events["ts_event_valid"].eq(True)]
    ts_folds = _fit_fold_set(ts_events, DYNAMIC_8, ("E_T_ts_kwh", "Q_T_ts_kwh"), experiments)
    rr_folds = _fit_fold_set(valid_events, DYNAMIC_8, ("E_PD_kwh", "Q_PD_kwh"), experiments)
    cache: dict[str, pd.DataFrame] = {}
    tables: list[pd.DataFrame] = []
    for _, record in catalog.iterrows():
        cycle_name = str(record["cycle_name"])
        if cycle_name not in point_by_cycle.index:
            continue
        point = point_by_cycle.loc[cycle_name]
        if isinstance(point, pd.DataFrame):
            point = point.iloc[0]
        if not bool(point.get("valid", False)):
            continue
        heating_start = _timestamp(record.get("heating_start"))
        preparation = _timestamp(
            point.get("candidate_end", point.get("t_actual_preparation"))
        ) or _timestamp(record.get("defrost_preparation_start"))
        if heating_start is None or preparation is None:
            continue
        stable = heating_start + pd.Timedelta(minutes=9)
        candidates = _candidate_times(
            stable,
            preparation,
            step_seconds=candidate_step_seconds,
        )
        if not candidates:
            continue
        experiment = str(record["experiment_id"])
        current = _load_frame(loader, cycle_name, cache)
        ts_minutes = _setpoint_duration(current, heating_start)
        ts_stable = heating_start + pd.Timedelta(minutes=ts_minutes)
        main_fold = fold_sets["dynamic_8"][experiment]
        ts_fold = ts_folds[experiment]
        rr_fold = rr_folds[experiment]
        candidate_index = pd.DatetimeIndex(candidates)
        feature_table = _pre_action_feature_table(current, candidate_index, heating_start)
        main_predictions = _candidate_model_predictions(
            feature_table,
            main_fold["E_T_observed_kwh"],
            main_fold["Q_T_observed_kwh"],
        )
        ts_predictions = _candidate_model_predictions(
            feature_table,
            ts_fold["E_T_ts_kwh"],
            ts_fold["Q_T_ts_kwh"],
        )
        rr_predictions = _candidate_model_predictions(
            feature_table,
            rr_fold["E_PD_kwh"],
            rr_fold["Q_PD_kwh"],
        )
        integrals = {
            "electricity": _candidate_integral_table(
                current, stable, candidate_index, "power_total"
            ),
            "water_heat": _candidate_integral_table(current, stable, candidate_index, "water_heat"),
            "unit_heat": _candidate_integral_table(
                current, stable, candidate_index, "heating_capacity"
            ),
            "compressor_electricity": _candidate_integral_table(
                current, stable, candidate_index, "compressor_power"
            ),
            "ts_electricity": _candidate_integral_table(
                current, ts_stable, candidate_index, "power_total"
            ),
            "ts_heat": _candidate_integral_table(current, ts_stable, candidate_index, "water_heat"),
            "rr_electricity": _candidate_integral_table(
                current, heating_start, candidate_index, "power_total"
            ),
            "rr_heat": _candidate_integral_table(
                current, heating_start, candidate_index, "water_heat"
            ),
        }
        rows: list[dict[str, object]] = []
        for candidate_position, candidate in enumerate(candidates):
            features = feature_table.iloc[candidate_position].to_dict()
            main_prediction = main_predictions.iloc[candidate_position]
            ts_prediction = ts_predictions.iloc[candidate_position]
            rr_prediction = rr_predictions.iloc[candidate_position]
            e_hat = float(main_prediction["E"])
            q_hat = float(main_prediction["Q"])
            supported = bool(main_prediction["supported"])
            e_distance = float(main_prediction["E_distance"])
            q_distance = float(main_prediction["Q_distance"])
            e_ts = float(ts_prediction["E"])
            q_ts = float(ts_prediction["Q"])
            ts_supported = bool(ts_prediction["supported"])
            e_pd = float(rr_prediction["E"])
            q_pd = float(rr_prediction["Q"])
            rr_supported = bool(rr_prediction["supported"])
            electricity = integrals["electricity"].iloc[candidate_position]
            water_heat = integrals["water_heat"].iloc[candidate_position]
            unit_heat = integrals["unit_heat"].iloc[candidate_position]
            compressor_electricity = integrals["compressor_electricity"].iloc[
                candidate_position
            ]
            ts_electricity = integrals["ts_electricity"].iloc[candidate_position]
            ts_heat = integrals["ts_heat"].iloc[candidate_position]
            rr_electricity = integrals["rr_electricity"].iloc[candidate_position]
            rr_heat = integrals["rr_heat"].iloc[candidate_position]
            e_total = float(electricity["energy"]) + e_hat
            q_total = float(water_heat["energy"]) + q_hat
            ts_e_total = float(ts_electricity["energy"]) + e_ts
            ts_q_total = float(ts_heat["energy"]) + q_ts
            rr_e_total = float(rr_electricity["energy"]) + e_pd
            rr_q_total = float(rr_heat["energy"]) + q_pd
            measurement = bool(electricity["valid"] and water_heat["valid"])
            ts_measurement = bool(ts_electricity["valid"] and ts_heat["valid"])
            rr_measurement = bool(rr_electricity["valid"] and rr_heat["valid"])
            rows.append(
                {
                    "cycle_name": cycle_name,
                    "experiment_id": experiment,
                    "candidate_time": candidate,
                    "cycle_start": heating_start,
                    "heating_start": heating_start,
                    "stable_start_fixed9": stable,
                    "ts_stable_start": ts_stable,
                    "candidate_end": preparation,
                    "actual_preparation_time": _timestamp(point.get("t_actual_preparation")),
                    "t_RB": _timestamp(point.get("t_RB")),
                    "rb_status": point.get("rb_status"),
                    **features,
                    "heating_electricity_kwh": electricity["energy"],
                    "water_heating_kwh": water_heat["energy"],
                    "unit_heating_kwh": unit_heat["energy"],
                    "heating_compressor_electricity_kwh": compressor_electricity["energy"],
                    "integration_coverage": min(
                        float(electricity["coverage"]), float(water_heat["coverage"])
                    ),
                    "max_gap_seconds": max(
                        float(electricity["maximum_gap_seconds"]),
                        float(water_heat["maximum_gap_seconds"]),
                    ),
                    "measurement_eligible": measurement,
                    "heating_electricity_measurement_eligible": bool(electricity["valid"]),
                    "water_heating_measurement_eligible": bool(water_heat["valid"]),
                    "heating_compressor_measurement_eligible": bool(
                        compressor_electricity["valid"]
                    ),
                    "rr_measurement_eligible": rr_measurement,
                    "E_T_hat_kwh": e_hat,
                    "Q_T_hat_kwh": q_hat,
                    "E_T_supported": bool(main_prediction["E_supported"]),
                    "Q_T_supported": bool(main_prediction["Q_supported"]),
                    "cycle_electricity_kwh": e_total,
                    "cycle_net_heat_kwh": q_total,
                    "E_support_distance": e_distance,
                    "Q_support_distance": q_distance,
                    "supported": supported,
                    "model_supported": supported,
                    "physical_valid": bool(e_total > 0 and q_total > Q_MIN_KWH),
                    "J_model": _raw_ratio(e_total, q_total),
                    "E_T_ts_hat_kwh": e_ts,
                    "Q_T_ts_hat_kwh": q_ts,
                    "ts_supported": bool(ts_supported and ts_measurement),
                    "ts_physical_valid": bool(ts_e_total > 0 and ts_q_total > Q_MIN_KWH),
                    "J_ts_model": _raw_ratio(ts_e_total, ts_q_total),
                    "E_PD_hat_kwh": e_pd,
                    "Q_PD_hat_kwh": q_pd,
                    "rr_heating_electricity_kwh": rr_electricity["energy"],
                    "rr_water_heating_kwh": rr_heat["energy"],
                    "rr_supported": bool(rr_supported and rr_measurement),
                    "rr_physical_valid": bool(rr_e_total > 0 and rr_q_total > Q_MIN_KWH),
                    "J_rr_model": _raw_ratio(rr_e_total, rr_q_total),
                    "model_training_experiment_ids": ",".join(
                        sorted(set(main_fold["E_T_observed_kwh"].training_groups.astype(str)))
                    ),
                    "E_alpha": main_fold["E_T_observed_kwh"].alpha,
                    "Q_alpha": main_fold["Q_T_observed_kwh"].alpha,
                    "model_provenance": json.dumps(
                        {
                            "E_T": json.loads(_model_provenance(main_fold["E_T_observed_kwh"])),
                            "Q_T": json.loads(_model_provenance(main_fold["Q_T_observed_kwh"])),
                        },
                        sort_keys=True,
                    ),
                    "algorithm": "v2.6.8",
                    "formula": "(E_H_water_boundary+E_T_hat)/(Q_H_water+Q_T_hat)",
                    "phase_interval_convention": PHASE_INTERVAL_CONVENTION,
                    "integral_sampling_convention": INTEGRAL_SAMPLING_CONVENTION,
                    "mixed_heat_basis": False,
                    "prediction_clipped": False,
                    "interpolated": False,
                    "endpoint_extrapolated": False,
                }
            )
        curve = finalize_v268_curve(pd.DataFrame(rows))
        curve = _copy_sensitivity_result(
            curve,
            prefix="ts",
            cost="J_ts_model",
            supported="ts_supported",
            physical="ts_physical_valid",
        )
        curve = _copy_sensitivity_result(
            curve,
            prefix="rr",
            cost="J_rr_model",
            supported="rr_supported",
            physical="rr_physical_valid",
        )
        eligible = curve["optimization_eligible"].fillna(False)
        minimum = pd.to_numeric(curve.loc[eligible, "J_model"], errors="coerce").min()
        curve["relative_regret"] = (
            pd.to_numeric(curve["J_model"], errors="coerce") / minimum - 1
        ).where(eligible)
        curve["near_optimal_1pct"] = eligible & curve["relative_regret"].le(0.01)
        curve["near_optimal_5pct"] = eligible & curve["relative_regret"].le(0.05)
        curve["cycle_cop"] = 1 / pd.to_numeric(curve["J_model"], errors="coerce")
        curve["cycle_status"] = np.where(
            curve["diagnostic_minimum"].notna(), "identified_curve", "model_support_limited"
        )
        curve["decision_status"] = "diagnostic_observational_v268"
        curve["t_star_semantics"] = "model_implied_diagnostic_minimum"
        curve["recommended_time"] = pd.NaT
        curve["hard_label_eligible"] = False
        tables.append(curve)
    if tables:
        columns = list(dict.fromkeys(column for frame in tables for column in frame.columns))
        table = pd.concat(
            [frame.dropna(axis=1, how="all") for frame in tables],
            ignore_index=True,
            sort=False,
        ).reindex(columns=columns)
    else:
        table = pd.DataFrame()
    bootstrap = _bootstrap_minima(
        table,
        events,
        fold_sets["dynamic_8"],
        replicates=bootstrap_replicates,
        n_jobs=n_jobs,
    )
    return table, {"validation": validation, "bootstrap": bootstrap, "events": events}


def _long_support_runs(times: pd.Series, selected: pd.Series) -> pd.Series:
    result = pd.Series(False, index=selected.index)
    chosen = np.flatnonzero(selected.to_numpy(dtype=bool))
    if not chosen.size:
        return result
    parsed = pd.to_datetime(times, errors="coerce")
    positive = parsed.sort_values().diff().dropna()
    typical = positive.median() if not positive.empty else pd.Timedelta(minutes=1)
    breaks = np.flatnonzero(
        (np.diff(chosen) != 1)
        | (
            parsed.iloc[chosen[1:]].to_numpy() - parsed.iloc[chosen[:-1]].to_numpy()
            > 1.5 * typical.to_timedelta64()
        )
    )
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(chosen) - 1]
    for left, right in zip(starts, ends, strict=True):
        indices = chosen[left : right + 1]
        if parsed.iloc[indices[-1]] - parsed.iloc[indices[0]] >= pd.Timedelta(minutes=5):
            result.iloc[indices] = True
    return result


def _connected_basin(
    values: pd.Series, eligible: pd.Series, optimum_index: int, fraction: float
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    threshold = float(values.iloc[optimum_index]) * (1 + fraction)
    within = eligible & values.le(threshold)
    left = right = optimum_index
    while left > 0 and bool(within.iloc[left - 1]):
        left -= 1
    while right + 1 < len(within) and bool(within.iloc[right + 1]):
        right += 1
    return left, right, float(right - left)


def finalize_v268_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Keep the full model curve and identify minima only on long supported runs."""
    result = curve.sort_values("candidate_time", kind="stable").reset_index(drop=True).copy()
    base = (
        result["supported"].fillna(False)
        & result["pre_action_window_valid"].fillna(False)
        & result.get("measurement_eligible", pd.Series(True, index=result.index)).fillna(False)
        & result["physical_valid"].fillna(False)
        & pd.to_numeric(result["J_model"], errors="coerce").notna()
    )
    result["continuous_support"] = _long_support_runs(result["candidate_time"], base)
    result["optimization_eligible"] = base & result["continuous_support"]
    result["diagnostic_minimum"] = pd.NaT
    result["raw_t_star"] = pd.NaT
    result["t_star"] = pd.NaT
    for percent in (1, 5):
        result[f"basin_{percent}pct_start"] = pd.NaT
        result[f"basin_{percent}pct_end"] = pd.NaT
        result[f"basin_{percent}pct_width_minutes"] = np.nan
    eligible = result["optimization_eligible"]
    if eligible.any():
        values = pd.to_numeric(result["J_model"], errors="coerce")
        minimum = values.loc[eligible].min()
        optimum_index = int(result.index[eligible & values.eq(minimum)][0])
        optimum = pd.Timestamp(result.loc[optimum_index, "candidate_time"])
        result[["diagnostic_minimum", "raw_t_star", "t_star"]] = optimum
        for percent in (1, 5):
            left, right, _ = _connected_basin(values, eligible, optimum_index, percent / 100)
            start = pd.Timestamp(result.loc[left, "candidate_time"])
            end = pd.Timestamp(result.loc[right, "candidate_time"])
            result[f"basin_{percent}pct_start"] = start
            result[f"basin_{percent}pct_end"] = end
            result[f"basin_{percent}pct_width_minutes"] = (end - start).total_seconds() / 60
    result["J"] = result["J_model"]
    result["inverse_cop"] = result["J_model"]
    result["valid"] = result["optimization_eligible"]
    result["t_star_model_supported"] = result["diagnostic_minimum"].notna()
    return result
