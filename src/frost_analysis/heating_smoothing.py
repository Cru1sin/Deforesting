"""Small, stage-bounded heating-capacity smoothers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt  # type: ignore[import-untyped]
from scipy.linalg import solve_banded
from scipy.optimize import isotonic_regression
from scipy.signal import butter, savgol_filter, sosfiltfilt

METHODS = (
    "median_centered_60s",
    "ewma_tau30s",
    "median30s_ewma30s",
    "savgol_70s",
    "adaptive_offline",
    "wavelet_offline",
    "wavelet_monotonic_offline",
    "nearly_isotonic_offline",
    "robust_monotone_offline",
)
OFFLINE_METHODS = (
    "median_centered_60s",
    "savgol_70s",
    "adaptive_offline",
    "wavelet_offline",
)
ONLINE_METHODS = ("ewma_tau30s", "median30s_ewma30s")
SHAPE_METHODS = (
    "wavelet_monotonic_offline",
    "nearly_isotonic_offline",
    "robust_monotone_offline",
)
_ADAPTIVE_MEDIAN_SECONDS = (30, 60, 90)
_ADAPTIVE_TAU_SECONDS = (20, 30, 45, 60, 90)


def water_heating_capacity(frame: pd.DataFrame) -> pd.Series:
    """Return the water-side audit signal in kW for flow expressed in m³/h."""
    return (
        1.161
        * pd.to_numeric(frame["water_flow"], errors="coerce")
        * (
            pd.to_numeric(frame["water_out_temperature"], errors="coerce")
            - pd.to_numeric(frame["water_in_temperature"], errors="coerce")
        )
    )


def smooth_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Append stage-bounded smoothers without filling target gaps."""
    required = {"timestamp", "cycle_stage", "heating_capacity"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"cycle frame is missing smoothing columns: {missing}")

    result = frame.copy()
    result["water_heating_capacity_audit"] = water_heating_capacity(result)
    times = pd.to_datetime(result["timestamp"], errors="coerce")
    values = pd.to_numeric(result["heating_capacity"], errors="coerce").to_numpy(
        dtype=float
    )
    outputs = {method: np.full(len(result), np.nan) for method in METHODS}
    median_windows = np.full(len(result), np.nan)
    lowpass_taus = np.full(len(result), np.nan)
    stages = result["cycle_stage"].astype("string")
    blocks = stages.ne(stages.shift()).fillna(True).cumsum()
    for indices in result.groupby(blocks, sort=False).indices.values():
        positions = np.asarray(indices, dtype=int)
        stage = str(stages.iloc[positions[0]])
        for run in _finite_runs(values[positions], times.iloc[positions].notna().to_numpy()):
            selected = positions[run]
            median_window, lowpass_tau = _smooth_run(
                values[selected],
                times.iloc[selected],
                pd.to_numeric(
                    result["water_heating_capacity_audit"].iloc[selected], errors="coerce"
                ).to_numpy(dtype=float),
                outputs,
                selected,
            )
            median_windows[selected] = median_window
            lowpass_taus[selected] = lowpass_tau
        wavelet_positions = positions[np.isfinite(outputs["wavelet_offline"][positions])]
        if stage == "frost_development":
            outputs["wavelet_monotonic_offline"][wavelet_positions] = np.asarray(
                isotonic_regression(
                    outputs["wavelet_offline"][wavelet_positions], increasing=False
                ).x,
                dtype=float,
            )
            shape_values = values[wavelet_positions]
            outputs["nearly_isotonic_offline"][wavelet_positions] = _nearly_isotonic(
                shape_values
            )
            outputs["robust_monotone_offline"][wavelet_positions] = _robust_monotone(
                shape_values
            )
        else:
            for method in SHAPE_METHODS:
                outputs[method][positions] = outputs["wavelet_offline"][positions]
    for method, output in outputs.items():
        result[method] = output
    result["adaptive_median_window_seconds"] = median_windows
    result["adaptive_lowpass_tau_seconds"] = lowpass_taus
    return result


def score_methods(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure noise suppression and distortion once per stage and method."""
    required = {"timestamp", "cycle_stage", "heating_capacity", *METHODS}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"smoothed frame is missing score columns: {missing}")
    cycle_name = (
        str(frame["cycle_name"].iloc[0]) if "cycle_name" in frame and len(frame) else "unknown"
    )
    rows: list[dict[str, object]] = []
    for stage, stage_frame in frame.groupby("cycle_stage", sort=False, dropna=False):
        raw = pd.to_numeric(stage_frame["heating_capacity"], errors="coerce").to_numpy(
            dtype=float
        )
        times = pd.to_datetime(stage_frame["timestamp"], errors="coerce")
        water = (
            pd.to_numeric(stage_frame["water_heating_capacity_audit"], errors="coerce")
            .to_numpy(dtype=float)
            if "water_heating_capacity_audit" in stage_frame
            else water_heating_capacity(stage_frame).to_numpy(dtype=float)
        )
        raw_steps = _steps(raw)
        raw_step_mad = _mad(raw_steps)
        raw_p90 = _percentile_abs(raw_steps, 90)
        raw_p99 = _percentile_abs(raw_steps, 99)
        raw_energy = _integral(raw, times)
        finite_raw = raw[np.isfinite(raw)]
        shortfall_reference = (
            float(np.median(finite_raw[: min(6, len(finite_raw))]))
            if len(finite_raw)
            else np.nan
        )
        raw_shortfall = _integral(np.maximum(shortfall_reference - raw, 0.0), times)
        for method in METHODS:
            values = pd.to_numeric(stage_frame[method], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(raw) & np.isfinite(values)
            steps = _steps(values)
            step_mad = _mad(steps)
            noise_ratio = _ratio(step_mad, raw_step_mad)
            spike_ratio = _ratio(_percentile_abs(steps, 99), raw_p99)
            energy = _integral(values, times)
            water_valid = valid & np.isfinite(water)
            water_error = values[water_valid] - water[water_valid]
            water_bias = float(np.median(water_error)) if len(water_error) else np.nan
            water_rmse = (
                float(np.sqrt(np.mean((water_error - water_bias) ** 2)))
                if len(water_error)
                else np.nan
            )
            rows.append(
                {
                    "cycle_name": cycle_name,
                    "cycle_stage": str(stage),
                    "method": method,
                    "mode": (
                        "offline"
                        if method in OFFLINE_METHODS
                        else "online"
                        if method in ONLINE_METHODS
                        else "constrained"
                    ),
                    "n": int(valid.sum()),
                    "metric_status": "available" if valid.sum() >= 4 else "insufficient_data",
                    "noise_ratio": noise_ratio,
                    "noise_reduction": 1.0 - noise_ratio if np.isfinite(noise_ratio) else np.nan,
                    "spike_ratio": spike_ratio,
                    "spike_reduction": (
                        1.0 - spike_ratio if np.isfinite(spike_ratio) else np.nan
                    ),
                    "water_bias_kw": water_bias,
                    "water_rmse_offset_kw": water_rmse,
                    "energy_error_pct": 100.0 * _relative_error(energy, raw_energy),
                    "shortfall_area_error_pct": 100.0
                    * _relative_error(
                        _integral(np.maximum(shortfall_reference - values, 0.0), times),
                        raw_shortfall,
                    ),
                    "transient_retention": _ratio(
                        _percentile_abs(steps, 90), raw_p90
                    ),
                    "lag_seconds": _lag_seconds(raw, values, times),
                }
            )
    return pd.DataFrame(rows)


def recommend_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one offline and one causal method per cycle stage by rank sum."""
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["cycle_name", "cycle_stage"]
    for (cycle_name, stage), group in metrics.groupby(keys, sort=False, dropna=False):
        row: dict[str, object] = {"cycle_name": cycle_name, "cycle_stage": stage}
        for mode, methods in (("offline", OFFLINE_METHODS), ("online", ONLINE_METHODS)):
            selected, score = _best_method(group.loc[group["method"].isin(methods)])
            row[f"{mode}_method"] = selected
            row[f"{mode}_rank_sum"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def global_method_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank one offline method across all valid frost-development cycles."""
    available = metrics.loc[
        metrics["metric_status"].eq("available")
        & metrics["cycle_stage"].eq("frost_development")
        & metrics["method"].isin(OFFLINE_METHODS)
    ].copy()
    if available.empty:
        return pd.DataFrame(columns=["method", "cycles", "global_mean_rank_sum"])
    available["transient_error"] = (available["transient_retention"] - 1.0).abs()
    available["absolute_lag"] = available["lag_seconds"].abs()
    rank_columns = (
        "water_rmse_offset_kw",
        "noise_ratio",
        "energy_error_pct",
        "transient_error",
        "absolute_lag",
    )
    available["rank_sum"] = 0.0
    for _, indices in available.groupby("cycle_name", sort=False).groups.items():
        for column in rank_columns:
            values = pd.to_numeric(available.loc[indices, column], errors="coerce")
            available.loc[indices, "rank_sum"] += values.fillna(np.inf).rank(method="min")
    return (
        available.groupby("method", as_index=False)
        .agg(
            cycles=("cycle_name", "nunique"),
            global_mean_rank_sum=("rank_sum", "mean"),
        )
        .sort_values("global_mean_rank_sum", ignore_index=True)
    )


def cost_method_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank offline curves by distortion of quantities entering the cost integral."""
    methods = (*OFFLINE_METHODS, *SHAPE_METHODS)
    available = metrics.loc[
        metrics["metric_status"].eq("available")
        & metrics["cycle_stage"].eq("frost_development")
        & metrics["method"].isin(methods)
    ].copy()
    if available.empty:
        return pd.DataFrame(columns=["method", "cycles", "cost_mean_rank_sum"])
    if "spike_ratio" not in available:
        available["spike_ratio"] = 1.0 - available["spike_reduction"]
    eligible_methods = (
        available.groupby("method")["spike_reduction"].median().loc[lambda x: x >= 0.1].index
    )
    available = available.loc[available["method"].isin(eligible_methods)].copy()
    if available.empty:
        return pd.DataFrame(columns=["method", "cycles", "cost_mean_rank_sum"])
    available["transient_error"] = (available["transient_retention"] - 1.0).abs()
    rank_columns = (
        "spike_ratio",
        "shortfall_area_error_pct",
        "energy_error_pct",
        "water_rmse_offset_kw",
        "transient_error",
    )
    available["cost_rank_sum"] = 0.0
    for _, indices in available.groupby("cycle_name", sort=False).groups.items():
        for column in rank_columns:
            values = pd.to_numeric(available.loc[indices, column], errors="coerce")
            available.loc[indices, "cost_rank_sum"] += values.fillna(np.inf).rank(
                method="min"
            )
    return (
        available.groupby("method", as_index=False)
        .agg(cycles=("cycle_name", "nunique"), cost_mean_rank_sum=("cost_rank_sum", "mean"))
        .sort_values("cost_mean_rank_sum", ignore_index=True)
    )


def _finite_runs(values: np.ndarray, valid_time: np.ndarray) -> list[np.ndarray]:
    valid = np.isfinite(values) & valid_time
    positions = np.flatnonzero(valid)
    if not len(positions):
        return []
    return [run for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)]


def _smooth_run(
    values: np.ndarray,
    times: pd.Series,
    water: np.ndarray,
    outputs: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[int, int]:
    indexed = pd.Series(values, index=pd.DatetimeIndex(times), dtype=float)
    minimum = min(3, len(indexed))
    outputs["median_centered_60s"][positions] = indexed.rolling(
        "60s", center=True, min_periods=minimum
    ).median()
    causal_median = indexed.rolling("30s", min_periods=1).median()
    outputs["ewma_tau30s"][positions] = _first_order(values, times, 30.0)
    outputs["median30s_ewma30s"][positions] = _first_order(
        causal_median.to_numpy(dtype=float), times, 30.0
    )
    outputs["savgol_70s"][positions] = _savgol(values, times)
    wavelet = _wavelet_denoise(values)
    outputs["wavelet_offline"][positions] = wavelet
    adaptive, median_window, lowpass_tau = _adaptive_offline(values, times, water)
    outputs["adaptive_offline"][positions] = adaptive
    return median_window, lowpass_tau


def _first_order(values: np.ndarray, times: pd.Series, tau_seconds: float) -> np.ndarray:
    output = values.astype(float, copy=True)
    nanoseconds = pd.to_datetime(times).astype("int64").to_numpy(dtype=np.int64)
    for index in range(1, len(output)):
        delta_seconds = max(0.0, (nanoseconds[index] - nanoseconds[index - 1]) / 1e9)
        alpha = 1.0 - np.exp(-delta_seconds / tau_seconds)
        output[index] = output[index - 1] + alpha * (values[index] - output[index - 1])
    return output


def _savgol(values: np.ndarray, times: pd.Series) -> np.ndarray:
    if len(values) < 5:
        return values.astype(float, copy=True)
    deltas = pd.DatetimeIndex(times).to_series().diff().dt.total_seconds().dropna()
    interval = float(deltas.median()) if not deltas.empty else 10.0
    window = max(5, int(round(70.0 / max(interval, 1e-9))))
    window += 1 - window % 2
    largest_odd = len(values) if len(values) % 2 else len(values) - 1
    window = min(window, largest_odd)
    return np.asarray(
        savgol_filter(values, window_length=window, polyorder=2, mode="interp"),
        dtype=float,
    )


def _wavelet_denoise(values: np.ndarray) -> np.ndarray:
    """Denoise one finite stage run with db4 universal soft thresholding."""
    wavelet = pywt.Wavelet("db4")
    level = pywt.dwt_max_level(len(values), wavelet.dec_len)
    if level < 1 or np.allclose(values, values[0]):
        return values.astype(float, copy=True)
    coefficients = pywt.wavedec(values, wavelet, mode="symmetric", level=level)
    finest = coefficients[-1]
    sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.67448975)
    if sigma <= np.finfo(float).eps:
        sigma = float(np.std(finest))
    threshold = sigma * np.sqrt(2.0 * np.log(len(values)))
    details = [pywt.threshold(c, threshold, mode="soft") for c in coefficients[1:]]
    denoised = [coefficients[0], *details]
    return np.asarray(
        pywt.waverec(denoised, wavelet, mode="symmetric")[: len(values)], dtype=float
    )


def _nearly_isotonic(
    values: np.ndarray, penalty: float = 0.15, rho: float = 1.0
) -> np.ndarray:
    """Penalize, rather than forbid, local increases using a small ADMM solve."""
    count = len(values)
    if count < 2 or np.allclose(values, values[0]):
        return values.astype(float, copy=True)
    diagonal = np.full(count, 1.0 + 2.0 * rho)
    diagonal[[0, -1]] = 1.0 + rho
    system = np.zeros((3, count))
    system[0, 1:] = -rho
    system[1] = diagonal
    system[2, :-1] = -rho
    estimate = values.astype(float, copy=True)
    difference = np.diff(estimate)
    dual = np.zeros(count - 1)
    for _ in range(500):
        rhs_difference = difference - dual
        rhs = values.astype(float, copy=True)
        rhs[0] -= rho * rhs_difference[0]
        rhs[-1] += rho * rhs_difference[-1]
        if count > 2:
            rhs[1:-1] += rho * (rhs_difference[:-1] - rhs_difference[1:])
        estimate = solve_banded((1, 1), system, rhs)
        shifted = np.diff(estimate) + dual
        threshold = penalty / rho
        previous_difference = difference.copy()
        difference = np.where(
            shifted < 0.0,
            shifted,
            np.where(shifted <= threshold, 0.0, shifted - threshold),
        )
        primal_residual = np.diff(estimate) - difference
        dual += primal_residual
        if max(
            np.max(np.abs(primal_residual)),
            rho * np.max(np.abs(difference - previous_difference)),
        ) <= 1e-8:
            break
    return np.asarray(estimate, dtype=float)


def _robust_monotone(
    values: np.ndarray, huber_delta: float = 0.15, smoothness: float = 2.0
) -> np.ndarray:
    """Fit a smooth non-increasing trend with Huber data fidelity."""
    if len(values) < 2 or np.allclose(values, values[0]):
        return values.astype(float, copy=True)
    estimate = np.asarray(
        isotonic_regression(values, increasing=False).x, dtype=float
    )
    momentum = estimate.copy()
    acceleration = 1.0
    step = 1.0 / (1.0 + 32.0 * smoothness)
    for _ in range(800):
        curvature = np.diff(momentum, n=2)
        curvature_gradient = np.zeros(len(values))
        if len(values) >= 3:
            curvature_gradient[:-2] += curvature
            curvature_gradient[1:-1] -= 2.0 * curvature
            curvature_gradient[2:] += curvature
        gradient = np.clip(momentum - values, -huber_delta, huber_delta)
        gradient += 2.0 * smoothness * curvature_gradient
        candidate = np.asarray(
            isotonic_regression(
                momentum - step * gradient, increasing=False
            ).x,
            dtype=float,
        )
        if np.max(np.abs(candidate - estimate)) <= 1e-8:
            estimate = candidate
            break
        next_acceleration = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * acceleration**2))
        momentum = candidate + (acceleration - 1.0) / next_acceleration * (
            candidate - estimate
        )
        estimate = candidate
        acceleration = next_acceleration
    return estimate


def _adaptive_offline(
    values: np.ndarray, times: pd.Series, water: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Pick the quietest zero-phase candidate with near-best water-side shape fit."""
    indexed = pd.Series(values, index=pd.DatetimeIndex(times), dtype=float)
    candidates: list[tuple[np.ndarray, int, int, float, float, float]] = []
    for median_seconds in _ADAPTIVE_MEDIAN_SECONDS:
        median = indexed.rolling(
            f"{median_seconds}s", center=True, min_periods=min(3, len(indexed))
        ).median()
        for tau_seconds in _ADAPTIVE_TAU_SECONDS:
            candidate = _zero_phase_lowpass(median.to_numpy(dtype=float), times, tau_seconds)
            valid = np.isfinite(candidate) & np.isfinite(water)
            if valid.sum() >= 4:
                error = candidate[valid] - water[valid]
                error -= np.median(error)
                water_rmse = float(np.sqrt(np.mean(error**2)))
            else:
                water_rmse = np.nan
            energy_error = _relative_error(
                _integral(candidate, times), _integral(values, times)
            )
            candidates.append(
                (
                    candidate,
                    median_seconds,
                    tau_seconds,
                    water_rmse,
                    energy_error,
                    _mad(_steps(candidate)),
                )
            )
    finite_rmse = [item[3] for item in candidates if np.isfinite(item[3])]
    if finite_rmse:
        threshold = min(finite_rmse) * 1.05 + 0.01
        eligible = [
            item
            for item in candidates
            if item[3] <= threshold and (not np.isfinite(item[4]) or item[4] <= 0.01)
        ]
    else:
        eligible = [item for item in candidates if item[1] == 90 and item[2] == 60]
    if not eligible:
        eligible = candidates
    best = min(eligible, key=lambda item: item[5] if np.isfinite(item[5]) else np.inf)
    return best[0], best[1], best[2]


def _zero_phase_lowpass(
    values: np.ndarray, times: pd.Series, tau_seconds: int
) -> np.ndarray:
    if len(values) < 5 or np.allclose(values, values[0]):
        return values.astype(float, copy=True)
    deltas = pd.DatetimeIndex(times).to_series().diff().dt.total_seconds().dropna()
    interval = float(deltas.median()) if not deltas.empty else 10.0
    normalized_cutoff = min(0.99, interval / (np.pi * tau_seconds))
    sos = butter(2, normalized_cutoff, btype="lowpass", output="sos")
    padlen = min(len(values) - 1, 9)
    return np.asarray(sosfiltfilt(sos, values, padlen=padlen), dtype=float)


def _steps(values: np.ndarray) -> np.ndarray:
    pairs = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    return np.asarray(np.diff(values)[pairs], dtype=float)


def _mad(values: np.ndarray) -> float:
    if not len(values):
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _percentile_abs(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.abs(values), percentile)) if len(values) else np.nan


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _relative_error(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or abs(reference) < 1e-12:
        return np.nan
    return float(abs(value - reference) / abs(reference))


def _integral(values: np.ndarray, times: pd.Series) -> float:
    seconds = pd.to_datetime(times).astype("int64").to_numpy(dtype=float) / 1e9
    pairs = (
        np.isfinite(values[:-1])
        & np.isfinite(values[1:])
        & np.isfinite(seconds[:-1])
        & np.isfinite(seconds[1:])
    )
    if not pairs.any():
        return np.nan
    widths = seconds[1:] - seconds[:-1]
    areas = 0.5 * (values[:-1] + values[1:]) * widths
    return float(np.sum(areas[pairs]))


def _lag_seconds(raw: np.ndarray, values: np.ndarray, times: pd.Series) -> float:
    valid = np.isfinite(raw) & np.isfinite(values)
    x = raw[valid]
    y = values[valid]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    best_lag = 0
    best_correlation = -np.inf
    limit = min(12, len(x) // 3)
    for lag in range(-limit, limit + 1):
        left = x[max(0, -lag) : len(x) - max(0, lag)]
        right = y[max(0, lag) : len(y) - max(0, -lag)]
        if len(left) < 4 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    deltas = pd.DatetimeIndex(times).to_series().diff().dt.total_seconds().dropna()
    interval = float(deltas.median()) if not deltas.empty else 10.0
    return float(best_lag * interval)


def _best_method(group: pd.DataFrame) -> tuple[str, float]:
    available = group.loc[group["metric_status"].eq("available")].copy()
    if available.empty:
        return "unavailable", np.nan
    available["transient_error"] = (available["transient_retention"] - 1.0).abs()
    available["absolute_lag"] = available["lag_seconds"].abs()
    columns = (
        "water_rmse_offset_kw",
        "noise_ratio",
        "energy_error_pct",
        "transient_error",
        "absolute_lag",
    )
    rank_sum = pd.Series(0.0, index=available.index)
    for column in columns:
        values = pd.to_numeric(available[column], errors="coerce")
        rank_sum += values.fillna(np.inf).rank(method="min", ascending=True)
    best = rank_sum.idxmin()
    return str(available.loc[best, "method"]), float(rank_sum.loc[best])
