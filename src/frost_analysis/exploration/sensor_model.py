"""Small, leakage-safe helpers for the 59-cycle sensor model study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceModel:
    """Linear normal-operation reference fitted on early training rows."""

    features: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.features].apply(pd.to_numeric, errors="coerce")
        standardized = (values.to_numpy(dtype=float) - self.center) / self.scale
        design = np.column_stack([np.ones(len(frame)), standardized])
        prediction = design @ self.coefficients
        prediction[~np.isfinite(standardized).all(axis=1)] = np.nan
        return np.asarray(prediction, dtype=float)


@dataclass(frozen=True)
class RidgeModel:
    """Cycle-weighted ridge model with stored training normalization."""

    features: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    @property
    def intercept(self) -> float:
        return float(self.coefficients[0])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.features].apply(pd.to_numeric, errors="coerce")
        standardized = (values.to_numpy(dtype=float) - self.center) / self.scale
        design = np.column_stack([np.ones(len(frame)), standardized])
        prediction = design @ self.coefficients
        prediction[~np.isfinite(standardized).all(axis=1)] = np.nan
        return np.asarray(prediction, dtype=float)


def split_replication_cohort(
    cycles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Freeze old positions 0–48, prospective positions 49–58, and cycle 11."""
    scoped = cycles.iloc[:59].copy()
    old = scoped.iloc[:49].loc[lambda value: value["status"].astype(str).eq("valid")]
    new = scoped.iloc[49:59].loc[lambda value: value["status"].astype(str).eq("valid")]
    stress = scoped.loc[scoped["cycle_name"].astype(str).eq("frost_cycle_000011")]
    return old.reset_index(drop=True), new.reset_index(drop=True), stress.reset_index(drop=True)


def shared_complete_cases(
    frame: pd.DataFrame,
    *,
    target: str,
    models: dict[str, list[str]],
) -> pd.DataFrame:
    """Use one row set when comparing models with different inputs."""
    features = list(dict.fromkeys(feature for values in models.values() for feature in values))
    return frame.dropna(subset=[target, *features])


def fit_reference_model(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    early: str,
    ridge: float = 1e-6,
) -> ReferenceModel:
    """Fit a normal-operation relation using early training rows only."""
    selected = frame.loc[frame[early].astype(bool), [target, *features]].apply(
        pd.to_numeric, errors="coerce"
    )
    selected = selected.dropna()
    if len(selected) < len(features) + 2:
        raise ValueError("normal reference has too few complete early rows")
    center = selected[features].median().to_numpy(dtype=float)
    scale = selected[features].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    standardized = (selected[features].to_numpy(dtype=float) - center) / scale
    design = np.column_stack([np.ones(len(selected)), standardized])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ selected[target].to_numpy(dtype=float),
    )
    return ReferenceModel(tuple(features), center, scale, coefficients)


def apply_reference_model(
    model: ReferenceModel,
    frame: pd.DataFrame,
    *,
    observed: str,
    cycle: str,
    early: str,
) -> pd.Series:
    """Apply a frozen reference and calibrate only one early offset per cycle."""
    result = pd.Series(model.predict(frame), index=frame.index, dtype=float)
    observed_values = pd.to_numeric(frame[observed], errors="coerce")
    for _, indices in frame.groupby(cycle, sort=False).groups.items():
        index = pd.Index(indices)
        calibration = index[frame.loc[index, early].astype(bool)]
        offsets = observed_values.loc[calibration] - result.loc[calibration]
        if offsets.notna().any():
            result.loc[index] += float(offsets.median())
        else:
            result.loc[index] = np.nan
    return result


def add_cycle_future(
    frame: pd.DataFrame,
    *,
    column: str,
    horizon_steps: int,
    cycle: str,
    time: str | None = None,
    step_minutes: float = 1.0,
) -> pd.Series:
    """Shift a target only within its original cycle."""
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    result = frame.groupby(cycle, sort=False)[column].shift(-horizon_steps)
    if time is not None:
        future_time = frame.groupby(cycle, sort=False)[time].shift(-horizon_steps)
        elapsed = future_time - pd.to_numeric(frame[time], errors="coerce")
        expected = horizon_steps * step_minutes
        result = result.where(np.isclose(elapsed, expected, atol=1e-6))
    return result


def fit_weighted_ridge(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    cycle: str,
    ridge: float = 1e-3,
) -> RidgeModel:
    """Fit ridge regression while giving each cycle equal total weight."""
    selected = frame[[target, cycle, *features]].dropna().copy()
    if len(selected) < len(features) + 2:
        raise ValueError("ridge model has too few complete rows")
    counts = selected.groupby(cycle)[cycle].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    values = selected[features].to_numpy(dtype=float)
    center = np.average(values, axis=0, weights=weights)
    variance = np.average(np.square(values - center), axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    standardized = (selected[features].to_numpy(dtype=float) - center) / scale
    design = np.column_stack([np.ones(len(selected)), standardized])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = selected[target].to_numpy(dtype=float) * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return RidgeModel(tuple(features), center, scale, coefficients)
