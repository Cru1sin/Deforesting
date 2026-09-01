"""Minimal, falsifiable frost state–performance analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def relative_degradation(
    observed: np.ndarray | pd.Series | list[float],
    healthy: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    """Return fractional performance loss relative to a healthy reference."""
    measured = np.asarray(observed, dtype=float)
    reference = np.asarray(healthy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - measured / reference


def monotonicity_metrics(
    values: np.ndarray | pd.Series | list[float],
) -> dict[str, float | int]:
    """Measure downward violations for a degradation state expected to increase."""
    observed = np.asarray(values, dtype=float)
    steps = np.diff(observed[np.isfinite(observed)])
    negative = np.maximum(-steps, 0.0)
    total = float(np.abs(steps).sum())
    return {
        "violation_fraction": float(negative.sum() / total) if total else 0.0,
        "violating_steps": int(np.sum(steps < 0.0)),
    }


@dataclass(frozen=True)
class HingeFit:
    """Fit of ``loss = slope * max(state - threshold, 0)``."""

    threshold: float
    slope: float
    rmse: float

    def predict(self, state: np.ndarray | pd.Series) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        return self.slope * np.maximum(values - self.threshold, 0.0)


def select_valid_catalog_positions(
    cycles: pd.DataFrame, *, last_position: int = 48
) -> pd.DataFrame:
    """Select valid rows from zero-based catalog positions 0..last_position."""
    if last_position < 0:
        raise ValueError("last_position must be non-negative")
    scoped = cycles.iloc[: last_position + 1]
    return scoped.loc[scoped["status"].astype(str).eq("valid")].reset_index(drop=True)


def fit_hinge(state: np.ndarray | pd.Series, loss: np.ndarray | pd.Series) -> HingeFit:
    """Fit the smallest one-threshold degradation law by deterministic grid search."""
    x = np.asarray(state, dtype=float)
    y = np.asarray(loss, dtype=float)
    observed = np.isfinite(x) & np.isfinite(y)
    x = x[observed]
    y = y[observed]
    if x.size < 3 or np.nanmax(x) <= np.nanmin(x):
        raise ValueError("at least three varying observed states are required")
    thresholds = np.unique(np.concatenate(([0.0], np.quantile(x, np.linspace(0.02, 0.80, 241)))))
    best: HingeFit | None = None
    for threshold in thresholds:
        active = np.maximum(x - threshold, 0.0)
        denominator = float(active @ active)
        if denominator <= 0.0:
            continue
        slope = max(0.0, float(active @ y) / denominator)
        rmse = float(np.sqrt(np.mean(np.square(y - slope * active))))
        candidate = HingeFit(float(threshold), slope, rmse)
        if best is None or candidate.rmse < best.rmse:
            best = candidate
    if best is None:
        raise ValueError("hinge fit is not identifiable")
    return best


def leave_group_out_reference(
    frame: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    group: str,
    early: str,
    cycle: str,
    ridge: float = 1e-6,
) -> pd.Series:
    """Predict normal values from other groups, then early-calibrate each cycle.

    Only rows marked by ``early`` are used to learn the context relationship.
    The held-out cycle's early observations contribute a single additive offset,
    never a slope or future value.
    """
    if not features:
        raise ValueError("at least one reference feature is required")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for held_out in pd.unique(frame[group].dropna()):
        test_mask = frame[group].eq(held_out)
        train_mask = ~test_mask & frame[early].astype(bool)
        train = frame.loc[train_mask, [target, *features]].apply(pd.to_numeric, errors="coerce")
        test = frame.loc[test_mask, [target, cycle, early, *features]].copy()
        usable = train.dropna()
        if len(usable) < len(features) + 2:
            continue
        center = usable[features].median()
        scale = usable[features].std(ddof=0).replace(0.0, 1.0)
        x_train = (usable[features] - center) / scale
        design = np.column_stack([np.ones(len(x_train)), x_train.to_numpy(dtype=float)])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ usable[target].to_numpy(dtype=float),
        )
        numeric_test = test[features].apply(pd.to_numeric, errors="coerce")
        complete = numeric_test.notna().all(axis=1)
        test_design = np.column_stack(
            [
                np.ones(int(complete.sum())),
                ((numeric_test.loc[complete] - center) / scale).to_numpy(dtype=float),
            ]
        )
        predictions = pd.Series(np.nan, index=test.index, dtype=float)
        predictions.loc[complete] = test_design @ coefficients
        observed_target = pd.to_numeric(test[target], errors="coerce")
        for _, indices in test.groupby(cycle, sort=False).groups.items():
            index = pd.Index(indices)
            calibration = index[test.loc[index, early].astype(bool)]
            offsets = observed_target.loc[calibration] - predictions.loc[calibration]
            offset = float(offsets.median()) if offsets.notna().any() else 0.0
            predictions.loc[index] += offset
        result.loc[test.index] = predictions
    return result
