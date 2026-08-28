"""Assign image states from candidate-level empirical cost regret."""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_cost_state_targets(states: pd.Series, task: str) -> pd.Series:
    """Map shared cost states to dense binary or three-class targets."""
    names = (
        ("pre_optimal", "post_optimal")
        if task == "binary"
        else ("pre_optimal", "near_optimal", "post_optimal")
    )
    if task not in {"binary", "three"}:
        raise ValueError(f"unknown classification task: {task}")
    return states.map({name: index for index, name in enumerate(names)}).astype("Int64")


def high_confidence_coverage(
    label_balance: pd.DataFrame, camera_group: str, threshold: float
) -> float:
    """Return retained pre/post images as a fraction of candidate-domain images."""
    rows = label_balance.loc[
        label_balance["camera_group"].eq(camera_group)
        & label_balance["regret_threshold"].eq(threshold)
        & label_balance["cost_state"].isin(
            ("pre_optimal", "near_optimal", "post_optimal")
        )
    ]
    retained = rows.loc[rows["cost_state"].ne("near_optimal"), "image_count"].sum()
    return float(retained / rows["image_count"].sum())


def _curve_support(
    curve: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, int | None]:
    ordered = (
        curve.assign(
            candidate_time=pd.to_datetime(
                curve["candidate_time"], errors="coerce", format="mixed"
            )
        )
        .sort_values("candidate_time", kind="stable")
        .reset_index(drop=True)
    )
    regret = pd.to_numeric(ordered["relative_regret"], errors="coerce")
    eligible = regret.map(np.isfinite) & ordered["optimization_eligible"].fillna(False)
    run = eligible.ne(eligible.shift(fill_value=False)).cumsum()
    support = eligible & eligible.groupby(run).transform("sum").ge(2)
    optimum_position = int(regret.loc[eligible].idxmin()) if eligible.any() else None
    return ordered, regret, eligible, run, support, optimum_position


def curve_label_exclusion_reason(curve: pd.DataFrame) -> str | None:
    """Return the candidate-curve reason that prevents image labeling."""
    _, _, eligible, _, support, optimum_position = _curve_support(curve)
    if not eligible.any():
        return "no_eligible_candidates"
    if optimum_position is None or not support.iloc[optimum_position]:
        return "t_star_not_in_interpolatable_run"
    return None


def complete_catalog_cycle_names(catalog: pd.DataFrame) -> list[str]:
    """Return valid catalog cycles with observed heating and completed defrost."""
    required = {"cycle_name", "status", "stable_heating_start", "defrost_start", "defrost_end"}
    missing = required - set(catalog)
    if missing:
        raise ValueError(f"catalog is missing complete-cycle fields: {sorted(missing)}")
    boundaries = catalog[["stable_heating_start", "defrost_start", "defrost_end"]].apply(
        pd.to_datetime, errors="coerce", format="mixed"
    )
    observed = catalog.loc[catalog["status"].eq("valid") & boundaries.notna().all(axis=1)]
    return sorted(observed["cycle_name"].astype(str).unique())


def complete_observed_cycle_names(
    catalog: pd.DataFrame, curves: pd.DataFrame
) -> list[str]:
    """Return complete catalog cycles supported by uncensored current curves."""
    complete = complete_catalog_cycle_names(catalog)
    scoped = curves.loc[curves["cycle_name"].isin(complete)].copy()
    if "is_censored" in scoped:
        censored = (
            scoped["is_censored"]
            .fillna(True)
            .astype(bool)
            .groupby(scoped["cycle_name"])
            .any()
        )
        scoped = scoped.loc[scoped["cycle_name"].isin(censored.index[~censored])]
    return sorted(scoped["cycle_name"].astype(str).unique())


def assign_image_cost_states(
    image_times: pd.Series | pd.DatetimeIndex,
    curve: pd.DataFrame,
    *,
    regret_threshold: float,
) -> pd.DataFrame:
    """Interpolate regret and label images without filling disconnected low-cost regions."""
    ordered, candidate_regret, _, run, support, optimum_position = _curve_support(curve)
    times = pd.Series(
        pd.to_datetime(image_times, errors="coerce", format="mixed")
    ).reset_index(drop=True)
    regret = pd.Series(np.nan, index=times.index, dtype=float)
    state = pd.Series(pd.NA, index=times.index, dtype="string")
    if optimum_position is not None and support.iloc[optimum_position]:
        for _, positions in ordered.loc[support].groupby(run.loc[support], sort=False):
            inside = times.between(
                positions["candidate_time"].iloc[0],
                positions["candidate_time"].iloc[-1],
            )
            regret.loc[inside] = np.interp(
                times.loc[inside].astype("int64"),
                positions["candidate_time"].astype("int64"),
                candidate_regret.loc[positions.index],
            )
        labeled = regret.notna()
        optimum = ordered.loc[optimum_position, "candidate_time"]
        state.loc[labeled & times.lt(optimum)] = "pre_optimal"
        state.loc[labeled & times.ge(optimum)] = "post_optimal"
        state.loc[labeled & regret.le(regret_threshold)] = "near_optimal"
    binary = state.mask(state.eq("near_optimal"), pd.NA)
    return pd.DataFrame(
        {
            "image_time": times,
            "relative_regret": regret,
            "cost_state": state,
            "three_class_state": state,
            "binary_state": binary,
        }
    )
