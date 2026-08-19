"""Assign image states from candidate-level empirical cost regret."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_image_cost_states(
    image_times: pd.Series | pd.DatetimeIndex,
    curve: pd.DataFrame,
    *,
    regret_threshold: float,
) -> pd.DataFrame:
    """Interpolate regret and label images without filling disconnected low-cost regions."""
    ordered = curve.assign(
        candidate_time=pd.to_datetime(curve["candidate_time"], errors="coerce")
    ).sort_values("candidate_time")
    times = pd.Series(pd.to_datetime(image_times, errors="coerce")).reset_index(drop=True)
    first = ordered["candidate_time"].iloc[0]
    last = ordered["candidate_time"].iloc[-1]
    optimum = ordered.loc[ordered["relative_regret"].idxmin(), "candidate_time"]
    inside = times.between(first, last)
    regret = pd.Series(np.nan, index=times.index, dtype=float)
    regret.loc[inside] = np.interp(
        times.loc[inside].astype("int64"),
        ordered["candidate_time"].astype("int64"),
        ordered["relative_regret"],
    )
    state = pd.Series("outside_candidate_domain", index=times.index, dtype="string")
    state.loc[inside & times.lt(optimum)] = "pre_optimal"
    state.loc[inside & times.ge(optimum)] = "post_optimal"
    state.loc[inside & regret.le(regret_threshold)] = "near_optimal"
    return pd.DataFrame({"image_time": times, "relative_regret": regret, "cost_state": state})
