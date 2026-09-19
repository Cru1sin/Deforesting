"""Continuous candidate-interval grouping used by the COP reference."""
import numpy as np
import pandas as pd


def five_minute_support_runs(times: pd.Series, selected: pd.Series) -> pd.Series:
    """Keep candidate runs spanning at least five continuous minutes."""
    result = pd.Series(False, index=selected.index)
    chosen = np.flatnonzero(selected.to_numpy(dtype=bool))
    if not chosen.size:
        return result
    parsed = pd.to_datetime(times, errors="coerce")
    breaks = np.flatnonzero(
        (np.diff(chosen) != 1)
        | (
            parsed.iloc[chosen[1:]].to_numpy() - parsed.iloc[chosen[:-1]].to_numpy()
            > np.timedelta64(90, "s")
        )
    )
    for left, right in zip(np.r_[0, breaks + 1], np.r_[breaks, len(chosen) - 1], strict=True):
        positions = chosen[left : right + 1]
        if parsed.iloc[positions[-1]] - parsed.iloc[positions[0]] >= pd.Timedelta(minutes=5):
            result.iloc[positions] = True
    return result
