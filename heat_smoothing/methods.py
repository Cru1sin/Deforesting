"""离线传感器插值与60秒过去窗口均值。"""

import numpy as np
import pandas as pd


def smooth_heat(frame, start, window_seconds, *, column="heating_capacity"):
    """Trailing time integral of the same linear segments as window_audit.

    Short missing intervals bridge only once both observations have arrived;
    original NaNs remain NaN. Gaps > MAXIMUM_GAP_SECONDS restart history.
    The first observation is the zero-duration limit of the startup mean.
    """
    from defrost_event_models.training_data import MAXIMUM_GAP_SECONDS

    result = frame[column].copy()
    if window_seconds == 0:
        return result
    valid = frame.loc[frame.timestamp.ge(start) & result.notna()]
    groups = valid.timestamp.diff().dt.total_seconds().gt(MAXIMUM_GAP_SECONDS).cumsum()
    for _, part in valid.groupby(groups):
        t = (part.timestamp - part.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        q = part[column].to_numpy()
        area = np.r_[0.0, np.cumsum(np.diff(t) * (q[1:] + q[:-1]) / 2)]
        left = np.maximum(0, t - window_seconds)
        j = np.searchsorted(t, left, side="right") - 1
        next_j = np.minimum(j + 1, len(t) - 1)
        slope = np.divide(
            q[next_j] - q[j], t[next_j] - t[j], out=np.zeros(len(t)), where=next_j != j
        )
        offset = left - t[j]
        left_area = area[j] + q[j] * offset + slope * offset**2 / 2
        result.loc[part.index] = np.divide(area - left_area, t - left, out=q.copy(), where=t > left)
    return result


PROCESSED_CHANNELS = ("heating_capacity", "evaporating_pressure", "coil_temperature")
MEASUREMENT_PROCESSING = "dataset_offline_linear_60s"


def measurement_values(frame):
    """Read model inputs from a published offline Processed table."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result.timestamp)
    for name in list(result):
        for suffix in ("_interpolated", "_filled"):
            if name.endswith(suffix):
                result[name.removesuffix(suffix)] = result[name]
    result.attrs["measurement_processing"] = MEASUREMENT_PROCESSING
    return result


def process_measurements(source, target, *, channels=None):
    """Original -> offline time interpolation -> trailing 60 s time mean.

    Interpolate every internal continuous-channel gap within a cycle, without
    a length limit. Endpoints remain missing; discrete channels use past values.
    One row flag records whether any continuous channel was interpolated.
    """
    from dataset_tools.builder.channel_mapping import load_channels
    from dataset_tools.builder.features import calculate_derived_features

    definitions = channels if channels is not None else load_channels()
    keys = ["experiment_id", "cycle_id"]
    raw = (
        source.sort_values("timestamp", kind="stable")
        .drop_duplicates(keys + ["timestamp"], keep="last")
        .copy()
    )
    continuous = [n for n, d in definitions.items() if d["kind"] == "continuous" and n in raw]
    discrete = [
        n
        for n, d in definitions.items()
        if d["kind"] in {"step", "event", "categorical"} and n in raw
    ]
    legacy = ("heating_capacity_low", "heating_capacity_high", "heating_capacity_mean")
    result = target.drop(
        columns=[n for n in target if n.endswith("__imputed") or n.startswith(legacy)],
        errors="ignore",
    ).copy()
    result["is_interpolated"] = False
    for _, rows in result.groupby(keys, sort=False, dropna=False):
        part = raw.loc[
            raw.experiment_id.eq(rows.experiment_id.iloc[0])
            & raw.cycle_id.eq(rows.cycle_id.iloc[0])
        ].set_index("timestamp")
        if part.empty:
            continue
        # Include absent sensor timestamps as well as explicit NaNs on the 1 s grid.
        grid = pd.date_range(
            min(part.index.min(), rows.timestamp.min()),
            max(part.index.max(), rows.timestamp.max()),
            freq="s",
        )
        grid = grid.union(part.index).union(pd.DatetimeIndex(rows.timestamp)).sort_values()
        original = part.reindex(grid)
        filled = (
            original[continuous]
            .apply(pd.to_numeric, errors="coerce")
            .interpolate(method="time", limit_area="inside")
        )
        flag = (original[continuous].isna() & filled.notna()).any(axis=1)
        selected = pd.DatetimeIndex(rows.timestamp)
        result.loc[rows.index, "is_interpolated"] = flag.reindex(selected).to_numpy()
        for name in continuous:
            result.loc[rows.index, name] = original[name].reindex(selected).to_numpy()
            result.loc[rows.index, name + "_interpolated"] = (
                filled[name].reindex(selected).to_numpy()
            )
        for name in discrete:
            result.loc[rows.index, name] = original[name].reindex(selected).to_numpy()
            filled[name] = original[name].ffill()
            result.loc[rows.index, name + "_filled"] = filled[name].reindex(selected).to_numpy()
        derived = calculate_derived_features(filled, definitions)
        raw_derived = calculate_derived_features(original, definitions)
        for name, definition in definitions.items():
            if definition["kind"] == "derived" and definition.get("formula") in {
                "cop",
                "evaporator_capacity",
                "pressure_ratio",
                "water_delta_temperature",
                "superheat_calculated",
            }:
                result.loc[rows.index, name] = raw_derived[name].reindex(selected).to_numpy()
                result.loc[rows.index, name + "_interpolated"] = (
                    derived[name].reindex(selected).to_numpy()
                )
        stages = original.cycle_stage.ffill()
        stages.loc[selected] = rows.cycle_stage.to_numpy()
        filled["timestamp"] = grid
        for name in PROCESSED_CHANNELS:
            processed = pd.Series(np.nan, index=grid)
            if name in filled:
                for _, stage in filled.groupby(stages.ne(stages.shift()).cumsum(), sort=False):
                    if name == "heating_capacity" and stages.loc[stage.index[0]] not in {
                        "recovery",
                        "frost_development",
                    }:
                        continue
                    processed.loc[stage.index] = smooth_heat(
                        stage, stage.timestamp.iloc[0], 60, column=name
                    ).to_numpy()
            result.loc[rows.index, name + "_processed"] = processed.reindex(selected).to_numpy()
    # Keep each original/interpolated/processed group adjacent in the table.
    columns = [n for n in result if not n.endswith(("_interpolated", "_filled", "_processed"))]
    for name in list(columns):
        extras = [
            name + suffix
            for suffix in ("_interpolated", "_filled", "_processed")
            if name + suffix in result
        ]
        position = columns.index(name) + 1
        columns[position:position] = extras
    columns += [n for n in result if n not in columns]
    return result.loc[:, columns]
