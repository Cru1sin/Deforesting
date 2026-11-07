"""Causal controller context and fixed-ceiling actual-frequency headroom."""

import numpy as np
import pandas as pd

COMMANDS = {
    "frequency_command_hz": "p1__设定频率<1_00>",
    "water_temperature_setpoint": "p1__Ts(37.1)'2_31",
}
RAW_CONTROL = {
    **COMMANDS,
    "actual_frequency_hz": "p1__运行频率'1_00",
    "water_out_temperature": "p1__Twout(27.1)'2_31'",
    "control_ambient_c": "p1__T4(22.1)'2_31'",
}
HISTORIES = {
    "baseline_slope": ["frequency_delta_hz", "frequency_slope_hz_per_min"],
    "changes": ["frequency_change_60s_hz", "frequency_change_300s_hz"],
}
COMPENSATION = [
    "control_setpoint_c",
    "temperature_error_c",
    "frequency_headroom_fraction",
    "actual_frequency_change_300s_hz",
    "temperature_error_change_300s_c",
]


def stopping_inputs(frequency_input, control_history):
    frequency = {
        "headroom100": "frequency_headroom_hz",
        "headroom_fraction": "frequency_headroom_fraction",
        "command": "frequency_command_hz",
    }[frequency_input]
    single = "RGB_command" if frequency_input == "command" else "RGB_headroom"
    history = HISTORIES[control_history]
    context = ["water_temperature_setpoint", frequency, *history]
    return {
        "RGB_stopping": [],
        single: [frequency],
        "RGB_response": [frequency, *history],
        "RGB_control": context,
        "Control_only": context,
    }


def native_control_context(raw, actions, stable_start):
    """Freeze first 5 min median after it completes; backward snapshots <=30 s."""
    raw = raw.reindex(columns=["timestamp", *RAW_CONTROL]).copy()
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    for column in RAW_CONTROL:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    raw["frequency_command_hz"] = raw.frequency_command_hz.where(
        raw.frequency_command_hz.between(1, 200)
    )
    raw["actual_frequency_hz"] = raw.actual_frequency_hz.where(
        raw.actual_frequency_hz.ge(0) & np.isfinite(raw.actual_frequency_hz)
    )
    raw["water_temperature_setpoint"] = raw.water_temperature_setpoint.where(
        raw.water_temperature_setpoint.between(-20, 80)
    )
    raw["water_out_temperature"] = raw.water_out_temperature.where(
        raw.water_out_temperature.between(-20, 80)
    )
    raw["control_ambient_c"] = raw.control_ambient_c.where(raw.control_ambient_c.between(-30, 50))

    def snapshot(times):
        query = pd.DataFrame({"candidate_defrost_time": pd.to_datetime(times)}).sort_values(
            "candidate_defrost_time"
        )
        return pd.merge_asof(
            query,
            raw,
            left_on="candidate_defrost_time",
            right_on="timestamp",
            direction="backward",
            allow_exact_matches=False,
            tolerance=pd.Timedelta(seconds=30),
        )

    result = actions.sort_values("candidate_defrost_time").copy()
    now = snapshot(result.candidate_defrost_time)
    result["control_timestamp"] = now.timestamp.to_numpy()
    for column in RAW_CONTROL:
        result[column] = now[column].to_numpy()
    # User-assumed 100 Hz ceiling; unavailable readings remain NaN.
    result["frequency_headroom_hz"] = (100.0 - result.actual_frequency_hz).clip(lower=0)
    result["frequency_headroom_fraction"] = result.frequency_headroom_hz / 100.0  # 0.2 denotes 20%.
    past = snapshot(result.candidate_defrost_time - pd.Timedelta(seconds=300))
    result["temperature_error_c"] = result.water_temperature_setpoint - result.water_out_temperature
    result["actual_frequency_change_300s_hz"] = (
        result.actual_frequency_hz - past.actual_frequency_hz.to_numpy()
    )
    result["temperature_error_change_300s_c"] = (
        result.temperature_error_c
        - (past.water_temperature_setpoint - past.water_out_temperature).to_numpy()
    )
    start = pd.Timestamp(stable_start)
    initial = snapshot(
        pd.date_range(start + pd.Timedelta(seconds=30), periods=10, freq="30s")
    ).frequency_command_hz
    baseline = initial.median() if initial.notna().all() else np.nan
    result["frequency_baseline_hz"] = np.where(
        result.candidate_defrost_time.ge(start + pd.Timedelta(minutes=5)), baseline, np.nan
    )
    result["frequency_delta_hz"] = result.frequency_command_hz - result.frequency_baseline_hz
    history = np.stack(
        [
            snapshot(
                result.candidate_defrost_time - pd.Timedelta(seconds=offset)
            ).frequency_command_hz.to_numpy()
            for offset in range(0, 301, 30)
        ],
        axis=1,
    )
    result["frequency_change_60s_hz"] = history[:, 0] - history[:, 2]
    result["frequency_change_300s_hz"] = history[:, 0] - history[:, 10]
    # Median of pairwise slopes on seven trailing 30 s samples, in Hz/min.
    slopes = [
        (history[:, a] - history[:, b]) / ((b - a) * 0.5) for a in range(7) for b in range(a + 1, 7)
    ]
    result["frequency_slope_hz_per_min"] = np.median(np.stack(slopes, axis=1), axis=1)
    before_start = result.candidate_defrost_time.lt(start + pd.Timedelta(minutes=3))
    result.loc[before_start, "frequency_slope_hz_per_min"] = np.nan
    return result


def load_control_context(dataset, name, actions):
    from dataset_tools import DatasetLoader

    raw = DatasetLoader(dataset).load_cycle_original(
        name, columns=["timestamp", *RAW_CONTROL.values()]
    )
    raw = raw.rename(columns={value: key for key, value in RAW_CONTROL.items()})
    result = native_control_context(raw, actions, actions.stable_heating_start.iloc[0])
    print(
        f"[control inputs] {name}: {result.frequency_command_hz.notna().sum()}/{len(result)} fresh commands",
        flush=True,
    )
    return result
