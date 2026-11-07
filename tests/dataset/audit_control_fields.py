"""Read-only controller-field audit against the published cycle cohort."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from dataset_tools.load_dataset import DatasetLoader
from defrost_decision.baselines.rule_based import RAW_COLUMNS, controller_snapshots

CODE = "p1__化霜条件'1_00"


def audit_raw_dates(data, dates, output, *, replay_rb=False):
    """Audit newly unpacked dates with the shared raw reader and event detector."""
    from dataset_tools.builder.channel_mapping import CHANNELS
    from dataset_tools.builder.dataset_settings import CycleSettings
    from dataset_tools.builder.detect_cycles import (
        _debounce_state,
        _defrost_runs,
        _fill_short_state_gaps,
        _normalize_state,
        find_defrost_preparation_start,
    )
    from dataset_tools.builder.prepare_measurements import _read_sensor_table

    settings = CycleSettings()
    rows, daily, exceptions = [], [], []
    for date in dates:
        paths = sorted((data / date).glob("*参数1.xls"))
        if not paths:
            print(f"[missing] {date}", flush=True)
            continue
        frame = pd.concat(
            [_read_sensor_table(p, "时间", infer_types=True) for p in paths], ignore_index=True
        )
        frame = frame.rename(columns={c: "p1__" + c for c in frame if c != "timestamp"})
        frame = controller_snapshots(frame).reset_index(drop=True)
        for name in (
            "water_temperature_setpoint",
            "water_out_temperature",
            "compressor_frequency_setpoint",
            "compressor_frequency",
            "coil_temperature",
            "ambient_temperature",
        ):
            frame[name] = pd.to_numeric(frame[CHANNELS[name]["source_names"][0]], errors="coerce")
        fan = "p1__设定风速<1_00>"
        frame[fan] = pd.to_numeric(frame[fan], errors="coerce")
        actual_fan = "p1__实际风速'2__20"
        frame[actual_fan] = pd.to_numeric(frame[actual_fan], errors="coerce")
        state, gaps = _fill_short_state_gaps(
            frame.timestamp,
            frame["p1__Deforst"].map(_normalize_state),
            settings.maximum_state_gap_seconds,
        )
        events = _defrost_runs(
            frame.timestamp,
            _debounce_state(frame.timestamp, state, settings.debounce_seconds),
            gaps,
        )
        heating = frame.loc[state.eq(False) & frame.compressor_frequency.gt(0)]
        for field in ("water_temperature_setpoint", fan, actual_fan, "water_out_temperature"):
            v = heating[field].dropna()
            daily.append(
                {
                    "date": date,
                    "field": field,
                    "samples": len(v),
                    "minimum": v.min(),
                    "median": v.median(),
                    "maximum": v.max(),
                    "at_or_above_setpoint_fraction": heating.water_out_temperature.ge(
                        heating.water_temperature_setpoint
                    ).mean()
                    if field == "water_out_temperature"
                    else None,
                    "top_values": json.dumps(v.value_counts().head(5).to_dict()),
                }
            )
        for i, event in enumerate(events):
            start = events[i - 1]["end"] if i else frame.timestamp.iloc[0]
            prep = find_defrost_preparation_start(frame, start, event["start"], settings)
            record = {
                "cycle_name": f"{date}_event_{i + 1:03}",
                "experiment_date": date,
                "assets": {"csv": "raw/event.csv"},
                "boundaries": {"defrost_preparation_start": prep, "defrost_start": event["start"]},
            }
            row, _ = event_evidence(frame, record)
            segment = heating.loc[
                heating.timestamp.ge(start) & heating.timestamp.lt(prep or event["start"])
            ]
            row["heating_start"] = start
            row["defrost_end"] = event["end"]
            row["partial_first_event"] = i == 0
            if replay_rb:
                from defrost_decision.baselines.rule_based import first_trigger

                replay = first_trigger(frame, start, prep or event["start"])
                row.update(replay)
                for label, boundary in [("preparation", prep), ("defrost", event["start"])]:
                    row["replay_minus_" + label + "_seconds"] = (
                        (replay["t_RB"] - boundary).total_seconds()
                        if pd.notna(replay["t_RB"]) and boundary is not None
                        else None
                    )
                observed = frame.loc[frame.timestamp.ge(start) & frame.timestamp.le(event["start"])]
                codes = pd.to_numeric(observed[CODE], errors="coerce")
                onset = observed.loc[codes.gt(0) & codes.shift().eq(0), "timestamp"]
                row["recorded_trigger_time"] = onset.iloc[0] if len(onset) else pd.NaT
                if replay["rb_status"] != "triggered":
                    # Post-action samples diagnose logging/threshold differences only.
                    trace = first_trigger(
                        frame, start, event["start"] + pd.Timedelta(seconds=1), return_trace=True
                    )
                    trace = trace.loc[
                        trace.index >= (prep or event["start"]) - pd.Timedelta(seconds=60)
                    ].copy()
                    trace.insert(0, "event", record["cycle_name"])
                    trace.insert(1, "actual_preparation", prep)
                    trace.insert(2, "actual_defrost", event["start"])
                    exceptions.append(trace.rename_axis("timestamp").reset_index())
            for field, label in [
                ("water_temperature_setpoint", "setpoint_C"),
                (fan, "fan_command"),
                ("water_out_temperature", "water_out_C"),
            ]:
                v = segment[field].dropna()
                row[label + "_median"] = v.median()
                row[label + "_maximum"] = v.max()
            valid = segment.dropna(subset=["water_out_temperature", "water_temperature_setpoint"])
            row["at_or_above_setpoint_samples"] = int(
                valid.water_out_temperature.ge(valid.water_temperature_setpoint).sum()
            )
            row["within_1C_setpoint_samples"] = int(
                valid.water_out_temperature.ge(valid.water_temperature_setpoint - 1).sum()
            )
            rows.append(row)
        print(f"[raw audit] {date}: {len(events)} defrost events", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "event_conditions.csv", index=False)
    pd.DataFrame(daily).to_csv(output / "daily_controls.csv", index=False)
    if exceptions:
        pd.concat(exceptions, ignore_index=True).to_csv(
            output / "untriggered_end_trace.csv", index=False
        )


def event_evidence(frame, record):
    """Use the same observed preparation-to-defrost window as the RB audit."""
    frame = controller_snapshots(frame)
    bounds = record["boundaries"]
    preparation = pd.to_datetime(bounds.get("defrost_preparation_start") or pd.NaT)
    defrost = pd.to_datetime(bounds.get("defrost_start") or pd.NaT)
    after = frame.loc[frame.timestamp.ge(preparation) & frame.timestamp.le(defrost)]
    codes = pd.to_numeric(after[CODE], errors="coerce").dropna()
    codes = codes[codes.gt(0)]
    before = frame.loc[frame.timestamp.lt(preparation)].dropna(subset=[RAW_COLUMNS[-2]])
    formal = frame.loc[
        frame.timestamp.ge(defrost) & frame.timestamp.le(defrost + pd.Timedelta(seconds=30))
    ].dropna(subset=[CODE])
    row = {
        "cycle_name": record["cycle_name"],
        "experiment_date": record["experiment_date"],
        "current_group": Path(record["assets"]["csv"]).parent.name,
        "preparation": preparation,
        "defrost": defrost,
        "condition_code": codes.iloc[0] if len(codes) else None,
        "event_codes": json.dumps(sorted(codes.unique().tolist())),
        "last_T1_seconds": before[RAW_COLUMNS[-2]].iloc[-1] if len(before) else None,
        "last_T2_seconds": before[RAW_COLUMNS[-1]].iloc[-1] if len(before) else None,
        "last_timer_sample": before.timestamp.iloc[-1] if len(before) else None,
        "formal_start_code": formal[CODE].iloc[0] if len(formal) else None,
        "formal_code_sample": formal.timestamp.iloc[0] if len(formal) else None,
    }
    trace = frame.loc[
        frame.timestamp.ge(preparation - pd.Timedelta(seconds=30))
        & frame.timestamp.le(defrost + pd.Timedelta(seconds=30)),
        ["timestamp", CODE, *RAW_COLUMNS[-3:]],
    ].copy()
    trace.insert(0, "cycle_name", record["cycle_name"])
    return row, trace


def test_event_evidence_ignores_previous_latched_code():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-10-04", periods=4, freq="s"),
            CODE: [50, 0, 20, 20],
            RAW_COLUMNS[-3]: [-10] * 4,
            RAW_COLUMNS[-2]: [0, 2399, 0, 0],
            RAW_COLUMNS[-1]: [0, 2800, 0, 0],
        }
    )
    record = {
        "cycle_name": "example",
        "experiment_date": "2026-10-04",
        "assets": {"csv": "cycles/example.csv"},
        "boundaries": {
            "defrost_preparation_start": frame.timestamp.iloc[2],
            "defrost_start": frame.timestamp.iloc[3],
        },
    }
    row, _ = event_evidence(frame, record)
    assert row["condition_code"] == 20
    assert row["last_T1_seconds"] == 2399
    record["boundaries"]["defrost_preparation_start"] = None
    row, trace = event_evidence(frame, record)
    assert row["condition_code"] is None and trace.empty
    assert row["formal_start_code"] == 20


def audit(dataset: Path, output: Path, prefixes=("p1__",), n_jobs=6) -> pd.DataFrame:
    loader = DatasetLoader(dataset)

    def inspect(record):
        rows = []
        name = record["cycle_name"]
        path = loader.dataset_root / record["assets"]["original_csv"]
        columns = pd.read_csv(path, nrows=0).columns
        columns = [c for c in columns if c.startswith(tuple(prefixes))]
        selected = list(dict.fromkeys([*columns, "timestamp", CODE, *RAW_COLUMNS[-3:]]))
        frame = loader.load_cycle_original(name, columns=selected)
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        bounds = record["boundaries"]
        start = pd.to_datetime(bounds.get("heating_start"))
        end = pd.to_datetime(bounds.get("defrost_preparation_start") or bounds.get("defrost_start"))
        heating = frame.loc[frame.timestamp.ge(start)] if pd.notna(start) else frame.iloc[:0]
        if pd.notna(end):
            heating = heating.loc[heating.timestamp.lt(end)]
        for phase, source in (("whole_cycle", frame), ("heating", heating)):
            for field in columns:
                if field == "timestamp":
                    continue
                values = source[field].dropna()
                numeric = pd.to_numeric(values, errors="coerce").dropna()
                counts = values.astype(str).value_counts()
                rows.append(
                    {
                        "cycle_name": name,
                        "experiment_date": record["experiment_date"],
                        "current_group": Path(record["assets"]["csv"]).parent.name,
                        "phase": phase,
                        "field": field,
                        "observations": len(values),
                        "unique_values": len(counts),
                        "mode": counts.index[0] if len(counts) else None,
                        "mode_fraction": counts.iloc[0] / len(values) if len(values) else None,
                        "minimum": numeric.min(),
                        "maximum": numeric.max(),
                        "zero_count": int(numeric.eq(0).sum()),
                        "value_7200_count": int(numeric.eq(7200).sum()),
                        "top_values": json.dumps(counts.head(3).to_dict(), ensure_ascii=False),
                    }
                )
        event, trace = event_evidence(frame, record)
        return rows, event, trace

    rows, events, traces = [], [], []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for index, (fields, event, trace) in enumerate(
            executor.map(inspect, loader.catalog["cycles"]), 1
        ):
            rows.extend(fields)
            events.append(event)
            traces.append(trace)
            if index % 50 == 0:
                print(f"[audit] {index}/{len(loader.catalog['cycles'])}", flush=True)
    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "control_fields.csv", index=False)
    pd.DataFrame(events).to_csv(output / "event_conditions.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(output / "event_traces.csv", index=False)
    print(f"Audited {len(loader.catalog['cycles'])} cycles; {len(result)} field/phase rows")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/99_历史归档/legacy_test/control_fields")
    )
    parser.add_argument("--prefixes", nargs="+", default=["p1__"])
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--raw-data", type=Path)
    parser.add_argument("--date", action="append")
    parser.add_argument("--replay-rb", action="store_true")
    args = parser.parse_args()
    if args.raw_data:
        audit_raw_dates(args.raw_data, args.date, args.output, replay_rb=args.replay_rb)
    else:
        audit(args.dataset, args.output, args.prefixes, args.n_jobs)
