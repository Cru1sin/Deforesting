from __future__ import annotations

import pandas as pd
import pytest

from dataset_tools.builder.schema import (
    build_processed_frame,
    drop_image_columns,
    merge_registries,
    registry_from_frame,
)


@pytest.mark.parametrize(
    "code,mode,directory",
    [
        (23, "rb", "cycles_RB"),
        (50, "timed", "cycles"),
        (0, "unknown", "cycles"),
        (99, "unknown", "cycles"),
    ],
)
def test_control_group_uses_current_defrost_not_previous_latched_code(code, mode, directory):
    from dataset_tools.cycle_metadata import cycle_assets, defrost_control

    times = pd.date_range("2026-07-14", periods=4, freq="s")
    frame = pd.DataFrame({"timestamp": times, "p1__化霜条件'1_00": [50, 0, code, code]})
    result = defrost_control(
        frame, {"defrost_preparation_start": times[2], "defrost_start": times[3]}
    )
    assert result["mode"] == mode
    assert result["condition_code"] == code
    rb = mode == "rb"
    assert cycle_assets("example", rb=rb)["csv"] == f"{directory}/example.csv"
    result = defrost_control(frame, {"defrost_start": times[3]})
    assert result["mode"] == mode
    assert defrost_control(frame, {})["mode"] == "unknown"


@pytest.mark.parametrize("date,value", [("2026-09-20", "medium"), ("2026-09-21", "low")])
def test_user_side_fan_setting_is_campaign_annotation(date, value):
    from dataset_tools.cycle_metadata import user_side_fan_setting

    assert user_side_fan_setting(date) == value


def test_new_campaign_fan_uses_setpoint_without_relabeling_history():
    from dataset_tools.cycle_metadata import cycle_operating_condition, user_side_fan_setting

    values = pd.Series([45.0, 50.0, 55.0, 60.0, 65.0, float("nan")])
    fans = user_side_fan_setting("2026-09-28", values)
    assert fans.tolist() == ["high", "high", "unknown", "low", "low", "unknown"]
    assert user_side_fan_setting("2026-09-20", 60) == "medium"
    assert user_side_fan_setting("2026-09-21", 60) == "low"
    assert user_side_fan_setting("2026-10-10", 50) == "high"
    assert user_side_fan_setting("2026-09-28", 65) == "low"
    assert user_side_fan_setting("2026-09-20", 65) == "medium"
    frame = pd.DataFrame(
        dict(water_temperature_setpoint=[50, 60], user_side_fan_setting=["high", "low"])
    )
    assert cycle_operating_condition(frame) == "50C_high; 60C_low"


def _registry() -> dict[str, object]:
    return {
        "resample_interval_seconds": 10,
        "channels": {
            "signal": {
                "kind": "continuous",
                "unit": "C",
                "role": "performance",
                "resample": "mean",
                "formula": None,
                "dependencies": None,
                "scale": None,
                "offset": None,
                "analysis_candidate": True,
                "expected_frost_direction": None,
                "coverage_required": True,
            }
        },
        "columns": ["timestamp", "experiment_id", "cycle_id", "signal"],
    }


def test_add_user_side_fan_annotation_preserves_other_values(tmp_path):
    from dataset_tools.dataset_paths import write_json
    from dataset_tools.measurements import add_missing_channel

    assets = {"original_csv": "original.csv", "csv": "cycle.csv", "parquet": "cycle.parquet"}
    write_json(
        {"cycles": [{"experiment_date": "2026-09-21", "assets": assets}]},
        tmp_path / "cycle_catalog.json",
    )
    write_json(_registry(), tmp_path / "channel_registry.json")
    pd.DataFrame({"timestamp": ["2026-09-21"], "raw": ["001.2345678901234567"]}).to_csv(
        tmp_path / "original.csv", index=False
    )
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-21"]), "signal": [1.23456789]})
    frame.to_parquet(tmp_path / "cycle.parquet", index=False)
    add_missing_channel(tmp_path, "user_side_fan_setting")
    updated = pd.read_parquet(tmp_path / "cycle.parquet")
    pd.testing.assert_frame_equal(updated.drop(columns="user_side_fan_setting"), frame)
    assert updated.user_side_fan_setting.eq("low").all()
    original = pd.read_csv(tmp_path / "original.csv", dtype=str, keep_default_na=False)
    assert original.user_side_fan_setting.eq("low").all()
    assert original.raw.iloc[0] == "001.2345678901234567"

    # Repair an empty resampling bucket without moving an existing annotation.
    updated["user_side_fan_setting__imputed"] = False
    updated.loc[0, "user_side_fan_setting"] = None
    updated.to_parquet(tmp_path / "cycle.parquet", index=False)
    from dataset_tools.dataset_paths import read_json

    registry = read_json(tmp_path / "channel_registry.json")
    registry["columns"].append("user_side_fan_setting__imputed")
    write_json(registry, tmp_path / "channel_registry.json")
    add_missing_channel(tmp_path, "user_side_fan_setting")
    repaired = pd.read_parquet(tmp_path / "cycle.parquet")
    assert repaired.columns.tolist() == updated.columns.tolist()
    assert repaired.user_side_fan_setting.eq("low").all()
    pd.testing.assert_frame_equal(
        repaired.drop(columns="user_side_fan_setting"),
        updated.drop(columns="user_side_fan_setting"),
    )
    catalog = read_json(tmp_path / "cycle_catalog.json")
    catalog["cycles"][0]["experiment_date"] = "2026-09-20"
    write_json(catalog, tmp_path / "cycle_catalog.json")
    add_missing_channel(tmp_path, "user_side_fan_setting")
    assert pd.read_parquet(tmp_path / "cycle.parquet").user_side_fan_setting.eq("medium").all()
    assert pd.read_csv(tmp_path / "original.csv").user_side_fan_setting.eq("medium").all()


def test_add_missing_channel_preserves_data_assets_and_existing_values(tmp_path):
    from dataset_tools.dataset_paths import read_json, write_json
    from dataset_tools.measurements import add_missing_channel

    assets = {"original_csv": "original.csv", "csv": "cycle.csv", "parquet": "cycle.parquet"}
    write_json(
        {"cycles": [{"cycle_name": "cycle", "assets": assets}]},
        tmp_path / "cycle_catalog.json",
    )
    write_json(_registry(), tmp_path / "channel_registry.json")
    original = pd.DataFrame({"timestamp": ["2026-10-04"], "raw": ["001.2345678901234567"]})
    original.to_csv(tmp_path / "original.csv", index=False)
    frame = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "timestamp": pd.to_datetime(["2026-10-04"]),
            "signal": [1.23456789],
        }
    )
    frame.to_parquet(tmp_path / "cycle.parquet", index=False)
    frame.to_csv(tmp_path / "cycle.csv", index=False)
    image = tmp_path / "cycle.png"
    image.write_bytes(b"untouched")
    image_time = image.stat().st_mtime_ns
    catalog = (tmp_path / "cycle_catalog.json").read_bytes()

    add_missing_channel(tmp_path, "Def_EntrTm")
    updated = pd.read_parquet(tmp_path / "cycle.parquet")
    pd.testing.assert_frame_equal(updated.drop(columns="Def_EntrTm"), frame)
    assert updated.Def_EntrTm.isna().all()
    saved = pd.read_csv(tmp_path / "original.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(saved.drop(columns="Def_EntrTm"), original)
    assert saved.Def_EntrTm.eq("").all()
    assert read_json(tmp_path / "channel_registry.json")["channels"]["Def_EntrTm"]["unit"] == "s"
    assert (tmp_path / "cycle_catalog.json").read_bytes() == catalog
    assert image.stat().st_mtime_ns == image_time
    paths = [tmp_path / name for name in assets.values()]
    times = [p.stat().st_mtime_ns for p in paths]
    add_missing_channel(tmp_path, "Def_EntrTm")
    assert [p.stat().st_mtime_ns for p in paths] == times


def test_drop_image_columns_removes_all_image_triples() -> None:
    frame = pd.DataFrame(
        columns=[
            "timestamp",
            "image_front_path",
            "image_front_time",
            "image_front_offset_seconds",
            "signal",
        ]
    )
    assert drop_image_columns(frame).columns.tolist() == ["timestamp", "signal"]


def test_registry_is_explicit_and_merge_adds_new_columns() -> None:
    candidate = registry_from_frame(
        pd.DataFrame(columns=["timestamp", "experiment_id", "cycle_id", "humidity"]),
        {
            "humidity": {
                "kind": "continuous",
                "unit": "%",
                "resample": "mean",
            }
        },
        resample_interval_seconds=10,
    )
    merged = merge_registries(_registry(), candidate)
    assert list(merged["channels"]) == ["signal", "humidity"]
    assert merged["columns"] == [
        "timestamp",
        "experiment_id",
        "cycle_id",
        "signal",
        "humidity",
    ]


def test_registry_rejects_changed_scientific_definition() -> None:
    changed = _registry()
    changed["channels"] = {"signal": {**changed["channels"]["signal"], "unit": "K"}}
    with pytest.raises(ValueError, match="channel definition changed"):
        merge_registries(_registry(), changed)


def test_processed_frame_has_only_cycle_and_scientific_identity() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-14 10:00:00"]),
            "experiment_id": ["exp"],
            "cycle_id": ["cycle_001"],
            "signal": [1.0],
        }
    )
    result = build_processed_frame(
        frame,
        _registry(),
        cycle_name="frost_cycle_000001",
        cycle_uid="exp::cycle_001",
    )
    assert result.columns.tolist() == [
        "cycle_name",
        "cycle_uid",
        "timestamp",
        "experiment_id",
        "cycle_id",
        "signal",
    ]
    assert not {
        "dataset_id",
        "dataset_schema_version",
        "dataset_cycle_index",
    } & set(result)


def test_no_defrost_inherits_rb_without_inventing_an_event():
    from dataset_tools.cycle_metadata import defrost_control

    previous = {"cycle_name": "frost_cycle_000329", "defrost_control": {"mode": "rb"}}
    result = defrost_control(pd.DataFrame(), {}, previous)
    assert result == {
        "mode": "rb",
        "condition_code": None,
        "sample_time": None,
        "inherited_from": "frost_cycle_000329",
    }
    following = defrost_control(
        pd.DataFrame(), {}, {"cycle_name": "frost_cycle_000330", "defrost_control": result}
    )
    assert following["mode"] == "rb"
    assert following["inherited_from"] == "frost_cycle_000330"
    for code, mode in [(50, "timed"), (99, "unknown")]:
        frame = pd.DataFrame({"timestamp": ["2026-09-21"], "p1__化霜条件'1_00": [code]})
        observed = defrost_control(frame, {"defrost_start": "2026-09-21"}, previous)
        assert observed["mode"] == mode
        assert "inherited_from" not in observed
    assert defrost_control(pd.DataFrame(), {})["mode"] == "unknown"
