from pathlib import Path

import pandas as pd
import pytest

from dataset_tools.builder.dataset_settings import Config
from dataset_tools.builder.raw import discover_inputs


def test_rb_prior_is_read_without_replay_and_requires_requested_cycles(tmp_path, monkeypatch):
    from dataset_tools import DatasetLoader
    from defrost_decision.baselines import rule_based

    monkeypatch.setattr(rule_based, "calculate_cycle", lambda *a, **kw: pytest.fail("RB replay"))
    loader = DatasetLoader.__new__(DatasetLoader)
    loader.dataset_root = tmp_path
    monkeypatch.setattr(
        loader,
        "list_cycles",
        lambda: pd.DataFrame({"cycle_name": ["a", "b"], "experiment_id": ["exp", "exp"]}),
    )
    (tmp_path / "priors").mkdir()
    path = tmp_path / "priors/rb_reference.csv"
    rows = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "experiment_id": ["exp", "exp"],
            "t_RB": ["2026-01-01 01:00:00", None],
            "rb_status": ["triggered", "right_censored"],
        }
    )
    rows.to_csv(path, index=False)
    result = loader.load_rb_reference(["a", "b"])
    assert result.loc["a", "t_RB"] == pd.Timestamp("2026-01-01 01:00:00")
    assert pd.isna(result.loc["b", "t_RB"])
    with pytest.raises(ValueError, match="missing cycles"):
        loader.load_rb_reference(["new"])
    rows.assign(experiment_id="wrong").to_csv(path, index=False)
    with pytest.raises(ValueError, match="experiment IDs"):
        loader.load_rb_reference(["a"])


def test_discover_inputs_reads_root_sensors_and_one_level_images(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "0715"
    camera = raw / "192.168.1.1_1"
    camera.mkdir(parents=True)
    (raw / "parameters.xls").write_text("sensor", encoding="utf-8")
    image = camera / "20260715080000000.jpg"
    image.write_bytes(b"image")
    nested = camera / "nested"
    nested.mkdir()
    (nested / "ignored.jpg").write_bytes(b"image")
    config = Config(
        project_root=tmp_path,
        experiment_id="exp_test",
        experiment_date="2026-07-15",
        input_dir=raw,
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="time",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
    )

    inputs = discover_inputs(config)

    assert inputs.sensor_files == (raw / "parameters.xls",)
    assert inputs.image_files == (image,)


@pytest.mark.parametrize("sheet", ["参数页4", "参数页5", "text"])
def test_def_entry_timer_accepts_zero_and_seconds_without_filling(tmp_path, sheet):
    from dataset_tools.builder.channel_mapping import load_channels
    from dataset_tools.builder.prepare_measurements import _combine_channel, _load_channel_frames

    times = pd.date_range("2026-10-04", periods=3, freq="s")
    path = tmp_path / "controller.xlsx"
    if sheet == "text":
        path = tmp_path / "参数5.xls"
        pd.DataFrame({"时间": times, "Def_EntrTm": [0, 7200, None]}).to_csv(
            path, sep="\t", index=False
        )
    else:
        with pd.ExcelWriter(path) as writer:
            for page in dict.fromkeys(["参数页0", "参数页1", "参数页4", sheet]):
                frame = pd.DataFrame({"DateTime": times})
                if page == sheet:
                    frame["Def_EntrTm"] = [0, 7200, None]
                frame.to_excel(writer, sheet_name=page, index=False)
    channels = load_channels()
    frames = _load_channel_frames((path,), channels, "时间")
    values = _combine_channel(frames["Def_EntrTm"], channels["Def_EntrTm"], pd.Series(times))
    assert values.value.iloc[:2].tolist() == [0, 7200]
    assert pd.isna(values.value.iloc[2])
    assert channels["Def_EntrTm"]["missing"] == "none"
