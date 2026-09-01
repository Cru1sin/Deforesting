from pathlib import Path

from frost_analysis.dataset.config import Config
from frost_analysis.dataset.raw import discover_inputs


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
        channels_path=tmp_path / "channels.yaml",
        sensor_globs=("*.xls",),
        image_extensions=(".jpg",),
        timestamp_column="time",
        expected_sensor_interval_seconds=1,
        image_match_tolerance_seconds=2,
        edf_pair_tolerance_seconds=1.0,
        cycles={},
        process={},
    )

    inputs = discover_inputs(config)

    assert inputs.sensor_files == (raw / "parameters.xls",)
    assert inputs.image_files == (image,)
