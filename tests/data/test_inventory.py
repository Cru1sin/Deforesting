from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frost_analysis.config import load_app_config
from frost_analysis.data.inventory import classify_file, inventory_directory, read_monitoring_table
from frost_analysis.data.registry import load_feature_registry


def test_load_app_config_exposes_independent_stage_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_app_config(root / "configs" / "0715.yaml")
    assert config.date == "0715"
    assert config.paths.prepared_data.name == "prepared_data.parquet"
    assert config.paths.processed_data.name == "processed_data.parquet"
    assert config.paths.correlation_results.name == "correlation_results.csv"
    assert config.process.resample_interval_seconds == 30


def test_load_app_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "configs" / "0715.yaml"
    path = tmp_path / "invalid.yaml"
    path.write_text(source.read_text(encoding="utf-8") + "\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top level"):
        load_app_config(path)


def test_gb18030_tab_export_is_decoded_and_timed(tmp_path: Path) -> None:
    path = tmp_path / "2026-07-15 00-00-00参数1.xls"
    path.write_bytes(
        "时间\t环境温度\tDeforst\r\n"
        "2026-07-15 00:00:01\t-3.0\tOFF\r\n"
        "2026-07-15 00:00:02\t-3.1\tON\r\n".encode("gb18030")
    )
    table, metadata = read_monitoring_table(path)
    assert metadata.encoding == "gb18030"
    assert metadata.delimiter == "\t"
    assert table.shape == (2, 3)
    assert table.iloc[1, 1] == "-3.1"
    assert metadata.time_start == pd.Timestamp("2026-07-15 00:00:01")
    assert metadata.sampling_median_s == 1.0


def test_inventory_recurses_without_decoding_images(tmp_path: Path) -> None:
    sensor = tmp_path / "nested" / "x参数5.xls"
    sensor.parent.mkdir()
    sensor.write_text("时间\tTe\n2026-07-15 00:00:00\t-2\n", encoding="utf-8")
    image = tmp_path / "camera" / "bad.jpg"
    image.parent.mkdir()
    image.write_bytes(b"not-an-image")
    rows, columns = inventory_directory(tmp_path)
    assert set(rows["file_class"]) == {"monitoring_table", "image"}
    assert classify_file(image) == "image"
    assert rows.loc[rows["file_class"].eq("image"), "status"].item() == "classified_only"
    assert columns["source_column"].tolist() == ["Te"]


def test_registry_keeps_qcomp_and_excludes_ccq() -> None:
    registry = load_feature_registry(Path("configs/feature_registry.yaml"))
    assert registry["heating_capacity"].raw_source == "p1__QComp10W'2_32"
    assert all("CCQ_Comp" not in (item.raw_source or "") for item in registry.values())
