import json

import pytest

import output_layout as layout
from image_models import checkpoints
from tools import output_migration as migration


def test_camera_temporary_run_keeps_training_quality_and_archive_together(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "PROJECT", tmp_path)
    monkeypatch.setattr(layout, "OUTPUT", tmp_path / "output")
    output = tmp_path / "temp/test/camera"
    paths = layout.camera_outputs(output, "front")
    assert all(path.is_relative_to(output) for path in paths.values())
    formal = layout.OUTPUT / "01_核心结果/六机位比较"
    paths = layout.camera_outputs(formal, "front")
    assert (
        paths["training"]
        == layout.OUTPUT / layout.REPRODUCTION / "01_视觉模型训练/六机位_正面_LSTM/training"
    )
    assert paths["archive"] == layout.OUTPUT / layout.ARCHIVE / "training_data/六机位比较/front"


def test_archive_preserves_models_and_updates_dataset_references(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(layout, "PROJECT", tmp_path)
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    monkeypatch.setattr(layout, "OUTPUT", output)
    monkeypatch.setattr(migration, "OUTPUT", output)
    monkeypatch.setattr(checkpoints, "ROOT", output / layout.REPRODUCTION / "training_checkpoint")
    run = output / layout.REPRODUCTION / "training_checkpoint/obsolete"
    run.mkdir(parents=True)
    (run / "weights.pkl").write_bytes(b"frozen weights")
    ridge = tmp_path / "dataset/除霜预测模型/old"
    ridge.mkdir(parents=True)
    (ridge / "parameters.json").write_text('{"coefficient": 1.25}')
    source = tmp_path / "dataset/source.json"
    source.write_text(json.dumps({"model": str(ridge / "parameters.json")}))
    target = output / layout.ARCHIVE / "experiments/obsolete"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        migration.archive([run, ridge])
    assert run.exists() and ridge.exists()
    target.rmdir()
    migration.archive([run, ridge])
    assert (target / "weights.pkl").read_bytes() == b"frozen weights"
    archived_ridge = output / layout.ARCHIVE / "ridge_models/old/parameters.json"
    assert archived_ridge.read_text() == '{"coefficient": 1.25}'
    assert json.loads(source.read_text())["model"] == str(archived_ridge)
    assert not run.exists() and not ridge.exists()


def test_replay_output_cannot_write_published_data_or_models(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "PROJECT", tmp_path)
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    monkeypatch.setattr(layout, "OUTPUT", tmp_path / "output")
    for target in [
        tmp_path / "output" / layout.REPRODUCTION / "analysis/replay",
        tmp_path / "temp/test/replay",
    ]:
        assert layout.analysis_output(target, replay=True) == target
    for target in [
        tmp_path / "dataset/training_data/replay",
        tmp_path / "dataset/除霜预测模型/replay",
        tmp_path / "output/99_历史归档/淘汰共享材料/replay",
        tmp_path / "output/90_复现材料/01_视觉模型训练/replay",
    ]:
        with pytest.raises(ValueError, match="Replay output"):
            layout.analysis_output(target, replay=True)


def test_result_folders_move_nested_files_and_remain_findable(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(layout, "PROJECT", tmp_path)
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    monkeypatch.setattr(layout, "OUTPUT", output)
    monkeypatch.setattr(migration, "OUTPUT", output)
    monkeypatch.setattr(checkpoints, "ROOT", output / layout.REPRODUCTION / "training_checkpoint")
    source = output / "analysis/camera_LSTM25_FiLM_20261025"
    source.mkdir(parents=True)
    figure = source / "camera_core_grouped.png"
    figure.write_bytes(b"unchanged figure")
    (source / "comparison_summary.csv").write_text("gain\n1.2\n")
    reference = tmp_path / "reference.md"
    reference.write_text(str(figure))
    nested = output / "analysis/霜层等级分析/霜态构建_论文核心单图_20261012"
    nested.mkdir(parents=True)
    (nested / "source.csv").write_text("original")
    old = output / "archive/old"
    old.mkdir(parents=True)
    (old / "history.csv").write_text("frozen")
    migration.relocate()
    topic = output / "01_核心结果/六机位比较"
    assert (topic / "01_共同集与跨工况比较.png").read_bytes() == b"unchanged figure"
    assert reference.read_text() == str(topic / "01_共同集与跨工况比较.png")
    assert (
        output / "03_数据与物理依据/视觉霜态构建_Harmony与MST/source.csv"
    ).read_text() == "original"
    assert (output / layout.ARCHIVE / "old/history.csv").read_text() == "frozen"
    assert not (output / "index.html").exists()
    assert (topic / "comparison_summary.csv").is_file()
    with pytest.raises(ValueError, match="actual named result"):
        layout.analysis_output(output / "analysis/camera_LSTM25_FiLM_20261025")
    assert layout.analysis_output(topic) == topic
    with pytest.raises(ValueError, match="Model output"):
        layout.training_output(output / "new_model")
    assert not migration.relocation_plan()


def test_main_model_output_is_exact_or_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(layout, "PROJECT", tmp_path)
    monkeypatch.setattr(migration, "PROJECT", tmp_path)
    monkeypatch.setattr(layout, "OUTPUT", tmp_path / "output")
    for root in (
        tmp_path / "temp/test",
        tmp_path / "output" / layout.REPRODUCTION / "01_视觉模型训练",
    ):
        target = root / "demo"
        assert layout.model_output(target) == target
        assert (
            checkpoints.checkpoint_path(target / "training", "LSTM.pkl")
            == target / "checkpoints/LSTM/LSTM.pkl"
        )
    with pytest.raises(ValueError, match="Model output"):
        layout.model_output(tmp_path / "output/demo")
