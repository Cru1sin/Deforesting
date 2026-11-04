from pathlib import Path

import pandas as pd

from dataset_tools import DatasetLoader


def test_append_publication_keeps_existing_inputs_and_rejects_changed_membership(
    tmp_path, monkeypatch
):
    import json

    import pytest

    import dataset_tools.training_data as publication

    dataset = tmp_path / "dataset"
    training = dataset / "training_data"
    training.mkdir(parents=True)
    admission = tmp_path / "admission"
    admission.mkdir()
    prior = pd.DataFrame(dict(cycle_name=["old"], evaluation_eligible=[True]))
    prior.to_csv(training / "cohort.csv", index=False)
    full = pd.DataFrame(
        dict(cycle_name=["old", "new"], measurement_processing="dataset_offline_linear_60s")
    )
    full.to_parquet(training / "candidate_decisions.parquet", index=False)
    full.to_parquet(admission / "full.parquet", index=False)
    pd.DataFrame(dict(cycle_name=["old", "new"])).to_csv(admission / "quality.csv", index=False)
    pd.DataFrame(dict(cycle_name=["old", "new"])).to_parquet(
        admission / "decision_inputs.parquet", index=False
    )
    settings = dict(
        input_rule="rgb70",
        limits={},
        decisions=str(admission / "full.parquet"),
        quality=str(admission / "quality.csv"),
        exclude_rb=False,
        minimum_terminal_decline=None,
        cohort_rule="observed",
    )
    (admission / "admission_settings.json").write_text(json.dumps(settings))

    class Loader:
        registry = {"measurement_processing": "dataset_offline_linear_60s"}

        def list_cycles(self):
            return full

    monkeypatch.setattr(publication, "DatasetLoader", lambda path: Loader())
    selected = pd.DataFrame(dict(cycle_name=["old", "new"], selected=[True, False]))
    monkeypatch.setattr(publication, "screen_cv_cycles", lambda *a, **kw: selected)
    before = (training / "cohort.csv").read_bytes()
    result, _ = publication.publish_training_data(dataset, admission, append=True)
    assert (
        result.cycle_name.tolist() == ["old"] and (training / "cohort.csv").read_bytes() == before
    )
    import plots.dataset as maintenance

    rendered = []
    monkeypatch.setattr(
        maintenance,
        "render_effective_cop_assets",
        lambda dataset, path, **kw: rendered.extend(pd.read_parquet(path).cycle_name),
    )
    publication.publish_training_data(dataset, admission, append=True, figures=True)
    assert rendered == ["old", "new"]  # Rejected new cycles also need Dataset COP figures.
    assert (training / "cohort.csv").read_bytes() == before
    selected["selected"] = False
    with pytest.raises(ValueError, match="historical admission"):
        publication.publish_training_data(dataset, admission, append=True)


def test_training_loader_is_self_contained_and_holds_out_whole_experiments(tmp_path):
    loader = object.__new__(DatasetLoader)
    loader.dataset_root = tmp_path
    for name in ("training_data", "image_DINO_feature"):
        (tmp_path / name).mkdir()
    cohort = pd.DataFrame(
        dict(
            cycle_name=["a", "b", "c", "d"],
            experiment_id=["one", "one", "two", "three"],
            complete_reward=[True, False, True, True],
            evaluation_eligible=[True, False, True, True],
        )
    )
    cohort.to_csv(tmp_path / "training_data/cohort.csv", index=False)
    for row in cohort.itertuples():
        table = pd.DataFrame(
            dict(
                cycle_name=[row.cycle_name],
                experiment_id=[row.experiment_id],
                file_name=["x.jpg"],
                candidate_defrost_time=[pd.Timestamp("2026-01-01")],
                input_valid=[True],
            )
        )
        table.to_parquet(tmp_path / "training_data" / f"{row.cycle_name}.parquet", index=False)
        features = pd.DataFrame(
            dict(
                cycle_name=[row.cycle_name],
                camera_role=["front"],
                file_name=["x.jpg"],
                **{f"dinov2_{i:03d}": [float(i)] for i in range(384)},
                **{f"dinov2_patchmean_{i:03d}": [float(2 * i)] for i in range(384)},
            )
        )
        features.to_parquet(
            tmp_path / "image_DINO_feature" / f"{row.cycle_name}.parquet", index=False
        )
    cohort.to_parquet(tmp_path / "training_data/reference.parquet", index=False)
    rows, reference = loader.load_training_data(objective="set_stop")
    assert len(reference) == 4 and rows.dinov2_383.eq(383).all()
    patch, _ = loader.load_training_data(
        visual_columns=[f"dinov2_patchmean_{i:03d}" for i in range(384)]
    )
    assert patch.dinov2_patchmean_383.eq(766).all() and "dinov2_383" not in patch
    assert patch.input_valid.equals(rows.input_valid)
    split = loader.split_training_data(rows, "one", ["two"])
    assert set(split.loc[split.split.eq("test"), "cycle_name"]) == {"a"}
    assert not split.loc[split.split.eq("train"), "experiment_id"].isin(["one", "two"]).any()
    assert split.loc[split.cycle_name.eq("b"), "split"].item() == "excluded"
    cdf, _ = loader.load_training_data(objective="cdf_ramp")
    assert cdf.training_eligible.all()
    import pytest

    with pytest.raises(ValueError, match="unknown training objective"):
        loader.load_training_data(objective="expected_cop")


def test_loader_folds_never_put_outer_experiment_in_validation():
    rows = pd.DataFrame(
        dict(cycle_name=list("abcdef"), experiment_id=list("abcdef"), evaluation_eligible=True)
    )
    folds = DatasetLoader.training_folds(rows, validation_count=5)
    assert all(
        test not in validation and len(set(validation)) == 4 for test, validation in folds.items()
    )


def test_validation_contains_reviewers_not_just_next_training_experiment():
    rows = pd.DataFrame(
        dict(
            cycle_name=list("abcdef"),
            experiment_id=list("abcdef"),
            evaluation_eligible=[True, False, False, False, False, True],
        )
    )
    folds = DatasetLoader.training_folds(rows, validation_count=2)
    assert folds["a"][0] == "f" and folds["f"][0] == "a"


def test_experiment_records_purpose_and_rejects_changed_configuration(tmp_path):
    import json

    import pytest

    from image_models.experiments import record_experiment

    run = tmp_path / "output" / "gru"
    record_experiment(run, {"seed": 0}, "Test causal history")
    assert json.loads((run / "settings.json").read_text())["purpose"] == "Test causal history"
    assert "Test causal history" in (run / "README.md").read_text()
    assert (run / "log.txt").read_text()
    with pytest.raises(ValueError, match="settings changed"):
        record_experiment(run, {"seed": 1}, "Test causal history")
    with pytest.raises(ValueError, match="temp/test"):
        record_experiment(tmp_path / "output/test/smoke", {}, "Smoke")


import pytest


@pytest.mark.slow
def test_published_model_data_reads_only_dataset(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "dataset"
    if not (root / "training_data/cohort.csv").exists():
        pytest.skip("No published research Dataset")
    loader = DatasetLoader(root / "training_data")
    for name in ["read_csv", "read_parquet"]:
        original = getattr(pd, name)

        def read(path, *args, _original=original, **kwargs):
            assert Path(path).resolve().is_relative_to(root.resolve())
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(pd, name, read)
    rows, reference = loader.load_training_data()
    assert len(reference) and rows.cycle_name.nunique() == len(loader.training_cohort())
    assert (
        rows.loc[rows.input_valid, "image_time"]
        .le(rows.loc[rows.input_valid, "candidate_defrost_time"])
        .all()
    )
    for test, validation in loader.training_folds(rows).items():
        split = loader.split_training_data(rows, test, validation)
        assert not set(split.loc[split.split.eq("train"), "experiment_id"]) & {test, *validation}
        assert split.loc[split.split.eq("test"), "evaluation_eligible"].all()


def test_training_loader_uses_camera_specific_actions_and_vectors(tmp_path):
    loader = object.__new__(DatasetLoader)
    loader.dataset_root = tmp_path
    training = tmp_path / "training_data"
    training.mkdir()
    features = tmp_path / "image_DINO_feature"
    features.mkdir()
    pd.DataFrame(
        dict(
            cycle_name=["a", "a", "b"],
            camera_role=["front", "top", "top"],
            complete_reward=True,
            evaluation_eligible=True,
        )
    ).to_csv(training / "cohort.csv", index=False)
    for name in ["a", "b"]:
        for camera, value in [("front", 1.0), ("top", 2.0)]:
            suffix = "" if camera == "front" else "__top"
            pd.DataFrame(
                dict(
                    cycle_name=[name],
                    file_name=[f"{camera}.jpg"],
                    input_valid=True,
                    image_time=[pd.Timestamp("2026-01-01")],
                )
            ).to_parquet(training / f"{name}{suffix}.parquet", index=False)
        pd.DataFrame(
            dict(
                cycle_name=[name, name],
                camera_role=["front", "top"],
                file_name=["front.jpg", "top.jpg"],
                dinov2_000=[1.0, 2.0],
            )
        ).to_parquet(features / f"{name}.parquet", index=False)
    pd.DataFrame(dict(cycle_name=["a", "b"])).to_parquet(
        training / "reference.parquet", index=False
    )
    front, _ = loader.load_training_data(visual_columns=["dinov2_000"])
    top, _ = loader.load_training_data(camera_role="top", visual_columns=["dinov2_000"])
    assert front.cycle_name.tolist() == ["a"] and front.dinov2_000.eq(1.0).all()
    assert top.cycle_name.tolist() == ["a", "b"] and top.dinov2_000.eq(2.0).all()


@pytest.mark.slow
def test_six_camera_publication_on_real_cached_cycles(tmp_path, monkeypatch):
    """Real input audit → authoritative admission → flat publication → camera loader/folds."""
    import json
    import shutil
    from types import SimpleNamespace

    from dataset_tools.cycle_metadata import image_root
    from image_models import camera_data as study

    root = Path(__file__).resolve().parents[2] / "dataset"
    source = DatasetLoader(root)
    if not image_root(root).exists():
        pytest.skip("External images are not available")
    selected = []
    experiments = set()
    for row in source.training_cohort().itertuples():
        if row.experiment_id in experiments:
            continue
        path = root / "image_DINO_feature" / f"{row.cycle_name}.parquet"
        roles = pd.read_parquet(path, columns=["camera_role"]).camera_role.unique()
        if set(roles) == set(study.CAMERAS):
            selected.append(row.cycle_name)
            experiments.add(row.experiment_id)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        pytest.skip("Need four experiments with completed six-camera frozen inputs")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ["cycles", "cycles_original"]:
        (dataset / name).symlink_to(root / name, target_is_directory=True)
    shutil.copy2(root / "channel_registry.json", dataset / "channel_registry.json")
    manifest = source.manifest
    manifest["images_root"] = str(image_root(root).resolve())
    (dataset / "dataset_manifest.json").write_text(json.dumps(manifest))
    catalog = source.catalog
    catalog["cycles"] = [r for r in catalog["cycles"] if r["cycle_name"] in selected]
    (dataset / "cycle_catalog.json").write_text(json.dumps(catalog))
    training = dataset / "training_data"
    training.mkdir()
    (training / "settings.json").write_text(json.dumps(source.training_settings))
    source.training_cohort().iloc[:0].to_csv(training / "cohort.csv", index=False)
    full = pd.read_parquet(root / "training_data/candidate_decisions.parquet")
    full.loc[full.cycle_name.isin(selected)].to_parquet(
        training / "candidate_decisions.parquet", index=False
    )
    features = dataset / "image_DINO_feature"
    features.mkdir()
    for name in selected:
        shutil.copy2(root / "image_DINO_feature" / f"{name}.parquet", features / f"{name}.parquet")
    output = tmp_path / "comparison"
    output.mkdir()
    monkeypatch.setattr(
        study, "run", lambda command: None
    )  # All fixture features are already frozen.
    study.prepare(
        SimpleNamespace(dataset=dataset, output=output, device="cpu", batch_size=64, n_jobs=1)
    )
    loader = DatasetLoader(dataset)
    for camera in study.CAMERAS:
        rows, reference = loader.load_training_data(camera_role=camera)
        assert set(rows.cycle_name) == set(selected) and len(reference)
        assert (
            rows.loc[rows.input_valid, "image_time"]
            .le(rows.loc[rows.input_valid, "candidate_defrost_time"])
            .all()
        )
    assert set(pd.read_csv(output / "intersection.csv").cycle_name) == set(selected)


def test_image_metadata_follows_configured_external_images(tmp_path):
    from dataset_tools.dataset_paths import write_json

    root = tmp_path / "dataset"
    images = tmp_path / "external_images"
    for path in (root / "cycles", root / "cycles_original", images):
        path.mkdir(parents=True)
    write_json(
        {
            "dataset_schema_version": 3,
            "dataset_id": "frost_cycle_dataset",
            "experiments": [],
            "images_root": str(images),
        },
        root / "dataset_manifest.json",
    )
    write_json({"cycles": []}, root / "cycle_catalog.json")
    write_json({}, root / "channel_registry.json")
    pd.DataFrame({"cycle_name": ["example"]}).to_parquet(images / "image_metadata.parquet")
    loader = DatasetLoader(root)
    assert loader.load_image_metadata().cycle_name.tolist() == ["example"]
    assert not (root / "images").exists()
    (images / "image_metadata.parquet").unlink()
    offline = DatasetLoader(root)
    assert offline.list_cycles().empty
    with pytest.raises(FileNotFoundError, match="image metadata"):
        offline.load_image_metadata()
