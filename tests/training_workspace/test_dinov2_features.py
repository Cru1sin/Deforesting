import importlib.util

import numpy as np
import pandas as pd
import pytest


def test_exact_key_cache_is_teacher_independent_and_additive(tmp_path):
    assert importlib.util.find_spec("image_models.dinov2_features") is not None
    from image_models.dinov2_features import DINO_COLUMNS, KEYS, merge_cached_features

    rows = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "camera_role": ["front"] * 2,
            "file_name": ["old.jpg", "new.jpg"],
        }
    )
    old = rows.iloc[:1].assign(target=1)
    old = pd.concat(
        [old.reset_index(drop=True), pd.DataFrame(np.ones((1, 384)), columns=DINO_COLUMNS)], axis=1
    )
    path = tmp_path / "cycle.parquet"
    old.to_parquet(path)
    result = merge_cached_features(rows, [path])
    assert result.columns.tolist() == KEYS + DINO_COLUMNS
    assert result.loc[0, DINO_COLUMNS].eq(1).all()
    assert result.loc[1, DINO_COLUMNS].isna().all()


def test_grouped_zip_payload_matches_direct_member(tmp_path):
    from zipfile import ZIP_DEFLATED, ZipFile

    from dataset_tools.cloud_images import _read_zip_member, _zip_member_index

    archive = tmp_path / "cycle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("front/a.jpg", b"rgb" * 100)
    size = archive.stat().st_size
    member = _zip_member_index(archive, size)["front/a.jpg"]
    direct = _read_zip_member(archive, member, archive_size=size)
    grouped = _read_zip_member(
        archive,
        member,
        archive_size=size,
        payload=memoryview(archive.read_bytes())[member[0] :],
    )
    assert direct == grouped == b"rgb" * 100


@pytest.mark.parametrize("camera", ["front", "top", "top_close", "left", "left_close", "extreme"])
def test_local_only_leaves_missing_images_blank_without_cloud(tmp_path, monkeypatch, camera):
    import torch

    from image_models import dinov2_features as features

    rows = pd.DataFrame(
        {
            "cycle_name": ["cycle", "cycle"],
            "camera_role": [camera] * 2,
            "file_name": ["local.jpg", "missing.jpg"],
        }
    )
    rows = rows.reindex(columns=features.KEYS + features.DINO_COLUMNS)
    folder = tmp_path / f"images/cycle/{camera}"
    folder.mkdir(parents=True)
    (folder / "local.jpg").write_bytes(b"local")
    monkeypatch.setattr(features, "_plan_image_members", lambda *a: pytest.fail("cloud called"))
    monkeypatch.setattr(features, "_embed", lambda payloads, *a: np.ones((len(payloads), 384)))
    result, audit = features._complete_cycle(
        rows,
        dataset=tmp_path,
        model=None,
        device=torch.device("cpu"),
        n_jobs=1,
        batch_size=2,
        local_only=True,
    )
    assert result.loc[0, features.DINO_COLUMNS].eq(1).all()
    assert result.loc[1, features.DINO_COLUMNS].isna().all()
    assert audit["generated_count"] == 1 and audit["cloud_source_count"] == 0
    assert audit["camera_role"] == camera


def test_camera_keys_do_not_mix_identical_filenames(tmp_path):
    from image_models.dinov2_features import DINO_COLUMNS, merge_cached_features

    keys = pd.DataFrame(
        dict(cycle_name=["cycle"] * 2, camera_role=["front", "top"], file_name=["same.jpg"] * 2)
    )
    values = np.repeat([[1.0], [2.0]], 384, axis=1)
    saved = pd.concat([keys, pd.DataFrame(values, columns=DINO_COLUMNS)], axis=1)
    path = tmp_path / "cycle.parquet"
    saved.to_parquet(path, index=False)
    result = merge_cached_features(keys.iloc[[1]], [path])
    assert result.camera_role.tolist() == ["top"]
    assert result[DINO_COLUMNS].eq(2.0).all().all()


def test_remote_request_keeps_camera_directory(tmp_path, monkeypatch):
    import torch

    from image_models import dinov2_features as features

    rows = pd.DataFrame(
        dict(cycle_name=["frost_cycle_1"], camera_role=["left_close"], file_name=["same.jpg"])
    ).reindex(columns=features.KEYS + features.DINO_COLUMNS)
    calls = []

    def plan(dataset, cycle, names, *args):
        calls.append(names)
        return [], [0], 0.0

    monkeypatch.setattr(features, "_plan_image_members", plan)
    with pytest.raises(FileNotFoundError, match="unavailable"):
        features._complete_cycle(
            rows, dataset=tmp_path, model=None, device=torch.device("cpu"), n_jobs=1, batch_size=1
        )
    assert calls == [["left_close/same.jpg"]]


def test_multicamera_cli_resume_preserves_front_cache(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from dataset_tools.local_images import RGB_CAMERA_ORDER
    from image_models import dinov2_features as features

    metadata = pd.DataFrame(
        dict(
            cycle_name=["frost_cycle_1"] * 6,
            camera_role=RGB_CAMERA_ORDER,
            file_name=["same.jpg"] * 6,
        )
    )
    saved = pd.concat(
        [
            metadata,
            pd.DataFrame(
                np.repeat(np.arange(6.0)[:, None], 384, axis=1), columns=features.DINO_COLUMNS
            ),
        ],
        axis=1,
    )
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    saved.to_parquet(source / "frost_cycle_1.parquet", index=False)
    front = saved.loc[saved.camera_role.eq("front")]
    front.to_parquet(output / "frost_cycle_1.parquet", index=False)
    images = tmp_path / "images"
    images.mkdir()
    metadata.to_parquet(images / "image_metadata.parquet", index=False)
    loader = SimpleNamespace(
        load_image_metadata=lambda: metadata,
        list_cycles=lambda **kwargs: metadata[["cycle_name"]].drop_duplicates(),
        manifest={"dataset_schema_version": 3},
    )
    monkeypatch.setattr(features, "DatasetLoader", lambda *a: loader)
    monkeypatch.setattr(
        features, "_load_backbone", lambda *a: pytest.fail("re-embedded cached image")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "features",
            "--dataset",
            str(tmp_path),
            "--output",
            str(output),
            "--source-cache",
            str(source),
            "--camera-roles",
            *RGB_CAMERA_ORDER,
            "--local-only",
        ],
    )
    for _ in range(2):
        assert features.main() == 0
        actual = (
            pd.read_parquet(output / "frost_cycle_1.parquet")
            .sort_values("camera_role")
            .reset_index(drop=True)
        )
        expected = saved.sort_values("camera_role").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_unreadable_image_embedding_stays_blank():
    import torch

    from image_models.dinov2_features import _embed

    values = _embed([b"broken jpeg"], None, torch.device("cpu"))
    assert values.shape == (1, 384) and np.isnan(values).all()
