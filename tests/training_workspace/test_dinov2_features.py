import importlib.util

import numpy as np
import pandas as pd


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
