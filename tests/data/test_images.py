from __future__ import annotations

from pathlib import Path

from frost_analysis.config import load_camera_mapping
from frost_analysis.data.images import build_image_manifest, extract_ip
from frost_analysis.data.inventory import inventory_directory


def _manifest(input_dir: Path, *, experiment_id: str = "exp_20260715"):
    inventory, _ = inventory_directory(input_dir)
    camera_roles, unknown_role = load_camera_mapping(Path("configs/camera_mapping.yaml"))
    return build_image_manifest(
        input_dir,
        inventory,
        experiment_id=experiment_id,
        camera_roles=camera_roles,
        unknown_role=unknown_role,
    )


def test_builds_manifest_from_inventory_with_explicit_and_unknown_camera_roles(
    tmp_path: Path, write_image
) -> None:
    write_image(tmp_path / "192.168.1.13_1" / "20260715120000123.jpg", size=(32, 18))
    write_image(tmp_path / "192.168.1.99_1" / "bad-name.jpg")

    manifest = _manifest(tmp_path)

    assert len(manifest) == 2
    known = manifest.loc[manifest["ip_address"] == "192.168.1.13"].iloc[0]
    unknown = manifest.loc[manifest["ip_address"] == "192.168.1.99"].iloc[0]
    assert known["camera_role"] == "中部正视"
    assert (known["width"], known["height"]) == (32, 18)
    assert bool(known["image_ok"]) is True
    assert bool(known["timestamp_ok"]) is True
    assert unknown["camera_role"] == "未映射"
    assert bool(unknown["timestamp_ok"]) is False
    assert known["image_path"] == "192.168.1.13_1/20260715120000123.jpg"


def test_manifest_reports_corrupt_images_and_has_stable_empty_schema(tmp_path: Path) -> None:
    corrupt = tmp_path / "named_camera" / "20260715120000123.png"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"not an image")

    manifest = _manifest(tmp_path, experiment_id="exp_test")
    empty = _manifest(tmp_path / "empty", experiment_id="exp_test")

    assert len(manifest) == 1
    assert bool(manifest.loc[0, "image_ok"]) is False
    assert manifest.loc[0, "image_error"]
    assert empty.empty
    assert empty.columns.tolist() == manifest.columns.tolist()


def test_manifest_keeps_ids_unique_across_repeated_ip_exports(tmp_path: Path, write_image) -> None:
    filename = "20260715120000123.jpg"
    write_image(tmp_path / "part_a" / "192.168.1.13_1" / filename)
    write_image(tmp_path / "part_b" / "192.168.1.13_2" / filename)

    manifest = _manifest(tmp_path)

    assert manifest["sample_id"].is_unique
    assert manifest["camera_id"].nunique() == 1
    assert extract_ip(Path("999.168.1.1_bad")) is None
    assert extract_ip(Path("camera_without_ip")) is None


def test_repository_camera_mapping_contains_six_confirmed_views() -> None:
    roles, unknown = load_camera_mapping(Path("configs/camera_mapping.yaml"))
    assert len(roles) == 6
    assert roles["192.168.1.1"] == "中上俯视"
    assert roles["192.168.1.14"] == "左侧斜视"
    assert unknown == "未映射"
