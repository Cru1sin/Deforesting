from __future__ import annotations

import importlib.util
import inspect
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/rgb/process_rgb_cycle_assets.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("process_rgb_cycle_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_module():  # type: ignore[no-untyped-def]
    path = Path("scripts/rgb/extract_rgb_feature_shards.py")
    spec = importlib.util.spec_from_file_location("extract_rgb_feature_shards_defaults", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_task_shards_creates_finite_binary_and_three_class_targets(tmp_path) -> None:
    module = _module()
    shard = pd.DataFrame(
        {
            "cost_state": ["pre_optimal", "near_optimal", "post_optimal"],
            "feature_000": [0.0, 1.0, 2.0],
        }
    )

    outputs = module.write_task_shards(shard, tmp_path, "frost_cycle_000001")

    binary = pd.read_parquet(outputs["binary"])
    three = pd.read_parquet(outputs["three"])
    assert binary["target"].tolist() == [0, 1]
    assert three["target"].tolist() == [0, 1, 2]
    assert binary["target"].notna().all() and three["target"].notna().all()


def test_copy_optimal_views_requires_and_exports_six_readable_images(tmp_path) -> None:
    module = _module()
    cycle_name = "frost_cycle_000001"
    cycle_dir = tmp_path / "images" / cycle_name
    rows = []
    for role in module.ROLE_ORDER:
        source = cycle_dir / role / f"{role}.jpg"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), (120, 130, 140)).save(source)
        rows.append(
            {
                "cycle_name": cycle_name,
                "camera_role": role,
                "file_name": source.name,
                "relative_path": f"{cycle_name}/{role}/{source.name}",
                "available": True,
            }
        )

    manifest_path = module.copy_optimal_views(
        cycle_dir, pd.DataFrame(rows), tmp_path / "optimal"
    )

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 6
    assert manifest["exported"].all()
    for relative_path in manifest["relative_path"]:
        with Image.open(tmp_path / "optimal" / relative_path) as image:
            image.verify()


def test_copy_optimal_views_records_unavailable_roles(tmp_path: Path) -> None:
    module = _module()
    cycle_name = "frost_cycle_000007"
    cycle_dir = tmp_path / "images" / cycle_name
    available_role = module.ROLE_ORDER[0]
    source = cycle_dir / available_role / "frame.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (120, 130, 140)).save(source)
    manifest = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": role,
                "file_name": "frame.jpg" if role == available_role else None,
                "relative_path": f"{cycle_name}/{role}/frame.jpg",
                "available": role == available_role,
            }
            for role in module.ROLE_ORDER
        ]
    )

    manifest_path = module.copy_optimal_views(cycle_dir, manifest, tmp_path / "optimal")

    exported = pd.read_csv(manifest_path)
    assert exported["camera_role"].tolist() == list(module.ROLE_ORDER)
    assert exported["exported"].tolist() == [True, False, False, False, False, False]


def test_copy_optimal_views_rejects_incomplete_role_manifest(tmp_path: Path) -> None:
    module = _module()
    manifest = pd.DataFrame(
        [
            {
                "cycle_name": "frost_cycle_000007",
                "camera_role": role,
                "available": True,
            }
            for role in module.ROLE_ORDER[:-1]
        ]
    )

    with pytest.raises(ValueError, match="six roles"):
        module.copy_optimal_views(tmp_path, manifest, tmp_path / "optimal")


def test_copy_optimal_views_records_cycle_without_available_images(tmp_path: Path) -> None:
    module = _module()
    cycle_name = "frost_cycle_000007"
    output = tmp_path / "optimal"
    manifest = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": role,
                "available": False,
            }
            for role in module.ROLE_ORDER
        ]
    )

    manifest_path = module.copy_optimal_views(tmp_path, manifest, output)

    exported = pd.read_csv(manifest_path)
    assert len(exported) == 6
    assert not exported["available"].any()
    assert not exported["exported"].any()
    assert set(output.rglob("*")) == {output / cycle_name, manifest_path}


def test_copy_optimal_views_rejects_missing_available_source(tmp_path: Path) -> None:
    module = _module()
    cycle_name = "frost_cycle_000007"
    manifest = pd.DataFrame(
        [
            {
                "cycle_name": cycle_name,
                "camera_role": role,
                "file_name": "missing.jpg",
                "relative_path": f"{cycle_name}/{role}/missing.jpg",
                "available": True,
            }
            for role in module.ROLE_ORDER
        ]
    )

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        module.copy_optimal_views(tmp_path, manifest, tmp_path / "optimal")


def test_main_passes_minimum_free_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module, "process_cycle_assets", lambda *args, **kwargs: seen.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_rgb_cycle_assets.py",
            "--cycles",
            "frost_cycle_000001",
            "--minimum-free-gib",
            "7.5",
        ],
    )

    module.main()

    assert seen["minimum_free_gib"] == 7.5


def test_process_defaults_use_full_representations_and_safe_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    seen: dict[str, object] = {}
    defaults = inspect.signature(module.process_cycle_assets).parameters
    monkeypatch.setattr(
        module, "process_cycle_assets", lambda *args, **kwargs: seen.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["process_rgb_cycle_assets.py", "--cycles", "frost_cycle_000001"],
    )

    module.main()

    assert defaults["maximum_per_group"].default == 48
    assert defaults["minimum_free_gib"].default == 21
    assert seen["maximum_per_group"] == 48
    assert seen["minimum_free_gib"] == 21
    assert tuple(seen["backbones"]) == module.DEEP_REPRESENTATIONS


def test_backbones_help_explains_expensive_default_and_opt_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setattr(sys, "argv", ["process_rgb_cycle_assets.py", "--help"])

    with pytest.raises(SystemExit):
        module.main()

    help_text = " ".join(capsys.readouterr().out.split())
    assert "all 7 pretrained representations" in help_text
    assert "bare --backbones disables" in help_text


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [(["--backbones"], []), (["--backbones", "dinov2"], ["dinov2"])],
)
def test_process_allows_explicit_empty_or_subset_backbones(
    arguments: list[str],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        module, "process_cycle_assets", lambda *args, **kwargs: seen.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_rgb_cycle_assets.py",
            "--cycles",
            "frost_cycle_000001",
            *arguments,
        ],
    )

    module.main()

    assert seen["backbones"] == expected


def test_extract_defaults_pass_sampling_and_free_space_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _extract_module()
    seen: dict[str, object] = {}
    labels = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000001"],
            "camera_role": ["top"],
            "file_name": ["frame.jpg"],
            "image_time": [pd.Timestamp("2026-01-01")],
            "relative_regret": [0.1],
            "cost_state_01pct": ["pre_optimal"],
        }
    )

    def materialize(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["minimum_free_gib"] = kwargs.get("minimum_free_gib")
        return nullcontext(tmp_path)

    def shard(rows, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen["maximum_per_group"] = kwargs["maximum_per_group"]
        return rows.copy(), pd.DataFrame()

    monkeypatch.setattr(module.pd, "read_parquet", lambda path: labels.copy())
    monkeypatch.setattr(module, "materialize_cycle_images", materialize)
    monkeypatch.setattr(module, "cycle_feature_shard", shard)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_rgb_feature_shards.py",
            "--labels",
            str(tmp_path / "labels.parquet"),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    module.main()

    assert seen == {"minimum_free_gib": 21, "maximum_per_group": 48}
