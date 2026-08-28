from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/labels/audit_rgb_cycle_assets.py")
    spec = importlib.util.spec_from_file_location("audit_rgb_cycle_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_covers_complete_allowed_missing_and_missing_shard(tmp_path: Path) -> None:
    module = _module()
    dataset = tmp_path / "dataset"
    root = tmp_path / "transaction"
    output = tmp_path / "audit"
    cycles = ["frost_cycle_000001", "frost_cycle_000007", "frost_cycle_000003"]
    records = []
    for cycle in cycles:
        panel = dataset / "cycles" / f"{cycle}_rgb_panel.png"
        panel.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2)).save(panel)
        records.append(
            {"cycle_name": cycle, "assets": {"rgb_panel": f"cycles/{panel.name}"}}
        )
        rows = []
        for role in module.ROLES:
            available = not (cycle == "frost_cycle_000007" and role == "extreme")
            relative_path = f"{cycle}/{role}/{role}.jpg" if available else ""
            rows.append(
                {
                    "cycle_name": cycle,
                    "camera_role": role,
                    "available": available,
                    "exported": available,
                    "relative_path": relative_path,
                }
            )
            if available:
                image = root / "optimal" / relative_path
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2)).save(image)
        target = root / "optimal" / cycle / "optimal_rgb_views_manifest.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target, index=False)
        for task in ("binary", "three"):
            if cycle == "frost_cycle_000003" and task == "binary":
                continue
            shard = root / "features" / task / "cycles" / f"{cycle}.parquet"
            shard.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"value": [1, 2]}).to_parquet(shard, index=False)
    (dataset / "cycle_catalog.json").write_text(
        json.dumps({"cycles": records}), encoding="utf-8"
    )
    labels = tmp_path / "labels.parquet"
    pd.DataFrame({"cycle_name": cycles, "relative_regret": [0.0, 0.1, 0.2]}).to_parquet(
        labels, index=False
    )

    audit, views = module.audit_rgb_cycle_assets(dataset, labels, root, output)

    passed = audit.set_index("cycle_name")["passed"].to_dict()
    assert passed == {
        "frost_cycle_000001": True,
        "frost_cycle_000003": False,
        "frost_cycle_000007": True,
    }
    assert len(views) == 18
    assert audit.set_index("cycle_name").loc["frost_cycle_000001", "views_export_readable"] == 6
    missing = views.loc[
        views["cycle_name"].eq("frost_cycle_000007") & views["camera_role"].eq("extreme")
    ].iloc[0]
    assert not missing["available"] and missing["allowed_missing"] and missing["view_pass"]
    assert (output / "cycle_asset_audit.csv").is_file()
    assert (output / "optimal_rgb_views_manifest.csv").is_file()
    absent, exists, valid = module._cycle_views("absent", root)
    assert len(absent) == 6 and not exists and not valid
    broken = root / "optimal" / "frost_cycle_000001" / "top" / "top.jpg"
    broken.unlink()
    missing_audit, missing_views = module.audit_rgb_cycle_assets(dataset, labels, root, output)
    assert not missing_audit.set_index("cycle_name").loc["frost_cycle_000001", "passed"]
    assert not missing_views.loc[
        missing_views["cycle_name"].eq("frost_cycle_000001")
        & missing_views["camera_role"].eq("top"),
        "view_pass",
    ].item()
    broken.write_bytes(b"not an image")
    corrupt_audit, _ = module.audit_rgb_cycle_assets(dataset, labels, root, output)
    assert not corrupt_audit.set_index("cycle_name").loc["frost_cycle_000001", "passed"]
    with pytest.raises(RuntimeError, match="frost_cycle_000003"):
        module.audit_rgb_cycle_assets(dataset, labels, root, output, strict=True)


def test_empty_labels_write_stable_headers_and_strict_fails(tmp_path: Path) -> None:
    module = _module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "cycle_catalog.json").write_text('{"cycles": []}', encoding="utf-8")
    labels = tmp_path / "labels.parquet"
    pd.DataFrame({"cycle_name": ["cycle"], "relative_regret": [None]}).to_parquet(labels)
    output = tmp_path / "audit"

    audit, views = module.audit_rgb_cycle_assets(dataset, labels, tmp_path, output)

    assert tuple(audit.columns) == module.CYCLE_COLUMNS
    assert tuple(views.columns) == module.EMPTY_VIEW_COLUMNS
    assert pd.read_csv(output / "cycle_asset_audit.csv").empty
    assert pd.read_csv(output / "optimal_rgb_views_manifest.csv").empty
    with pytest.raises(RuntimeError, match="no cycles with valid relative_regret"):
        module.audit_rgb_cycle_assets(dataset, labels, tmp_path, output, strict=True)
