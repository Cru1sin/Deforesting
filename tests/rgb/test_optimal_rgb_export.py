from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _module():  # type: ignore[no-untyped-def]
    path = Path("scripts/rgb/export_optimal_rgb_views.py")
    spec = importlib.util.spec_from_file_location("optimal_rgb_export", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_selects_six_roles_near_earliest_inverse_cop_minimum() -> None:
    module = _module()
    optimum = pd.Timestamp("2026-01-01 00:10")
    curves = pd.DataFrame(
        {
            "cycle_name": ["partial_complete"] * 3,
            "candidate_time": [optimum - pd.Timedelta(minutes=1), optimum, optimum],
            "inverse_cop": [0.6, 0.5, 0.5],
        }
    )
    metadata = pd.DataFrame(
        [
            {
                "cycle_name": "partial_complete",
                "camera_role": role,
                "file_name": f"{role}.jpg",
                "image_time": optimum + pd.Timedelta(seconds=index),
            }
            for index, role in enumerate(module.RGB_CAMERA_ORDER)
        ]
    )

    manifest = module.build_optimal_view_manifest(curves, metadata)

    assert manifest["camera_role"].tolist() == list(module.RGB_CAMERA_ORDER)
    assert manifest["available"].all()
    assert manifest["target_time"].eq(optimum).all()
    assert manifest["relative_path"].tolist() == [
        f"partial_complete/{role}/{role}.jpg" for role in module.RGB_CAMERA_ORDER
    ]
    assert manifest["relative_regret"].eq(0.0).all()
    assert manifest["source_relative_path"].tolist() == [
        f"images/partial_complete/{role}/{role}.jpg" for role in module.RGB_CAMERA_ORDER
    ]
    summary = module.optimal_view_report(manifest.assign(exported=True))
    assert "完整观测循环：1" in summary
    assert "已导出图像：6/6" in summary


def test_manifest_keeps_missing_or_more_than_two_minute_roles_as_rows() -> None:
    module = _module()
    optimum = pd.Timestamp("2026-01-01 00:10")
    curves = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "candidate_time": [optimum],
            "inverse_cop": [0.5],
        }
    )
    metadata = pd.DataFrame(
        {
            "cycle_name": ["cycle"],
            "camera_role": [module.RGB_CAMERA_ORDER[0]],
            "file_name": ["far.jpg"],
            "image_time": [optimum + pd.Timedelta(seconds=121)],
        }
    )

    manifest = module.build_optimal_view_manifest(curves, metadata)

    assert len(manifest) == 6
    assert not manifest["available"].any()
