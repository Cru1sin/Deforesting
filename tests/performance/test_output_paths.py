from __future__ import annotations

import importlib.util
from pathlib import Path


def _module(name: str):
    path = Path("scripts/performance") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_performance_analysis_defaults_stay_under_cost_function_test_output() -> None:
    root = Path.cwd()
    expected = {
        "analyze_image_apparent_conductance": (
            root / "output/test/成本函数/其他/表观导热分析_cycles_020_030"
        ),
        "analyze_sensor_model_59": (
            root / "output/test/成本函数/其他/01_制热量与退化/传感器模型_59循环"
        ),
        "analyze_degradation_law": (
            root / "output/test/成本函数/其他/01_制热量与退化/退化规律_0至48循环"
        ),
    }

    for name, output in expected.items():
        assert output == _module(name).OUT
