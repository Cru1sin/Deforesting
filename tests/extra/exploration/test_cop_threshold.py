from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def test_best_threshold_uses_first_downward_crossing() -> None:
    path = Path("scripts/exploration/analyze_cop_threshold.py")
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("analyze_cop_threshold", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    best, errors = module.best_threshold(
        [
            (np.array([0, 5, 10]), np.array([3.0, 2.0, 1.0]), 5.0),
            (np.array([0, 5, 10]), np.array([2.8, 2.2, 1.8]), 10.0),
        ],
        np.array([2.0, 2.2]),
    )

    assert best == 2.0
    assert errors.tolist() == [0.0, 0.0]
