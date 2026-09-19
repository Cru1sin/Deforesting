import numpy as np
import pandas as pd
import pytest
from plots.event_calibration import event_calibration_metrics

def test_event_calibration_keeps_seed_repeats_separate():
    rows = pd.DataFrame({
        "event_id": ["a", "b", "a", "b"],
        "experiment_id": ["e1", "e2", "e1", "e2"],
        "representation": ["model"] * 4,
        "seed": [0, 0, 1, 1],
        "target": [1.0, 2.0, 1.0, 2.0],
        "predicted_target": [1.1, 1.9, 1.2, 1.8],
    })

    metrics = event_calibration_metrics(rows, ["target"])

    assert metrics.events.tolist() == [2, 2]
    assert metrics.seed.tolist() == [0, 1]
