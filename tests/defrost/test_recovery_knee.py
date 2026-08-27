import numpy as np

from frost_analysis.recovery_knee import find_global_knee


def test_find_global_knee_prefers_larger_later_transition() -> None:
    minutes = np.arange(0.0, 30.0, 1.0 / 6.0)
    temperature = np.piecewise(
        minutes,
        [minutes <= 5.0, (minutes > 5.0) & (minutes <= 13.0), minutes > 13.0],
        [
            lambda value: 25.0 + 2.0 * value,
            lambda value: 35.0 + 1.5 * (value - 5.0),
            lambda value: 47.0 + 0.08 * (value - 13.0),
        ],
    )

    knee = find_global_knee(minutes, temperature)

    assert knee is not None
    assert abs(knee - 13.0) <= 1.0
