import numpy as np
import pandas as pd


def test_batched_heat_integral_matches_single_window_with_missingness():
    from defrost_event_models.training_data import candidate_integral_table, window_audit

    rng = np.random.default_rng(17)
    t = np.cumsum(rng.choice([1, 2, 7, 35], 160))
    start = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(t, unit="s"),
            "heating_capacity": rng.normal(7, 1, len(t)),
            "water_out_temperature": rng.normal(40, 0.5, len(t)),
        }
    )
    frame.loc[::17, "heating_capacity"] = np.nan
    frame.loc[::13, "water_out_temperature"] = np.nan
    ends = [
        start,
        start + pd.Timedelta(seconds=1),
        *frame.timestamp.iloc[::7].tolist(),
        frame.timestamp.iloc[-1] + pd.Timedelta(seconds=50),
    ]
    expected = pd.DataFrame([window_audit(frame, start, end, "heating_capacity") for end in ends])
    actual = candidate_integral_table(frame, start, ends, "heating_capacity")
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
