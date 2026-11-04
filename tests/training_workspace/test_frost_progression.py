def test_frost_interval_excludes_preparation_recovery_but_keeps_partial():
    import pandas as pd

    from image_models.frost_progression import frost_interval

    rows = pd.DataFrame(
        {
            "image_time": pd.date_range("2026-01-01", periods=5, freq="30s"),
            "cycle_stage": [
                "recovery",
                "partial",
                "frost_development",
                "defrost_preparation",
                "frost_development",
            ],
        }
    )
    assert frost_interval(rows, pd.NaT).index.tolist() == [1, 2]
    assert frost_interval(rows, rows.image_time.iloc[2]).index.tolist() == [1]
