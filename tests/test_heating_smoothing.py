from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.heating_smoothing import (
    METHODS,
    _nearly_isotonic,
    cost_method_ranking,
    global_method_ranking,
    recommend_methods,
    score_methods,
    smooth_cycle,
    water_heating_capacity,
)


def _frame(values: list[float], stages: list[str] | None = None) -> pd.DataFrame:
    count = len(values)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=count, freq="10s"),
            "cycle_stage": stages or ["frost_development"] * count,
            "heating_capacity": values,
            "water_flow": [1.0] * count,
            "water_in_temperature": [40.0] * count,
            "water_out_temperature": [45.0] * count,
        }
    )


def test_smoothing_is_stage_bounded_and_preserves_missing_samples() -> None:
    frame = _frame(
        [0.0, 0.0, 12.0, 0.0, np.nan, 0.0, 100.0, 100.0, 100.0, 100.0],
        ["frost_development"] * 6 + ["defrost"] * 4,
    )

    result = smooth_cycle(frame)

    assert result.loc[2, "median_centered_60s"] == 0.0
    assert abs(result.loc[2, "adaptive_offline"]) < 1.0
    assert pd.isna(result.loc[4, list(METHODS)]).all()
    assert (result.loc[6:, "median_centered_60s"] == 100.0).all()
    assert result.loc[6:, "adaptive_offline"].min() > 99.0
    assert result.loc[6:, "wavelet_offline"].min() > 99.0
    assert result["adaptive_median_window_seconds"].dropna().isin([30, 60, 90]).all()
    assert result["adaptive_lowpass_tau_seconds"].dropna().isin([20, 30, 45, 60, 90]).all()


def test_wavelet_offline_removes_an_isolated_spike_without_changing_length() -> None:
    values = [5.0] * 31
    values[15] = 20.0

    result = smooth_cycle(_frame(values))

    assert len(result["wavelet_offline"]) == len(values)
    assert result.loc[15, "wavelet_offline"] < 12.0
    assert np.max(np.abs(result["wavelet_offline"] - 5.0)) < 7.0


def test_monotonic_curve_only_decreases_during_frost_development() -> None:
    frame = _frame([5.0, 7.0, 6.0, 8.0, 4.0, 5.0, 3.0, 4.0])

    result = smooth_cycle(frame)

    assert (result["wavelet_monotonic_offline"].diff().dropna() <= 1e-12).all()


def test_shape_constrained_smoothers_reduce_rises_and_avoid_isotonic_steps() -> None:
    frame = _frame([10.0, 9.8, 12.0, 9.4, 9.0, 9.6, 8.3, 8.0, 7.7, 7.4, 7.1])

    result = smooth_cycle(frame)

    raw_rises = result["heating_capacity"].diff().clip(lower=0).sum()
    nearly_rises = result["nearly_isotonic_offline"].diff().clip(lower=0).sum()
    robust = result["robust_monotone_offline"]
    isotonic = result["wavelet_monotonic_offline"]
    assert 0 <= nearly_rises < raw_rises
    assert (robust.diff().dropna() <= 1e-10).all()
    assert np.square(np.diff(robust, n=2)).sum() < np.square(np.diff(isotonic, n=2)).sum()


def test_nearly_isotonic_penalty_has_a_real_effect() -> None:
    values = np.array([0.0, 2.0, 0.0])

    weak = _nearly_isotonic(values, penalty=0.1)
    strong = _nearly_isotonic(values, penalty=1.0)

    assert np.maximum(np.diff(strong), 0.0).sum() < np.maximum(
        np.diff(weak), 0.0
    ).sum()


def test_online_methods_do_not_change_when_only_future_values_change() -> None:
    first = _frame([0.0] * 8)
    second = _frame([0.0] * 5 + [100.0] * 3)

    first_result = smooth_cycle(first)
    second_result = smooth_cycle(second)

    for method in ("ewma_tau30s", "median30s_ewma30s"):
        np.testing.assert_allclose(
            first_result.loc[:4, method], second_result.loc[:4, method]
        )


def test_water_heating_capacity_uses_m3_per_hour_and_returns_kw() -> None:
    result = water_heating_capacity(_frame([1.0]))

    assert result.iloc[0] == 5.805


def test_scoring_recommends_one_offline_and_one_causal_method() -> None:
    frame = _frame([5.0, 5.1, 8.0, 5.2, 5.1, 5.0, 4.9, 5.0, 5.1])
    frame["cycle_name"] = "cycle_test"
    smoothed = smooth_cycle(frame)

    metrics = score_methods(smoothed)
    recommendations = recommend_methods(metrics)
    global_ranking = global_method_ranking(metrics)
    cost_ranking = cost_method_ranking(metrics)

    assert set(metrics["method"]) == set(METHODS)
    assert smoothed["median30s_ewma30s"].notna().all()
    assert metrics["energy_error_pct"].notna().all()
    assert recommendations.loc[0, "offline_method"] in {
        "median_centered_60s",
        "savgol_70s",
        "adaptive_offline",
        "wavelet_offline",
    }
    assert recommendations.loc[0, "online_method"] in {
        "ewma_tau30s",
        "median30s_ewma30s",
    }
    assert global_ranking.iloc[0]["method"] in set(METHODS)
    assert global_ranking["global_mean_rank_sum"].is_monotonic_increasing
    assert cost_ranking["cost_mean_rank_sum"].is_monotonic_increasing


def test_cost_ranking_excludes_identity_like_curves_that_do_not_reduce_spikes() -> None:
    metrics = pd.DataFrame(
        {
            "cycle_name": ["cycle_test", "cycle_test"],
            "cycle_stage": ["frost_development", "frost_development"],
            "method": ["nearly_isotonic_offline", "wavelet_offline"],
            "metric_status": ["available", "available"],
            "spike_reduction": [0.0, 0.2],
            "shortfall_area_error_pct": [0.0, 0.2],
            "energy_error_pct": [0.0, 0.01],
            "water_rmse_offset_kw": [0.5, 0.5],
            "transient_retention": [1.0, 0.8],
        }
    )

    ranking = cost_method_ranking(metrics)

    assert ranking["method"].tolist() == ["wavelet_offline"]


def test_analysis_script_writes_curves_tables_figures_and_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "cycles"
    output_dir = tmp_path / "report"
    input_dir.mkdir()
    frame = _frame([5.0, 5.1, 8.0, 5.2, 5.1, 5.0, 4.9, 5.0, 5.1])
    frame["cycle_name"] = "cycle_test"
    frame.to_parquet(input_dir / "cycle_test.parquet", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_heating_capacity_smoothing.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert not (output_dir / "源数据" / "smoothed_curves.parquet").exists()
    assert (output_dir / "源数据" / "method_metrics.csv").is_file()
    assert (output_dir / "源数据" / "cycle_recommendations.csv").is_file()
    assert (output_dir / "源数据" / "cost_method_ranking.csv").is_file()
    assert (output_dir / "图表" / "循环图" / "cycle_test.png").is_file()
    assert (output_dir / "图表" / "method_overview.png").is_file()
    assert (output_dir / "报告.md").is_file()
