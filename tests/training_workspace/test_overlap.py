import numpy as np
import pandas as pd


def test_first_crossing_uses_history_without_repairing_reversals():
    from image_models.overlap import crossing_intervals

    curve = pd.DataFrame(
        {
            "visual_frost_score": [0.0, 0.8, 0.3, 0.6, 0.9],
            "near_optimal": [False, False, False, True, True],
        }
    )
    assert crossing_intervals(curve, near_only=True) == [(0.8, 0.9)]
    assert crossing_intervals(curve) == [(0.0, 0.9)]
    assert curve.visual_frost_score.tolist() == [0.0, 0.8, 0.3, 0.6, 0.9]


def test_initial_high_score_and_later_gap_do_not_invent_observability():
    from image_models.overlap import action_intervals

    curve = pd.DataFrame(
        {
            "cycle_name": "a",
            "visual_frost_score": [0.6, 0.3, 0.4, np.nan, 0.7, 0.8],
            "near_optimal": [False] * 5 + [True],
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=6, freq="30s"),
        }
    )
    spans = action_intervals(curve)
    assert spans.kind.tolist() == ["observed", "feasible"]
    assert spans.left.tolist() == [0.7, 0.7] and spans.right.tolist() == [0.8, 0.8]


def test_first_crossing_plateaus_gaps_and_exact_coverage_denominators():
    from image_models.overlap import coverage_table, crossing_intervals

    curve = pd.DataFrame(
        {
            "visual_frost_score": [0.0, 0.2, 0.2, 0.4, 0.6, 0.8],
            "near_optimal": [False, True, True, False, True, False],
        }
    )
    assert crossing_intervals(curve, near_only=True) == [(0.0, 0.2), (0.4, 0.6)]
    curve.loc[3, "visual_frost_score"] = np.nan
    assert crossing_intervals(curve, near_only=True) == [(0.0, 0.2)]
    observed = pd.DataFrame({"cycle_name": ["a", "b"], "left": [0.0, 0.1], "right": [1.0, 0.5]})
    feasible = pd.DataFrame({"cycle_name": ["a", "b"], "left": [0.2, 0.3], "right": [0.4, 0.5]})
    table = coverage_table(observed, feasible)
    at = table.set_index("right")
    assert at.loc[0.4, "coverage"] == 1 and at.loc[0.4, "n_observed"] == 2
    assert at.loc[0.5, "coverage"] == 0.5 and at.loc[0.5, "n_observed"] == 2
    assert at.loc[1.0, "n_observed"] == 1 and at.loc[1.0, "coverage"] == 0


def test_loeo_maximum_components_and_first_crossing():
    from image_models.overlap import maximum_intervals, teacher_crossing

    table = pd.DataFrame(
        {
            "left": [0.0, 0.2, 0.3, 0.4, 0.5],
            "right": [0.2, 0.3, 0.4, 0.5, 0.6],
            "coverage": [0.5, 1.0, 1.0, 0.5, 1.0],
            "n_observed": [2] * 5,
        }
    )
    peaks = maximum_intervals(table)
    assert [(p["left"], p["right"]) for p in peaks] == [(0.2, 0.4), (0.5, 0.6)]
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=4, freq="30s"),
            "visual_frost_score": [0.0, 0.3, 0.6, 0.8],
            "cop_norm": [0.9, 0.995, 0.98, 0.9],
        }
    )
    result = teacher_crossing(curve, 0.3)
    assert result["hit_1pct"] and np.isclose(result["cop_regret"], 0.005)
    assert result["crossing_time"] == curve.candidate_defrost_time.iloc[1]
    assert teacher_crossing(curve, 0.9)["no_crossing"]
    curve.loc[1, "visual_frost_score"] = np.nan
    assert teacher_crossing(curve, 0.5)["status"] == "reference_gap"
    curve.loc[1, "visual_frost_score"] = 0.3
    curve.loc[2, "cop_norm"] = np.nan
    unsupported = teacher_crossing(curve, 0.5)
    assert unsupported["status"] == "unsupported_cop"
    assert not unsupported["hit_1pct"] and np.isnan(unsupported["cop_regret"])


def test_fixed_cohort_support_excludes_gaps_and_small_denominator_peaks():
    from image_models.overlap import coverage_table

    observed = pd.DataFrame(
        {"cycle_name": ["a", "a", "b"], "left": [0.0, 0.5, 0.1], "right": [0.3, 1.0, 0.8]}
    )
    feasible = pd.DataFrame(
        {"cycle_name": ["a", "b", "a"], "left": [0.2, 0.2, 0.9], "right": [0.3, 0.3, 1.0]}
    )
    conditional = coverage_table(observed, feasible).set_index("right")
    assert conditional.loc[1.0, "coverage"] == 1  # Only cycle a remains.
    fixed = coverage_table(observed, feasible, common_cycles=["a", "b"]).set_index("right")
    assert fixed.loc[0.3, "coverage"] == 1 and fixed.loc[0.3, "n_observed"] == 2
    assert np.isnan(fixed.loc[0.5, "coverage"])  # Internal unknown crossing, not min–max support.
    assert np.isnan(fixed.loc[1.0, "coverage"])
    assert fixed.loc[fixed.common_support, "n_observed"].eq(2).all()
    missing = coverage_table(observed, feasible, common_cycles=["a", "b", "c"])
    assert not missing.common_support.any() and missing.coverage.isna().all()


def test_reference_basin_preserves_gaps_singletons_and_one_vote_per_cycle():
    from image_models.overlap import interval_cycle_counts, reference_basin_intervals

    g = pd.DataFrame(
        dict(
            cycle_name="a",
            Ts=50.0,
            candidate_defrost_time=pd.date_range("2026-01-01", periods=5, freq="30s"),
            raw_effective_cop=[1.0, 1.0, 0.9, 1.0, 1.0],
            relative_effective_cop=[0.8, 0.8, 1.0, 0.8, 0.8],
            visual_frost_score=[0.1, 0.2, 0.5, 0.8, 0.9],
            slingshot_grade=[10, 20, 50, 80, 90],
        )
    )
    r = reference_basin_intervals(g)
    raw = r.loc[r.method.eq("effective_cop") & r.coordinate.eq("visual_frost_score")]
    assert len(raw) == 2
    np.testing.assert_array_equal(
        interval_cycle_counts(pd.concat([raw, raw]), [0.1, 0.15, 0.5, 0.85, 0.9], closed=True),
        [1, 1, 0, 1, 1],
    )
    ratio = r.loc[r.method.eq("frontier_ratio") & r.coordinate.eq("slingshot_grade")]
    np.testing.assert_array_equal(
        interval_cycle_counts(ratio, [49, 50, 51], closed=True), [0, 1, 0]
    )
