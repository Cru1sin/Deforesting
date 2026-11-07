import itertools

import numpy as np
import pandas as pd


def test_exact_pair_counts_and_cycle_only_time_ties():
    from tests.training_workspace.reference_ranking import (
        experiment_summary,
        pair_counts,
        plateau_gap_counts,
        ranking_counts,
    )

    rng = np.random.default_rng(4)
    for n in [0, 1, 2, 15, 40]:
        x, y = rng.integers(0, 4, size=(2, n))
        expected = np.zeros(3, dtype=np.int64)
        for i, j in itertools.combinations(range(n), 2):
            if x[i] == x[j]:
                continue
            order = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            expected += [1, 1 + order, int(y[i] == y[j])]
        np.testing.assert_array_equal(pair_counts(x, y), expected)

    frames = pd.DataFrame(
        dict(
            reference=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            score=[0.0, 0.2, 0.1, 0.3, 0.3, 0.4],
            cycle_name=["a", "a", "a", "b", "b", "c"],
            experiment_id=["e1", "e1", "e1", "e2", "e2", "e2"],
            image_time=pd.date_range("2026-01-01", periods=6, freq="10min"),
        )
    )
    actual = ranking_counts(frames)
    for scope in [
        "within_cycle_reference",
        "cross_cycle_reference",
        "cross_experiment_reference",
        "within_cycle_time_ties",
        "within_cycle_combined",
    ]:
        expected = np.zeros(3, dtype=np.int64)
        for i, j in itertools.combinations(range(len(frames)), 2):
            a, b = frames.iloc[i], frames.iloc[j]
            same_cycle = a.cycle_name == b.cycle_name
            same_ref = a.reference == b.reference
            use_time = scope in ["within_cycle_time_ties", "within_cycle_combined"] and same_ref
            keep = {
                "within_cycle_reference": same_cycle and not same_ref,
                "cross_cycle_reference": not same_cycle and not same_ref,
                "cross_experiment_reference": a.experiment_id != b.experiment_id and not same_ref,
                "within_cycle_time_ties": same_cycle and same_ref,
                "within_cycle_combined": same_cycle,
            }[scope]
            if keep:
                direction = -1 if use_time else np.sign(a.reference - b.reference)
                order = direction * np.sign(a.score - b.score)
                expected += [1, int(1 + order), int(a.score == b.score)]
        np.testing.assert_array_equal(actual[scope], expected)
    frames["time_order"] = frames.image_time.astype("int64")
    np.testing.assert_array_equal(plateau_gap_counts(frames, 5), [1, 2, 0])
    np.testing.assert_array_equal(plateau_gap_counts(frames, 11), [0, 0, 0])
    pairs = pd.DataFrame(
        dict(
            reference_min=[0, 0, 0],
            experiment_a=["a", "a", "b"],
            experiment_b=["b", "c", "c"],
            pairs=[1, 20, 40],
            concordance=[0.5, 0.5, 0.5],
        )
    )
    intervals = experiment_summary(pairs, bootstrap=100)
    np.testing.assert_allclose(intervals[["concordance", "ci_low", "ci_high"]], 0.5)
