"""Separate longitudinal signal from experiment offsets without editing embeddings."""

import numpy as np
import pandas as pd


def test_pc_audit_separates_time_and_batch_with_equal_cycle_weights():
    from image_models.representation_audit import pc_diagnostics

    t = np.tile(np.linspace(0.0, 1.0, 20), 4)
    batch = np.repeat(["a", "a", "b", "b"], 20)
    cycles = np.repeat(["a1", "a2", "b1", "b2"], 20)
    frames = pd.DataFrame(
        {
            "cycle_name": cycles,
            "experiment_id": batch,
            "image_date": batch,
            "t_rel": t,
            "existing_reference": t,
        }
    )
    z = np.column_stack([t, (batch == "b").astype(float), t + 3 * (batch == "b")])
    table, correlations = pc_diagnostics(frames, z, np.array([0.5, 0.3, 0.2]), n_jobs=1)
    table = table.set_index("pc")
    assert table.loc[1, "median_cycle_spearman"] > 0.99
    assert table.loc[1, "experiment_eta2"] < 1e-10
    assert table.loc[2, "experiment_eta2"] > 0.99
    assert table.loc[2, "experiment_partial_r2_after_time"] > 0.99
    assert np.isnan(table.loc[2, "median_cycle_spearman"])
    assert table.loc[3, "experiment_partial_r2_after_time"] > 0.99
    assert correlations.cycle_name.nunique() == 4


def test_state_support_counts_observation_not_continuous_range():
    from image_models.representation_audit import state_support

    frames = pd.DataFrame(
        {
            "experiment_id": ["a", "a", "b"],
            "cycle_name": ["a1", "a1", "b1"],
            "existing_reference": [0.0, 1.0, 0.5],
        }
    )
    counts, summary = state_support(frames)
    assert (
        counts.loc[(counts.experiment_id == "a") & (counts.state_bin == 5), "cycle_count"].iloc[0]
        == 0
    )
    assert (
        counts.loc[(counts.experiment_id == "a") & (counts.state_bin == 9), "cycle_count"].iloc[0]
        == 1
    )
    assert summary.loc[summary.state_bin == 5, "experiment_count"].iloc[0] == 1
    assert summary.loc[summary.state_bin == 0, "experiment_count"].iloc[0] == 1
