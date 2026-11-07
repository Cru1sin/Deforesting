import numpy as np
import pandas as pd


def test_visual_lag_is_causal_cycle_local_and_keeps_the_action_mask():
    from image_models.inputs import causal_visual_history

    times = pd.date_range("2026-01-01", periods=4, freq="1min")
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=list(times) * 2,
            image_time=list(times - pd.Timedelta(seconds=1)) * 2,
            input_valid=[True, True, False, True] * 2,
            v=[1.0, 2.0, np.nan, 4.0, 10.0, 20.0, np.nan, 40.0],
        )
    )
    result = causal_visual_history(rows, 2, columns=["v"])
    np.testing.assert_allclose(
        result.history_change_000, [0.0, 1.0, np.nan, 2.0, 0.0, 10.0, np.nan, 20.0], equal_nan=True
    )
    assert result.input_valid.equals(rows.input_valid)
    assert (
        result.loc[result.input_valid, "history_image_time"]
        .le(result.loc[result.input_valid, "candidate_defrost_time"])
        .all()
    )
    # The first valid frame pads the unavailable pre-cycle history, without dropping early actions.
    assert result.history_padded.tolist() == [True, True, False, False] * 2
    changed = rows.copy()
    changed.loc[3, "v"] = 999.0
    pd.testing.assert_frame_equal(
        causal_visual_history(changed, 2, columns=["v"]).iloc[:3], result.iloc[:3]
    )
    zero = causal_visual_history(rows, 0, columns=["v"])
    assert zero.loc[zero.input_valid, "history_change_000"].eq(0).all()
    gap = causal_visual_history(rows, 1, columns=["v"])
    assert gap.loc[3, "history_missing"] and gap.loc[3, "history_change_000"] == 0


def test_current_history_control_has_same_width_without_past_information():
    from image_models.inputs import causal_visual_history

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            image_time=times - pd.Timedelta(seconds=1),
            input_valid=[True, True, False, True],
            v=[1.0, 2.0, np.nan, 4.0],
        )
    )
    control = causal_visual_history(rows, 1, columns=["v"], mode="current")
    real = causal_visual_history(rows, 1, columns=["v"])
    assert control.columns.equals(real.columns)
    np.testing.assert_allclose(control.history_change_000, rows.v, equal_nan=True)
    assert control.input_valid.equals(rows.input_valid)
    changed = rows.copy()
    changed.loc[:1, "v"] = 999.0
    assert (
        causal_visual_history(changed, 1, columns=["v"], mode="current").history_change_000.iloc[3]
        == 4.0
    )


def test_cycle_change_audit_keeps_new_failures_and_both_directions():
    from image_models.comparisons import paired_cycle_changes

    records = []
    for name, left, right in [
        ("improve", 1.0, 1.1),
        ("worsen", 1.1, 1.0),
        ("lost", 1.0, np.nan),
        ("restore", np.nan, 1.0),
    ]:
        for version, value in [("base", left), ("new", right)]:
            records.append(
                dict(
                    cycle_name=name,
                    experiment_id="exp",
                    seed=0,
                    version=version,
                    selected_cop=value,
                    regret_pct=100 * (1 - value / 2),
                    trigger_status="late",
                )
            )
    result = paired_cycle_changes(pd.DataFrame(records), "base", "new").set_index("cycle_name")
    assert result.change.to_dict() == {
        "improve": "improved",
        "worsen": "worsened",
        "lost": "newly_unsupported",
        "restore": "newly_evaluable",
    }
    assert result.loc["lost", "delta_cop"] != result.loc["lost", "delta_cop"]
    assert np.isclose(result.loc["worsen", "delta_cop"], -0.1)


def test_visual_anchor_is_cycle_local_causal_and_translation_invariant():
    from image_models.inputs import causal_visual_history

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=list(times) * 2,
            image_time=list(times - pd.Timedelta(seconds=1)) * 2,
            input_valid=True,
            v=[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0],
        )
    )
    result = causal_visual_history(rows, 5, columns=["v"], mode="anchor")
    pd.testing.assert_frame_equal(
        result, causal_visual_history(rows, 5, columns=["v"], mode="anchor_only")
    )
    np.testing.assert_allclose(result.history_change_000, [0, 1, 2, 3, 0, 2, 4, 6])
    shifted = rows.copy()
    shifted.loc[shifted.cycle_name.eq("a"), "v"] += 100
    np.testing.assert_allclose(
        causal_visual_history(shifted, 5, columns=["v"], mode="anchor").history_change_000,
        result.history_change_000,
    )
    rows.loc[3, "v"] = 999
    np.testing.assert_allclose(
        causal_visual_history(rows, 5, columns=["v"], mode="anchor").history_change_000.iloc[:3],
        result.history_change_000.iloc[:3],
    )
    assert result.history_image_time.le(result.candidate_defrost_time).all()


def test_anchor_only_retains_original_input_width_and_reference_difference():
    from types import SimpleNamespace
    from unittest.mock import patch

    from image_models import training as stopping
    from image_models.inputs import CHANGE

    args = SimpleNamespace(
        cohort="training",
        recipe="temperature",
        ambient_context=False,
        version="anchor_only",
        architecture="GRU32",
        visual_history="anchor_only",
    )
    with (
        patch.object(stopping.DatasetLoader, "split_training_data", return_value=pd.DataFrame()),
        patch.object(stopping, "fit_case") as fit,
    ):
        stopping.fit_fold(args, pd.DataFrame(), pd.DataFrame(), "test", [], 0)
    assert fit.call_args.kwargs["visual_columns"] == CHANGE
    assert len(CHANGE) == 384


def test_film_visual_skip_preserves_initial_baseline_and_current_image_path():
    import torch

    from image_models.model import build_model

    base = build_model("GRU32_film", 2, 4, seed=0)
    skip = build_model("GRU32_film_skip", 2, 4, seed=0)
    # Creating the skip layer changes later RNG consumption; copy the shared state.
    shared = skip.state_dict()
    shared.update(base.state_dict())
    skip.load_state_dict(shared)
    base.eval()
    skip.eval()
    values = torch.randn(1, 3, 6)
    torch.testing.assert_close(skip(values), base(values))
    with torch.no_grad():
        skip.visual_skip.weight.fill_(1.0)
    torch.testing.assert_close(skip(values) - base(values), values[..., 2:].sum(-1))
    assert skip.visual_dropout == base.visual_dropout == 0.1


def test_compensation_progress_is_causal_cycle_local_and_skips_missing_input():
    from image_models.inputs import causal_compensation_progress

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=list(times) * 2,
            frequency_headroom_fraction=[0.1, 0.4, 0.9, 0.2, 0.05, 0.1, 0.2, 0.3],
            input_valid=[True, True, False, True] * 2,
        )
    )
    result = causal_compensation_progress(rows)
    np.testing.assert_allclose(
        result.frequency_used_margin_fraction, [0, 0, np.nan, 0.2, 0, 0, np.nan, 0], equal_nan=True
    )
    assert result.input_valid.equals(rows.input_valid)
    changed = rows.copy()
    changed.loc[3, "frequency_headroom_fraction"] = 0.8
    pd.testing.assert_frame_equal(causal_compensation_progress(changed).iloc[:3], result.iloc[:3])
