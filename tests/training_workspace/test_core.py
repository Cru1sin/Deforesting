import pandas as pd
import torch


def test_reference_fits_exclude_test_validation_and_own_experiment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from image_models import inputs

    (tmp_path / "base").mkdir()
    (tmp_path / "ridge").mkdir()
    pd.DataFrame({"cycle_name": ["cycle"]}).to_parquet(tmp_path / "base/cycle.parquet")
    captured = []

    def fit(events, excluded):
        captured.append(excluded)
        return {"training_experiment_ids": sorted(set(events.experiment_id) - excluded)}

    monkeypatch.setattr(inputs, "ridge_parameters", fit)
    monkeypatch.setattr(inputs, "apply_reference", lambda base, *args, **kwargs: base)
    args = SimpleNamespace(output=tmp_path, rgb="off", allow_extrapolation=True)
    cohort = pd.DataFrame({"cycle_name": ["cycle"], "experiment_id": ["own"]})
    events = pd.DataFrame({"experiment_id": ["test", "validation", "own", "training"]})
    inputs.build_fold_rows(args, cohort, events, ("test", "validation"))
    assert captured == [frozenset(("test", "validation", "own"))]


def test_current_architectures_and_cycle_balanced_loss():
    from image_models.core import contract, model
    from image_models.policy import _batch_loss

    for name, width, count in [
        ("F_core", 192, 18281),
        ("B0", 384, 449194),
        ("G_core", 576, 190977),
    ]:
        numeric, visual = contract(name)
        assert len(numeric) + len(visual) == width
        net = model(name, len(numeric), len(visual), 0)
        assert sum(p.numel() for p in net.parameters()) == count
        assert net(torch.zeros(2, width)).shape == ((2, 2) if name == "B0" else (2,))
    rows = pd.DataFrame(dict(cycle_name=["a", "a", "b"], after_optimum_target=[0.0, 1.0, 1.0]))
    logits = torch.tensor([-0.2, 0.5, 0.1])
    loss = _batch_loss(logits, rows, "after_optimum")
    repeated = pd.concat([rows.iloc[:2]] * 3 + [rows.iloc[2:]], ignore_index=True)
    repeated_logits = torch.cat([logits[:2]] * 3 + [logits[2:]])
    torch.testing.assert_close(loss, _batch_loss(repeated_logits, repeated, "after_optimum"))


def test_saved_models_reproduce_formal_predictions():
    import pickle
    from pathlib import Path
    import numpy as np
    from image_models.core import common_actions
    from image_models.policy import predict_policy, policy_cycle_metrics

    run = Path("output/image_models/core_comparison_ridge106")
    if not run.exists():
        import pytest

        pytest.skip("local formal results are not distributed with code")
    for name in ("F_core", "B0", "G_core"):
        path = next(iter(sorted((run / name / "seed_0/folds").glob("*.pkl"))))
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        reference = pd.read_parquet(run / "ridge" / f"{path.stem}_reference.parquet")
        actions = common_actions(reference)
        prediction = predict_policy(actions, saved["checkpoint"])
        np.testing.assert_allclose(
            prediction.probability, saved["predictions"].probability, atol=1e-7, rtol=1e-6
        )
        metrics = policy_cycle_metrics(
            prediction, reference, 0.5, loss_name="after_optimum", controller_reference=reference
        )
        np.testing.assert_allclose(metrics.selected_cop, saved["cycle_metrics"].selected_cop)


def test_common_reference_precedes_clock_and_missing_input():
    from image_models.core import common_actions

    times = pd.date_range("2026-01-01", periods=7, freq="10s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            cycle_cop=[1, 2, 4, 3, 2, 1, 1],
            cycle_cop_eligible=True,
            stable_heating_start=times[0],
            observation_end=times[-1],
            rgb_available=False,
        )
    )
    actions = common_actions(rows)
    assert len(actions) == 2
    assert actions.optimal_time.eq(times[2]).all()
    assert actions.after_optimum_target.tolist() == [0.0, 1.0]
    assert actions.optimal_cop.eq(4).all()


def test_extrapolated_reference_remains_eligible_and_dashed(monkeypatch):
    import numpy as np
    import matplotlib.pyplot as plt
    from image_models import inputs as relative_cop
    from plots.cop import plot_objectives

    times = pd.date_range("2026-01-01", periods=7, freq="10s")
    base = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            sensor_timestamp=times,
            stable_heating_start=times[0],
            observation_end=times[-1],
            pre_defrost_heat_kwh=[1, 2, 4, 3, 2, 1, 1],
            pre_defrost_electricity_kwh=1.0,
            pre_defrost_electricity_measurement_valid=True,
            pre_defrost_heat_measurement_valid=True,
            pre_defrost_feature_window_valid=True,
        )
    )
    prediction = pd.DataFrame(
        dict(
            prediction=[1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
            support_distance=9.0,
            support_threshold=2.0,
        )
    )
    monkeypatch.setattr(relative_cop, "predict_with_model_parameters", lambda *a: prediction)
    rows = relative_cop.apply_reference(base, {}, "off", allow_model_extrapolation=True)
    assert rows.cycle_cop_eligible.tolist() == [True] * 5 + [False, True]
    assert rows.cycle_cop_uses_model_extrapolation.iloc[2]
    from image_models.core import common_actions

    assert common_actions(rows).optimal_cop.eq(2).all()
    fig, axis = plt.subplots()
    plot_objectives(axis, rows, times[0], metrics=("cycle_cop",))
    assert axis.lines[0].get_linestyle() == "--"
    assert np.isfinite(axis.lines[0].get_ydata()).any()
    assert not np.isfinite(axis.lines[1].get_ydata()).any()
    plt.close(fig)
