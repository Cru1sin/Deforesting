import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from image_models.relative_cop import (
    RelativeCOP,
    cycle_weights,
    feature_columns,
    history_features,
    normalize_curve,
)


def test_solution_u_structure_and_input_contract():
    columns = feature_columns()
    assert len(columns) == 583 and columns[-1] == "elapsed_minutes"
    assert not any("humidity" in c or c.endswith("_missing") for c in columns)
    model = RelativeCOP(len(columns))
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [
        (583, 60),
        (60, 60),
        (60, 32),
        (32, 32),
        (32, 1),
    ]
    assert all(torch.count_nonzero(m.bias) == 0 for m in layers)
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [0.2, 0.2]
    assert not any(isinstance(m, (nn.ReLU, nn.Sigmoid)) for m in model.modules())
    assert model(torch.zeros(3, 583)).shape == (3,)


def test_sensor_inputs_and_pinn_derivatives_follow_official_contract():
    from image_models.relative_cop import dynamic_network, pinn_forward

    assert len(feature_columns(rgb="off")) == 199
    assert not any(c.startswith("dinov2") for c in feature_columns(rgb="off"))
    u = RelativeCOP(199)
    g = dynamic_network(399)
    layers = [m for m in g.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [(399, 60), (60, 60), (60, 1)]
    values, residual = pinn_forward(u, g, torch.randn(4, 199))
    (values.square().mean() + residual.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in u.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())


def test_pairing_and_peak_weights_preserve_cycle_objective():
    from image_models.relative_cop import adjacent_pairs, direction_loss, peak_pair_weights

    frame = pd.DataFrame(
        {"cycle_name": ["a", "a", "a", "b", "b", "c"], "target": [0.3, 1.0, 0.8, 0.9, 1.0, 0.5]}
    )
    left, right = adjacent_pairs(frame)
    assert list(zip(left, right, strict=True)) == [(0, 1), (1, 2), (3, 4)]
    weights = peak_pair_weights(frame, left, right)
    errors = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    got = 0.5 * ((weights[left] * errors[left]).mean() + (weights[right] * errors[right]).mean())
    assert np.isclose(got, 0.5 * ((1 + 5 * 4 + 9) / 7 + (16 + 5 * 25) / 6))
    y1, y2 = torch.tensor([0.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])
    assert direction_loss(y1, y2, y1, y2) == 0
    assert direction_loss(y2, y1, y1, y2) == 2


def test_official_learning_rate_keeps_full_horizon_for_short_refits():
    from image_models.relative_cop import official_learning_rates

    rates = official_learning_rates(200)
    assert len(rates) == 200 and rates[0] == 5e-4 and rates[9] == 1e-3
    assert np.isclose(rates[10], 1e-3) and rates[-1] > 1e-4


def test_pinn_minibatch_checkpoint_uses_u_only_for_inference():
    from image_models.relative_cop import fit_regression, predict_regression

    rng = np.random.default_rng(0)
    columns = feature_columns("off")
    frame = pd.DataFrame(rng.normal(size=(9, len(columns))), columns=columns).assign(
        cycle_name=["a"] * 5 + ["b"] * 4, experiment_id="train", target=0.9, input_available=True
    )
    fitted = fit_regression(
        frame, None, epochs=2, patience=1, seed=0, architecture="pinn4soh", rgb="off", batch_size=3
    )
    assert fitted["losses"].optimizer_steps.eq(3).all()
    assert fitted["dynamic_state_dict"] is not None
    expected = predict_regression(frame, fitted).prediction
    del fitted["dynamic_state_dict"]
    pd.testing.assert_series_equal(expected, predict_regression(frame, fitted).prediction)
    assert len(fitted["feature_columns"]) == 199


def test_pinn_dynamic_input_contains_actual_partial_derivatives():
    from image_models.relative_cop import pinn_forward

    class LinearSolution(nn.Module):
        def forward(self, x):
            return 2 * x[:, 0] + 3 * x[:, 1]

    class InspectDynamics(nn.Module):
        def forward(self, values):
            torch.testing.assert_close(values[:, -2:], torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
            return values[:, -1:] * 0

    u, residual = pinn_forward(
        LinearSolution(), InspectDynamics(), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    torch.testing.assert_close(u, torch.tensor([8.0, 18.0]))
    torch.testing.assert_close(residual, torch.tensor([3.0, 3.0]))


def test_sensor_cache_reuse_does_not_access_photos(tmp_path):
    from types import SimpleNamespace

    from image_models.relative_cop import prepare_cycle

    source, output = tmp_path / "source", tmp_path / "output"
    (source / "base").mkdir(parents=True)
    (output / "base").mkdir(parents=True)
    pd.DataFrame(
        {
            "effective_heat_rule": ["outlet_at_least_recovery_temperature"],
            "rgb_available": [False],
            "dinov2_000": [np.nan],
        }
    ).to_parquet(source / "base" / "cycle.parquet")
    prepare_cycle(
        SimpleNamespace(reference_run=source, output=output, rgb="off"),
        SimpleNamespace(cycle_name="cycle"),
    )
    cached = pd.read_parquet(output / "base" / "cycle.parquet")
    assert "dinov2_000" not in cached


def test_online_trigger_uses_confirmation_time_and_keeps_missing_slots():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "target": [0.7, 0.8, 0.9, 1.0, 0.95],
            "cycle_cop": [2.8, 3.2, 3.6, 4.0, 3.8],
            "cycle_cop_eligible": True,
        }
    )
    stream = curve.assign(score=[0.995, np.nan, 0.98, 0.999, 0.995])
    results = online_trigger_metrics(curve, stream, 0.99)
    assert len(results) == 1
    assert results[0]["trigger_time"] == times[4]
    assert np.isclose(results[0]["relative_cop_loss"], 0.05)
    changed = stream.copy()
    changed.loc[1:, "score"] = 5.0
    assert online_trigger_metrics(curve, changed, 0.99)[0]["trigger_time"] == times[1]


def test_online_no_trigger_and_unsupported_rb_are_not_forced_to_optimum():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curve = pd.DataFrame(
        {
            "candidate_defrost_time": times,
            "target": [np.nan, 1.0, 0.9],
            "cycle_cop": [np.nan, 4.0, 3.6],
            "cycle_cop_eligible": [True, False, True],
        }
    )
    assert (
        online_trigger_metrics(curve, curve.assign(score=0.98), 0.99)[0]["status"] == "no_trigger"
    )
    result = online_trigger_metrics(curve, curve.assign(score=1.0), 0.99)[0]
    assert result["status"] == "trigger_outside_reference_support"
    assert pd.isna(result["relative_cop_loss"])


def test_rb_trace_preserves_original_first_trigger():
    from defrost_decision.baselines.rule_based import first_trigger

    times = pd.date_range("2026-01-01", periods=60, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "coil_temperature": 0.0,
            "ambient_temperature": 0.0,
            "water_out_temperature": 45.0,
            "p1__T3o'2_20": 0.0,
            "p1__DefTim1'2_20": 151 * 60,
            "p1__DefTim2'2_20": 6 * 60,
        }
    )
    end = times[-1] + pd.Timedelta(seconds=1)
    original = first_trigger(frame, times[0], end)
    trace = first_trigger(frame, times[0], end, return_trace=True)
    assert original["t_RB"] == trace.index[trace.triggered][0] == times[0]
    assert original["trigger_type"] == "Case8"


def test_state_consistency_is_discrete_and_trains_both_networks():
    from image_models.relative_cop import PHYSICAL_STATE, state_transition_loss

    assert len(PHYSICAL_STATE) == 16
    predictor = nn.Linear(20, 1)
    transition = nn.Linear(34, 1)
    previous, current = torch.randn(4, 20), torch.randn(4, 20)
    u0, u1 = predictor(previous).squeeze(1), predictor(current).squeeze(1)
    captured = []
    hook = transition.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].detach())
    )
    loss = state_transition_loss(transition, previous, current, u0, u1, list(range(16)))
    loss.backward()
    hook.remove()
    assert captured[0].shape == (4, 34)
    torch.testing.assert_close(captured[0][:, :16], current[:, :16])
    torch.testing.assert_close(captured[0][:, 16:32], previous[:, :16])
    torch.testing.assert_close(captured[0][:, 32], current[:, -1])
    torch.testing.assert_close(captured[0][:, 33], u0.detach())
    assert predictor.weight.grad.abs().sum() > 0
    assert transition.weight.grad.abs().sum() > 0


def test_fixed_boundary_heat_uses_only_past_samples():
    from defrost_event_models.training_data import window_audit

    times = pd.date_range("2026-01-01", periods=31, freq="s")
    data = pd.DataFrame(
        {
            "timestamp": times,
            "heating_capacity": 10.0,
            "water_out_temperature": np.linspace(49.0, 51.0, 31),
        }
    )
    expected = window_audit(
        data, times[0], times[20], "heating_capacity", minimum_outlet_temperature=50.0
    )
    data.loc[data.timestamp.ge(times[20]), ["heating_capacity", "water_out_temperature"]] = 1000.0
    changed = window_audit(
        data, times[0], times[20], "heating_capacity", minimum_outlet_temperature=50.0
    )
    assert expected == changed


def test_state_fit_does_not_train_direction_or_bridge_missing_frames():
    from image_models.relative_cop import fit_regression, state_diagnostics

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(6, 583)), columns=feature_columns()).assign(
        cycle_name="a",
        experiment_id="x",
        target=0.9,
        input_available=True,
        candidate_defrost_time=pd.Timestamp("2026-01-01")
        + pd.to_timedelta([0, 10, 20, 50, 60, 70], unit="s"),
    )
    result = fit_regression(
        frame, None, epochs=2, patience=1, seed=0, architecture="state-consistency", batch_size=3
    )
    assert result["losses"].direction_loss.eq(0).all()
    assert result["state_transition_pairs"] == 4
    assert result["state_transition_excluded_pairs"] == 1
    assert result["dynamic_state_dict"]["0.weight"].shape == (60, 34)
    assert state_diagnostics(frame, result)["pairs"] == 4


def test_cycle_weighting_is_invariant_to_repeating_one_cycle():
    groups = pd.Series(["a", "a", "b"])
    errors = np.array([1.0, 3.0, 8.0])
    assert np.dot(cycle_weights(groups), errors) == 5.0
    assert np.dot(cycle_weights(pd.Series(["a"] * 4 + ["b"])), [1.0, 3.0, 1.0, 3.0, 8.0]) == 5.0


def test_normalization_does_not_redefine_peak_when_rgb_missing():
    curve = pd.DataFrame(
        {
            "cycle_cop": [2.0, 4.0, 3.0],
            "cycle_cop_eligible": [True] * 3,
            "rgb_available": [True, False, True],
        }
    )
    result = normalize_curve(curve)
    np.testing.assert_allclose(result.target, [0.5, 1.0, 0.75])
    assert result.reference_max.eq(4).all()


def test_history_is_strictly_past_and_uses_absolute_cop():
    t = pd.date_range("2026-01-01", periods=12, freq="10s")
    frame = pd.DataFrame(
        {"cycle_name": "a", "candidate_defrost_time": t, "cycle_cop": np.arange(12, dtype=float)}
    )
    original = history_features(frame)
    frame.loc[6:, "cycle_cop"] = 999.0
    changed = history_features(frame)
    pd.testing.assert_frame_equal(original.iloc[:7], changed.iloc[:7])
    assert original.stat_candidate_cop_current.iloc[6] == 5.0


def test_ridge_exclusions_include_outer_inner_and_training_experiment(monkeypatch):
    from image_models import relative_cop as module

    events = pd.DataFrame(
        {
            "experiment_id": list("abcdef"),
            "energy_event_valid": True,
            "defrost_event_electricity_observed_kwh": 1.0,
        }
    )
    captured = []

    def fit(rows, features, target):
        captured.extend(rows.experiment_id)
        return rows

    monkeypatch.setattr(module, "fit_model_on_all_experiments", fit)
    monkeypatch.setattr(module, "model_to_parameters", lambda model: model)
    module.ridge_parameters(events, {"a", "b", "c"})
    assert captured == ["d", "e", "f"]


def test_prediction_peak_metrics_separate_coverage_from_network_error():
    from image_models.relative_cop import cycle_metrics

    times = pd.date_range("2026-01-01", periods=3, freq="min")
    rows = normalize_curve(
        pd.DataFrame(
            {
                "cycle_name": "a",
                "experiment_id": "x",
                "candidate_defrost_time": times,
                "cycle_cop": [4.0, 3.8, 3.6],
                "cycle_cop_eligible": True,
                "rgb_available": [False, True, True],
                "prediction": [np.nan, 0.7, 0.9],
                "t_RB": times[1],
            }
        )
    )
    row = cycle_metrics(rows).iloc[0]
    assert np.isclose(row.relative_cop_loss, 0.1)
    assert np.isclose(row.input_coverage_loss, 0.05)
    assert np.isclose(row.network_cop_loss, 0.05)
    assert row.predicted_time == times[2]
    assert row.rb_cop == 3.8


def test_cold_start_base_uses_confirmed_boundary_not_standby():
    from pathlib import Path

    import pytest

    root = Path(__file__).resolve().parents[2]
    path = root / "output/image_models/relative_cop/base/frost_cycle_000023.parquet"
    if not path.exists():
        pytest.skip("local prepared regression data unavailable")
    frame = pd.read_parquet(path)
    bounds = pd.read_csv(
        root / "output/defrost_decisions/effective_cop_zero/recovery_boundaries.csv"
    )
    start = pd.Timestamp(
        bounds.loc[bounds.cycle_name.eq("frost_cycle_000023"), "heating_start"].iloc[0]
    )
    assert start == pd.Timestamp("2026-07-21 08:56:35")
    expected = (pd.to_datetime(frame.candidate_defrost_time) - start).dt.total_seconds() / 60
    np.testing.assert_allclose(frame.heating_elapsed_minutes, expected)


def test_preprocessing_fits_training_only_and_drops_empty_columns():
    from image_models.relative_cop import fit_regression

    columns = feature_columns()
    train = pd.DataFrame(1.0, index=range(4), columns=columns).assign(
        cycle_name="a", experiment_id="train", target=0.9, input_available=True
    )
    valid = pd.DataFrame(100.0, index=range(2), columns=columns).assign(
        cycle_name="b", experiment_id="heldout", target=0.8, input_available=True
    )
    train["stat_water_flow_skew"] = np.nan
    result = fit_regression(train, valid, epochs=1, patience=1, seed=0)
    assert result["training_experiments"] == ["train"]
    assert "stat_water_flow_skew" not in result["feature_columns"]
    np.testing.assert_allclose(result["preprocessor"].named_steps["standardscaler"].mean_, 1.0)
    assert result["feature_columns"][-1] == "elapsed_minutes"


def test_d32_cop_reuses_branch_encoder_and_time_linear_head():
    from image_models.outcome_representation import OutcomeRepresentation
    from image_models.relative_cop import D32COP, fit_regression, predict_regression

    columns = feature_columns()
    model = D32COP(columns)
    assert isinstance(model, OutcomeRepresentation)
    assert model.sensor_width == 196 and model.accounting_width == 2
    assert (model.visual.in_features, model.visual.out_features) == (384, 32)
    assert (model.fusion.in_features, model.fusion.out_features) == (66, 32)
    assert (model.head.in_features, model.head.out_features) == (33, 1)
    assert model(torch.zeros(3, 583)).shape == (3,)
    train = pd.DataFrame(1.0, index=range(4), columns=columns).assign(
        cycle_name="a", experiment_id="train", target=0.9, input_available=True
    )
    fitted = fit_regression(train, None, epochs=2, patience=1, seed=0, architecture="d32")
    assert fitted["architecture"] == "d32"
    assert predict_regression(train, fitted).prediction.notna().all()


def test_binary_reference_and_fit_use_rgb_only_and_full_curve():
    from image_models.relative_cop import RGB, binary_labels, fit_regression, predict_regression

    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 384)), columns=RGB)
    frame = frame.assign(
        cycle_name="a",
        experiment_id="train",
        candidate_defrost_time=times,
        cycle_cop=[2.0, 4.0, 4.0, 3.0, 2.0, 1.0],
        cycle_cop_eligible=True,
        target=[0.5, 1.0, 1.0, 0.75, 0.5, 0.25],
        input_available=True,
        rgb_available=[True, False, True, True, True, True],
    )
    labels = binary_labels(frame)
    assert labels.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fitted = fit_regression(
        frame,
        frame.assign(experiment_id="validation"),
        epochs=2,
        patience=1,
        seed=0,
        architecture="dinov2-binary",
        batch_size=2,
    )
    assert fitted["feature_columns"] == RGB
    assert fitted["training_experiments"] == ["train"]
    assert "validation_cross_entropy" in fitted["losses"]
    result = predict_regression(frame, fitted)
    assert result.prediction.dropna().between(0, 1).all()
    assert pd.isna(result.prediction.iloc[1])
    assert result.trigger_positive.equals(result.prediction.ge(0.5))
    altered = frame.copy()
    altered["target"] = 999.0
    pd.testing.assert_series_equal(
        result.prediction, predict_regression(altered, fitted).prediction
    )
    weights = cycle_weights(pd.Series(["a", "b", "b", "b"]))
    assert np.isclose(weights[0], weights[1:].sum())


def test_online_missing_reference_and_input_remain_visible():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curve = pd.DataFrame(
        dict(
            candidate_defrost_time=times,
            cycle_cop=[2.0, 4.0, 3.0],
            cycle_cop_eligible=True,
            target=[0.5, 1.0, 0.75],
        )
    )
    assert (
        online_trigger_metrics(curve, curve.assign(score=np.nan), 0.5)[0]["status"]
        == "no_available_input"
    )
    absent = curve.assign(cycle_cop_eligible=False, target=np.nan)
    assert (
        online_trigger_metrics(absent, curve.assign(score=1.0), 0.5)[0]["status"]
        == "no_supported_reference"
    )


def test_online_renderer_keeps_full_cohort_denominators(tmp_path):
    from plots.pareto_learning import render_online_cop

    rows = pd.DataFrame(
        dict(
            model=["binary"] * 2,
            cycle_name=["a", "b"],
            status=["scored", "no_trigger"],
            trigger_time=[pd.Timestamp("2026-01-01"), pd.NaT],
            relative_cop_loss=[0.01, np.nan],
            time_error_minutes=[-1.0, np.nan],
            absolute_time_error_minutes=[1.0, np.nan],
            within_1pct=[1.0, np.nan],
            within_2pct=[1.0, np.nan],
            within_5pct=[1.0, np.nan],
        )
    )
    summary = render_online_cop(rows, tmp_path)
    own = summary.loc[summary.scope.eq("own_scored")].iloc[0]
    assert own.hit_2pct_of_cohort == 0.5 and own.no_trigger_rate == 0.5
    assert own.median_absolute_time_error_minutes == 1.0
    assert not any("first_positive" in p.name for p in tmp_path.iterdir())


def test_binary_validation_loss_is_cycle_equal_and_training_only_scaler():
    from image_models.relative_cop import RGB, binary_labels, fit_regression, regression_model

    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(4, 384)), columns=RGB).assign(
        cycle_name="train-cycle",
        experiment_id="train",
        candidate_defrost_time=pd.date_range("2026-01-01", periods=4, freq="30s"),
        cycle_cop=[2.0, 4.0, 3.0, 2.0],
        cycle_cop_eligible=True,
        target=[0.5, 1.0, 0.75, 0.5],
        input_available=True,
        rgb_available=True,
    )
    validation = pd.concat(
        [train.iloc[:1].assign(cycle_name="short"), train.assign(cycle_name="long")],
        ignore_index=True,
    )
    validation[RGB] += 10
    validation["experiment_id"] = "held-out-validation"
    fitted = fit_regression(
        train, validation, epochs=1, patience=1, seed=0, architecture="dinov2-binary"
    )
    scaler = fitted["preprocessor"].steps[-1][1]
    np.testing.assert_allclose(scaler.mean_, train[RGB].mean().to_numpy())
    model = regression_model(RGB, "dinov2-binary")
    model.load_state_dict(fitted["model_state_dict"])
    with torch.no_grad():
        logits = model(
            torch.tensor(fitted["preprocessor"].transform(validation[RGB]), dtype=torch.float32)
        )
        losses = nn.functional.cross_entropy(
            logits,
            torch.tensor(binary_labels(validation).to_numpy(), dtype=torch.long),
            reduction="none",
        ).numpy()
    assert np.isclose(
        fitted["losses"].validation_cross_entropy.iloc[0], 0.5 * (losses[0] + losses[1:].mean())
    )


def test_binary_cli_selects_one_cached_recipe(monkeypatch):
    import sys
    from pathlib import Path

    import train_pareto_boundary
    from image_models import relative_cop

    calls = []
    monkeypatch.setattr(relative_cop, "run", calls.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pareto_boundary.py",
            "--task",
            "effective-cop-binary",
            "--dataset",
            "../../dataset",
        ],
    )
    train_pareto_boundary.main()
    args = calls[0]
    assert args.regression_architecture == "dinov2-binary"
    assert args.rgb == "on" and args.trigger_threshold == 0.5
    assert args.processing_seconds == 30 and args.n_jobs == 6
    assert args.output == Path("output/image_models/dinov2_binary_tref")


def test_binary_inference_needs_no_cop_or_time_labels():
    from image_models.relative_cop import RGB, fit_regression, predict_regression

    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 384)), columns=RGB).assign(
        cycle_name="a",
        experiment_id="train",
        candidate_defrost_time=pd.date_range("2026-01-01", periods=4, freq="30s"),
        cycle_cop=[2.0, 4.0, 3.0, 2.0],
        cycle_cop_eligible=True,
        target=[0.5, 1.0, 0.75, 0.5],
        input_available=True,
        rgb_available=True,
    )
    fitted = fit_regression(frame, None, epochs=1, patience=1, seed=0, architecture="dinov2-binary")
    expected = predict_regression(frame, fitted)
    actual = predict_regression(frame[[*RGB, "rgb_available"]], fitted)
    pd.testing.assert_series_equal(actual.prediction, expected.prediction)
    pd.testing.assert_series_equal(actual.trigger_positive, expected.trigger_positive)


def test_joint_near_optimal_labels_and_global_negative_regression():
    from image_models.relative_cop import near_optimal_labels, classification_regression_loss, regression_model

    target = pd.Series([.4, 1., .98, .995, .8, np.nan])
    labels = near_optimal_labels(target)
    assert labels.iloc[:5].tolist() == [0, 1, 0, 1, 0]
    assert pd.isna(labels.iloc[-1])
    outputs = torch.zeros(5, 2, requires_grad=True)
    classification_regression_loss(outputs, torch.tensor(target.iloc[:5].values)).backward()
    assert (outputs.grad[:, 1] != 0).all()  # negatives also train the COP head
    assert (outputs.grad[:, 0] != 0).all()
    model = regression_model(feature_columns('off'), 'cop-classification-regression')
    assert model(torch.zeros(3, 199)).shape == (3, 2)


def test_joint_thresholds_and_missing_confirmation_slots():
    from image_models.relative_cop import calibrate_thresholds, online_trigger_metrics

    curve = pd.DataFrame(dict(
        cycle_name=['a'] * 5,
        candidate_defrost_time=pd.date_range('2025-01-01', periods=5, freq='30s'),
        cycle_cop=[1., 1.99, 2., 1.8, 1.7], cycle_cop_eligible=True,
        binary_target=[0, 1, 1, 0, 0], probability=[.2, .9, .95, .3, .1],
    ))
    choices, table = calibrate_thresholds(curve)
    assert choices['first_positive']['threshold'] == .95
    assert choices['two_of_three']['threshold'] == .9
    stream = curve.assign(score=[.9, np.nan, .9, .1, .1])
    assert online_trigger_metrics(curve, stream, .5, 'first_positive')[0]['trigger_time'] == curve.candidate_defrost_time.iloc[0]
    assert online_trigger_metrics(curve, stream, .5, 'two_of_three')[0]['trigger_time'] == curve.candidate_defrost_time.iloc[2]
    choices, _ = calibrate_thresholds(curve.assign(binary_target=0))
    assert np.isnan(choices['first_positive']['threshold'])


def test_joint_singletons_validation_loss_and_train_only_preprocessor():
    from image_models.relative_cop import fit_regression, predict_regression, classification_regression_loss

    columns = feature_columns('off')
    frame = pd.DataFrame(np.zeros((3, len(columns))), columns=columns).assign(
        cycle_name=['a', 'b', 'c'], experiment_id='train', target=[.4, 1., .995], input_available=True,
        candidate_defrost_time=pd.date_range('2025-01-01', periods=3, freq='10s'),
    )
    valid = frame.assign(experiment_id='valid')
    valid[columns[0]] = 100.
    fit = fit_regression(frame, valid, epochs=1, patience=1, seed=0, rgb='off',
                         architecture='cop-classification-regression', batch_size=2)
    assert fit['training_experiments'] == ['train']
    assert fit['preprocessor'][-1].mean_[0] == 0
    assert fit['losses'].optimizer_steps.iloc[0] == 2
    pred = predict_regression(valid, fit)
    out = torch.tensor(pred[['classification_logit', 'prediction']].values)
    loss = classification_regression_loss(out, torch.tensor(pred.target.values))
    assert np.isclose(loss.item(), fit['losses'].validation_total_loss.iloc[0])
    assert 'trigger_positive' not in pred


def test_pure_classification_has_one_logit_no_cop_loss_or_prediction():
    from image_models.relative_cop import classification_regression_loss, fit_regression, predict_regression

    logits = torch.tensor([-.5, .5], requires_grad=True)
    y = torch.tensor([.1, 1.])
    loss = classification_regression_loss(logits, y)
    torch.testing.assert_close(loss, nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])))
    torch.testing.assert_close(loss, classification_regression_loss(logits, torch.tensor([.98, .995]), weight=100))
    frame = pd.DataFrame(np.zeros((2,199)), columns=feature_columns('off')).assign(
        cycle_name=['a','b'], experiment_id='train', target=[.1,1.], input_available=True)
    fit = fit_regression(frame, None, epochs=1, patience=1, seed=0, rgb='off', architecture='cop-classification')
    assert fit['model_state_dict']['predictor.3.weight'].shape == (1,32)
    out = predict_regression(frame, fit)
    assert out.probability.notna().all() and out.prediction.isna().all()
    assert fit['dynamic_state_dict'] is None


def test_after_optimum_labels_stay_positive_and_ignore_missing_peak_input():
    from image_models.relative_cop import fit_regression, predict_regression, binary_labels

    frame = pd.DataFrame(np.zeros((5,199)), columns=feature_columns('off')).assign(
        cycle_name='a', experiment_id='train', target=[.5,.99,1.,.98,.4],
        cycle_cop=[2.,3.96,4.,3.92,1.6], cycle_cop_eligible=True,
        input_available=[True,True,False,True,True],
        candidate_defrost_time=pd.date_range('2025-01-01',periods=5,freq='10s'))
    assert binary_labels(frame).tolist() == [0.,0.,1.,1.,1.]
    fit = fit_regression(frame, frame, epochs=1, patience=1, seed=0,
                         rgb='off', architecture='cop-classification', classification_label='after-optimum')
    predicted = predict_regression(frame, fit)
    assert predicted.binary_target.tolist() == [0.,0.,1.,1.,1.]
    valid = predicted.loc[predicted.input_available]
    expected = nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(valid.classification_logit.values), torch.tensor(valid.binary_target.values))
    assert np.isclose(expected.item(), fit['losses'].validation_total_loss.iloc[0])
    assert predicted.prediction.isna().all()


def test_complete_peak_screen_requires_observed_both_sides_and_no_gap_bridging():
    from image_models.relative_cop import complete_peak_screen

    curve = pd.DataFrame(dict(cycle_name='a', experiment_id='e',
        candidate_defrost_time=pd.date_range('2025-01-01', periods=17, freq='10s'),
        cycle_cop=[.9]*7+[.995,1.,.995]+[.9]*7, cycle_cop_eligible=True))
    assert complete_peak_screen(curve).selected.iloc[0]
    assert not complete_peak_screen(curve.iloc[:9]).selected.iloc[0]
    assert not complete_peak_screen(curve.assign(cycle_cop=[.9]*7+[.995,1.]+[.995]*8)).selected.iloc[0]
    missing = curve.copy(); missing.loc[13,'cycle_cop_eligible'] = False
    assert not complete_peak_screen(missing).selected.iloc[0]
    assert not complete_peak_screen(curve.drop(index=[12,13,14,15])).selected.iloc[0]
    assert complete_peak_screen(curve.assign(prediction=100, rgb_available=False)).selected.iloc[0]


def test_dataset_peak_quality_preserves_prior_status_and_updates_only_requested_cycles(tmp_path):
    from dataset_tools.cycle_metadata import write_catalog, read_catalog, update_effective_cop_quality

    records = [dict(cycle_name=n, status=status, status_reason='original', pipeline_status='valid')
               for n, status in [('a','valid'),('b','invalid'),('c','partial')]]
    write_catalog(tmp_path, {'cycles':records})
    curve = pd.DataFrame(dict(cycle_name='a', experiment_id='e',
        candidate_defrost_time=pd.date_range('2025-01-01', periods=17, freq='10s'),
        cycle_cop=[.9]*7+[.995,1.,.995]+[.9]*7, cycle_cop_eligible=True))
    rows = pd.concat([curve,curve.assign(cycle_name='b')],ignore_index=True)
    update_effective_cop_quality(tmp_path, rows, cycle_names=['a','b','c'])
    got=read_catalog(tmp_path)['cycles']
    assert [(r['status'],r['status_reason']) for r in got]==[
        ('valid','original'),('invalid','original'),('partial','original')]
    assert got[2]['pre_cop_status']['status']=='partial'
    update_effective_cop_quality(tmp_path, curve.iloc[:9])
    got=read_catalog(tmp_path)['cycles']
    assert got[0]['status']=='valid' and got[0]['pre_cop_status']['status']=='valid'
    assert got[1]['status_reason']=='original'
    update_effective_cop_quality(tmp_path, curve)
    assert read_catalog(tmp_path)['cycles'][0]['status']=='valid'
    reviewed = read_catalog(tmp_path)
    reviewed['cycles'][1]['review_status'] = 'valid'
    write_catalog(tmp_path, reviewed)
    update_effective_cop_quality(tmp_path, rows)
    assert read_catalog(tmp_path)['cycles'][1]['status']=='invalid'
    assert read_catalog(tmp_path)['cycles'][1]['pre_cop_status']['status']=='invalid'


def test_rgb_validity_is_presence_independent_of_peak_and_predictions(tmp_path):
    from dataset_tools.cycle_metadata import write_catalog, read_catalog, update_rgb_validity
    write_catalog(tmp_path, {'cycles':[dict(cycle_name='a',status='valid')]})
    frame=pd.DataFrame(dict(cycle_name='a',candidate_defrost_time=pd.date_range('2025-01-01',periods=6,freq='30s'),
        cycle_cop=[1.,2.,3.,2.9,2.8,2.7],cycle_cop_eligible=True,rgb_available=[True,True,False,True,False,True],prediction=0.))
    assert update_rgb_validity(tmp_path,frame).rgb_valid.iloc[0]
    assert read_catalog(tmp_path)['cycles'][0]['status']=='valid'
    missing=frame.drop(index=[4]);missing.loc[missing.index[-1],'candidate_defrost_time']+=pd.Timedelta(minutes=2)
    assert update_rgb_validity(tmp_path,missing).rgb_valid.iloc[0]
    assert not update_rgb_validity(tmp_path, frame.assign(rgb_available=False)).rgb_valid.iloc[0]
    pd.DataFrame({'cycle_name':['a'], 'camera_role':['left']}).to_parquet(tmp_path/'image_metadata.parquet')
    assert update_rgb_validity(tmp_path, frame.assign(rgb_available=False)).rgb_valid.iloc[0]
    assert update_rgb_validity(tmp_path,frame.assign(rgb_available=[True,True,False,False,False,False])).rgb_valid.iloc[0]


def test_rgb_projection_preserves_numeric_inputs_and_checkpoint_prediction():
    from image_models.relative_cop import regression_model, fit_regression, predict_regression, Sin

    model = regression_model(feature_columns(), 'cop-classification', rgb_projection=32)
    assert isinstance(model.rgb_projection[1], Sin)
    assert model.encoder[0].in_features == 231
    assert sum(p.numel() for p in model.parameters()) == 32941
    x = torch.randn(4, 583, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad[:, :384].abs().sum() > 0 and x.grad[:, 384:].abs().sum() > 0
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(4,583)), columns=feature_columns()).assign(
        cycle_name='a', experiment_id='e', target=[.5,1.,.9,.8], input_available=[True,True,False,True])
    fitted = fit_regression(frame, None, epochs=1, patience=1, seed=0,
                           architecture='cop-classification', rgb_projection=32)
    assert fitted['rgb_projection'] == 32
    predicted = predict_regression(frame, fitted)
    assert predicted.probability.notna().tolist() == [True,True,False,True]


def test_sensor_matched_rgb_cache_keeps_availability(tmp_path):
    from types import SimpleNamespace
    from image_models.relative_cop import prepare_cycle

    source, output = tmp_path/'source', tmp_path/'output'
    (source/'base').mkdir(parents=True); (output/'base').mkdir(parents=True)
    pd.DataFrame({'effective_heat_rule':['outlet_at_least_recovery_temperature']*2,
                  'rgb_available':[True,False], 'dinov2_000':[1.,np.nan]}).to_parquet(source/'base/c.parquet')
    prepare_cycle(SimpleNamespace(reference_run=source, output=output, rgb='off', require_rgb_input=True),
                  SimpleNamespace(cycle_name='c'))
    assert pd.read_parquet(output/'base/c.parquet').rgb_available.tolist() == [True,False]


def test_cached_base_allows_legacy_row_without_observed_preparation(tmp_path):
    from types import SimpleNamespace
    from image_models.relative_cop import prepare_cycle

    output = tmp_path / "output"
    (output / "base").mkdir(parents=True)
    pd.DataFrame({
        "effective_heat_rule": ["outlet_at_least_recovery_temperature"],
        "candidate_defrost_time": [pd.Timestamp("2026-01-01")],
        "observed_defrost_preparation_start": [pd.Timestamp("2026-01-01")],
    }).to_parquet(output / "base/c.parquet")

    prepare_cycle(SimpleNamespace(output=output), SimpleNamespace(cycle_name="c"))


def test_empty_supported_validation_preserves_unavailable_model_without_predictions():
    from image_models.relative_cop import fit_regression, predict_regression

    frame = pd.DataFrame(np.zeros((3,199)), columns=feature_columns('off')).assign(
        cycle_name='a', experiment_id='e', target=[.5,1.,.8], cycle_cop=[2.,4.,3.2],
        cycle_cop_eligible=True, input_available=True,
        candidate_defrost_time=pd.date_range('2025-01-01', periods=3, freq='10s'))
    fit = fit_regression(frame, frame.assign(target=np.nan, cycle_cop_eligible=False),
                         epochs=1, patience=1, seed=0, rgb='off', architecture='cop-classification',
                         classification_label='after-optimum')
    assert fit['status'] == 'no_usable_validation_samples' and fit['selected_epoch'] == 0
    out = predict_regression(frame, fit)
    assert out.probability.isna().all() and out.input_available.all()
    assert out.binary_target.tolist() == [0.,1.,1.]


def test_domain_extrapolation_is_scored_but_invalid_measurements_are_not():
    from image_models.relative_cop import online_trigger_metrics
    times = pd.date_range('2025-01-01', periods=3, freq='30s')
    curve = pd.DataFrame(dict(candidate_defrost_time=times, cycle_cop=[2.,3.,2.8],
        cycle_cop_eligible=[True,True,False], cycle_cop_measurements_valid=True,
        cycle_cop_physically_valid=True, pre_defrost_feature_window_valid=True,
        defrost_event_electricity_prediction_available=True,
        defrost_event_net_heat_prediction_available=True))
    stream = pd.DataFrame(dict(candidate_defrost_time=times, score=[0.,0.,1.]))
    result = online_trigger_metrics(curve, stream, .5, 'first_positive', allow_model_extrapolation=True)[0]
    assert result['status']=='scored' and result['outside_reference_support']
    assert result['trigger_cop']==2.8
    curve.loc[2,'cycle_cop_measurements_valid']=False
    result = online_trigger_metrics(curve, stream, .5, 'first_positive', allow_model_extrapolation=True)[0]
    assert result['status']!='scored'


def test_extrapolated_trigger_cannot_count_as_formal_near_optimal():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curve = pd.DataFrame({
        "candidate_defrost_time": times, "cycle_cop": [2., 3., 4.],
        "cycle_cop_eligible": [True, True, False],
        "cycle_cop_measurements_valid": True, "cycle_cop_physically_valid": True,
        "pre_defrost_feature_window_valid": True,
        "defrost_event_electricity_prediction_available": True,
        "defrost_event_net_heat_prediction_available": True,
    })
    result = online_trigger_metrics(
        curve, pd.DataFrame({"candidate_defrost_time": times, "score": [0., 0., 1.]}),
        .5, "first_positive", allow_model_extrapolation=True,
    )[0]

    assert result["outside_reference_support"] and result["trigger_cop"] == 4.
    assert result["extrapolated_reference_gap"] == pytest.approx(1 - 4 / 3)
    assert pd.isna(result["relative_cop_loss"])
    assert all(pd.isna(result[f"within_{percent}pct"]) for percent in (1, 2, 5))


def test_online_trigger_keeps_reliable_cop_without_a_supported_peak():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curve = pd.DataFrame({
        "candidate_defrost_time": times, "cycle_cop": [2., 3., 4.],
        "cycle_cop_eligible": False,
        "cycle_cop_measurements_valid": True, "cycle_cop_physically_valid": True,
        "pre_defrost_feature_window_valid": True,
        "defrost_event_electricity_prediction_available": True,
        "defrost_event_net_heat_prediction_available": True,
    })

    result = online_trigger_metrics(
        curve, curve.assign(score=[0., 1., 1.]), .5,
        allow_model_extrapolation=True,
    )[0]

    assert result["status"] == "scored"
    assert result["trigger_time"] == times[-1]
    assert result["trigger_cop"] == 4.
    assert result["outside_reference_support"]
    assert pd.isna(result["reference_cop"])
    assert pd.isna(result["relative_cop_loss"])


def test_online_trigger_before_accounting_start_is_never_given_virtual_cop():
    from image_models.relative_cop import online_trigger_metrics

    times = pd.date_range("2026-01-01", periods=4, freq="30s")
    curve = pd.DataFrame({
        "candidate_defrost_time": times, "heating_accounting_start": times[3],
        "cycle_cop": [9., 9., 9., 3.], "cycle_cop_eligible": [False] * 3 + [True],
    })

    result = online_trigger_metrics(curve, curve.assign(score=1.), .5)[0]

    assert result["trigger_time"] == times[1]
    assert result["status"] == "before_reference_accounting_start"
    assert pd.isna(result["trigger_cop"])
    assert pd.isna(result["reference_cop"])


def test_common_policy_reference_uses_the_deduplicated_old_new_union(tmp_path, monkeypatch):
    from image_models import relative_cop

    times = pd.date_range("2026-01-01", periods=3, freq="10s")
    shared = {
        "cycle_name": "cycle", "experiment_id": "experiment",
        "heating_start": times[0], "stable_heating_start": times[0],
        "heating_accounting_start": times[0],
        "t_RB": times[1],
    }
    current = pd.DataFrame({**shared, "candidate_defrost_time": times[[0, 2]]})
    legacy = pd.DataFrame({**shared, "candidate_defrost_time": times})
    current_path, legacy_path = tmp_path / "current.parquet", tmp_path / "legacy.parquet"
    current.to_parquet(current_path); legacy.to_parquet(legacy_path)
    monkeypatch.setattr(
        relative_cop, "apply_reference",
        lambda rows, parameters, rgb, include_history: rows.assign(applied=True),
    )

    result = relative_cop._common_policy_reference(
        current_path, legacy_path, {"training_experiment_ids": []}
    )

    assert result.candidate_defrost_time.tolist() == list(times)
    assert result.applied.all()
    legacy["heating_accounting_start"] = times[1]
    legacy.to_parquet(legacy_path)
    with pytest.raises(ValueError, match="heating_accounting_start"):
        relative_cop._common_policy_reference(
            current_path, legacy_path, {"training_experiment_ids": []}
        )


def test_policy_percent_change_rejects_nonpositive_rb_cop():
    from image_models.relative_cop import _percent_change

    result = _percent_change(pd.Series([2., 1.]), pd.Series([1., 0.]))
    assert result.iloc[0] == 100.
    assert pd.isna(result.iloc[1])


def test_audit_cycle_keeps_truth_support_separate_from_oracle_input():
    from image_models.relative_cop import audit_cycle

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    curve = pd.DataFrame({
        "cycle_name": "a", "experiment_id": "e",
        "candidate_defrost_time": times,
        "cycle_cop": [9.8, 9.9, 10.0, 12.0, 9.7],
        "cycle_cop_eligible": [True, True, True, False, True],
        "pre_defrost_electricity_measurement_valid": True,
        "pre_defrost_heat_measurement_valid": True,
        "sensor_timestamp": times,
        "rgb_available": True,
        "t_RB": times[1],
        "observed_defrost_preparation_start": times[-1],
    })

    result = audit_cycle(curve)

    assert result["reference_peak_cop"] == 10.0
    assert result["trusted_label_count"] == 4
    assert result["rb_reference_supported"]
    assert result["rb_cop"] == 9.9
    assert result["rb_headroom"] == pytest.approx((10 - 9.9) / 9.9)
    assert result["oracle_1pct_trigger_time"] == times[2]
    assert result["oracle_1pct_hit"]
    assert result["oracle_2pct_trigger_time"] == times[1]
    assert result["oracle_2pct_hit"]


def test_grouped_audit_folds_rotate_sorted_experiments_without_targets():
    from image_models.relative_cop import grouped_audit_folds

    rows = pd.DataFrame({
        "experiment_id": ["a", "a", "b", "c", "c", "c", "d", "e"],
        "cycle_name": list("abcdefgh"),
    })
    folds = grouped_audit_folds(rows)
    assert sorted(name for names in folds.values() for name in names) == ["a", "b", "c", "d", "e"]
    assert all(set(left).isdisjoint(right) for left in folds.values() for right in folds.values() if left is not right)
    assert folds == grouped_audit_folds(rows.sample(frac=1, random_state=2))


def test_audit_oracle_does_not_compress_missing_30_second_slots():
    from image_models.relative_cop import audit_cycle

    times = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:01:30"])
    curve = pd.DataFrame({
        "cycle_name": "a", "experiment_id": "e", "candidate_defrost_time": times,
        "cycle_cop": [1., 1.], "cycle_cop_eligible": True,
        "pre_defrost_electricity_measurement_valid": True,
        "pre_defrost_heat_measurement_valid": True, "sensor_timestamp": times,
        "rgb_available": True, "t_RB": pd.NaT,
        "observed_defrost_preparation_start": times[-1],
    })

    result = audit_cycle(curve)
    assert result["processing_30s_count"] == 4
    assert pd.isna(result["oracle_1pct_trigger_time"])
    assert not result["oracle_1pct_hit"]
    assert "no_rb_trigger" in result["missing_data_reason"]
    assert "rb_outside_reference_support" not in result["missing_data_reason"]
    assert result["observed_defrost_preparation_start"] == times[-1]

    missing_actual = audit_cycle(curve.assign(observed_defrost_preparation_start=pd.NaT))
    assert pd.isna(missing_actual["observed_defrost_preparation_start"])
    assert "no_observed_preparation_boundary" in missing_actual["missing_data_reason"]
    assert "actual_preparation_outside_reference_support" not in missing_actual["missing_data_reason"]


def test_audit_gate_requires_one_joint_rb_comparison_not_universal_rb_support():
    from image_models.relative_cop import summarize_audit

    rows = pd.DataFrame({
        "rgb_valid_cohort": True,
        "reference_peak_cop": [1., 1.],
        "oracle_1pct_hit": [True, True], "oracle_2pct_hit": [True, True],
        "rb_reference_supported": [True, False],
        "joint_input_available_count": [1, 1],
    })

    summary = summarize_audit(rows)
    assert summary.classifier_readiness.eq("classifier_development_ready").all()
    assert summary.cop_benefit_readiness.eq("cop_benefit_comparison_available").all()


def test_audit_reports_reference_sensor_and_shared_oracles_independently():
    from image_models.relative_cop import audit_cycle

    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    curve = pd.DataFrame({
        "cycle_name": "a", "experiment_id": "e", "candidate_defrost_time": times,
        "cycle_cop": [1., 1., .5], "cycle_cop_eligible": True,
        "pre_defrost_electricity_measurement_valid": True,
        "pre_defrost_heat_measurement_valid": True,
        "sensor_timestamp": times, "rgb_available": False,
        "t_RB": times[0], "observed_defrost_preparation_start": times[-1],
    })

    result = audit_cycle(curve)
    assert result["reference_available_count"] == 3
    assert result["sensor_input_available_count"] == 3
    assert result["front_input_available_count"] == 0
    assert result["oracle_reference_1pct_hit"]
    assert result["oracle_sensor_1pct_hit"]
    assert not result["oracle_shared_1pct_hit"]
    assert result["processing_grid_complete"]


def test_support_decomposition_keeps_oof_and_full_domains_and_event_outcome_separate():
    from image_models.relative_cop import support_decomposition_rows
    from defrost_event_models.ridge_models import DYNAMIC_STATE_8

    times = pd.date_range("2026-01-01", periods=2, freq="1min")
    common = pd.DataFrame({
        "cycle_name": "a", "experiment_id": "e", "candidate_defrost_time": times,
        "t_RB": times[0], "observed_defrost_preparation_start": times[1],
        "pre_defrost_electricity_measurement_valid": True,
        "pre_defrost_heat_measurement_valid": True, "sensor_timestamp": times,
        "rgb_available": [True, False],
        **{name: 1. for name in DYNAMIC_STATE_8},
    })
    oof = common.assign(
        defrost_event_electricity_prediction_available=True,
        defrost_event_electricity_in_training_domain=[False, True],
        cycle_cop_eligible=[False, True],
    )
    full = common.assign(
        defrost_event_electricity_prediction_available=True,
        defrost_event_electricity_in_training_domain=True,
        cycle_cop_eligible=True,
    )
    events = pd.DataFrame({
        "cycle_name": ["a"], "energy_event_valid": [False],
        "defrost_event_electricity_observed_kwh": [np.nan],
    })

    rows = support_decomposition_rows(oof, full, events)
    rb = rows.loc[rows.point.eq("rb")].iloc[0]
    actual = rows.loc[rows.point.eq("actual_preparation")].iloc[0]
    assert rb.oof_root_reason == "ridge_outside_support"
    assert rb.full_root_reason == "supported"
    assert rb.shared_input_available
    assert not actual.front_input_available
    assert not rb.own_event_outcome_available
    assert actual.oof_root_reason == "supported"
    assert rb.rb_vs_actual_prep_minutes == 1.

    absent = support_decomposition_rows(
        oof.assign(t_RB=times[0] + pd.Timedelta(seconds=30)),
        full.assign(t_RB=times[0] + pd.Timedelta(seconds=30)), events,
    )
    assert absent.loc[absent.point.eq("rb"), "oof_root_reason"].item() == "candidate_not_found"
    outside = support_decomposition_rows(oof, full.assign(
        defrost_event_electricity_in_training_domain=False, cycle_cop_eligible=False
    ), events)
    assert outside.loc[outside.point.eq("rb"), "support_change"].item() == "outside_both"


def test_audit_summary_conditions_only_on_trusted_reference_grid_rows():
    from image_models.relative_cop import summarize_audit

    rows = pd.DataFrame({
        "reference_peak_cop": [1., 1.], "trusted_reference_grid_count": [1, 0],
        "oracle_reference_1pct_hit": [True, False], "oracle_reference_2pct_hit": [True, False],
        "oracle_sensor_1pct_hit": [True, False], "oracle_sensor_2pct_hit": [True, False],
        "oracle_shared_1pct_hit": [False, False], "oracle_shared_2pct_hit": [False, False],
        "oracle_1pct_hit": [False, False], "oracle_2pct_hit": [False, False],
        "rb_reference_supported": [False, False], "joint_input_available_count": [0, 0],
    })

    summary = summarize_audit(rows)
    assert summary.trusted_reference_grid_denominator.eq(1).all()
    reference = summary.loc[summary.oracle_mode.eq("reference")]
    assert reference.oracle_hit_rate_full_cohort.eq(.5).all()
    assert reference.oracle_hit_rate_trusted_reference_grid.eq(1.).all()
    assert summary.classifier_readiness.eq("classifier_development_ready").all()
