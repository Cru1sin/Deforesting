import numpy as np
import pandas as pd
import torch


def rows(experiment: str, offset: float = 0) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "row_id": [f"{experiment}:{index}" for index in range(3)],
            "cycle_name": experiment,
            "experiment_id": experiment,
            "candidate_defrost_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "defrost_preparation_start": pd.Timestamp("2026-01-01 00:02"),
            "stat_temperature_mean": np.arange(3.0) + offset,
            "stat_temperature_mean_missing": False,
            "online_pre_defrost_heat_kwh": np.arange(3.0),
            "online_pre_defrost_electricity_kwh": np.arange(3.0),
            "online_pre_defrost_compressor_electricity_kwh": np.arange(3.0),
            "online_pointwise_valid": True,
            "rgb_missing": False,
            "rgb_age_seconds": 0.0,
            "defrost_event_electricity_kwh": np.arange(3.0) + 10,
        }
    )
    rgb = pd.DataFrame(
        np.random.default_rng(0).normal(size=(3, 384)),
        columns=[f"dinov2_{index:03d}" for index in range(384)],
    )
    return pd.concat([result, rgb], axis=1)


def events(experiments: str) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "cycle_name": list(experiments), "experiment_id": list(experiments),
            "event_id": list(experiments), "event_valid": True,
        }
    )
    names = (
        "defrost_event_electricity_observed_kwh", "defrost_event_net_heat_observed_kwh",
        "defrost_event_compressor_electricity_observed_kwh",
        "defrost_event_duration_observed_minutes",
    )
    for index, name in enumerate(names, 1):
        result[name] = np.arange(len(result), dtype=float) + index
    for name in ("energy_event_valid", "heat_event_valid", "compressor_event_valid",
                 "duration_event_valid"):
        result[name] = True
    return result


def test_event_rows_use_only_actual_preparation_start():
    from image_models.outcome_representation import outcome_event_rows

    result = outcome_event_rows(pd.concat([rows("a"), rows("b")]), events("ab"))
    assert result.row_id.tolist() == ["a:2", "b:2"]
    assert result.event_id.tolist() == ["a", "b"]


def test_event_rows_accept_current_event_table_key():
    from image_models.outcome_representation import outcome_event_rows

    event_table = events("a").drop(columns="cycle_name")
    result = outcome_event_rows(rows("a"), event_table)

    assert result[["cycle_name", "event_id"]].iloc[0].tolist() == ["a", "a"]


def test_outcome_encoder_reloads_identical_latents_without_test_fit():
    from image_models.outcome_representation import (
        encode_outcome_rows,
        fit_outcome_representation,
        outcome_event_rows,
        predict_outcomes,
    )

    torch.set_num_threads(1)
    all_rows = {name: rows(name, 1000 if name == "d" else 0) for name in "abcd"}
    event_table = events("abcd")
    samples = {name: outcome_event_rows(value, event_table) for name, value in all_rows.items()}
    result = fit_outcome_representation(
        pd.concat([samples["a"], samples["b"]]), samples["c"],
        pd.concat([samples["a"], samples["b"], samples["c"]]),
        use_rgb=True, maximum_epochs=3, patience=2,
    )
    latent = encode_outcome_rows(all_rows["d"], result["checkpoint"])
    replay = encode_outcome_rows(all_rows["d"], result["checkpoint"])
    np.testing.assert_allclose(latent.filter(like="z_").to_numpy(),
                               replay.filter(like="z_").to_numpy())
    scaler = result["checkpoint"]["preprocessor"].named_steps["standardscaler"]
    assert scaler.mean_[0] < 100
    assert latent.shape[0] == 3
    predicted = predict_outcomes(samples["d"], result["checkpoint"])
    neural = predicted[[name for name in predicted if name.startswith("predicted_")]]
    assert neural.shape == (1, 4)
    assert predicted.ridge_predicted_defrost_event_electricity_observed_kwh.iloc[0] == 12
    assert np.isfinite(neural).all(axis=None)


def test_fixed_epoch_outcome_fit_uses_all_rows_once():
    from image_models.outcome_representation import (
        fit_outcome_fixed,
        outcome_event_rows,
    )

    event_rows = outcome_event_rows(
        pd.concat([rows(name) for name in "abc"]), events("abc")
    )
    event_rows["minutes_since_heating_start"] = [1.0, 2.0, 3.0]
    checkpoint, losses = fit_outcome_fixed(
        event_rows, use_rgb=True, event_head="multimodal_time_linear", epochs=2, seed=0
    )

    assert checkpoint["selected_epoch"] == 2
    assert checkpoint["training_events"] == 3
    assert losses.epoch.max() == 2


def test_latent_event_readout_matches_the_frozen_full_model():
    from image_models.outcome_representation import (
        encode_outcome_rows,
        fit_outcome_representation,
        outcome_event_rows,
        predict_outcomes,
        predict_outcomes_from_latent,
    )

    torch.set_num_threads(1)
    all_rows = {name: rows(name) for name in "abcd"}
    event_table = events("abcd")
    samples = {name: outcome_event_rows(value, event_table) for name, value in all_rows.items()}
    fitted = fit_outcome_representation(
        pd.concat([samples["a"], samples["b"]]), samples["c"],
        pd.concat([samples["a"], samples["b"], samples["c"]]),
        use_rgb=True, maximum_epochs=2, patience=1,
    )
    checkpoint = fitted["checkpoint"]
    latent = encode_outcome_rows(samples["d"], checkpoint)
    direct = predict_outcomes(samples["d"], checkpoint)
    replay = predict_outcomes_from_latent(latent, checkpoint)

    columns = [name for name in direct if name.startswith("predicted_")]
    assert replay.columns.tolist() == ["row_id", *columns]
    np.testing.assert_allclose(replay[columns], direct[columns], rtol=0, atol=1e-7)


def test_heating_time_uses_training_scale_and_flags_both_extrapolation_sides():
    from image_models.outcome_representation import normalized_heating_time

    values = normalized_heating_time(pd.Series([30., 60., 240.]), 60., 180.)

    np.testing.assert_allclose(values.normalized, [1 / 6, 1 / 3, 4 / 3])
    assert values.below_training_range.tolist() == [True, False, False]
    assert values.above_training_range.tolist() == [False, False, True]


def test_quadratic_spline_basis_partitions_unity_and_extrapolates():
    from image_models.outcome_representation import quadratic_spline_basis

    basis = quadratic_spline_basis(np.array([-.25, 0., .5, 1., 1.25]))

    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.sum(axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(basis[1], [1., 0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(basis[3], [0., 0., 0., 1.], atol=1e-12)
    assert np.isfinite(basis[[0, -1]]).all()


def test_equal_varying_coefficients_reduce_to_one_fixed_head_for_batch_one():
    from image_models.outcome_representation import (
        OutcomeRepresentation,
        quadratic_spline_basis,
    )

    torch.manual_seed(7)
    model = OutcomeRepresentation(
        1, 0, use_rgb=False, event_head="multimodal_time_varying"
    ).eval()
    common = torch.randn(4, 33)
    with torch.no_grad():
        model.coefficients.copy_(common.unsqueeze(0).repeat(4, 1, 1))
    values = torch.ones((1, 1))
    basis = torch.tensor(quadratic_spline_basis(np.array([1.2])), dtype=torch.float32)
    latent = model.encode(values)
    expected = torch.nn.functional.linear(latent, common[:, :32], common[:, 32])

    actual = model(values, basis)

    np.testing.assert_allclose(actual.detach(), expected.detach(), rtol=0, atol=1e-6)


def test_varying_latent_readout_matches_full_model():
    from image_models.outcome_representation import (
        encode_outcome_rows,
        fit_outcome_representation,
        outcome_event_rows,
        predict_outcomes,
        predict_outcomes_from_latent,
    )

    torch.set_num_threads(1)
    all_rows = {name: rows(name) for name in "abcd"}
    for frame in all_rows.values():
        frame["minutes_since_heating_start"] = [60., 90., 120.]
    event_table = events("abcd")
    samples = {name: outcome_event_rows(value, event_table) for name, value in all_rows.items()}
    fitted = fit_outcome_representation(
        pd.concat([samples["a"], samples["b"]]), samples["c"],
        pd.concat([samples["a"], samples["b"], samples["c"]]),
        use_rgb=True, event_head="multimodal_time_varying_regularized",
        maximum_epochs=2, patience=1,
    )
    checkpoint = fitted["checkpoint"]
    latent = encode_outcome_rows(samples["d"], checkpoint)
    direct = predict_outcomes(samples["d"], checkpoint)
    replay = predict_outcomes_from_latent(latent, checkpoint)

    columns = [name for name in direct if name.startswith("predicted_")]
    np.testing.assert_allclose(replay[columns], direct[columns], rtol=0, atol=1e-7)
    assert {"time_below_training_range", "time_above_training_range"} <= set(replay)
    assert checkpoint["time_min_minutes"] == 120.
    assert checkpoint["time_scale_minutes"] == 120.
    assert {"event_loss", "coefficient_loss"} <= set(fitted["losses"])


def test_state_recipes_change_only_latent_width_and_curvature():
    from image_models.outcome_representation import (
        CURVATURE_HEADS,
        LATENT_WIDTHS,
        OutcomeRepresentation,
    )

    expected = {
        "multimodal_time_linear_z16": 16,
        "multimodal_time_linear": 32,
        "multimodal_time_linear_z64": 64,
        "multimodal_time_linear_z32_curvature": 32,
        "multimodal_time_linear_z64_curvature": 64,
    }
    assert {name: LATENT_WIDTHS[name] for name in expected} == expected
    assert CURVATURE_HEADS == {
        "multimodal_time_linear_z32_curvature",
        "multimodal_time_linear_z64_curvature",
    }
    for name, width in expected.items():
        model = OutcomeRepresentation(2, 3, use_rgb=True, event_head=name)
        assert model.fusion.out_features == width
        assert model.head.in_features == width + 1


def test_feature_groups_cover_checkpoint_inputs_without_overlap():
    from image_models.outcome_representation import outcome_feature_groups

    columns = [
        "stat_temperature_mean", "stat_temperature_mean_missing",
        "stat_temperature_valid_count", "stat_temperature_age_seconds",
        "online_pre_defrost_heat_kwh", "rgb_missing", "rgb_age_seconds",
        "dinov2_000", "dinov2_001",
    ]
    groups = outcome_feature_groups(columns)

    assert groups == {
        "rgb": ["dinov2_000", "dinov2_001"],
        "sensor": ["stat_temperature_mean"],
        "ledger": ["online_pre_defrost_heat_kwh"],
        "quality": [
            "stat_temperature_mean_missing", "stat_temperature_valid_count",
            "stat_temperature_age_seconds", "rgb_missing", "rgb_age_seconds",
        ],
    }
    assert sorted(sum(groups.values(), [])) == sorted(columns)


def test_temporal_triplets_require_exact_spacing_and_stable_observation_status():
    from image_models.outcome_representation import temporal_triplet_indices

    frame = pd.DataFrame({
        "cycle_name": ["a"] * 5,
        "candidate_defrost_time": pd.to_datetime([
            "2026-01-01 00:00:00", "2026-01-01 00:00:10",
            "2026-01-01 00:00:20", "2026-01-01 00:00:30",
            "2026-01-01 00:00:41",
        ]),
        "is_teacher_candidate": True,
        "sensor_timestamp": pd.to_datetime(["2026-01-01"] * 5),
        "rgb_missing": [False, False, False, True, True],
        "stat_temperature_mean_missing": False,
        "online_pre_defrost_heat_uses_measurement_reconstruction": False,
    })

    result = temporal_triplet_indices(frame)

    assert len(result) == 1
    np.testing.assert_array_equal(result[0], [[0, 1, 2]])

    assert temporal_triplet_indices(frame, seconds=60) == []


def test_latent_curvature_is_zero_for_linear_path_and_has_gradient_for_reversal():
    from image_models.outcome_representation import latent_curvature_loss

    linear = torch.tensor([[0., 0.], [1., 2.], [2., 4.]], requires_grad=True)
    reversal = torch.tensor([[0., 0.], [1., 2.], [0., 0.]], requires_grad=True)
    triplets = [np.array([[0, 1, 2]])]

    assert latent_curvature_loss(linear, triplets).item() == 0
    loss = latent_curvature_loss(reversal, triplets)
    assert loss.item() > 0
    loss.backward()
    assert reversal.grad is not None


def test_scheduled_curvature_encodes_only_selected_triplet_rows():
    from image_models.outcome_representation import (
        latent_curvature_loss,
        scheduled_latent_curvature_loss,
    )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def encode(self, values):
            self.seen += len(values)
            return values

    values = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    selected = [np.array([[1, 2, 3], [5, 6, 7]])]
    encoder = Encoder()

    actual = scheduled_latent_curvature_loss(encoder, values, selected)
    expected = latent_curvature_loss(values, selected)

    assert encoder.seen == 6
    assert actual.item() == expected.item()


def test_latent_scale_weights_cycles_equally_and_reports_constant_dimensions():
    from image_models.outcome_representation import fit_latent_scale

    latents = pd.DataFrame({
        "z_00": [0., 0., 0., 10.],
        "z_01": [3., 3., 3., 3.],
    })
    scale = fit_latent_scale(latents, pd.Series(["long", "long", "long", "short"]))

    np.testing.assert_allclose(scale["mean"], [5., 3.])
    np.testing.assert_allclose(scale["scale"], [5., 0.])
    assert scale["zero_variance_dimensions"] == 1


def test_local_pathway_time_only_change_has_zero_interaction():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from image_models.outcome_representation import (
        OutcomeRepresentation,
        local_pathway_sensitivity,
    )

    frame = pd.DataFrame({
        "row_id": ["a:0", "a:1", "a:2"],
        "cycle_name": "a",
        "experiment_id": "experiment-a",
        "candidate_defrost_time": pd.date_range(
            "2026-01-01", periods=3, freq="10s"
        ),
        "is_teacher_candidate": True,
        "minutes_since_heating_start": [1., 2., 3.],
        "stat_temperature_mean": 4.,
        "online_pre_defrost_heat_kwh": 2.,
        "rgb_missing": False,
    })
    columns = [
        "stat_temperature_mean", "online_pre_defrost_heat_kwh", "rgb_missing"
    ]
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    ).fit(frame[columns])
    target_scaler = StandardScaler().fit(np.arange(12, dtype=float).reshape(3, 4))
    model = OutcomeRepresentation(
        1, 2, use_rgb=False, event_head="multimodal_time_linear"
    ).eval()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "feature_columns": columns,
        "sensor_width": 1,
        "accounting_width": 2,
        "use_rgb": False,
        "event_head": "multimodal_time_linear",
        "time_min_minutes": 1.,
        "time_scale_minutes": 3.,
    }

    result = local_pathway_sensitivity(frame, checkpoint, (10,))

    interaction = result.loc[result.pathway.eq("interaction"), "standardized_rms"]
    assert len(interaction) == 4
    np.testing.assert_allclose(interaction, 0., atol=1e-7)
    assert set(result.pathway) == {
        "rgb", "sensor", "ledger", "quality", "time", "actual", "interaction"
    }


def test_historical_time_linear_checkpoint_defaults_to_32_latent_dimensions():
    from image_models.outcome_representation import OutcomeRepresentation, _model

    expected = OutcomeRepresentation(
        1, 0, use_rgb=False, event_head="multimodal_time_linear"
    )
    checkpoint = {
        "model_state_dict": expected.state_dict(),
        "sensor_width": 1,
        "accounting_width": 0,
        "use_rgb": False,
        "event_head": "multimodal_time_linear",
    }

    assert _model(checkpoint).latent_width == 32


def test_trajectory_roughness_is_zero_for_linear_paths_and_detects_reversal():
    from image_models.outcome_representation import TARGETS, trajectory_roughness

    rows = pd.DataFrame({
        "row_id": ["a:0", "a:1", "a:2"],
        "cycle_name": "a",
        "experiment_id": "experiment-a",
        "candidate_defrost_time": pd.date_range(
            "2026-01-01", periods=3, freq="10s"
        ),
        "is_teacher_candidate": True,
        "sensor_timestamp": pd.Timestamp("2026-01-01"),
        "rgb_missing": False,
    })
    latents = pd.DataFrame({
        "row_id": rows.row_id,
        "z_00": [0., 1., 2.],
        "z_01": [0., 2., 4.],
    })
    predictions = pd.DataFrame({"row_id": rows.row_id})
    for index, target in enumerate(TARGETS):
        predictions[f"predicted_{target}"] = [0., 1., 0.] if index == 0 else 0.
        rows[f"predicted_{target}"] = 999.
    scale = {"mean": [0., 0.], "scale": [1., 2.]}

    result = trajectory_roughness(
        rows, latents, predictions, scale, np.ones(4), (10, 60)
    )

    assert result.delta_seconds.tolist() == [10]
    assert result.triplets.tolist() == [1]
    assert result.latent_second_difference_rms_median.iloc[0] == 0
    assert result.outcome_second_difference_rms_median.iloc[0] > 0


def test_group_balanced_distance_does_not_favor_wider_feature_groups():
    from image_models.outcome_representation import group_balanced_squared_distance

    query = np.zeros(4)
    reference = np.array([
        [1., 1., 1., 0.],
        [0., 0., 0., 1.],
    ])

    distance = group_balanced_squared_distance(query, reference, [[0, 1, 2], [3]])

    np.testing.assert_allclose(distance, [0.5, 0.5])
