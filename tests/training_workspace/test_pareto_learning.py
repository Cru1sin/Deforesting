import numpy as np
import pandas as pd
import torch

from image_models.pareto_learning import (
    PROBE_RECIPES,
    STATE_TRANSFER_RECIPES,
    ParetoBoundaryModel,
    StopHead,
    predict_pareto_rows,
    predict_stop_rows,
    probe_feature_columns,
    relation_pairs,
    signed_relation_loss,
    stop_feature_columns,
    train_pareto_fold,
    train_stop_fixed,
    train_stop_fold,
)


def samples(experiment):
    rows = pd.DataFrame({
        "row_id": np.arange(6), "experiment_id": experiment, "cycle_name": experiment,
        "image_time": pd.date_range("2026-01-01", periods=6, freq="min"),
        "is_frame": [True, True, False, True, True, True],
        "is_knee": [False, False, True, False, False, False],
        "target": [0, 0, .5, 1, 1, 1],
        "pareto_selection_score": [.1, .8, 1, .9, .3, np.nan],
        "relation_branch": "chord", "relation_support_run": "0",
        "stat_temperature_mean": np.arange(6),
        "pre_defrost_heat_kwh": np.arange(6) / 10,
        "pre_defrost_electricity_kwh": np.arange(6) / 20,
        "economic_c": [np.nan, 2, 3, 2.5, 2, 1], "economic_h": np.arange(6),
    })
    rgb = pd.DataFrame(
        np.random.default_rng(0).normal(size=(6, 384)),
        columns=[f"dinov2_{index:03d}" for index in range(384)],
    )
    return pd.concat([rows, rgb], axis=1)


def test_pairs_stay_on_same_strict_side_branch_and_support():
    rows = samples("a")
    assert relation_pairs(rows).tolist() == [[1, 0], [3, 4]]
    rows.loc[1, "relation_support_run"] = "1"
    rows.loc[4, "relation_branch"] = "fallback"
    assert relation_pairs(rows).shape == (0, 2)


def test_signed_relation_loss_rewards_smaller_margin_near_boundary():
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    pairs = torch.tensor([[0, 1], [2, 3]])
    ordered = signed_relation_loss(torch.tensor([-1.0, -3.0, 1.0, 3.0]), target, pairs)
    reversed_order = signed_relation_loss(
        torch.tensor([-3.0, -1.0, 3.0, 1.0]), target, pairs
    )
    assert ordered < reversed_order


def test_stop_head_uses_explicit_recipe_and_supervised_scaler_rows():
    base = samples("a")
    base[[f"z_{index:02d}" for index in range(2)]] = np.arange(12).reshape(6, 2)
    base["online_c_current"] = np.arange(6.0)
    base["online_h_current"] = np.arange(6.0)
    base["online_c_mean"] = np.arange(6.0)
    base["online_h_mean"] = np.arange(6.0)
    base["online_o_current"] = np.arange(6.0)
    assert stop_feature_columns(base, "s0") == ["z_00", "z_01"]
    assert "online_c_mean" in stop_feature_columns(base, "s2")
    assert "online_o_current" in stop_feature_columns(base, "s3")
    unsupervised = base.iloc[[0]].copy()
    unsupervised["row_id"] = 999
    unsupervised["is_frame"] = False
    unsupervised["is_knee"] = False
    unsupervised["pareto_selection_score"] = np.nan
    unsupervised["z_00"] = 1e9
    train = pd.concat([base, unsupervised], ignore_index=True)
    result = train_stop_fold(train, base, base, base, method="s0", maximum_epochs=2, patience=1)
    scaler = result["checkpoint"]["preprocessor"].named_steps["standardscaler"]
    assert scaler.mean_[0] == base.z_00.mean()
    assert "stat_temperature_mean" not in result["predictions"]
    assert "online_c_current" in result["predictions"]
    sparse = train_stop_fold(
        base, base, base, base, method="s4", base_method="s2",
        maximum_epochs=2, patience=1,
    )
    with_unused_grid = train_stop_fold(
        train, base, train, base, method="s4", base_method="s2",
        maximum_epochs=2, patience=1,
    )
    np.testing.assert_allclose(sparse["predictions"].logit, with_unused_grid["predictions"].logit)


def test_fixed_epoch_stop_head_reuses_reference_budget_and_reloads():
    train = samples("train")
    test = samples("test")
    for table in (train, test):
        table[[f"z_{index:02d}" for index in range(2)]] = np.arange(12).reshape(6, 2)
        for objective in ("c", "h"):
            table[f"online_{objective}_current"] = np.arange(6.0)
            table[f"online_{objective}_current_missing"] = False
            table[f"online_{objective}_valid_count"] = np.arange(1, 7)
            table[f"online_{objective}_age_seconds"] = 0.0
    result = train_stop_fixed(train, test, method="s1", epochs=3, seed=0)

    assert result["checkpoint"]["selected_epoch"] == 3
    assert result["losses"].epoch.max() == 3
    replay = predict_stop_rows(test, result["checkpoint"])
    np.testing.assert_allclose(result["predictions"].logit, replay.logit)
    ridge_changed = test.assign(economic_c=1e9, economic_h=-1e9)
    np.testing.assert_allclose(
        replay.logit, predict_stop_rows(ridge_changed, result["checkpoint"]).logit
    )
    latent_changed = test.copy()
    latent_changed["z_00"] += 100
    assert not np.allclose(
        replay.logit, predict_stop_rows(latent_changed, result["checkpoint"]).logit
    )


def test_fixed_epoch_stop_head_accepts_the_selected_transfer_columns():
    train = samples("train")
    test = samples("test")
    for table in (train, test):
        table["raw"] = np.arange(6.0)
    result = train_stop_fixed(
        train, test, method="raw_ridge_ch_mlp", epochs=2,
        feature_columns=["raw"], hidden_width=16,
    )

    assert result["checkpoint"]["feature_columns"] == ["raw"]
    assert result["checkpoint"]["hidden_width"] == 16


def test_probe_recipes_have_explicit_matched_inputs_and_head_capacity():
    rows = samples("a")
    rows[[f"z_{index:02d}" for index in range(32)]] = np.arange(6 * 32).reshape(6, 32)
    raw = [f"raw_{index:03d}" for index in range(782)]
    rows = pd.concat(
        [rows, pd.DataFrame(0.0, index=rows.index, columns=raw)], axis=1
    )
    quality = {}
    for objective in ("c", "h"):
        for suffix in ("current", "current_missing", "valid_count", "age_seconds"):
            quality[f"online_{objective}_{suffix}"] = 0.0
    rows = pd.concat([rows, pd.DataFrame(quality, index=rows.index)], axis=1)

    expected = {
        "latent_ridge_ch_linear": (40, None),
        "latent_ridge_ch_mlp": (40, 16),
        "latent_raw_ridge_ch_mlp": (822, 16),
        "raw_ridge_ch_mlp": (790, 16),
    }
    for method, (width, hidden_width) in expected.items():
        columns = probe_feature_columns(rows, method, raw)
        assert len(columns) == width
        assert PROBE_RECIPES[method].hidden_width == hidden_width
        assert len(columns) == len(set(columns))


def test_state_transfer_recipes_keep_only_d32_t32_and_raw():
    rows = samples("a")
    rows[[f"z_{index:02d}" for index in range(32)]] = 0.0
    raw = [f"raw_{index:03d}" for index in range(782)]
    rows = pd.concat([rows, pd.DataFrame(0.0, index=rows.index, columns=raw)], axis=1)
    for objective in ("c", "h"):
        for suffix in ("current", "current_missing", "valid_count", "age_seconds"):
            rows[f"online_{objective}_{suffix}"] = 0.0

    assert set(STATE_TRANSFER_RECIPES) == {
        "d32_ridge_ch_mlp", "t32_ridge_ch_mlp", "raw_ridge_ch_mlp"
    }
    assert {
        method: len(probe_feature_columns(rows, method, raw))
        for method in STATE_TRANSFER_RECIPES
    } == {
        "d32_ridge_ch_mlp": 40,
        "t32_ridge_ch_mlp": 40,
        "raw_ridge_ch_mlp": 790,
    }


def test_probe_mlp_checkpoint_reloads_the_same_predictions():
    train = samples("train")
    test = samples("test")
    for table in (train, test):
        table[[f"z_{index:02d}" for index in range(2)]] = np.arange(12).reshape(6, 2)
        for objective in ("c", "h"):
            for suffix in ("current", "current_missing", "valid_count", "age_seconds"):
                table[f"online_{objective}_{suffix}"] = 0.0
    columns = probe_feature_columns(train, "latent_ridge_ch_mlp", [])
    result = train_stop_fold(
        train, test, train, test, method="latent_ridge_ch_mlp",
        feature_columns=columns, hidden_width=16, maximum_epochs=2, patience=1,
    )

    assert isinstance(StopHead(len(columns), hidden_width=16).output, torch.nn.Sequential)
    assert result["checkpoint"]["hidden_width"] == 16
    replay = predict_stop_rows(test, result["checkpoint"])
    np.testing.assert_allclose(result["predictions"].logit, replay.logit)


def test_economic_mask_and_nonvisual_are_exact():
    torch.manual_seed(0)
    model = ParetoBoundaryModel(1, use_economic_context=False).eval()
    x = torch.randn(3, 389)
    changed = x.clone()
    changed[:, 3:5] += 100
    assert torch.equal(model(x), model(changed))
    model.use_economic_context = True
    assert not torch.equal(model(x), model(changed))
    model.nonvisual = True
    changed = x.clone()
    changed[:, 5:] += 100
    assert torch.equal(model(x), model(changed))


def test_fold_smoke_reloads_and_never_fits_test_scaler():
    torch.set_num_threads(1)
    a, b, c, d = [samples(name) for name in "abcd"]
    d["stat_temperature_mean"] += 1000
    result = train_pareto_fold(
        pd.concat([a, b], ignore_index=True), c,
        pd.concat([a, b, c], ignore_index=True), d,
        use_economic_context=True, use_pareto_relation=True,
        maximum_epochs=3, patience=2,
    )
    predictions = result["predictions"]
    assert len(predictions) == 5
    assert np.isfinite(predictions["logit"]).all()
    scaler = result["checkpoint"]["preprocessor"].named_steps["standardscaler"]
    assert scaler.mean_[0] == 2.5
    replay = predict_pareto_rows(d, result["checkpoint"])
    np.testing.assert_allclose(predictions["logit"], replay["logit"])
    unlabelled = predict_pareto_rows(d.assign(target=np.nan), result["checkpoint"])
    np.testing.assert_allclose(predictions["logit"], unlabelled["logit"])
    assert set(result["losses"]["split"]) == {"train_side", "val_side", "train_rank"}
    assert set(result["pair_metrics"]["split"]) == {
        "inner_train", "inner_validation", "outer_train", "outer_test"
    }
