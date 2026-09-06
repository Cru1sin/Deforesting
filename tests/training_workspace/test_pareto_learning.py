import numpy as np
import pandas as pd
import torch

from image_models.pareto_learning import (
    ParetoBoundaryModel,
    predict_pareto_rows,
    relation_pairs,
    train_pareto_fold,
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
