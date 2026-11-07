"""The clean model settings must preserve frozen model behavior."""

import numpy as np
import pandas as pd
import pytest
import torch

from image_models.policy import fit_policy, predict_policy


@pytest.mark.parametrize(
    "old,new,film",
    [
        ("LSTM32_film", "LSTM", True),
        ("GRU32_film", "GRU", True),
        ("MLP_plain", "MLP", False),
    ],
)
def test_explicit_model_settings_preserve_training_and_frozen_predictions(old, new, film):
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=list(pd.date_range("2026-01-01", periods=4, freq="30s")) * 2,
            x=np.arange(8.0),
            visual=np.sin(np.arange(8.0)),
            input_valid=True,
            stopping_cdf_target=[0.0, 0.0, 0.5, 1.0] * 2,
            stopping_cdf_weight=1.0,
        )
    )
    options = dict(
        loss_name="cdf_ramp",
        numeric_columns=["x"],
        visual_columns=["visual"],
        seed=42,
        maximum_epochs=2,
        patience=3,
        cycles_per_batch=1,
        hidden_width=25,
        validation_metric="bce",
        trigger_rule="cdf",
        dropout_config=dict(
            visual_dropout=0.2, output_dropout=0.2, visual_time_shared=True, output_time_shared=True
        ),
    )
    frozen = fit_policy(rows, rows, architecture=old, **options)
    explicit = fit_policy(rows, rows, architecture=new, film=film, **options)
    pd.testing.assert_frame_equal(frozen["losses"], explicit["losses"], check_exact=True)
    for key, value in frozen["model_state_dict"].items():
        torch.testing.assert_close(value, explicit["model_state_dict"][key], rtol=0, atol=0)
    pd.testing.assert_frame_equal(
        predict_policy(rows, frozen), predict_policy(rows, explicit), check_exact=True
    )


def test_main_training_does_not_accept_overlapping_experiments(tmp_path):
    from image_models.training import fit_model

    rows = pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            experiment_id=["same", "same"],
            split=["train", "test"],
            input_valid=True,
            training_eligible=True,
        )
    )
    with pytest.raises(ValueError, match="experiments overlap"):
        fit_model(rows, pd.DataFrame(), {}, epochs=1, seed=42, output=tmp_path, name="LSTM")
