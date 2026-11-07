import pandas as pd
import pytest


def test_loss_budget_keeps_observable_actions_without_terminal_reward(tmp_path, monkeypatch):
    import json
    import pickle

    import numpy as np

    from image_models import policy
    from image_models.stopping_diagnostics import admission_diagnostics

    run = tmp_path / "run"
    run.mkdir()
    dataset = tmp_path / "dataset"
    (dataset / "training_data").mkdir(parents=True)
    (run / "settings.json").write_text(json.dumps(dict(dataset=str(dataset))))
    pd.DataFrame(dict(cycle_name=["a", "b"], water_temperature_setpoint=[50.0, 60.0])).to_parquet(
        dataset / "training_data/reference.parquet"
    )
    pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            version="model",
            seed=0,
            selected_cop=[2.0, 2.0],
            optimal_cop=[3.0, 3.0],
            rb_cop=[1.0, 1.0],
            terminal_cop=[np.nan, 2.8],
            fallback_terminal=False,
            input_missing=False,
            evaluable=True,
            regret_pct=100 / 3,
        )
    ).to_csv(run / "cycle_metrics.csv", index=False)
    predictions = pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            split="test",
            input_valid=True,
            cycle_cop_eligible=True,
            cycle_cop=[2.9, 2.7],
            candidate_defrost_time=pd.Timestamp("2026-01-01"),
        )
    )
    with (run / "model.pkl").open("wb") as stream:
        pickle.dump(dict(predictions=predictions), stream)
    monkeypatch.setattr(
        policy,
        "summarize_metrics",
        lambda rows: dict(
            headroom_captured_pct=100
            * (rows.selected_cop.sum() - rows.rb_cop.sum())
            / (rows.optimal_cop.sum() - rows.rb_cop.sum())
        ),
    )
    output = tmp_path / "diagnostics"
    admission_diagnostics(run, output)
    budget = pd.read_csv(output / "cycle_loss_budget.csv")
    np.testing.assert_allclose(budget.observable_oracle_cop, [2.9, 2.8])
    summary = pd.read_csv(output / "loss_budget_summary.csv")
    assert summary.evaluable_count.item() == 2
    assert summary.observable_oracle_H_RB_pct.item() == pytest.approx(92.5)


@pytest.mark.parametrize("architecture", ["GRU32", "GRU32_film"])
def test_visual_probe_keeps_controls_and_original_history(architecture):
    import numpy as np
    import torch

    from image_models.policy import fit_policy, predict_policy
    from image_models.stopping_diagnostics import visual_state_probes

    torch.set_num_threads(1)
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            experiment_id="e",
            input_valid=True,
            candidate_defrost_time=pd.date_range("2026", periods=6, freq="30s"),
            near_optimal=[False, False, False, True, True, False],
            control=np.arange(6),
            visual=np.arange(6) ** 2,
        )
    )
    c = fit_policy(
        rows,
        None,
        architecture=architecture,
        loss_name="set_stop",
        numeric_columns=["control"],
        visual_columns=["visual"],
        seed=0,
        maximum_epochs=2,
        patience=3,
        cycles_per_batch=1,
    )
    original = predict_policy(rows, c)
    probes = visual_state_probes(rows, c, offsets=(0, -30, 30))
    self_probe = probes.loc[probes.offset_seconds.eq(0)]
    np.testing.assert_allclose(self_probe.probe_margin, original.decision_margin, atol=1e-6)
    for offset, index, donor in [(-30, 3, 2), (30, 3, 4)]:
        prefix = rows.iloc[: index + 1].copy()
        prefix.loc[index, "visual"] = rows.visual.iloc[donor]
        expected = predict_policy(prefix, c).decision_margin.iloc[-1]
        actual = probes.loc[
            probes.offset_seconds.eq(offset)
            & probes.candidate_defrost_time.eq(rows.candidate_defrost_time.iloc[index]),
            "probe_margin",
        ].item()
        assert actual == pytest.approx(expected, abs=1e-6)
    frost = rows.assign(visual_frost_score=[0.02, 0.15, 0.25, 0.55, 0.85, 0.95])
    bands = visual_state_probes(frost, c, frost_bins=True)
    for lower, donor in [(0.0, 0), (0.9, 5)]:
        prefix = rows.iloc[:4].copy()
        prefix.loc[3, "visual"] = rows.visual.iloc[donor]
        expected = predict_policy(prefix, c).decision_margin.iloc[-1]
        actual = bands.loc[
            bands.frost_bin_lower.eq(lower)
            & bands.candidate_defrost_time.eq(rows.candidate_defrost_time.iloc[3]),
            "probe_margin",
        ].item()
        assert actual == pytest.approx(expected, abs=1e-6)


def test_unified_timing_uses_supported_near_end_and_keeps_missing_hits():
    import numpy as np

    from image_models.policy import policy_timing_metrics, summarize_metrics

    t = pd.Timestamp("2026-01-01")
    reference = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=[t + pd.Timedelta(minutes=m) for m in [10, 20, 30, 40] * 2],
            cycle_cop=[2.0, 1.99, 1.5, 3.0, 2.0, 1.99, 1.5, 3.0],
            cycle_cop_eligible=[True, True, True, False] * 2,
            t_RB=t + pd.Timedelta(minutes=10),
            cycle_start=t,
            heating_accounting_start=t,
        )
    )
    rows = pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            selected_time=[t + pd.Timedelta(minutes=25), pd.NaT],
            selected_cop=[1.99, np.nan],
            rb_cop=2.0,
            optimal_cop=2.0,
            early_trigger=False,
            soft_hard_gap=np.nan,
        )
    )
    result = policy_timing_metrics(rows, reference)
    assert result.delay_space_minutes.tolist() == [10.0, 10.0]
    assert result.delay_space_used_pct.iloc[0] == 150.0
    summary = summarize_metrics(result)
    assert summary["hit_rate_all_pct"] == 50.0
    assert summary["hit_rate_evaluable_pct"] == 100.0
    assert summary["delay_space_used_pct"] == 150.0
    assert summary["timing_missing_cycles"] == 1


def test_rescore_keeps_saved_time_and_does_not_fill_invalid_reward(tmp_path, monkeypatch):
    from dataset_tools import DatasetLoader
    from image_models import comparisons as labels
    from image_models.stopping_diagnostics import rescore_decisions

    t = pd.Timestamp("2026-01-01")
    times = pd.date_range(t, periods=3, freq="30s")
    reference = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            cycle_cop=[2.0, 9.0, 3.0],
            cycle_cop_eligible=[True, False, True],
            t_RB=t,
            cycle_start=t,
            heating_accounting_start=t,
        )
    )
    rows = pd.DataFrame(dict(cycle_name=["a"], optimal_cop=[3.0], optimal_time=[times[-1]]))
    monkeypatch.setattr(DatasetLoader, "__init__", lambda self, path: None)
    monkeypatch.setattr(DatasetLoader, "load_training_data", lambda self: (rows, reference))
    monkeypatch.setattr(DatasetLoader, "training_settings", property(lambda self: {}))
    monkeypatch.setattr(labels, "write_evaluation_metrics", lambda *args: None)
    old = pd.DataFrame(
        dict(
            cycle_name=["a"],
            experiment_id=["e"],
            selected_time=[times[1]],
            selected_cop=[100.0],
            optimal_cop=[100.0],
            version=["v"],
            seed=[0],
            split=["test"],
        )
    )
    old.to_csv(tmp_path / "cycle_metrics.csv", index=False)
    rescore_decisions(tmp_path, tmp_path / "new", tmp_path)
    new = pd.read_csv(tmp_path / "new/cycle_metrics.csv", parse_dates=["selected_time"])
    assert new.selected_time.iloc[0] == times[1]
    assert pd.isna(new.selected_cop.iloc[0])
    assert new.optimal_cop.iloc[0] == 3.0 and new.rb_cop.iloc[0] == 2.0


def test_candidate_selection_uses_validation_not_better_test_result(tmp_path, monkeypatch):
    import json
    import pickle

    from image_models import comparisons as labels
    from image_models.stopping_diagnostics import select_trigger_rule

    t = pd.Timestamp("2026-01-01")
    parts = []
    for split, values in [("validation", [1.0, 2.0]), ("test", [2.0, 1.0])]:
        parts.append(
            pd.DataFrame(
                dict(
                    cycle_name=split,
                    experiment_id=split,
                    split=split,
                    candidate_defrost_time=[t, t + pd.Timedelta(seconds=30)],
                    cycle_cop=values,
                    cycle_cop_eligible=True,
                    optimal_cop=2.0,
                    optimal_time=t,
                    terminal_time=t + pd.Timedelta(seconds=30),
                    near_optimal=[v == 2 for v in values],
                    input_valid=True,
                    t_RB=t,
                    cycle_start=t,
                    heating_accounting_start=t,
                )
            )
        )
    reference = pd.concat(parts, ignore_index=True)
    data = tmp_path / "dataset/training_data"
    data.mkdir(parents=True)
    reference.to_parquet(data / "reference.parquet")
    monkeypatch.setattr(labels, "write_evaluation_metrics", lambda *args: None)
    runs = []
    for version, probabilities in [("A", [0.9, 0.1]), ("B", [0.1, 0.9])]:
        run = tmp_path / version
        run.mkdir()
        runs.append(run)
        (run / "settings.json").write_text(
            json.dumps(dict(dataset=str(data.parent), inputs=[version], seeds=[0]))
        )
        predictions = reference.assign(probability=probabilities * 2)
        saved = dict(
            checkpoint=dict(
                seed=0,
                training_experiments=["validation"],
                selection_training_experiments=["train"],
            ),
            predictions=predictions,
            validation_predictions=predictions.loc[predictions.split.eq("validation")],
            cycle_metrics=pd.DataFrame(dict(outer_experiment=["test"], version=[version])),
        )
        with (run / "model_test_seed0.pkl").open("wb") as stream:
            pickle.dump(saved, stream)
    select_trigger_rule(
        runs[0], runs[0], tmp_path / "selected", rules=("hazard",), candidate_runs=[runs[1]]
    )
    result = pd.read_csv(tmp_path / "selected/cycle_metrics.csv")
    assert result.chosen_model.iloc[0] == "B"
    assert result.selected_cop.iloc[0] == 1.0  # Worse test COP must not reverse inner selection.

    # Same recipe name across optimizer runs must still select exactly one source.
    for index, run in enumerate(runs):
        path = run / "model_test_seed0.pkl"
        saved = pickle.load(path.open("rb"))
        saved["cycle_metrics"]["version"] = "same_recipe"
        saved["checkpoint"]["training_experiments"] = ["validation", "train"]
        saved["checkpoint"]["inner_validation_folds"] = [["validation"], ["train"]]
        saved["checkpoint"]["pooled_validation_scores"] = pd.DataFrame(
            dict(epoch=[1], rule=["cdf"], validation_evaluable=[1], validation_cop=[1.0 + index])
        )
        with path.open("wb") as stream:
            pickle.dump(saved, stream)
    select_trigger_rule(
        runs[0], runs[0], tmp_path / "pooled_selected", rules=("cdf",), candidate_runs=[runs[1]]
    )
    result = pd.read_csv(tmp_path / "pooled_selected/cycle_metrics.csv")
    choices = pd.read_csv(tmp_path / "pooled_selected/rule_selection.csv")
    assert result.chosen_source.iloc[0] == str(runs[1])
    assert result.selected_cop.iloc[0] == 1.0
    assert choices.selected.sum() == 1
