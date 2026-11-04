"""历史 LOEO 冻结基准的匹配消融与结果汇总。

--baseline 必须是 image_models.optimal_stopping 生成的按实验留出训练结果。
main_defrosting_model 的固定训练/后期测试结果不作为 LOEO 基准。
新网络与输入变体使用共享训练接口；不得把两种实验设计混作同一对照。
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from image_models.checkpoints import checkpoint_files
from image_models.comparisons import (
    compensation_comparisons,
    write_ablation_tables,
    write_evaluation_metrics,
)
from image_models.inputs import FREQUENCY, TEMPERATURE
from output_layout import PROJECT, analysis_output, experiment_root

NUMERIC = [*FREQUENCY, *TEMPERATURE, "elapsed_cycle_minutes", "frequency_used_margin_fraction"]


def ablation_recipes(architecture="LSTM32_film"):
    """Retained LSTM component comparisons; all other inputs stay fixed."""
    if not architecture.startswith("LSTM"):
        raise ValueError("Component recipes require the retained LSTM baseline")
    return [
        ("LSTM组件", label, overrides)
        for label, overrides in [
            ("无视觉", ["--visual-pooling", "zero"]),
            ("仅视觉", ["--drop-inputs", *NUMERIC]),
            ("无FiLM", ["--architecture", "LSTM_plain"]),
            ("无LSTM历史", ["--reset-history"]),
        ]
    ]


def read_metrics(run, version, seeds=None):
    frames = []
    for path in checkpoint_files(run):
        saved = pickle.load(path.open("rb"))
        if seeds is not None and saved["checkpoint"]["seed"] not in seeds:
            continue
        frames.append(
            saved["cycle_metrics"].loc[lambda x: x.split.eq("test")].assign(version=version)
        )
    return pd.concat(frames, ignore_index=True)


def stopping_evidence_figure(
    baseline, output, seed_study, component_study, width_search_runs=(), memory_result_study=None
):
    """Four evidence panels from frozen predictions; component probes retain their model identity."""
    from image_models.cycle_selection import screen_published_cycles
    from image_models.policy import apply_trigger_rule
    from plots.model_comparison import render_stopping_evidence

    settings = json.loads((baseline / "training/settings.json").read_text())
    screening = screen_published_cycles(settings["dataset"])
    screening.to_csv(output / "cycle_screening.csv", index=False)
    cohort = set(screening.loc[screening.selected_train, "cycle_name"])
    study = analysis_output(seed_study)
    configurations = pd.read_csv(study / "replication_configurations.csv").set_index("model")
    summary = pd.read_csv(study / "evaluation_summary.csv")
    models = ["MLP64x64_FiLM", "GRU32_FiLM", "LSTM25_FiLM"]
    summary = summary.loc[
        summary.version.isin(models) & summary.subset.eq("all") & summary.setpoint_c.eq("all")
    ]
    histories = []
    for model in models:
        run = Path(configurations.loc[model, "experiment"])
        config = json.loads((run / "training/settings.json").read_text())
        assert all(
            config[k] == settings[k]
            for k in [
                "data_definition",
                "folds",
                "seeds",
                "maximum_epochs",
                "validation_rule",
                "validation_utility",
                "validation_tiebreak",
                "trigger_rule",
                "threshold",
            ]
        )
        metrics = read_metrics(run, model)
        assert all(set(part.cycle_name) == cohort for _, part in metrics.groupby("seed"))
        scores = pd.read_csv(run / "training/ablation_summary.csv")
        summary.loc[summary.version.eq(model), "parameters"] = int(
            scores.loc[scores.scope.eq("outer_fold"), "parameters"].iloc[0]
        )
        history = pd.read_csv(run / "training/ablation_learning_curves.csv")
        histories.append(history.loc[history.scope.eq("inner_training_curve")].assign(model=model))
    summary.to_csv(output / "panel_b_models.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "panel_d_learning.csv", index=False)
    metrics = read_metrics(baseline, "LSTM25_FiLM")
    metrics.to_csv(output / "cycle_results.csv", index=False)
    means = metrics.groupby("cycle_name")[["selected_cop", "rb_cop", "regret_pct"]].mean()
    means["delta_cop"] = means.selected_cop - means.rb_cop
    best = means.sort_values(["delta_cop"], ascending=False, kind="stable").index[0]
    worst = (
        means.drop(index=best).sort_values("regret_pct", ascending=False, kind="stable").index[0]
    )
    pd.DataFrame(
        [
            dict(cycle_name=best, role="Largest COP gain over RB"),
            dict(cycle_name=worst, role="Largest remaining COP regret"),
        ]
    ).to_csv(output / "case_selection.csv", index=False)
    traces = []
    for path in checkpoint_files(baseline):
        saved = pickle.load(path.open("rb"))
        rows = saved["predictions"].loc[lambda x: x.split.eq("test")].copy()
        rows = apply_trigger_rule(rows, settings["trigger_rule"])
        traces.append(
            rows.loc[rows.cycle_name.isin([best, worst])].assign(seed=saved["checkpoint"]["seed"])
        )
    pd.concat(traces, ignore_index=True).to_csv(output / "panel_a_predictions.csv", index=False)
    study_path = analysis_output(component_study)
    quick_path = study_path / "LSTM快速组件消融.csv"
    probes = []
    labels = {
        "无视觉": "Remove vision",
        "仅视觉": "Remove controls",
        "无FiLM": "Remove FiLM",
        "无LSTM历史": "Reset LSTM memory",
        "无CDF": "Remove CDF",
        "无CDF且无LSTM历史": "Remove CDF + reset memory",
    }
    if quick_path.exists():
        components = pd.read_csv(quick_path)
        reference = components.loc[components.variant.eq("完整LSTM25")].iloc[0]
        for row in components.loc[~components.variant.eq("完整LSTM25")].itertuples():
            probes.append(
                dict(
                    label=labels[row.variant],
                    delta_cop=row.validation_cop - reference.validation_cop,
                    delta_H_RB=row.validation_headroom - reference.validation_headroom,
                    lower=np.nan,
                    upper=np.nan,
                    cycles=int(row.validation_cycles),
                    reference="LSTM25 + FiLM",
                    seed=int(row.seed),
                    source_run=row.source_run,
                    evaluation_scope="training_side_quick",
                    epochs=int(row.epoch),
                    H_RB=row.validation_headroom,
                    H_RB_std=np.nan,
                )
            )
    else:
        components = pd.read_csv(study_path / "LSTM组件消融/LSTM组件消融.csv")
        components = components.loc[
            components.subset.eq("all") & components.setpoint_c.eq("all") & ~components.baseline
        ]
        for row in components.sort_values("variant_order").itertuples():
            probes.append(
                dict(
                    label=labels[row.version],
                    delta_cop=row.experiment_equal_delta_cop,
                    lower=row.bootstrap_lower_95,
                    upper=row.bootstrap_upper_95,
                    cycles=int(row.paired_cycles_x),
                    reference="LSTM25 + FiLM",
                    seed=int(row.seed),
                    source_run=row.source_run,
                    evaluation_scope="full_experiment_loeo",
                    H_RB=row.headroom_captured_pct,
                    H_RB_std=np.nan,
                    delta_H_RB=row.right_headroom_captured_pct - row.left_headroom_captured_pct,
                )
            )
    if len(probes) not in (4, 6) or any(row["seed"] != 42 for row in probes):
        raise ValueError("Fig7 requires complete matched seed-42 LSTM component tests")
    if probes:
        formal = summary.loc[summary.version.eq("LSTM25_FiLM"), "headroom_captured_pct"]
        probes.insert(
            0,
            dict(
                label="Full LSTM25 + FiLM",
                H_RB=formal.mean(),
                H_RB_std=formal.std(),
                cycles=len(cohort),
                reference="LSTM25 + FiLM",
                seed=np.nan,
                source_run=str(baseline),
                evaluation_scope="full_experiment_loeo",
                epochs=np.nan,
                delta_cop=np.nan,
                delta_H_RB=np.nan,
                lower=np.nan,
                upper=np.nan,
            ),
        )
    if memory_result_study is not None:
        memory_path = analysis_output(memory_result_study)
        memory_summary = pd.read_csv(memory_path / "evaluation_summary.csv")
        reset = memory_summary.loc[
            memory_summary.version.eq("C_清零记忆_完整输入")
            & memory_summary.subset.eq("all")
            & memory_summary.setpoint_c.eq("all")
        ]
        assert set(reset.seed) == {42, 2026, 1010} and reset.cohort_cycles.eq(len(cohort)).all()
        plan = pd.read_csv(memory_path / "experiment_plan.csv").set_index("variant")
        for row in probes:
            if row["label"] == "Reset LSTM memory":
                row.update(
                    H_RB=reset.headroom_captured_pct.mean(),
                    H_RB_std=reset.headroom_captured_pct.std(),
                    cycles=len(cohort),
                    seed=np.nan,
                    epochs=np.nan,
                    evaluation_scope="full_experiment_loeo",
                    source_run=plan.loc["C_清零记忆_完整输入", "run"],
                )
    pd.DataFrame(probes).to_csv(output / "panel_c_components.csv", index=False)
    if width_search_runs:
        widths = pd.concat(
            [
                pd.read_csv(experiment_root(run) / "training/shared_lstm_grid.csv").assign(
                    source_run=str(run)
                )
                for run in width_search_runs
            ],
            ignore_index=True,
        )
        widths = widths.loc[
            widths.hidden_width.isin([16, 25, 32])
            & widths.learning_rate.eq(0.003)
            & widths.weight_decay.eq(0.0001)
            & widths.visual_dropout.eq(0.1)
            & widths.output_dropout.eq(0.2)
            & widths.recurrent_layers.eq(1)
        ].drop_duplicates("hidden_width")
        if set(widths.hidden_width) != {16, 25, 32} or not widths.validation_cycles.eq(99).all():
            raise ValueError("Width check requires the matched 16/25/32 training-side screens")
        widths.sort_values("hidden_width").to_csv(output / "panel_b_width_screen.csv", index=False)
    render_stopping_evidence(output)


def fit_quick_component(args, rows, reference, test, folds, label, overrides):
    from copy import copy

    from image_models.inputs import RGB
    from image_models.model_selection import fit_candidate

    local = copy(args)
    local.output = args.output / "training" / label
    if "--drop-inputs" in overrides:
        local.drop_inputs = NUMERIC
    if "--architecture" in overrides:
        local.architecture = overrides[overrides.index("--architecture") + 1]
    local.reset_history = "--reset-history" in overrides
    values = rows.copy() if "--visual-pooling" in overrides else rows
    if "--visual-pooling" in overrides:
        values[RGB] = 0.0
    result = fit_candidate(
        local,
        values,
        reference,
        test,
        folds,
        25,
        1,
        args.learning_rate,
        weight_decay=args.weight_decay,
        visual_dropout=args.visual_dropout,
        output_dropout=args.output_dropout,
    )
    result.update(
        variant=label,
        seed=args.seeds[0],
        source_run=str(local.output),
        evaluation_scope="training_side_quick",
        reset_history=local.reset_history,
        inputs=";".join(c for c in NUMERIC if c not in local.drop_inputs),
    )
    return result


def memory_factorial_study(baseline, output, n_jobs):
    """Matched LOEO refits with a frozen training-side epoch schedule."""
    import torch

    from dataset_tools import DatasetLoader
    from image_models.cycle_selection import screen_published_cycles
    from image_models.policy import _model_values, restore_policy

    torch.set_num_threads(1)
    settings = json.loads((baseline / "training/settings.json").read_text())
    screening = screen_published_cycles(settings["dataset"])
    screening.to_csv(output / "cycle_screening.csv", index=False)
    cohort = set(screening.loc[screening.selected_train, "cycle_name"])
    rows, _ = DatasetLoader(settings["dataset"]).load_training_data(objective="cdf_ramp")
    from image_models.inputs import causal_compensation_progress, elapsed_cycle_minutes

    rows = causal_compensation_progress(rows)
    rows["elapsed_cycle_minutes"] = elapsed_cycle_minutes(rows)
    schedule = []
    audit = []
    for path in checkpoint_files(baseline):
        saved = pickle.load(path.open("rb"))
        checkpoint = saved["checkpoint"]
        test = saved["cycle_metrics"].outer_experiment.iloc[0]
        seed = checkpoint["seed"]
        assert test not in checkpoint["training_experiments"]
        assert checkpoint["architecture"] == "LSTM32_film" and not checkpoint["reset_history"]
        schedule.append(
            dict(
                outer_experiment=test,
                seed=seed,
                epoch=checkpoint["selected_epoch"],
                source=str(path),
            )
        )
        sample = rows.loc[rows.experiment_id.eq(test) & rows.input_valid].groupby(
            "cycle_name", sort=False
        )
        model = restore_policy(checkpoint)
        model.eval()
        for cycle, part in sample:
            part = part.sort_values("candidate_defrost_time").head(32)
            values = torch.as_tensor(
                _model_values(
                    part,
                    [*checkpoint["numeric_columns"], *checkpoint["visual_columns"]],
                    checkpoint["preprocessor"],
                )
            )[None]
            with torch.no_grad():
                whole = model(values)
                state = None
                states = []
                for step in range(values.shape[1]):
                    hidden, state = model.gru(values[:, step : step + 1], state)
                    states.append(hidden)
                streamed = model.readout(
                    torch.cat(states, dim=1), values[..., : model.numeric_width]
                ).squeeze(-1)
                error = float((whole - streamed).abs().max())
                assert error < 1e-5
                model.reset_history = True
                reset = model(values)
                model.reset_history = False
            audit.append(
                dict(
                    cycle_name=cycle,
                    outer_experiment=test,
                    seed=seed,
                    streaming_error=error,
                    hidden_norm=float(state[0].norm()),
                    cell_norm=float(state[1].norm()),
                    reset_margin_mean_absolute_change=float((whole - reset).abs().mean()),
                    actions=len(part),
                )
            )
    pd.DataFrame(audit).to_csv(output / "memory_state_audit.csv", index=False)
    schedule_path = output / "frozen_epoch_schedule.csv"
    pd.DataFrame(schedule).to_csv(schedule_path, index=False)
    history = [*FREQUENCY[1:], "frequency_used_margin_fraction"]
    variants = [
        ("B_递归记忆_无频率历史", ["--drop-inputs", *history]),
        ("C_清零记忆_完整输入", ["--reset-history"]),
        ("D_清零记忆_无频率历史", ["--reset-history", "--drop-inputs", *history]),
    ]
    plan = [
        dict(variant="A_递归记忆_完整输入", run=str(baseline), overrides="reused formal reference")
    ]
    frames = [read_metrics(baseline, plan[0]["variant"])]
    contrasts = [
        ("A_递归记忆_完整输入", "C_清零记忆_完整输入"),
        ("B_递归记忆_无频率历史", "D_清零记忆_无频率历史"),
        ("A_递归记忆_完整输入", "B_递归记忆_无频率历史"),
        ("C_清零记忆_完整输入", "D_清零记忆_无频率历史"),
    ]
    for label, overrides in variants:
        run = baseline.parent / (output.name + "_" + label)
        plan.append(dict(variant=label, run=str(run), overrides=" ".join(overrides)))
        pd.DataFrame(plan).to_csv(output / "experiment_plan.csv", index=False)
        command = [
            sys.executable,
            "-m",
            "image_models.optimal_stopping",
            "--dataset",
            settings["dataset"],
            "--recipe",
            "temperature",
            "--cohort",
            "training",
            "--loss-name",
            "cdf_ramp",
            "--architecture",
            "LSTM32_film",
            "--hidden-width",
            "25",
            "--elapsed-context",
            "--compensation-progress",
            "--inner-folds",
            "2",
            "--refit-all",
            "--joint-trigger-selection",
            "--trigger-rule",
            "cdf",
            "--validation-utility",
            "headroom",
            "--validation-tiebreak",
            "set_loss",
            "--validation-rule",
            settings["validation_rule"],
            "--maximum-epochs",
            str(settings["maximum_epochs"]),
            "--learning-rate",
            str(settings["learning_rate"]),
            "--weight-decay",
            str(settings["weight_decay"]),
            "--visual-dropout",
            str(settings["visual_dropout"]),
            "--output-dropout",
            str(settings["output_dropout"]),
            "--seeds",
            *map(str, settings["seeds"]),
            "--n-jobs",
            str(n_jobs),
            "--epoch-schedule",
            str(schedule_path),
            "--version",
            label,
            "--output",
            str(run),
            "--purpose",
            "Recursive memory × explicit frequency history; matched frozen training-side epochs; practical memory gain margin 2 H_RB pp",
            *overrides,
        ]
        if not (run / "evaluation/policy/evaluation_summary.csv").exists():
            log = run / "training/execution.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[memory study start] {label}", flush=True)
            with log.open("a") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
        metrics = read_metrics(run, label)
        assert len(checkpoint_files(run)) == len(schedule)
        for _, part in metrics.groupby("seed"):
            assert set(part.cycle_name) == cohort
        frames.append(metrics)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output / "evaluation_cycles.csv", index=False)
        write_evaluation_metrics(combined, settings["dataset"], output)
        completed = set(combined.version)
        compensation_comparisons(
            combined, output, [pair for pair in contrasts if set(pair) <= completed]
        )
        print(f"[memory study complete] {label}", flush=True)


def memory_study_diagnostics(baseline, output):
    """Paired policy effects and an inference intervention; never choose a new trigger."""
    from dataset_tools import DatasetLoader
    from image_models.evaluation import score_stopping_predictions
    from image_models.inputs import causal_compensation_progress, elapsed_cycle_minutes
    from image_models.policy import apply_trigger_rule, predict_policy

    settings = json.loads((baseline / "training/settings.json").read_text())
    metrics = pd.read_csv(output / "evaluation_cycles.csv")
    a, b, c, d = [
        "A_递归记忆_完整输入",
        "B_递归记忆_无频率历史",
        "C_清零记忆_完整输入",
        "D_清零记忆_无频率历史",
    ]
    assert set(metrics.version) == {a, b, c, d}
    compensation_comparisons(metrics, output, [(a, c), (b, d), (a, b), (c, d)])
    effects = pd.read_csv(output / "experiment_bootstrap.csv")
    memory = effects.loc[effects.left.isin([a, b]) & effects.right.isin([c, d])].copy()
    memory["memory_gain_H_pp"] = -memory.pooled_H_delta_pp
    memory["memory_gain_lower_95"] = -memory.pooled_H_upper_95
    memory["memory_gain_upper_95"] = -memory.pooled_H_lower_95
    memory["practical_margin_H_pp"] = 2.0
    memory["limited_gain_confirmed"] = memory.memory_gain_upper_95.lt(2.0)
    memory.to_csv(output / "memory_gain_conclusions.csv", index=False)
    rows, reference = DatasetLoader(settings["dataset"]).load_training_data(objective="cdf_ramp")
    rows = causal_compensation_progress(rows)
    rows["elapsed_cycle_minutes"] = elapsed_cycle_minutes(rows)
    frames = []
    accumulation = []
    for path in checkpoint_files(baseline):
        saved = pickle.load(path.open("rb"))
        checkpoint = saved["checkpoint"]
        test = saved["cycle_metrics"].outer_experiment.iloc[0]
        seed = checkpoint["seed"]
        heldout = (
            rows.loc[rows.experiment_id.eq(test) & rows.training_eligible]
            .copy()
            .assign(split="test")
        )
        available = heldout.input_valid
        for reset in [False, True]:
            model = dict(checkpoint, reset_history=reset)
            predicted = heldout.copy()
            predicted["probability"] = np.nan
            predicted["decision_margin"] = np.nan
            fitted = predict_policy(heldout.loc[available], model)
            predicted.loc[available, ["probability", "decision_margin"]] = fitted[
                ["probability", "decision_margin"]
            ].to_numpy()
            label = "A_inference_reset" if reset else a
            result = score_stopping_predictions(
                predicted, reference, label, seed, test, "first_positive_terminal", "cdf"
            )
            frames.append(result)
            if not reset:
                original = (
                    saved["cycle_metrics"].loc[lambda x: x.split.eq("test")].set_index("cycle_name")
                )
                matched = result.set_index("cycle_name").reindex(original.index)
                assert np.allclose(
                    original.selected_cop, matched.selected_cop, atol=1e-10, equal_nan=True
                )
                transformed = apply_trigger_rule(predicted, "cdf")
                for cycle, part in transformed.groupby("cycle_name"):
                    valid = part.input_valid
                    margins = part.decision_margin.where(valid, -np.inf).to_numpy()
                    risk = np.cumsum(np.logaddexp(0.0, margins))
                    cdf = -np.expm1(-risk)
                    error = np.max(np.abs(cdf[valid] - part.loc[valid, "probability"]))
                    assert error < 1e-5
                    first = np.flatnonzero(valid.to_numpy() & (risk >= np.log(2.0)))
                    accumulation.append(
                        dict(
                            cycle_name=cycle,
                            seed=seed,
                            cdf_identity_error=error,
                            cumulative_risk_at_trigger=risk[first[0]] if len(first) else np.nan,
                            instantaneous_hazard_at_trigger=float(
                                1 / (1 + np.exp(-margins[first[0]]))
                            )
                            if len(first)
                            else np.nan,
                        )
                    )
    directory = output / "inference_reset"
    directory.mkdir(exist_ok=True)
    intervention = pd.concat(frames, ignore_index=True)
    intervention.to_csv(directory / "evaluation_cycles.csv", index=False)
    write_evaluation_metrics(intervention, settings["dataset"], directory)
    compensation_comparisons(intervention, directory, [(a, "A_inference_reset")])
    pd.DataFrame(accumulation).to_csv(output / "cdf_accumulation_audit.csv", index=False)
    print(
        memory[
            [
                "left",
                "memory_gain_H_pp",
                "memory_gain_lower_95",
                "memory_gain_upper_95",
                "limited_gain_confirmed",
            ]
        ].to_string(index=False),
        flush=True,
    )


def cdf_memory_diagnostics(baseline, completed, output, seeds):
    from dataset_tools import DatasetLoader
    from image_models.evaluation import score_stopping_predictions

    runs = {label: run for _, label, run in completed}
    a, b, c, d = "Full_CDF", "Reset_CDF", "Full_noCDF", "Reset_noCDF"
    sources = {a: baseline, b: runs["无LSTM历史"], c: runs["无CDF"], d: runs["无CDF且无LSTM历史"]}
    directory = output / "CDF与记忆"
    directory.mkdir(exist_ok=True)
    table = pd.concat(
        [read_metrics(run, label, seeds) for label, run in sources.items()], ignore_index=True
    )
    table.to_csv(directory / "evaluation_cycles.csv", index=False)
    settings = json.loads((baseline / "training/settings.json").read_text())
    write_evaluation_metrics(table, settings["dataset"], directory)
    compensation_comparisons(table, directory, [(a, b), (c, d), (a, c), (b, d)])
    effects = pd.read_csv(directory / "paired_effects.csv")
    reset = effects.loc[
        ((effects.left == a) & (effects.right == b)) | ((effects.left == c) & (effects.right == d))
    ].copy()
    reset["CDF_training_and_trigger"] = reset.left.eq(a)
    reset.to_csv(directory / "memory_effects.csv", index=False)
    interaction = reset.pivot(
        index="seed", columns="CDF_training_and_trigger", values="headroom_capture_delta_pp"
    )
    interaction["CDF_change_in_reset_gain_pp"] = interaction[True] - interaction[False]
    interaction.to_csv(directory / "memory_CDF_interaction_point_estimate.csv")
    _, reference = DatasetLoader(settings["dataset"]).load_training_data(objective="cdf_ramp")
    replay = []
    for label, run in list(sources.items())[:2]:
        for path in checkpoint_files(run):
            saved = pickle.load(path.open("rb"))
            checkpoint = saved["checkpoint"]
            seed = checkpoint["seed"]
            if seed not in seeds:
                continue
            predicted = saved["predictions"].loc[lambda x: x.split.eq("test")].copy()
            test = saved["cycle_metrics"].outer_experiment.iloc[0]
            for rule in ("cdf", "hazard"):
                replay.append(
                    score_stopping_predictions(
                        predicted,
                        reference,
                        label + "_" + rule,
                        seed,
                        test,
                        "first_positive_terminal",
                        rule,
                    )
                )
    directory = directory / "same_weights_trigger_replay"
    directory.mkdir(exist_ok=True)
    replay = pd.concat(replay, ignore_index=True)
    replay.to_csv(directory / "evaluation_cycles.csv", index=False)
    write_evaluation_metrics(replay, settings["dataset"], directory)
    compensation_comparisons(
        replay, directory, [(a + "_cdf", b + "_cdf"), (a + "_hazard", b + "_hazard")]
    )


def quick_lstm_components(baseline, output, epochs, n_jobs, seed):
    from types import SimpleNamespace

    from joblib import Parallel, delayed, parallel_config

    from dataset_tools import DatasetLoader
    from image_models.inputs import prepare_ablation_inputs as prepare
    from output_layout import training_output

    settings = json.loads((baseline / "training/settings.json").read_text())
    args = SimpleNamespace(
        dataset=Path(settings["dataset"]),
        recipe="temperature",
        loss_name="cdf_ramp",
        architecture=settings["recipes"][0][0],
        cohort="training",
        elapsed_context=True,
        compensation_progress=True,
        visual_pooling="cls",
        visual_history="none",
        candidate_set="exact",
        late_preference="none",
        maximum_epochs=epochs,
        shared_epoch_candidates=[epochs],
        seeds=[seed],
        visual_dropout=settings["visual_dropout"],
        output_dropout=settings["output_dropout"],
        visual_time_shared=True,
        output_time_shared=True,
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        drop_inputs=[],
        output=baseline.parent / output.name,
    )
    rows, reference = prepare(args)
    folds = DatasetLoader.training_folds(
        rows, nested=True, cohort="training", validation_rule=settings["validation_rule"]
    )
    test = next(iter(folds))
    variants = [
        ("完整LSTM25", []),
        *[(label, overrides) for _, label, overrides in ablation_recipes(args.architecture)],
    ]
    root = training_output(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "settings.json").write_text(
        json.dumps(
            dict(
                reference=str(baseline),
                seed=seed,
                epochs=epochs,
                development_experiment=test,
                validation_folds=folds[test],
                variants=variants,
                evaluation_scope="training_side_quick",
                interpretation="Fixed epoch, no parameter search, no outer LOEO scoring",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"[quick] {len(variants)} variants × 2 training-side folds × {epochs} fixed epochs; seed {seed}",
        flush=True,
    )
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        results = Parallel(max_nbytes=None)(
            delayed(fit_quick_component)(args, rows, reference, test, folds[test], label, overrides)
            for label, overrides in variants
        )
    table = pd.DataFrame(results)
    table.to_csv(output / "LSTM快速组件消融.csv", index=False)
    print(
        table[
            ["variant", "validation_headroom", "validation_cop", "parameters", "epoch"]
        ].to_string(index=False),
        flush=True,
    )


def publication_results(
    baseline, output, *, decision_only=False, decision_view="triptych", decision_events=None
):
    """Reuse frozen fold predictions, screening, comparisons and epoch logs."""
    from image_models.cycle_selection import screen_published_cycles
    from image_models.labels import stopping_cdf_targets
    from image_models.policy import apply_trigger_rule
    from plots.external_policy import render_stopping_results

    settings = json.loads((baseline / "training/settings.json").read_text())
    cohort = screen_published_cycles(settings["dataset"])
    cohort = cohort.loc[cohort.selected_train]
    metrics = read_metrics(baseline, "Model")
    if decision_only:
        for _, part in metrics.groupby("seed"):
            assert part.cycle_name.is_unique and set(part.cycle_name) == set(cohort.cycle_name)
        metrics = metrics.merge(
            cohort[["cycle_name", "setpoint_c"]], on="cycle_name", validate="many_to_one"
        )
        metrics["delta_cop"] = metrics.selected_cop - metrics.rb_cop
        metrics["rb_regret_pct"] = 100 * (1 - metrics.rb_cop / metrics.optimal_cop)
        if decision_view in ("electricity", "power", "heating_electricity"):
            from image_models.policy import predicted_defrost_electricity_metrics

            reference = pd.read_parquet(
                Path(settings["dataset"]) / "training_data/reference.parquet"
            )
            metrics = predicted_defrost_electricity_metrics(metrics, reference)
            if decision_view == "power":
                if decision_events is None:
                    raise ValueError(
                        "--decision-view power requires --decision-events with measured phase boundaries"
                    )
                from image_models.policy import fixed_duration_defrost_power_metrics

                metrics = fixed_duration_defrost_power_metrics(
                    metrics, pd.read_csv(decision_events)
                )
        metrics.to_csv(output / "cycle_results.csv", index=False)
        write_evaluation_metrics(metrics, settings["dataset"], output)
        figure_name = {
            "paired": "Fig2a_paired_decisions",
            "gains": "Fig2a_ranked_cycle_gains",
            "frequency": "Fig2b_ranked_frequency_reduction",
            "heating": "Fig2b_ranked_stable_heating_increase",
            "electricity": "Fig2c_ranked_defrost_electricity_change",
            "power": "Fig2d_ranked_defrost_mean_power_change",
            "heating_electricity": "Fig2e_heating_electricity_relationship",
            "triptych": "Fig2_decision_benefit",
        }[decision_view]
        render_stopping_results(output, figures=[figure_name], decision_view=decision_view)
        return
    if metrics.cycle_name.duplicated().any() or set(metrics.cycle_name) != set(cohort.cycle_name):
        raise ValueError("Results require one seed and the full screened outer-test cohort")
    metrics = metrics.merge(
        cohort[["cycle_name", "setpoint_c", "near_span_minutes"]],
        on="cycle_name",
        validate="one_to_one",
    )
    metrics["delta_cop"] = metrics.selected_cop - metrics.rb_cop
    metrics["rb_regret_pct"] = 100 * (1 - metrics.rb_cop / metrics.optimal_cop)
    metrics.to_csv(output / "cycle_results.csv", index=False)
    rb = metrics.assign(
        version="RB",
        selected_cop=metrics.rb_cop,
        selected_time=metrics.t_RB,
        regret_pct=metrics.rb_regret_pct,
        near_optimal_hit=metrics.rb_cop.ge(0.99 * metrics.optimal_cop),
    )
    compensation_comparisons(
        pd.concat([rb, metrics], ignore_index=True), output, [("RB", "Model")], ["RB", "Model"]
    )
    history = pd.read_csv(baseline / "training/ablation_learning_curves.csv")
    history = history.loc[
        history.experiment.eq(baseline.name) & history.scope.eq("inner_training_curve")
    ]
    if history.history_source.nunique() != 2 * metrics.experiment_id.nunique():
        raise ValueError("Expected two inner validation histories per outer experiment")
    history.to_csv(output / "inner_learning_curves.csv", index=False)
    # Deterministic selections include the failure; never select only successes.
    chosen = []
    middle = (
        ("Most fragmented near-set", "near_components")
        if metrics.near_components.max() > 1
        else ("Largest gain over RB", "gain_vs_rb_pct")
    )
    for role, field in [
        ("Wide near-window", "near_span_minutes"),
        middle,
        ("Largest decision regret", "regret_pct"),
    ]:
        eligible = metrics.loc[~metrics.cycle_name.isin([x["cycle_name"] for x in chosen])]
        row = eligible.sort_values([field, "cycle_name"], ascending=[False, True]).iloc[0]
        chosen.append(
            dict(
                role=role,
                cycle_name=row.cycle_name,
                near_components=row.near_components,
                near_span_minutes=row.near_span_minutes,
                regret_pct=row.regret_pct,
            )
        )
    cases = pd.DataFrame(chosen)
    cases.to_csv(output / "case_selection.csv", index=False)
    traces = []
    for path in checkpoint_files(baseline):
        saved = pickle.load(path.open("rb"))
        rows = saved["predictions"].loc[lambda x: x.split.eq("test")].copy()
        rows = stopping_cdf_targets(rows)
        rows = apply_trigger_rule(rows, settings["trigger_rule"])
        traces.append(rows.loc[rows.cycle_name.isin(cases.cycle_name)])
    traces = pd.concat(traces, ignore_index=True)
    traces.to_csv(output / "case_predictions.csv", index=False)
    render_stopping_results(output)


def collect_figures(baseline, completed, output, dimensions=None, *, render=True, seeds=None):
    """CSV is the plotting input; all summaries use the existing policy evaluator."""
    from plots.model_comparison import render_stopping_ablation

    groups = {}
    settings = json.loads((baseline / "training/settings.json").read_text())
    baseline_label = (
        f"MLP加FiLM宽度{settings.get('hidden_width', 32)}"
        if settings["recipes"][0][0] == "MLP_film"
        else "当前最佳基准"
    )
    base = read_metrics(baseline, baseline_label, seeds)
    for dimension, label, run in completed:
        if not dimensions or dimension in dimensions:
            groups.setdefault(dimension, []).append((label, read_metrics(run, label, seeds)))
    for dimension, variants in groups.items():
        if dimensions and dimension not in dimensions:
            continue
        directory = output / f"{dimension}消融"
        directory.mkdir(parents=True, exist_ok=True)
        table = pd.concat([base, *[g for _, g in variants]], ignore_index=True)
        table.to_csv(directory / "逐循环结果.csv", index=False)
        write_evaluation_metrics(table, settings["dataset"], directory)
        compensation_comparisons(
            table,
            directory,
            [(baseline_label, label) for label, _ in variants],
            list(table.version.unique()),
        )
        source = pd.read_csv(directory / "evaluation_summary.csv")
        # Show the admitted RB-early subset alongside each architecture's total score.
        early_columns = [
            "cohort_cycles",
            "rb_paired_cycles",
            "delta_cop_vs_rb",
            "gain_vs_rb_pct",
            "headroom_captured_pct",
        ]
        early = source.loc[
            source.subset.eq("rb_early"), ["version", "seed", "split", "setpoint_c", *early_columns]
        ].rename(columns={c: f"rb_early_{c}" for c in early_columns})
        source = source.merge(
            early, on=["version", "seed", "split", "setpoint_c"], how="left", validate="many_to_one"
        )
        uncertainty = pd.read_csv(directory / "experiment_bootstrap.csv").rename(
            columns={"right": "version"}
        )
        source = source.merge(uncertainty.drop(columns="left"), on="version", how="left")
        effects = pd.read_csv(directory / "paired_effects.csv").rename(columns={"right": "version"})
        effect_columns = [
            "version",
            "seed",
            "paired_cycles",
            "left_headroom_captured_pct",
            "right_headroom_captured_pct",
        ]
        source = source.merge(effects[effect_columns], on=["version", "seed"], how="left")
        # These paired estimates cover all conditions; never relabel them as subgroup CIs.
        global_rows = source.subset.eq("all") & source.setpoint_c.eq("all")
        pooled_columns = [
            c
            for c in [*uncertainty.columns, *effect_columns]
            if c not in ("left", "version", "seed")
        ]
        source.loc[~global_rows, pooled_columns] = np.nan
        source["dimension"] = dimension
        source["evaluation_scope"] = "full_experiment_loeo"
        source["baseline"] = source.version.eq(baseline_label)
        metadata = {}
        from image_models.policy import restore_policy

        local_runs = [(label, run) for group, label, run in completed if group == dimension]
        for label, run in [(baseline_label, baseline), *local_runs]:
            if label in set(source.version):
                model = pickle.load(checkpoint_files(run)[0].open("rb"))["checkpoint"]
                actual = restore_policy(model)
                metadata[label] = dict(
                    parameters=sum(p.numel() for p in actual.parameters()),
                    source_run=str(run),
                    architecture=model["architecture"],
                    hidden_width=model.get("hidden_width", 32),
                    recurrent_layers=model.get("recurrent_layers", 1),
                    reset_history=model.get("reset_history", False),
                    learning_rate=model["learning_rate"],
                    visual_dropout=getattr(actual, "visual_dropout", 0.0),
                    output_dropout=getattr(getattr(actual, "output_dropout", None), "p", 0.0),
                    visual_time_shared=getattr(actual, "visual_time_shared", False),
                    output_time_shared=getattr(actual, "output_time_shared", False),
                    inputs=";".join(model["numeric_columns"]),
                    loss=model["loss_name"],
                    cdf_score=model.get("cdf_score", "bce"),
                    cdf_link=model.get("cdf_link", "cumulative"),
                )
        for label in source.version.unique():
            metadata.setdefault(label, metadata[baseline_label])
        for field in metadata[baseline_label]:
            source[field] = source.version.map(
                {label: values[field] for label, values in metadata.items()}
            )
        source["parameter_difference_pct"] = 100 * (
            source.parameters / metadata[baseline_label]["parameters"] - 1
        )
        source["variant_order"] = source.version.map(
            {label: i for i, label in enumerate(table.version.unique())}
        )
        source.to_csv(directory / f"{dimension}消融.csv", index=False)
        if render:
            render_stopping_ablation(directory / f"{dimension}消融.csv")


def typical_model_comparison(
    baseline, output, seed_study=None, history=None, layout="ranked", subset="all"
):
    """Representative architectures on one common frozen evaluation cohort."""
    from image_models.cycle_selection import screen_published_cycles
    from image_models.policy import summarize_metrics
    from plots.model_comparison import render_model_comparison

    if seed_study is None:
        raise ValueError("Current model comparison requires --comparison-seed-study")
    study = analysis_output(seed_study)
    summary = pd.read_csv(study / "evaluation_summary.csv")
    summary = summary.loc[summary.subset.eq(subset) & summary.setpoint_c.eq("all")]
    configurations = pd.read_csv(study / "replication_configurations.csv").set_index("model")
    labels = {
        "LSTM25_FiLM": ("LSTM", "LSTM + FiLM (25)"),
        "GRU32_FiLM": ("GRU", "GRU + FiLM (32)"),
        "MLP64x64_FiLM": ("MLP", "MLP + FiLM (64–64)"),
    }
    metrics = pd.read_csv(study / "evaluation_cycles.csv")
    dataset = json.loads((baseline / "training/settings.json").read_text())["dataset"]
    cohort = screen_published_cycles(dataset)
    cohort = set(cohort.loc[cohort.selected_train, "cycle_name"])
    selected = cohort
    if subset == "rb_early":
        selected = cohort & set(
            pd.read_csv(Path(dataset) / "RB失败循环" / "早除霜" / "cohort.csv").cycle_name
        )
    rows = []
    for model, (family, label) in labels.items():
        values = summary.loc[summary.version.eq(model)].sort_values("seed")
        assert values.seed.tolist() == [42, 1010, 2026]
        current = metrics.loc[metrics.version.eq(model)]
        assert all(set(group.cycle_name) == cohort for _, group in current.groupby("seed"))
        run = Path(configurations.loc[model, "experiment"])
        table = pd.read_csv(run / "training/ablation_summary.csv")
        parameters = table.loc[table.scope.eq("outer_fold"), "parameters"].unique()
        assert len(parameters) == 1
        rows.append(
            dict(
                model=model,
                family=family,
                label=label,
                parameters=int(parameters[0]),
                cycles=len(selected),
                experiments=current.loc[
                    current.cycle_name.isin(selected), "experiment_id"
                ].nunique(),
                seeds=3,
                H_RB=values.headroom_captured_pct.mean(),
                H_RB_std=values.headroom_captured_pct.std(),
                H_RB_seeds=json.dumps(values.headroom_captured_pct.tolist()),
                COP_gain_pct=values.gain_vs_rb_pct.mean(),
                COP_gain_pct_std=values.gain_vs_rb_pct.std(),
                COP_gain_seeds=json.dumps(values.gain_vs_rb_pct.tolist()),
                COP_gain_absolute=values.delta_cop_vs_rb.mean(),
                source=str(run),
            )
        )
    source = pd.DataFrame(rows)
    if history is not None:
        historical = pd.read_csv(history)
        historical = historical.loc[
            ~historical.label.isin(source.label) & ~historical.label.str.contains("memory reset")
        ].copy()
        assert historical.cycles.eq(len(cohort)).all() and historical.experiments.eq(29).all()
        historical["seeds"] = 1
        if subset == "rb_early":
            detail = pd.read_csv(history.with_name(history.stem + "_逐循环.csv"))
            for index, row in historical.iterrows():
                current = detail.loc[detail.model.eq(row.model)]
                assert set(current.cycle_name) == cohort and current.seed.eq(0).all()
                current = current.loc[current.cycle_name.isin(selected)]
                values = summarize_metrics(current)
                historical.loc[
                    index,
                    [
                        "H_RB",
                        "COP_gain_pct",
                        "COP_gain_absolute",
                        "exact_hit_pct",
                        "cycles",
                        "experiments",
                    ],
                ] = [
                    values["headroom_captured_pct"],
                    values["gain_vs_rb_pct"],
                    values["delta_cop_vs_rb"],
                    values["hit_rate_exact_all_pct"],
                    len(current),
                    current.experiment_id.nunique(),
                ]
        source = pd.concat([historical, source], ignore_index=True)
    stem = "RB早除霜理想提升空间瀑布图" if subset == "rb_early" else "典型模型性能比较"
    source.to_csv(output / f"{stem}.csv", index=False)
    summary.loc[summary.version.isin(labels)].to_csv(output / f"{stem}_逐种子.csv", index=False)
    render_model_comparison(output / f"{stem}.csv", layout=layout)
    return


def replicate_typical_models(baseline, output, seeds, n_jobs):
    """Frozen structures and optimizers, common inner-only epoch selection for all seeds."""
    from image_models.experiments import collect_frozen_mlp_metrics

    methods = pd.read_csv(
        analysis_output((PROJECT / "output/99_历史归档/探索与旧版图表/所有网络方法比较_20261011"))
        / "所有方法比较.csv"
    ).set_index("方法")
    sources = [
        ("GRU32_FiLM_reset", Path(methods.loc["无历史GRU32 + FiLM", "source"])),
        ("MLP64x64_FiLM", baseline.parent / "MLP_FiLM通用组合_固定64x2_20261011"),
        ("LSTM25_FiLM", baseline.parent / "LSTM_FiLM分阶段优化_20261011_top1"),
        ("GRU32_FiLM", baseline),
    ]
    runs = []
    configurations = []
    for label, source in sources:
        settings = json.loads((source / "training/settings.json").read_text())
        destination = baseline.parent / f"四模型统一早停_{label}_20261011"
        (destination / "training").mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "image_models.optimal_stopping",
            "--dataset",
            settings["dataset"],
            "--output",
            str(destination),
            "--architecture",
            settings["recipes"][0][0],
            "--purpose",
            f"Frozen {label} architecture and optimizer; common 160-epoch two-inner-fold H_RB selection across seeds {seeds}; no outer-test selection or ensemble",
            "--loss-name",
            "cdf_ramp",
            "--recipe",
            "temperature",
            "--cohort",
            "training",
            "--inner-folds",
            "2",
            "--refit-all",
            "--joint-trigger-selection",
            "--validation-tiebreak",
            "set_loss",
            "--validation-rule",
            "stratified",
            "--validation-utility",
            "headroom",
            "--elapsed-context",
            "--compensation-progress",
            "--trigger-rule",
            "cdf",
            "--hidden-width",
            str(settings["hidden_width"]),
            "--recurrent-layers",
            str(settings.get("recurrent_layers", 1)),
            "--learning-rate",
            str(settings["learning_rate"]),
            "--weight-decay",
            str(settings.get("weight_decay", 1e-4)),
            "--visual-dropout",
            str(settings.get("visual_dropout", 0.1)),
            "--output-dropout",
            str(settings["output_dropout"]),
            "--maximum-epochs",
            "160",
            "--seeds",
            *map(str, seeds),
            "--n-jobs",
            str(n_jobs),
            "--version",
            label,
        ]
        if settings.get("reset_history"):
            command += ["--reset-history"]
        if settings["recipes"][0][0] == "MLP_film":
            command += ["--mlp-layers", str(settings.get("mlp_layers", 2))]
        configurations.append(
            dict(
                model=label,
                source=str(source),
                experiment=str(destination),
                seeds=json.dumps(seeds),
                architecture=settings["recipes"][0][0],
                hidden_width=settings["hidden_width"],
                learning_rate=settings["learning_rate"],
                maximum_epochs=160,
                epoch_rule="Two training-side folds: paired coverage, pooled H_RB, set loss, earliest epoch",
                validation_rule="stratified",
                visual_dropout=settings.get("visual_dropout", 0.1),
                output_dropout=settings["output_dropout"],
                reset_history=settings.get("reset_history", False),
            )
        )
        pd.DataFrame(configurations).to_csv(output / "replication_configurations.csv", index=False)
        print(f"[seed replication start] {label}: {len(seeds)} seeds × 29 LOEO folds", flush=True)
        with (destination / "training/runtime.log").open("a") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs.append(destination)
        print(f"[seed replication complete] {label}", flush=True)
    metrics = collect_frozen_mlp_metrics(runs, Path(settings["dataset"]), output)
    assert metrics.groupby(["version", "seed"]).cycle_name.nunique().eq(101).all()
    summary = pd.read_csv(output / "evaluation_summary.csv")
    fields = [
        "headroom_captured_pct",
        "delta_cop_vs_rb",
        "gain_vs_rb_pct",
        "hit_rate_exact_all_pct",
        "hit_rate_filled_all_pct",
        "mean_delay_vs_rb_minutes",
        "mean_cycle_duration_minutes",
        "delay_space_used_pct",
        "rb_paired_cycles",
    ]
    grouped = summary.groupby(["version", "subset", "setpoint_c"])[fields].agg(
        ["mean", "std", "min", "max"]
    )
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.reset_index().to_csv(output / "seed_mean_std.csv", index=False)
    print(
        "[complete] Four frozen model recipes, 348 outer checkpoints; seed-specific and mean/std results saved",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="历史 LOEO 实验目录，含 training/settings.json 和分折权重",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--dimensions", nargs="+", help="Optional dimension subset, using the recipe names"
    )
    parser.add_argument(
        "--ablation-seed",
        type=int,
        help="One seed for component tests; same-seed frozen reference is used",
    )
    parser.add_argument(
        "--epoch-schedule",
        type=Path,
        help="Reuse frozen reference training-side epochs for matched full-LOEO component refits",
    )
    parser.add_argument(
        "--cdf-memory-ablation",
        action="store_true",
        help="Add current-probability ramp BCE, with and without recurrent memory",
    )
    parser.add_argument(
        "--reset-run", type=Path, help="Reuse the completed matched full-input memory-reset study"
    )
    parser.add_argument(
        "--memory-diagnostics",
        action="store_true",
        help="Paired memory effects and frozen-model reset intervention after the factorial study",
    )
    parser.add_argument(
        "--memory-study",
        action="store_true",
        help="Full LOEO 2×2 memory/history refits using frozen baseline training-side epochs",
    )
    parser.add_argument(
        "--quick-epochs",
        type=int,
        help="Fixed-epoch two-fold training-side LSTM component screen; no LOEO",
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--rb-opportunity-only",
        action="store_true",
        help="Plot RB improvement headroom from the screened, frozen evaluation; no training",
    )
    parser.add_argument(
        "--publication-results-only",
        action="store_true",
        help="Figures 2–4 from saved test predictions and inner logs; no training",
    )
    parser.add_argument(
        "--decision-benefit-only",
        action="store_true",
        help="Saved single-model LOEO decision benefit; show each seed separately",
    )
    parser.add_argument(
        "--decision-view",
        choices=[
            "triptych",
            "paired",
            "gains",
            "frequency",
            "heating",
            "electricity",
            "power",
            "heating_electricity",
        ],
        default="triptych",
    )
    parser.add_argument(
        "--decision-events",
        type=Path,
        help="Measured phase boundaries for the fixed-duration mean-power view",
    )
    parser.add_argument(
        "--stopping-evidence-figure",
        action="store_true",
        help="Four-panel decisions, common-seed benchmarks, matched component probes and learning curves",
    )
    parser.add_argument(
        "--memory-result-study",
        type=Path,
        help="Completed matched LOEO study replaces the quick reset row in Fig7",
    )
    parser.add_argument(
        "--component-study", type=Path, help="Matched LSTM component ablation study for Fig7"
    )
    parser.add_argument(
        "--width-search-runs",
        type=Path,
        nargs="+",
        help="Frozen training-side width search runs; kept separate from LOEO",
    )
    parser.add_argument(
        "--typical-model-comparison",
        action="store_true",
        help="Representative fixed-configuration architecture comparison from frozen LOEO predictions",
    )
    parser.add_argument(
        "--comparison-seed-study",
        type=Path,
        help="Saved multi-seed study for the three MLP/GRU/LSTM comparison",
    )
    parser.add_argument(
        "--comparison-history",
        type=Path,
        help="Frozen historical comparison CSV; replace matching model labels and exclude memory reset",
    )
    parser.add_argument("--comparison-layout", choices=["ranked", "waterfall"], default="ranked")
    parser.add_argument("--comparison-subset", choices=["all", "rb_early"], default="all")
    parser.add_argument(
        "--replicate-seeds",
        type=int,
        nargs="+",
        help="Repeat four frozen representative model recipes across these seeds, then collect shared metrics",
    )
    parser.add_argument(
        "--wait-for-run",
        type=Path,
        help="Finish an already-running experiment before using the six workers",
    )
    args = parser.parse_args()
    if args.wait_for_run:
        while not (experiment_root(args.wait_for_run) / "training/ablation_summary.csv").exists():
            time.sleep(30)
    baseline = experiment_root(args.baseline)
    output = analysis_output(args.output)
    settings = json.loads((baseline / "training/settings.json").read_text())
    if not {"loss_name", "learning_rate", "recipes", "seeds", "folds"} <= settings.keys():
        parser.error(
            "--baseline requires a frozen LOEO experiment from image_models.optimal_stopping; fixed train/test experiments use main_defrosting_model.py evaluate"
        )
    if settings["loss_name"] != "cdf_ramp" or settings["learning_rate"] != 0.003:
        raise ValueError("This recipe requires the frozen G_lr003 CDF reference")
    from dataset_tools import DatasetLoader

    loader = DatasetLoader(settings["dataset"])
    if loader.training_settings != settings["data_definition"]:
        raise ValueError("Dataset definition changed; do not pool with the frozen reference")
    output.mkdir(parents=True, exist_ok=True)
    if args.memory_diagnostics:
        memory_study_diagnostics(baseline, output)
        return
    if args.memory_study:
        memory_factorial_study(baseline, output, args.n_jobs)
        return
    if args.quick_epochs:
        if args.quick_epochs < 1 or not settings["recipes"][0][0].startswith("LSTM"):
            parser.error("Quick component screen requires LSTM and positive epochs")
        quick_lstm_components(
            baseline, output, args.quick_epochs, args.n_jobs, args.ablation_seed or 42
        )
        return
    if args.replicate_seeds:
        if len(set(args.replicate_seeds)) != len(args.replicate_seeds):
            parser.error("Replication seeds must be distinct")
        replicate_typical_models(baseline, output, args.replicate_seeds, args.n_jobs)
        return
    if args.stopping_evidence_figure:
        if args.comparison_seed_study is None:
            parser.error("--stopping-evidence-figure requires --comparison-seed-study")
        if args.component_study is None:
            parser.error("--stopping-evidence-figure requires --component-study")
        stopping_evidence_figure(
            baseline,
            output,
            args.comparison_seed_study,
            args.component_study,
            args.width_search_runs or (),
            args.memory_result_study,
        )
        return
    if args.typical_model_comparison:
        typical_model_comparison(
            baseline,
            output,
            args.comparison_seed_study,
            args.comparison_history,
            args.comparison_layout,
            args.comparison_subset,
        )
        return
    if args.publication_results_only:
        publication_results(baseline, output)
        return
    if args.decision_benefit_only:
        publication_results(
            baseline,
            output,
            decision_only=True,
            decision_view=args.decision_view,
            decision_events=args.decision_events,
        )
        return
    if args.rb_opportunity_only:
        from image_models.cycle_selection import screen_published_cycles
        from image_models.policy import near_set_timing
        from plots.rb_rule_explainer import render_rb_opportunity, render_rb_timing

        screening = screen_published_cycles(settings["dataset"])
        screening.to_csv(output / "cycle_screening.csv", index=False)
        cohort = screening.loc[screening.selected_train]
        metrics = read_metrics(baseline, "reference")
        if metrics.cycle_name.duplicated().any() or set(metrics.cycle_name) != set(
            cohort.cycle_name
        ):
            raise ValueError(
                "Use one seed and all frozen outer-test cycles in the current screened cohort"
            )
        source = metrics[["cycle_name", "experiment_id", "rb_cop", "optimal_cop"]].merge(
            cohort[["cycle_name", "setpoint_c"]], on="cycle_name", validate="one_to_one"
        )
        if (
            not np.isfinite(source[["rb_cop", "optimal_cop"]]).all().all()
            or not source.rb_cop.gt(0).all()
        ):
            raise ValueError("Every paired RB/oracle value must be finite and RB COP positive")
        source["potential_gain_pct"] = 100 * (source.optimal_cop / source.rb_cop - 1)
        source["rb_within_1pct"] = source.rb_cop.ge(0.99 * source.optimal_cop)
        if source.potential_gain_pct.lt(-1e-8).any():
            raise ValueError("Oracle must include the RB candidate before comparing headroom")
        source.to_csv(output / "source_data.csv", index=False)
        summary = []
        for label, group in [("All", source), *source.groupby("setpoint_c", sort=True)]:
            summary.append(
                dict(
                    group=str(label),
                    cycles=len(group),
                    experiments=group.experiment_id.nunique(),
                    mean_rb_cop=group.rb_cop.mean(),
                    mean_optimal_cop=group.optimal_cop.mean(),
                    pooled_potential_gain_pct=100
                    * (group.optimal_cop.mean() / group.rb_cop.mean() - 1),
                    median_potential_gain_pct=group.potential_gain_pct.median(),
                    p90_potential_gain_pct=group.potential_gain_pct.quantile(0.9),
                    rb_within_1pct=int(group.rb_within_1pct.sum()),
                )
            )
        pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
        render_rb_opportunity(output / "source_data.csv", output / "summary.csv")
        reference = pd.read_parquet(Path(settings["dataset"]) / "training_data/reference.parquet")
        reference = reference.loc[reference.cycle_name.isin(cohort.cycle_name)].copy()
        reference["candidate_defrost_time"] = pd.to_datetime(
            reference.candidate_defrost_time, format="mixed"
        )
        peaks = metrics.set_index("cycle_name").optimal_cop
        reference["near_optimal"] = reference.cycle_cop_eligible & reference.cycle_cop.ge(
            0.99 * reference.cycle_name.map(peaks)
        )
        rb = metrics.assign(selected_time=pd.to_datetime(metrics.t_RB, format="mixed"))
        at_rb = rb[["cycle_name", "selected_time", "rb_cop"]].merge(
            reference[["cycle_name", "candidate_defrost_time", "cycle_cop", "cycle_cop_eligible"]],
            left_on=["cycle_name", "selected_time"],
            right_on=["cycle_name", "candidate_defrost_time"],
            how="left",
            validate="one_to_one",
        )
        if not at_rb.cycle_cop_eligible.eq(True).all() or not np.allclose(
            at_rb.rb_cop, at_rb.cycle_cop
        ):
            raise ValueError("RB timing requires the exact supported frozen RB COP candidate")
        rb["unsupported_trigger"] = False
        timing, bands = near_set_timing(reference, rb)
        timing = timing.merge(
            rb[["cycle_name", "selected_time", "rb_cop", "optimal_cop"]],
            on="cycle_name",
            validate="one_to_one",
        ).merge(
            cohort[["cycle_name", "experiment_id", "setpoint_c", "baseline_cycle"]],
            on="cycle_name",
            validate="one_to_one",
        )
        if (
            not timing.trigger_status.eq("hit")
            .eq(timing.rb_cop.ge(0.99 * timing.optimal_cop))
            .all()
        ):
            raise ValueError("Exact near-set timing and COP membership must agree")
        timing["outside_near_minutes"] = np.where(
            timing.trigger_status.eq("early"),
            (timing.near_left - timing.selected_time).dt.total_seconds() / 60,
            np.where(
                timing.trigger_status.eq("late"),
                (timing.selected_time - timing.near_right).dt.total_seconds() / 60,
                np.nan,
            ),
        )
        timing["rb_trigger_source"] = np.where(
            timing.baseline_cycle, "rb_acquired_cycle", "replayed_rb_in_experiment"
        )
        timing.to_csv(output / "rb_timing_source.csv", index=False)
        bands.to_csv(output / "rb_near_set_intervals.csv", index=False)
        render_rb_timing(output / "rb_timing_source.csv")
        return
    rows, _ = loader.load_training_data(objective="cdf_ramp")
    observed = rows.loc[rows.training_eligible]
    seeds = [args.ablation_seed] if args.ablation_seed is not None else settings["seeds"]
    if not set(seeds).issubset(settings["seeds"]):
        parser.error("Component seeds must exist in the frozen reference")
    frozen_parts = []
    for path in checkpoint_files(baseline):
        saved = pickle.load(path.open("rb"))
        if saved["checkpoint"]["seed"] == seeds[0]:
            frozen_parts.append(saved["predictions"].loc[lambda x: x.split.eq("test")])
    frozen = pd.concat(frozen_parts, ignore_index=True)
    keys = ["cycle_name", "candidate_defrost_time"]
    fields = ["cycle_cop", "optimal_cop", "input_valid", "near_optimal"]
    comparison = frozen[keys + fields].merge(
        observed[keys + fields],
        on=keys,
        how="outer",
        suffixes=("_frozen", "_current"),
        validate="one_to_one",
    )
    if (
        set(frozen.cycle_name) != set(observed.cycle_name)
        or not comparison.input_valid_frozen.eq(comparison.input_valid_current).all()
        or not comparison.near_optimal_frozen.eq(comparison.near_optimal_current).all()
        or any(
            not np.isclose(
                comparison[field + "_frozen"],
                comparison[field + "_current"],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ).all()
            for field in ("cycle_cop", "optimal_cop")
        )
    ):
        raise ValueError(
            "Teacher/action support changed; preserve the reference and rerun a matched baseline"
        )
    completed, plan = [], []
    recipes = ablation_recipes(settings["recipes"][0][0])
    if args.cdf_memory_ablation:
        recipes += [
            ("LSTM组件", "无CDF", ["--cdf-link", "instantaneous", "--trigger-rule", "hazard"]),
            (
                "LSTM组件",
                "无CDF且无LSTM历史",
                ["--cdf-link", "instantaneous", "--trigger-rule", "hazard", "--reset-history"],
            ),
        ]
    for number, (dimension, label, overrides) in enumerate(recipes, 1):
        if args.dimensions and dimension not in args.dimensions:
            continue
        version = f"AB{number:02d}"
        run = baseline.parent / f"{output.name}_{dimension}_{label}"
        started = baseline.parent / f"{output.name}_{version}"
        if started.exists():
            run = started
        if label == "无LSTM历史" and args.reset_run:
            run = experiment_root(args.reset_run)
            previous = json.loads((run / "training/settings.json").read_text())
            if not previous["reset_history"] or any(
                previous[k] != settings[k]
                for k in (
                    "data_definition",
                    "inputs",
                    "folds",
                    "learning_rate",
                    "weight_decay",
                    "visual_dropout",
                    "output_dropout",
                    "hidden_width",
                )
            ):
                raise ValueError("Reset reuse requires the identical frozen LSTM recipe and cohort")
        # Exact previous complete experiments can be reused, never partial screens.
        reuse = (
            None
            if settings["recipes"][0][0].startswith("LSTM")
            else {
                "无循环记忆": "G_lr003_全维度消融_20261009_时间记忆_无GRU历史",
                "LSTM宽度32": "LSTM消融_20261010_循环单元_LSTM宽度32",
                "LSTM参数匹配宽度25": "LSTM消融_20261010_循环单元_LSTM参数匹配宽度25",
                "无视觉": "cdf90_S_no_visual_lr003_20261009",
                "MLP加FiLM宽度84": "MLP消融_20261010_逐时刻模型_MLP加FiLM宽度84",
                "视觉GRU加FiLM": "G_lr003_全维度消融_20261009_融合方式_视觉GRU加FiLM",
                "lr=0.002": "cdf30_Z5_readoutdrop_lr2_20261009",
                "lr=0.01": "cdf90_I_lr010_20261009",
            }.get(label)
        )
        if (
            reuse
            and (baseline.parent / reuse / "evaluation/policy/evaluation_summary.csv").exists()
        ):
            run = baseline.parent / reuse
            previous = json.loads((run / "training/settings.json").read_text())
            for field in ("data_definition", "folds", "inputs", "seeds", "maximum_epochs"):
                if previous[field] != settings[field]:
                    raise ValueError(f"Reused experiment differs in {field}: {run}")
        plan.append(
            dict(dimension=dimension, variant=label, run=str(run), overrides=" ".join(overrides))
        )
        pd.DataFrame(plan).to_csv(output / "实验计划.csv", index=False)
        if (
            not (run / "evaluation/policy/evaluation_summary.csv").exists()
            and not args.collect_only
        ):
            command = [
                sys.executable,
                "-m",
                "image_models.optimal_stopping",
                "--dataset",
                settings["dataset"],
                "--recipe",
                "temperature",
                "--cohort",
                "training",
                "--loss-name",
                "cdf_ramp",
                "--architecture",
                settings["recipes"][0][0],
                "--hidden-width",
                str(settings["hidden_width"]),
                "--recurrent-layers",
                str(settings.get("recurrent_layers", 1)),
                "--elapsed-context",
                "--compensation-progress",
                "--inner-folds",
                "2",
                "--refit-all",
                "--joint-trigger-selection",
                "--trigger-rule",
                "cdf",
                "--validation-utility",
                settings["validation_utility"],
                "--validation-tiebreak",
                settings["validation_tiebreak"],
                "--validation-rule",
                settings["validation_rule"],
                "--maximum-epochs",
                str(settings["maximum_epochs"]),
                "--n-jobs",
                str(args.n_jobs),
                "--visual-dropout",
                str(settings["visual_dropout"]),
                "--output-dropout",
                str(settings["output_dropout"]),
                "--learning-rate",
                str(settings["learning_rate"]),
                "--weight-decay",
                str(settings["weight_decay"]),
                "--seeds",
                *[str(seed) for seed in seeds],
                "--version",
                version,
                "--output",
                str(run),
                "--compare-run",
                str(baseline),
                "--purpose",
                f"{dimension}: {label}; one-factor frozen reference comparison, same cohort/folds/seed and inner epoch rule",
                *(["--epoch-schedule", str(args.epoch_schedule)] if args.epoch_schedule else []),
                *overrides,
            ]
            output_log = run / "training" / "execution.log"
            output_log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[start {number}/{len(recipes)}] {dimension}: {label}", flush=True)
            with output_log.open("a") as stream:
                subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
        if (run / "evaluation/policy/evaluation_summary.csv").exists():
            matrix = read_metrics(run, label, seeds)
            if any(
                part.outer_experiment.nunique() != len(settings["folds"])
                or not part.cycle_name.is_unique
                or set(part.cycle_name) != set(observed.cycle_name)
                for _, part in matrix.groupby("seed")
            ):
                raise ValueError(f"Incomplete LOEO matrix: {run}")
            completed.append((dimension, label, run))
            collect_figures(baseline, completed, output, [dimension], seeds=seeds)
            print(f"[completed] {dimension}: {label}", flush=True)
    if not args.dimensions or "触发规则" in args.dimensions:
        collect_figures(baseline, completed, output, ["触发规则"])
    write_ablation_tables([baseline, *[run for _, _, run in completed]], output / "训练过程")
    if args.cdf_memory_ablation and len(completed) == 6:
        cdf_memory_diagnostics(baseline, completed, output, seeds)


if __name__ == "__main__":
    main()
