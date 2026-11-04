import numpy as np
import pandas as pd
import pytest
import torch


def test_cached_visual_inputs_preserve_exact_training_and_validation(monkeypatch):
    from image_models.policy import fit_policy, predict_policy

    visual = ["dinov2_000", "dinov2_001", "dinov2_002"]
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 4 + ["b"] * 4,
            candidate_defrost_time=list(pd.date_range("2026-01-01", periods=4, freq="30s")) * 2,
            x=np.arange(8.0),
            stopping_cdf_target=[0.0, 0.0, 0.5, 1.0] * 2,
            stopping_cdf_weight=1.0,
            input_valid=True,
            **{name: np.sin(np.arange(8.0) + i) for i, name in enumerate(visual)},
        )
    )
    kwargs = dict(
        architecture="LSTM32_film",
        loss_name="cdf_ramp",
        numeric_columns=["x"],
        visual_columns=visual,
        seed=42,
        maximum_epochs=3,
        patience=4,
        cycles_per_batch=1,
        validation_metric="bce",
        hidden_width=4,
        dropout_config=dict(
            visual_dropout=0.1, output_dropout=0.2, visual_time_shared=True, output_time_shared=True
        ),
    )
    drop = pd.DataFrame.drop

    def retain_inputs(frame, *args, **kw):
        return frame if kw.get("columns") == visual else drop(frame, *args, **kw)

    with monkeypatch.context() as context:
        context.setattr(pd.DataFrame, "drop", retain_inputs)
        baseline = fit_policy(rows, rows, **kwargs)
    compact = fit_policy(rows, rows, **kwargs)
    pd.testing.assert_frame_equal(baseline["losses"], compact["losses"], check_exact=True)
    for name, value in baseline["model_state_dict"].items():
        torch.testing.assert_close(value, compact["model_state_dict"][name], rtol=0, atol=0)
    pd.testing.assert_frame_equal(
        predict_policy(rows, baseline), predict_policy(rows, compact), check_exact=True
    )


def test_policy_timing_includes_unsupported_cop_and_pairs_missing_stops():
    from image_models.policy import policy_timing_metrics, summarize_timing

    reference = pd.DataFrame(
        dict(
            cycle_name=["a", "b", "c"],
            cycle_start=["2026-01-01"] * 3,
            heating_accounting_start=["2026-01-01 00:10"] * 3,
            t_RB=["2026-01-01 01:00"] * 3,
        )
    )
    metrics = pd.DataFrame(
        dict(
            cycle_name=["a", "b", "c"],
            selected_time=["2026-01-01 01:20", "2026-01-01 01:10", None],
            selected_cop=[2.0, np.nan, np.nan],
            forced_final=[False, True, False],
        )
    )
    result = summarize_timing(policy_timing_metrics(metrics, reference))
    assert result["timing_paired_cycles"] == 2 and result["timing_missing_cycles"] == 1
    assert result["timing_fallback_cycles"] == 1
    assert result["mean_delay_vs_rb_minutes"] == 15
    assert result["mean_cycle_duration_minutes"] == 75
    assert result["mean_rb_cycle_duration_minutes"] == 60
    assert result["mean_heating_duration_minutes"] == 65


def test_exact_and_filled_hits_keep_cop_gaps_and_missing_stops_distinct():
    from image_models.policy import policy_timing_metrics, summarize_metrics

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    reference = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            cycle_cop=[2.0, 1.9, 2.0, 1.8, 2.0],
            cycle_cop_eligible=True,
            on_action_grid=[True, True, True, True, False],
        )
    )
    reference = pd.concat(
        [reference.assign(cycle_name=name) for name in ["a", "b", "c"]], ignore_index=True
    )
    metrics = pd.DataFrame(
        dict(
            cycle_name=["a", "b", "c"],
            selected_time=[times[1], times[3], pd.NaT],
            selected_cop=[1.9, 1.8, np.nan],
            optimal_cop=2.0,
            rb_cop=1.8,
            early_trigger=False,
            soft_hard_gap=np.nan,
            no_trigger=[False, False, True],
        )
    )
    rows = policy_timing_metrics(metrics, reference)
    assert rows.filled_near_hit.tolist() == [True, False, False]
    result = summarize_metrics(rows)
    assert result["hit_cycles_exact"] == 0 and result["hit_cycles_filled"] == 1
    assert result["filled_only_hit_cycles"] == 1
    assert result["hit_rate_filled_all_pct"] == pytest.approx(100 / 3)


def test_set_stop_marginalizes_disjoint_candidates_with_stable_survival():
    from image_models.policy import _batch_loss, first_trigger_log_probabilities

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            near_optimal=[False, True, False, True, False],
        )
    )
    h = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.9], dtype=torch.float64)
    logits = torch.logit(h).requires_grad_()
    expected = -torch.log(torch.tensor(0.8 * 0.3 + 0.8 * 0.7 * 0.6 * 0.5, dtype=torch.float64))
    loss = _batch_loss(logits, rows, "set_stop")
    torch.testing.assert_close(loss, expected.to(torch.float64))
    loss.backward()
    assert logits.grad[-1] == 0  # Scores after the final candidate do not affect the event.
    logq, log_remaining = first_trigger_log_probabilities(logits.detach())
    torch.testing.assert_close(
        logq.exp().sum() + log_remaining.exp(), torch.tensor(1.0, dtype=torch.float64)
    )
    # Gaps are not filled, and row order is not interpreted as time order.
    permutation = [4, 2, 0, 3, 1]
    torch.testing.assert_close(
        _batch_loss(logits[permutation], rows.iloc[permutation], "set_stop"), loss
    )
    extreme = torch.tensor([1000.0, -1000.0, 1000.0, -1000.0, 1000.0], requires_grad=True)
    stable = _batch_loss(extreme, rows, "set_stop")
    stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()
    rows["near_optimal"] = False
    with pytest.raises(ValueError, match="candidate"):
        _batch_loss(logits, rows, "set_stop")


def test_near_set_timing_preserves_gaps_and_does_not_call_before_peak_an_error():
    from image_models.policy import near_set_timing

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            near_optimal=[False, True, False, True, False],
            visual_frost_score=[0, 0.8, 0.4, 0.6, 1],
        )
    )
    metrics = pd.DataFrame(
        dict(
            cycle_name=["a"],
            selected_time=[times[1]],
            no_trigger=[False],
            unsupported_trigger=[False],
        )
    )
    timing, bands = near_set_timing(rows, metrics)
    assert timing.trigger_status.item() == "hit"
    assert timing.near_set_position.item() == 0
    assert len(bands) == 2
    metrics["selected_time"] = times[2]
    timing, _ = near_set_timing(rows, metrics)
    assert timing.trigger_status.item() == "gap"
    assert timing.near_set_position.item() == 0.5
    for index, status in [(0, "early"), (4, "late")]:
        metrics["selected_time"] = times[index]
        assert near_set_timing(rows, metrics)[0].trigger_status.item() == status
    rows["near_optimal"] = rows.candidate_defrost_time.eq(times[1])
    metrics["selected_time"] = times[1]
    timing, _ = near_set_timing(rows, metrics)
    assert timing.trigger_status.item() == "hit" and np.isnan(timing.near_set_position.item())
    metrics["unsupported_trigger"] = True
    assert near_set_timing(rows, metrics)[0].trigger_status.item() == "unsupported"
    metrics["selected_time"] = pd.NaT
    metrics["no_trigger"] = True
    assert near_set_timing(rows, metrics)[0].trigger_status.item() == "no_trigger"


def test_event_probability_diagnoses_mass_without_deterministic_crossing():
    from image_models.evaluation import event_probabilities

    times = pd.date_range("2026-01-01", periods=5, freq="30s")
    # One unavailable slot forces continuation; all three allowed hazards stay below .5.
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            input_valid=[False, True, True, True, True],
            near_optimal=[False, True, True, True, False],
            decision_margin=[np.nan, *([np.log(0.3 / 0.7)] * 4)],
        )
    )
    result = event_probabilities(rows).iloc[0]
    assert result.probability_first_in_set == pytest.approx(1 - 0.7**3)
    assert result.max_hazard == pytest.approx(0.3)
    assert result.probability_before_set == 0
    assert result.probability_after_set == pytest.approx(0.7**3 * 0.3)
    assert result.probability_no_event == pytest.approx(0.7**4)
    # A real gap contributes outside-set mass rather than becoming an allowed point.
    rows.loc[2, "near_optimal"] = False
    result = event_probabilities(rows).iloc[0]
    assert result.probability_in_gap == pytest.approx(0.7 * 0.3)
    assert result.probability_first_in_set == pytest.approx(0.3 + 0.7**2 * 0.3)


from image_models.policy import (
    cycle_batches,
    replay_cycles,
    summarize_metrics,
)


def test_cycle_batches_never_split_a_trajectory():
    rows = pd.DataFrame({"cycle_name": ["a", "a", "b", "c", "c", "c"]})
    batches = list(cycle_batches(rows, cycles_per_batch=2, seed=4))
    assert sorted(np.concatenate(batches).tolist()) == list(range(len(rows)))
    assert all(rows.iloc[index].cycle_name.nunique() <= 2 for index in batches)
    assert all(
        sum(rows.iloc[index].cycle_name.eq(name).sum() for index in batches)
        == rows.cycle_name.eq(name).sum()
        for name in rows.cycle_name.unique()
    )


def test_summary_computes_ratios_from_mean_cops_not_mean_cycle_percentages():
    rows = pd.DataFrame(
        {
            "cycle_name": ["a", "b"],
            "selected_cop": [4.0, 8.0],
            "rb_cop": [2.0, 6.0],
            "optimal_cop": [5.0, 10.0],
            "evaluation_status": ["evaluated", "evaluated"],
            "early_trigger": [False, True],
            "soft_hard_gap": [1.0, 3.0],
        }
    )
    result = summarize_metrics(rows)
    assert result["mean_cycle_cop"] == 6
    assert result["gain_vs_rb_pct"] == pytest.approx(50)
    assert result["headroom_captured_pct"] == pytest.approx(100 * 2 / 3.5)
    assert result["headroom_remaining_pct"] == pytest.approx(100 - 100 * 2 / 3.5)


def test_recurrent_policy_is_causal_and_cycle_local():
    from image_models.policy import build_model, policy_logits

    model = build_model("GRU32", numeric_width=0, visual_width=2, seed=0)
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 3 + ["b"] * 2,
            candidate_defrost_time=pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"]
            ),
        )
    )
    x = torch.randn(5, 2)
    base = policy_logits(model, x, rows)
    x[2] += 100
    changed = policy_logits(model, x, rows)
    torch.testing.assert_close(base[:2], changed[:2])
    torch.testing.assert_close(base[3:], changed[3:])
    perm = np.array([4, 2, 0, 3, 1])
    torch.testing.assert_close(policy_logits(model, x[perm], rows.iloc[perm]), changed[perm])


def test_terminal_fallback_is_exact_and_censoring_is_not_forced_stop():
    times = pd.date_range("2026-01-01", periods=3, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            probability=[0.1, 0.2, 0.3],
            cycle_cop=[1.0, 2.0, 3.0],
            cycle_cop_eligible=[True, False, True],
            optimal_time=times[-1],
            optimal_cop=3.0,
            terminal_time=times[-1] + pd.Timedelta(seconds=7),
            terminal_cop=1.5,
        )
    )
    ref = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    dict(
                        cycle_name="a",
                        candidate_defrost_time=rows.terminal_time.iloc[0],
                        cycle_cop=1.5,
                        cycle_cop_eligible=True,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    result = replay_cycles(rows, 0.5, strategy="first_positive_terminal", reference=ref).iloc[0]
    assert (
        result.selected_cop == 1.5
        and result.forced_final
        and result.selected_time == rows.terminal_time.iloc[0]
    )
    rows.loc[1, "probability"] = 0.5
    result = replay_cycles(rows, 0.5, strategy="first_positive_terminal", reference=ref).iloc[0]
    assert (
        pd.isna(result.selected_cop)
        and result.selected_time == times[1]
        and not result.forced_final
    )
    rows.probability = 0.1
    rows.terminal_time = pd.NaT
    result = replay_cycles(rows, 0.5, strategy="first_positive_terminal", reference=ref).iloc[0]
    assert result.no_trigger and pd.isna(result.selected_time) and pd.isna(result.selected_cop)


@pytest.mark.parametrize("width,parameters", [(16, 19905), (32, 41345), (64, 88833)])
def test_gru_capacity_roundtrip_preserves_causal_predictions(width, parameters):
    from image_models.policy import build_model, policy_logits, restore_policy

    model = build_model("GRU32_film", numeric_width=7, visual_width=384, seed=0, hidden_width=width)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert model.gru.hidden_size == width and model.film.out_features == 2 * width
    checkpoint = dict(
        architecture="GRU32_film",
        numeric_columns=list(range(7)),
        visual_columns=list(range(384)),
        seed=0,
        hidden_width=width,
        model_state_dict=model.state_dict(),
    )
    model.eval()
    restored = restore_policy(checkpoint)
    rows = pd.DataFrame(
        dict(cycle_name="a", candidate_defrost_time=pd.date_range("2026", periods=5, freq="30s"))
    )
    x = torch.randn(5, 391)
    base = policy_logits(model, x, rows)
    torch.testing.assert_close(policy_logits(restored, x, rows), base)
    x[3:] += 10
    torch.testing.assert_close(policy_logits(restored, x, rows)[:3], base[:3])


def test_gru_final_score_backpropagates_through_cycle_prefix():
    from image_models.policy import build_model, policy_logits

    model = build_model("GRU32", numeric_width=0, visual_width=2, seed=0)
    rows = pd.DataFrame(
        dict(
            cycle_name=["a"] * 8 + ["b"] * 2,
            candidate_defrost_time=list(pd.date_range("2026", periods=8, freq="30s"))
            + list(pd.date_range("2026", periods=2, freq="30s")),
        )
    )
    values = torch.randn(10, 2, requires_grad=True)
    policy_logits(model, values, rows)[7].backward()
    assert values.grad[0].abs().sum() > 0
    assert values.grad[8:].abs().sum() == 0


def test_visual_dropout_is_locked_in_time_and_absent_at_inference():
    from image_models.policy import build_model

    base = build_model("GRU32", numeric_width=5, visual_width=384, seed=0)
    model = build_model("GRU32_visualdrop", numeric_width=5, visual_width=384, seed=0)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base.parameters())
    seen = []
    hook = model.gru.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    x = torch.ones(2, 7, 389)
    model.train()
    model(x)
    masked = seen[-1]
    torch.testing.assert_close(masked[..., :5], x[..., :5])
    torch.testing.assert_close(masked[:, 1:], masked[:, :1].expand(-1, 6, -1))
    assert (masked[..., 5:] == 0).any() and (masked[..., 5:] > 1).any()
    assert not torch.equal(masked[0], masked[1])
    model.eval()
    base.eval()
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    torch.testing.assert_close(seen[-1], x)
    hook.remove()


def test_filled_late_set_preserves_cop_labels_and_prefers_later_equal_mass():
    from image_models.labels import stopping_candidates
    from image_models.policy import _batch_loss

    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=pd.date_range("2026", periods=5, freq="30s"),
            near_optimal=[False, True, False, True, False],
            cycle_cop=[1.0, 2.0, 1.5, 2.0, 1.0],
        )
    )
    filled = stopping_candidates(rows, "filled", "linear")
    assert filled.stopping_candidate.tolist() == [False, True, True, True, False]
    pd.testing.assert_series_equal(filled.near_optimal, rows.near_optimal)
    pd.testing.assert_series_equal(filled.cycle_cop, rows.cycle_cop)
    np.testing.assert_allclose(np.exp(filled.stopping_log_weight.iloc[[1, 3]]), [0.0, 1.0])
    hazard = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.9], dtype=torch.float64)
    # Candidate weights at indices 1,2,3 are 0,.5,1; gaps keep their raw COP.
    expected = -torch.log(
        torch.tensor(0.8 * 0.7 * 0.4 * 0.5 + 0.8 * 0.7 * 0.6 * 0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(_batch_loss(torch.logit(hazard), filled, "set_stop"), expected)
    early = torch.tensor([-30.0, 10.0, -30.0, -30.0, -30.0], requires_grad=True)
    late = torch.tensor([-30.0, -30.0, -30.0, 10.0, -30.0], requires_grad=True)
    assert _batch_loss(late, filled, "set_stop") < _batch_loss(early, filled, "set_stop")
    exact = stopping_candidates(rows, "exact", "none")
    torch.testing.assert_close(
        _batch_loss(early, exact, "set_stop"), _batch_loss(early, rows, "set_stop")
    )
    single = rows.copy()
    single["near_optimal"] = [False, False, True, False, False]
    singleton = stopping_candidates(single, "filled", "linear")
    assert singleton.stopping_log_weight.eq(0).all()
    loss = _batch_loss(late, filled, "set_stop")
    loss.backward()
    assert torch.isfinite(late.grad).all() and late.grad[-1] == 0


def test_linear_distribution_matches_event_mass_instead_of_rightmost_point():
    from image_models.labels import stopping_candidates
    from image_models.policy import _batch_loss, first_trigger_log_probabilities

    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=pd.date_range("2026", periods=5, freq="30s"),
            near_optimal=[False, True, False, True, False],
        )
    )
    target = stopping_candidates(rows, "filled", "linear", "distribution")
    z = torch.tensor(
        [-20.0, -20.0, float(np.log(0.5)), 20.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    logq, _ = first_trigger_log_probabilities(z)
    loss = _batch_loss(z, target, "set_stop")
    torch.testing.assert_close(loss, -(logq[2] / 3 + 2 * logq[3] / 3))
    right = torch.tensor([-20.0, -20.0, -20.0, 20.0, 0.0], dtype=torch.float64)
    assert loss < _batch_loss(right, target, "set_stop")
    marginal = stopping_candidates(rows, "filled", "linear")
    assert _batch_loss(right, marginal, "set_stop") < _batch_loss(z, marginal, "set_stop")
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad[-1] == 0
    single = rows.assign(near_optimal=[False, False, True, False, False])
    single = stopping_candidates(single, "filled", "linear", "distribution")
    torch.testing.assert_close(_batch_loss(z, single, "set_stop"), -logq[2])


def test_output_dropout_keeps_recurrent_state_and_eval_unchanged():
    from image_models.policy import build_model

    base = build_model("GRU32", numeric_width=5, visual_width=384, seed=0)
    model = build_model("GRU32_drop", numeric_width=5, visual_width=384, seed=0)
    assert model.output_dropout.p == 0.2 and model.visual_dropout == 0
    x = torch.randn(2, 7, 389)
    hidden = []
    head = []
    a = model.gru.register_forward_hook(lambda _, args, out: hidden.append(out[0].detach().clone()))
    b = model.head.register_forward_pre_hook(lambda _, args: head.append(args[0].detach().clone()))
    model.train()
    model(x)
    model(x)
    torch.testing.assert_close(hidden[0], hidden[1])
    assert not torch.equal(head[0], head[1]) and (head[0] == 0).any()
    model.eval()
    base.eval()
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    a.remove()
    b.remove()


def test_configured_dropout_positions_and_time_masks_preserve_eval():
    from image_models.policy import build_model

    x = torch.ones(2, 9, 389)
    base = build_model("GRU32", numeric_width=5, visual_width=384, seed=0).eval()
    for place in ["visual", "output"]:
        for shared in [False, True]:
            model = build_model(
                "GRU32",
                numeric_width=5,
                visual_width=384,
                seed=0,
                dropout_config={place + "_dropout": 0.2, place + "_time_shared": shared},
            )
            before = []
            after = []
            layer = model.gru if place == "visual" else model.head
            handle = layer.register_forward_pre_hook(
                lambda _, args: after.append(args[0].detach().clone())
            )
            if place == "output":
                other = model.gru.register_forward_hook(
                    lambda _, args, out: before.append(out[0].detach().clone())
                )
            model.train()
            model(x)
            mask = after[0][..., 5:] if place == "visual" else after[0] / before[0]
            if shared:
                torch.testing.assert_close(mask, mask[:, :1].expand_as(mask))
            else:
                assert not torch.equal(mask, mask[:, :1].expand_as(mask))
            if place == "visual":
                torch.testing.assert_close(after[0][..., :5], x[..., :5])
            model.eval()
            torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
            handle.remove()
            if place == "output":
                other.remove()


def test_cdf_ramp_bounds_precede_input_mask_and_include_post_interval():
    from image_models.labels import stopping_cdf_targets
    from image_models.policy import _batch_loss, apply_trigger_rule

    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            near_optimal=[False, True, False, False, True, False],
            input_valid=[True, False, True, True, True, True],
        )
    )
    targets = stopping_cdf_targets(rows)
    np.testing.assert_allclose(targets.stopping_cdf_target, [0, 0, 1 / 3, 2 / 3, 1, 1])
    valid = targets.loc[targets.input_valid]
    logits = torch.tensor([-6.0, -4.0, -2.0, 0.0, 1.0], requires_grad=True)
    loss = _batch_loss(logits, valid, "cdf_ramp")
    survival = torch.nn.functional.logsigmoid(-logits).cumsum(0).exp()
    expected = torch.nn.functional.binary_cross_entropy(
        1 - survival, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0, 1.0])
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[-1] != 0
    extreme = torch.tensor([-1000.0, 1000.0, -1000.0, 1000.0, -1000.0], requires_grad=True)
    stable = _batch_loss(extreme, valid, "cdf_ramp")
    stable.backward()
    assert torch.isfinite(stable) and torch.isfinite(extreme.grad).all()
    replay = apply_trigger_rule(targets.assign(probability=0.1), "cdf")
    assert replay.probability.iloc[2] == pytest.approx(0.19)


def test_film_starts_as_identity_and_is_causal():
    from image_models.model import build_model

    base = build_model("GRU32", 3, 4, 0).eval()
    film = build_model("GRU32_film", 3, 4, 0).eval()
    values = torch.randn(2, 5, 7)
    torch.testing.assert_close(base(values), film(values))
    altered = values.clone()
    altered[:, 3:] = 100
    torch.testing.assert_close(film(values)[:, :3], film(altered)[:, :3])


def test_validation_cdf_loss_uses_observed_actions_and_shared_training_loss():
    from image_models.labels import stopping_cdf_targets
    from image_models.policy import _batch_loss, fit_policy, predict_policy

    times = pd.date_range("2026-01-01", periods=6, freq="30s")
    rows = pd.DataFrame(
        dict(
            cycle_name="a",
            candidate_defrost_time=times,
            x=range(6),
            input_valid=[True, True, False, True, True, True],
            near_optimal=[False, True, True, True, False, False],
            cycle_cop=[2.0, 2.99, 3.0, 2.99, 2.9, 2.5],
            cycle_cop_eligible=True,
            optimal_time=times[2],
            optimal_cop=3.0,
            terminal_time=times[-1],
            t_RB=times[0],
        )
    )
    rows = stopping_cdf_targets(rows)
    valid = rows.loc[rows.input_valid]
    fit = fit_policy(
        valid,
        rows,
        architecture="logistic",
        loss_name="cdf_ramp",
        numeric_columns=["x"],
        visual_columns=[],
        maximum_epochs=1,
        patience=2,
        cycles_per_batch=1,
        seed=0,
        controller_reference=rows,
        validation_strategy="first_positive_terminal",
        validation_complete_first=True,
        joint_trigger_selection=True,
        validation_tiebreak="set_loss",
        trigger_rule="cdf",
        initial_hazard=0.01,
    )
    predicted = predict_policy(valid, fit)
    expected = float(
        _batch_loss(torch.as_tensor(predicted.decision_margin.to_numpy()), valid, "cdf_ramp")
    )
    assert fit["losses"].validation_cdf_loss.item() == pytest.approx(expected)
    assert fit["losses"].validation_cdf_loss_cycles.item() == 1


def test_predicted_defrost_energy_uses_exact_fixed_decisions_and_rb_denominator():
    from image_models.policy import predicted_defrost_electricity_metrics

    reference = pd.DataFrame(
        dict(
            cycle_name=["a"] * 3,
            candidate_defrost_time=["2026-01-01 01:00", "2026-01-01 01:10", "2026-01-01 01:20"],
            cycle_cop=[2.0, 2.1, 2.2],
            defrost_event_electricity_kwh=[0.2, 0.3, 0.1],
            defrost_event_electricity_prediction_available=True,
        )
    )
    metrics = pd.DataFrame(
        dict(
            cycle_name=["a", "a"],
            selected_time=reference.candidate_defrost_time.iloc[1:].tolist(),
            t_RB=reference.candidate_defrost_time.iloc[0],
            selected_cop=[2.1, 2.2],
            rb_cop=2.0,
            seed=[42, 1010],
        )
    )
    result = predicted_defrost_electricity_metrics(metrics, reference)
    assert np.allclose(result.defrost_electricity_change_kwh, [0.1, -0.1])
    assert np.allclose(result.defrost_electricity_change_pct, [50.0, -50.0])
    pd.testing.assert_frame_equal(result[metrics.columns], metrics)
    with pytest.raises(ValueError, match="frozen COP"):
        predicted_defrost_electricity_metrics(metrics, reference.assign(cycle_cop=9.0))
    with pytest.raises(ValueError, match="positive RB"):
        predicted_defrost_electricity_metrics(
            metrics, reference.assign(defrost_event_electricity_kwh=0.0)
        )


def test_fixed_duration_power_converts_kwh_to_w_and_preserves_relative_change():
    from image_models.policy import fixed_duration_defrost_power_metrics

    metrics = pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            selected_defrost_electricity_kwh=[0.3, 0.3],
            rb_defrost_electricity_kwh=[0.2, 0.2],
        )
    )
    events = pd.DataFrame(
        dict(
            cycle_name=["a", "b"],
            defrost_preparation_start="2026-01-01 01:00",
            defrost_end=["2026-01-01 01:05", None],
            duration_event_valid=[True, False],
        )
    )
    result = fixed_duration_defrost_power_metrics(metrics, events)
    assert result.power_duration_available.tolist() == [True, False]
    assert np.isclose(result.selected_defrost_mean_power_w.iloc[0], 3600.0)
    assert np.isclose(result.rb_defrost_mean_power_w.iloc[0], 2400.0)
    assert np.isclose(result.defrost_mean_power_change_w.iloc[0], 1200.0)
    assert np.isclose(result.defrost_mean_power_change_pct.iloc[0], 50.0)
    assert result.defrost_mean_power_change_w.iloc[1:].isna().all()
    pd.testing.assert_frame_equal(result[metrics.columns], metrics)
