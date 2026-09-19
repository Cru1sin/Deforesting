import pandas as pd
import torch


def test_chain_shapes_and_r_endpoint():
    from image_models.upgrade_chain import contract, model
    from image_models.relative_cop import RelativeCOP

    expected = {'B0': 384, **{f'V{i}': 191 for i in range(7)},
                'V3b': 191, 'V7': 192, 'V8': 199, 'V9': 583,
                'X1': 575, 'X2': 959, 'X3': 959}
    for name, width in expected.items():
        numeric, visual = contract(name)
        assert len(numeric) + len(visual) == width
        net = model(name, len(numeric), len(visual), 0)
        assert net(torch.zeros(2, width)).shape in ((2,), (2, 2))
    torch.manual_seed(0)
    original = RelativeCOP(199)
    rebuilt = model('V8', 199, 0, 0)
    for a, b in zip(original.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_common_reference_precedes_clock_and_missing_input():
    from image_models.upgrade_chain import common_actions

    times = pd.date_range('2026-01-01', periods=7, freq='10s')
    rows = pd.DataFrame(dict(cycle_name='a', candidate_defrost_time=times,
        cycle_cop=[1, 2, 4, 3, 2, 1, 1], cycle_cop_eligible=True,
        stable_heating_start=times[0], observation_end=times[-1],
        rgb_available=False))
    actions = common_actions(rows)
    assert len(actions) == 2
    assert actions.optimal_time.eq(times[2]).all()
    assert actions.after_optimum_target.tolist() == [0., 1.]
    assert actions.optimal_cop.eq(4).all()


def test_extrapolated_reference_remains_eligible_and_dashed(monkeypatch):
    import numpy as np
    import matplotlib.pyplot as plt
    from image_models import relative_cop
    from plots.pareto_selection import plot_objectives

    times = pd.date_range('2026-01-01', periods=7, freq='10s')
    base = pd.DataFrame(dict(cycle_name='a', candidate_defrost_time=times,
        sensor_timestamp=times, stable_heating_start=times[0], observation_end=times[-1],
        pre_defrost_heat_kwh=[1, 2, 4, 3, 2, 1, 1], pre_defrost_electricity_kwh=1.,
        pre_defrost_electricity_measurement_valid=True,
        pre_defrost_heat_measurement_valid=True, pre_defrost_feature_window_valid=True))
    prediction = pd.DataFrame(dict(prediction=[1., 1., 1., 1., 1., np.nan, 1.],
                                   support_distance=9., support_threshold=2.))
    monkeypatch.setattr(relative_cop, 'predict_with_model_parameters', lambda *a: prediction)
    rows = relative_cop.apply_reference(base, {}, 'off', include_history=False,
                                       allow_model_extrapolation=True)
    assert rows.cycle_cop_eligible.tolist() == [True]*5 + [False, True]
    assert rows.cycle_cop_uses_model_extrapolation.iloc[2]
    from image_models.upgrade_chain import common_actions
    assert common_actions(rows).optimal_cop.eq(2).all()
    fig, axis = plt.subplots()
    plot_objectives(axis, rows, times[0], metrics=('cycle_cop',))
    assert axis.lines[0].get_linestyle() == '--'
    assert np.isfinite(axis.lines[0].get_ydata()).any()
    assert not np.isfinite(axis.lines[1].get_ydata()).any()
    plt.close(fig)


def test_final_core_and_frozen_replay():
    from image_models.upgrade_chain import contract, model, frozen_probabilities
    numeric, visual = contract('F_core')
    original, _ = contract('V2')
    assert numeric == original + ['pre_defrost_heat_kwh']
    assert len(numeric) == 192 and visual == []
    net = model('F_core', 192, 0, 0)
    assert sum(p.numel() for p in net.parameters()) == 18281
    assert sum(isinstance(layer, torch.nn.SiLU) for layer in net) == 4
    actions = pd.DataFrame(dict(cycle_name=['a']*3 + ['b'],
        candidate_defrost_time=pd.date_range('2026-01-01', periods=4, freq='30s'),
        cycle_cop=[1., 2., 3., 4.]))
    saved = actions.iloc[[0, 2]].assign(probability=[.1, .7], cycle_cop=99.)
    replay = frozen_probabilities(actions, saved)
    assert replay.cycle_name.tolist() == ['a']*3
    assert replay.probability.iloc[[0, 2]].tolist() == [.1, .7]
    assert pd.isna(replay.probability.iloc[1])
    assert replay.cycle_cop.tolist() == [1., 2., 3.]


def test_rgb_bridge_and_wide_core_are_explicit_recipes():
    from image_models.upgrade_chain import contract, model
    from image_models.stopping_loss_comparison import build_model

    numeric, _ = contract('F_core')
    for name, widths, count in [
        ('G_small', [576, 60, 60, 32, 32, 1], 41321),
        ('G_core', [576, 256, 128, 64, 32, 1], 190977),
        ('G_large', [576, 384, 256, 128, 64, 32, 1], 363393),
    ]:
        inputs, visual = contract(name)
        assert inputs == numeric and len(visual) == 384
        net = build_model(name, numeric_width=192, visual_width=384, seed=0)
        linear = [layer for layer in net if isinstance(layer, torch.nn.Linear)]
        assert [linear[0].in_features] + [layer.out_features for layer in linear] == widths
        assert sum(p.numel() for p in net.parameters()) == count
        assert sum(isinstance(layer, torch.nn.SiLU) for layer in net) == len(widths)-2
        assert not any(isinstance(layer, (torch.nn.Dropout, torch.nn.LayerNorm)) for layer in net)
        output = net(torch.randn(2, 576))
        assert output.shape == (2,)
        output.sum().backward()
        assert all(p.grad is not None for p in net.parameters())
