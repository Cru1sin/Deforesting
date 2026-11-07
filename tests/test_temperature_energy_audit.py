import numpy as np
import pandas as pd

from tests.cost_workspace.temperature_energy_audit import decompose


def test_frequency_figure_has_one_instantaneous_line_and_cost_in_wh(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    import plots.frequency as publication

    raw = pd.DataFrame(
        dict(
            cycle_minutes=[1.0, 2.0],
            compressor_frequency=[50.0, 60.0],
            instantaneous_cop=[3.0, 2.8],
            power_total=[2.0, 2.0],
            heating_capacity_processed=[6.0, 5.6],
        )
    )
    curve = pd.DataFrame(
        dict(
            cycle_minutes=[1.0, 2.0],
            cop_current=[2.0, 1.9],
            cop_fixed_cost=[2.0, 1.95],
            pre_defrost_electricity_kwh=[1.0, 1.2],
            pre_defrost_heat_kwh=[2.0, 2.4],
            defrost_event_electricity_kwh=[0.04, 0.06],
            fixed_event_cost_kwh=[0.04, 0.04],
            cycle_cop_uses_model_extrapolation=[False, True],
        )
    )
    origin = pd.Timestamp("2026-01-01")
    episode = dict(
        start_time=origin + pd.Timedelta(minutes=1),
        end_time=origin + pd.Timedelta(minutes=2),
        anchor_time=origin + pd.Timedelta(seconds=59),
        start_frequency_hz=50,
        end_frequency_hz=60,
        n_steps=2,
        fixed_event_cost_kwh=0.04,
        direction="increase",
        cycle_name="example",
        episode=1,
        maximum_absolute_cop_difference=0.05,
    )

    def verify(fig, *args, **kwargs):
        assert len(fig.axes) == 6
        np.testing.assert_allclose(fig.axes[5].lines[1].get_ydata(), [2.0, 2.4])
        # Vertical anchor markers are additional lines, not signal curves.
        assert len(fig.axes[2].lines) == 2
        np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [3.0, 2.8])
        np.testing.assert_allclose(fig.axes[4].lines[0].get_ydata(), [40.0, 60.0])
        np.testing.assert_allclose(fig.axes[4].lines[1].get_ydata(), [40.0, 40.0])
        plt.close(fig)

    monkeypatch.setattr(publication, "save_publication", verify)
    publication.render_frequency_episode(curve, raw, episode, origin, tmp_path, "example")


def test_consolidated_csv_keeps_cycle_identity_and_partial_reruns(tmp_path, monkeypatch):
    from tests.cost_workspace.temperature_energy_audit import consolidate_episode_csv

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "figures"
    output.mkdir()
    for cycle in ["000001", "000002"]:
        (output / f"frost_cycle_{cycle}_cop_curves.csv").write_text(
            "episode,cop_current\n1,2.123456789012345\n"
        )
    consolidate_episode_csv(output)
    table = pd.read_csv(output / "全部循环调频数据.csv", dtype=str)
    assert table.cycle_name.tolist() == ["frost_cycle_000001", "frost_cycle_000002"]
    assert table.cop_current.eq("2.123456789012345").all()
    (output / "frost_cycle_000001_cop_curves.csv").write_text("episode,cop_current\n1,3.5\n")
    consolidate_episode_csv(output, ["frost_cycle_000001"])
    table = pd.read_csv(output / "全部循环调频数据.csv", dtype=str).set_index("cycle_name")
    assert len(table) == 2 and table.loc["frost_cycle_000001", "cop_current"] == "3.5"
    assert table.loc["frost_cycle_000002", "cop_current"] == "2.123456789012345"
    assert len(list(output.glob("*.csv"))) == 1


def test_frequency_episodes_join_plateaus_but_split_reversals_and_gaps():
    from tests.cost_workspace.temperature_energy_audit import frequency_episodes

    f = pd.DataFrame(
        dict(
            timestamp=pd.date_range("2026-01-01", periods=10, freq="s"),
            compressor_frequency=[65, 70, 70, 70, 80, 90, 90, 85, 85, 75],
        )
    )
    episodes = frequency_episodes(f)
    assert [(e["start_frequency_hz"], e["end_frequency_hz"], e["n_steps"]) for e in episodes] == [
        (65, 90, 3),
        (90, 75, 2),
    ]
    f.loc[2, "compressor_frequency"] = np.nan
    assert len(frequency_episodes(f)) == 2  # Isolated channel dropout does not split a plateau.
    f.loc[4:, "timestamp"] += pd.Timedelta(minutes=1)
    assert len(frequency_episodes(f)) == 3
    f["compressor_frequency"] = 65
    assert frequency_episodes(f) == []


def test_balance_does_not_bridge_support_or_time_gaps():
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:30",
                    "2026-01-01 00:01:00",
                    "2026-01-01 00:02:00",
                ]
            ),
            cycle_cop_eligible=[True, True, False, True],
            pre_defrost_heat_kwh=[2, 2.1, 2.2, 2.3],
            pre_defrost_electricity_kwh=[1, 1.1, 1.2, 1.3],
            defrost_event_electricity_kwh=[0.5, 0.4, 0.4, 0.4],
        )
    )
    g["cycle_cop"] = g.pre_defrost_heat_kwh / (
        g.pre_defrost_electricity_kwh + g.defrost_event_electricity_kwh
    )
    result = decompose(g)
    assert result.supported_pair.tolist() == [True, False, False, False]
    np.testing.assert_allclose(result.net_margin_kw.iloc[0], 12)


def test_frozen_replay_scores_selected_action_without_reselection():
    from tests.cost_workspace.temperature_energy_audit import frozen_replay

    g = pd.DataFrame(
        dict(
            candidate_defrost_time=pd.date_range("2026-01-01", periods=3, freq="30s"),
            cycle_cop_eligible=[True, True, False],
            cycle_cop=[2.0, 1.98, 3.0],
            visual_frost_score=[0.3, 0.5, 0.7],
        )
    )
    r = frozen_replay(g, 1)
    np.testing.assert_allclose(r["frozen_regret_pct"], 1.0)
    assert r["frozen_hit_1pct"]
    assert r["frozen_cop_at_selected"] == 1.98
    np.testing.assert_allclose(r["reference_shift"], 0.2)
    assert r["trigger_shift_minutes"] == 0.5


def test_multistep_balance_preserves_units_and_rejects_internal_missing_support():
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=pd.date_range("2026-01-01", periods=11, freq="30s"),
            cycle_cop_eligible=True,
            pre_defrost_heat_kwh=np.linspace(2, 2.5, 11),
            pre_defrost_electricity_kwh=np.linspace(1, 1.2, 11),
            defrost_event_electricity_kwh=0.5,
        )
    )
    g["cycle_cop"] = g.pre_defrost_heat_kwh / (g.pre_defrost_electricity_kwh + 0.5)
    result = decompose(g, steps=10)
    assert result.supported_pair.sum() == 1
    np.testing.assert_allclose(result.q_kw.iloc[0], 6.0)
    np.testing.assert_allclose(result.p_kw.iloc[0], 2.4)
    g.loc[5, "cycle_cop_eligible"] = False
    assert not decompose(g, steps=10).supported_pair.any()


def test_frequency_onset_requires_stable_history_persistence_and_continuous_time():
    import numpy as np
    import pandas as pd

    from tests.cost_workspace.temperature_energy_audit import frequency_onsets

    grid = pd.DataFrame(
        dict(
            candidate_defrost_time=pd.date_range("2026-01-01", periods=16, freq="30s"),
            compressor_frequency=[30.0] * 10 + [33.0] * 6,
        )
    )
    assert np.flatnonzero(frequency_onsets(grid)).tolist() == [9]
    grid["compressor_frequency"] = [33.0] * 10 + [30.0] * 6
    assert np.flatnonzero(frequency_onsets(grid)).tolist() == [9]
    grid.loc[11, "compressor_frequency"] = 33.0
    assert not frequency_onsets(grid).any()
    grid["compressor_frequency"] = [30.0] * 10 + [33.0] * 6
    grid.loc[12:, "candidate_defrost_time"] += pd.Timedelta(seconds=30)
    assert not frequency_onsets(grid).any()


def test_one_second_balance_uses_declared_interval():
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=pd.date_range("2026-01-01", periods=3, freq="s"),
            cycle_cop_eligible=True,
            pre_defrost_heat_kwh=[2.0, 2.001, 2.002],
            pre_defrost_electricity_kwh=[1.0, 1.001, 1.002],
            defrost_event_electricity_kwh=[0.1, 0.11, 0.11],
        )
    )
    g["cycle_cop"] = g.pre_defrost_heat_kwh / (
        g.pre_defrost_electricity_kwh + g.defrost_event_electricity_kwh
    )
    assert decompose(g, step_seconds=1).supported_pair.tolist() == [True, True, False]
    assert not decompose(g).supported_pair.any()
