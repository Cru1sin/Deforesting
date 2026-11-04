import numpy as np
import pandas as pd

from tests.cost_workspace.cop_stopping_audit import downward_crossings, marginal_points


def test_variable_event_cost_changes_marginal_rule_and_gaps_do_not_cross():
    t = pd.date_range("2026-01-01", periods=5, freq="10s")
    g = pd.DataFrame(
        dict(
            candidate_defrost_time=t,
            cycle_cop_eligible=True,
            pre_defrost_heat_kwh=[2.0, 2.3, 2.6, 2.9, 3.2],
            pre_defrost_electricity_kwh=[1.0, 1.1, 1.2, 1.3, 1.4],
            defrost_event_electricity_kwh=[0.0, 0.0, 0.2, 0.4, 0.6],
        )
    )
    g["cycle_cop"] = g.pre_defrost_heat_kwh / (
        g.pre_defrost_electricity_kwh + g.defrost_event_electricity_kwh
    )
    result = marginal_points(g, 10)
    assert result.fixed_cost_margin.iloc[1] > 0
    assert result.corrected_margin.iloc[1] < 0
    assert len(downward_crossings(result, "corrected_margin")) == 1
    result.loc[1, "supported_pair"] = False
    assert downward_crossings(result, "corrected_margin").empty
    g.defrost_event_electricity_kwh = 0.1
    g.cycle_cop = g.pre_defrost_heat_kwh / (g.pre_defrost_electricity_kwh + 0.1)
    result = marginal_points(g, 10)
    np.testing.assert_allclose(result.fixed_cost_margin, result.corrected_margin)
