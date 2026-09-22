from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import select_defrost_time
from defrost_decision.baselines import rule_based


def _frame(
    seconds: int,
    *,
    t1_seconds: float = 0,
    t2_seconds: float = 360,
    t3: float | list[float] = 0,
    t4: float = 0,
    twout: float = 30,
    t3o_raw: float = 50,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=seconds, freq="s"),
            "coil_temperature": t3,
            "ambient_temperature": t4,
            "water_out_temperature": twout,
            "p1__T3o'2_20": t3o_raw,
            "p1__DefTim1'2_20": t1_seconds,
            "p1__DefTim2'2_20": t2_seconds,
        }
    )


def _trigger(frame: pd.DataFrame) -> dict[str, object]:
    start = pd.Timestamp(frame["timestamp"].iloc[0])
    end = pd.Timestamp(frame["timestamp"].iloc[-1]) + pd.Timedelta(seconds=1)
    return rule_based.first_trigger(frame, start, end)


@pytest.mark.parametrize(
    ("t4", "twout", "t3o", "expected"),
    [
        (-2, 35, 4, (40, 1)),
        (-2, 25, 4, (35, 1)),
        (-2, 24, 4, (30, 1)),
        (-5, 35, 4, (40, -1)),
        (-5, 25, 4, (38, -1)),
        (-5, 24, 4, (33, -1)),
        (-8, 35, 4, (80, -1)),
        (-8, 25, 4, (60, -1)),
        (-8, 24, 4, (40, -1)),
        (-10, 35, 4, (90, -15)),
        (-10, 25, 4, (70, -15)),
        (-10, 24, 4, (50, -15)),
        (-10.1, 35, 4, (150, -15.1)),
        (-10.1, 25, 4, (120, -15.1)),
        (-10.1, 24, 4, (90, -15.1)),
    ],
)
def test_rb_case_boundaries_and_limits(
    t4: float, twout: float, t3o: float, expected: tuple[float, float]
) -> None:
    assert rule_based.limits(t4, twout, t3o) == expected


@pytest.mark.parametrize(
    ("delta_seconds", "triggered"), [(49, False), (50, True), (600, True), (601, False)]
)
def test_condition1_uses_only_50_to_600_second_history(
    delta_seconds: int, triggered: bool
) -> None:
    t3 = [np.nan] * (delta_seconds + 1)
    t3[0] = 0
    t3[-1] = -1
    result = _trigger(_frame(len(t3), t1_seconds=36 * 60, t3=t3, t3o_raw=-100))
    assert result["rb_status"] == ("triggered" if triggered else "right_censored")


def test_case_temperature_requires_20_seconds_but_timer_is_checked_at_trigger() -> None:
    frame = _frame(25, t1_seconds=34 * 60, t3=1, t3o_raw=50)
    frame.loc[frame.index >= 1, "coil_temperature"] = 0
    frame.loc[frame.index >= 19, "p1__DefTim1'2_20"] = 35 * 60
    result = _trigger(frame)
    assert result["trigger_type"] == "Case1"
    assert result["t_RB"] == frame.loc[19, "timestamp"]


def test_missing_second_breaks_temperature_confirmation() -> None:
    frame = _frame(20, t1_seconds=40 * 60, t3=0, t3o_raw=50).drop(index=10)
    assert _trigger(frame)["rb_status"] == "right_censored"


def test_case7_wins_same_second_priority_and_case8_needs_no_temperatures() -> None:
    frame = _frame(20, t1_seconds=29 * 60, t2_seconds=0, t3=-20, t4=-5)
    frame.loc[19, "p1__DefTim1'2_20"] = 150 * 60
    assert _trigger(frame)["trigger_type"] == "Case7"

    forced = _frame(
        1,
        t1_seconds=150 * 60,
        t2_seconds=0,
        t3=np.nan,
        t4=np.nan,
        twout=np.nan,
        t3o_raw=np.nan,
    )
    result = _trigger(forced)
    assert result["trigger_type"] == "Case8"
    assert pd.isna(result["case"])


def test_first_trigger_does_not_depend_on_future_data() -> None:
    frame = _frame(80, t1_seconds=36 * 60, t3o_raw=-100)
    frame.loc[50:, "coil_temperature"] = -1
    expected = _trigger(frame)
    frame.loc[51:, ["coil_temperature", "ambient_temperature", "water_out_temperature"]] = 999
    actual = _trigger(frame)
    assert (actual["t_RB"], actual["trigger_type"]) == (
        expected["t_RB"],
        expected["trigger_type"],
    )


class _Loader:
    def __init__(self, frame: pd.DataFrame, preparation: pd.Timestamp | None) -> None:
        self.frame = frame
        self.preparation = preparation

    def get_cycle_record(self, _cycle_name: str) -> dict[str, object]:
        return {
            "cycle_name": "frost_cycle_000999",
            "cycle_id": "cycle_999",
            "experiment_id": "exp",
            "boundaries": {
                "heating_start": self.frame["timestamp"].iloc[0],
                "defrost_preparation_start": self.preparation,
            },
        }

    def load_cycle_original(
        self, _cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        return self.frame.loc[:, columns] if columns is not None else self.frame.copy()


def test_calculate_cycle_excludes_preparation_start() -> None:
    frame = _frame(20, t1_seconds=40 * 60, t3=0, t3o_raw=50)
    result = rule_based.calculate_cycle(_Loader(frame, frame.loc[19, "timestamp"]), "cycle")
    assert result["rb_status"] == "right_censored"
    assert result["observation_end_source"] == "actual_preparation"


def test_incomplete_cycle_uses_raw_cycle_end() -> None:
    frame = _frame(20, t1_seconds=40 * 60, t3=0, t3o_raw=50)
    result = rule_based.calculate_cycle(_Loader(frame, None), "cycle")
    assert result["trigger_type"] == "Case1"
    assert result["t_RB"] == frame.loc[19, "timestamp"]
    assert result["observation_end_source"] == "raw_cycle_end"


def test_condition1_integer_drop_parameter_preserves_historical_default():
    frame = _frame(51, t1_seconds=36*60, t3=[0]*50+[-1.5], t3o_raw=-100)
    start=frame.timestamp.iloc[0]; end=frame.timestamp.iloc[-1]+pd.Timedelta(seconds=1)
    assert rule_based.first_trigger(frame,start,end)['rb_status']=='triggered'
    assert rule_based.first_trigger(frame,start,end,condition1_drop_c=2)['rb_status']=='right_censored'


def test_last_snapshot_matches_prepared_xlsx_and_does_not_use_future():
    frame = _frame(1, t1_seconds=0)
    latest = frame.copy()
    latest["p1__DefTim1'2_20"] = 150 * 60
    frame = pd.concat([frame, latest], ignore_index=True)
    assert _trigger(frame)["trigger_type"] == "Case8"


def test_timeout_can_be_excluded_without_disabling_temperature_cases():
    frame = _frame(20, t1_seconds=161 * 60, t2_seconds=0, t3=0)
    start = frame.timestamp.iloc[0]
    end = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
    assert rule_based.first_trigger(frame, start, end, forced_timeout_min=0)["rb_status"] == "right_censored"
    frame["coil_temperature"] = -20
    assert rule_based.first_trigger(frame, start, end, forced_timeout_min=0)["trigger_type"] == "Case7"


def test_other_parameter_sheet_row_does_not_erase_controller_snapshot():
    frame = _frame(20, t1_seconds=40 * 60, t3=0, t3o_raw=50)
    other_sheet = frame.iloc[[10]].copy()
    other_sheet[["p1__T3o'2_20", "p1__DefTim1'2_20", "p1__DefTim2'2_20"]] = np.nan
    frame = pd.concat([frame, other_sheet], ignore_index=True).sort_values("timestamp", kind="stable")
    assert _trigger(frame)["trigger_type"] == "Case1"


def test_one_second_sampling_tolerance_recovers_pre_reset_boundary():
    frame = _frame(19, t1_seconds=2399, t3=0, t3o_raw=50)
    start = frame.timestamp.iloc[0]
    end = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
    assert rule_based.first_trigger(frame, start, end)["rb_status"] == "right_censored"
    result = rule_based.first_trigger(frame, start, end, sample_tolerance_seconds=1)
    assert result["trigger_type"] == "Case1"
    assert result["t_RB"] < end


@pytest.mark.parametrize("missing, violation, expected", [(3, False, True), (4, False, False), (3, True, False)])
def test_confirmation_gap_tolerance_never_ignores_observed_failure(missing, violation, expected):
    frame = _frame(20, t1_seconds=2400, t3=0, t3o_raw=50)
    frame.loc[5:4 + missing, "p1__T3o'2_20"] = np.nan
    if violation:
        frame.loc[12, "coil_temperature"] = 3
    result = rule_based.first_trigger(frame, frame.timestamp.iloc[0],
                                     frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1),
                                     confirmation_missing_seconds=3)
    assert (result["rb_status"] == "triggered") == expected


def test_gap_tolerance_requires_full_window_and_current_observation():
    frame = _frame(17, t1_seconds=2400, t3=0, t3o_raw=50)
    start = frame.timestamp.iloc[0]
    end = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
    assert rule_based.first_trigger(frame, start, end, confirmation_missing_seconds=3)["rb_status"] == "right_censored"
    frame = _frame(20, t1_seconds=2400, t3=0, t3o_raw=50)
    frame.loc[19, "coil_temperature"] = np.nan
    assert rule_based.first_trigger(frame, start, frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1),
                                    confirmation_missing_seconds=3)["rb_status"] == "right_censored"


def test_condition1_logging_resolution_tolerance_is_explicit():
    frame = _frame(51, t1_seconds=2160, t3=[0] * 50 + [-1.9], t3o_raw=-100)
    start = frame.timestamp.iloc[0]
    end = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
    assert rule_based.first_trigger(frame, start, end, condition1_drop_c=2)["rb_status"] == "right_censored"
    assert rule_based.first_trigger(frame, start, end, condition1_drop_c=2,
                                     condition1_tolerance_c=0.1)["trigger_type"] == "Condition1"


def test_replay_tolerances_remain_causal():
    frame = _frame(80, t1_seconds=2399, t3=0, t3o_raw=50)
    frame.loc[5:7, "p1__T3o'2_20"] = np.nan
    start = frame.timestamp.iloc[0]
    end = frame.timestamp.iloc[-1] + pd.Timedelta(seconds=1)
    settings = dict(sample_tolerance_seconds=1, confirmation_missing_seconds=3,
                    condition1_drop_c=2, forced_timeout_min=0)
    expected = rule_based.first_trigger(frame, start, end, **settings)
    frame.loc[frame.timestamp.gt(expected["t_RB"]), "coil_temperature"] = 999
    actual = rule_based.first_trigger(frame, start, end, **settings)
    assert (actual["t_RB"], actual["trigger_type"]) == (expected["t_RB"], expected["trigger_type"])


def test_environment_boundary_can_be_assigned_to_colder_case():
    assert rule_based.limits(-2, 35, -2.6, case_boundary="colder") == (40, -7.6)
    assert rule_based.limits(-5, 35, -3.7, case_boundary="colder") == (80, -8.7)
    assert rule_based.limits(-1.9, 35, -2.6, case_boundary="colder") == (40, -5.6)


def test_post_trigger_temperature_confirmation_is_audit_only():
    from calculate_rb_reference import audit_actual

    frame = _frame(55, t1_seconds=4000, t3=[0] * 50 + [-1.9, -1.9, -2, -2, -2], t3o_raw=-100)
    frame["p1__化霜条件'1_00"] = 0
    frame.loc[50:, "p1__化霜条件'1_00"] = 10
    frame.loc[50:, ["p1__DefTim1'2_20", "p1__DefTim2'2_20"]] = 0

    class Loader(_Loader):
        def get_cycle_record(self, name):
            record = super().get_cycle_record(name)
            record["assets"] = {"csv": "cycles_RB/frost_cycle_000999.csv"}
            record["boundaries"]["defrost_start"] = frame.timestamp.iloc[-1]
            return record

    loader = Loader(frame, frame.timestamp.iloc[50])
    replay = rule_based.calculate_cycle(loader, "frost_cycle_000999", condition1_drop_c=2)
    result = audit_actual(loader, pd.DataFrame([replay]), condition1_drop_c=2)
    assert pd.isna(result.t_RB.iloc[0])
    assert result.condition1_post_trigger_drop_delay_seconds.iloc[0] == 2
