from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


class MemoryLoader:
    def __init__(self) -> None:
        self.start = pd.Timestamp("2026-01-01 00:00:00")
        self.records: dict[str, dict[str, object]] = {}
        self.frames: dict[str, pd.DataFrame] = {}
        for index, experiment in enumerate(("heldout", "train_a", "train_b", "train_c")):
            name = f"cycle_{experiment}"
            stable = self.start + pd.Timedelta(hours=index)
            self.records[name] = {
                "cycle_name": name,
                "experiment_id": experiment,
                "status": "valid",
                "status_reason": "",
                "stable_heating_start": stable,
                "defrost_preparation_start": stable + pd.Timedelta(seconds=150),
                "defrost_start": stable + pd.Timedelta(seconds=180),
                "defrost_end": stable + pd.Timedelta(seconds=240),
            }
            seconds = pd.date_range(stable, stable + pd.Timedelta(seconds=300), freq="s")
            self.frames[name] = pd.DataFrame(
                {
                    "timestamp": seconds,
                    "power_total": 2.0 + index / 10,
                    "heating_capacity": 4.0 + index / 10,
                    "water_in_temperature": 40.0 + index,
                    "water_out_temperature": 45.0 + index,
                    "coil_temperature": -10.0 + index,
                    "evaporating_pressure": 0.2 + index / 100,
                    "water_temperature_setpoint": 50.0,
                }
            )

    @property
    def catalog(self) -> dict[str, object]:
        return {"cycles": list(self.records.values())}

    def list_cycles(self) -> pd.DataFrame:
        return pd.DataFrame(self.records.values())

    def get_cycle_record(self, cycle_name: str) -> dict[str, object]:
        return deepcopy(self.records[cycle_name])

    def load_cycle_original(
        self, cycle_name: str, *, columns: list[str] | None = None
    ) -> pd.DataFrame:
        result = self.frames[cycle_name].copy()
        return result if columns is None else result[columns]


def _sources(loader: MemoryLoader) -> dict[str, pd.DataFrame]:
    rows = []
    tickets = []
    recoveries = []
    for index, experiment in enumerate(("heldout", "train_a", "train_b", "train_c")):
        name = f"cycle_{experiment}"
        pe = 0.2 + index / 100
        rows.append(
            {
                "cycle_name": name,
                "status": "included",
                "experiment_id": experiment,
                "evaporating_pressure": pe,
                "inclusive_energy_kwh": 0.08 + pe / 10,
                "preparation_signed_heat_kwh": 0.03 + index / 100,
                "water_temperature_setpoint": 50.0,
            }
        )
        tickets.append(
            {
                "cycle_name": name,
                "valid": True,
                "experiment_id": experiment,
                "water_in_temperature": 40.0 + index,
                "water_out_temperature": 45.0 + index,
                "coil_temperature": -10.0 + index,
                "evaporating_pressure": pe,
                "rule_defrost_duration_minutes": 4.0 + index / 10,
                "defrost_absorbed_heat_kwh": 0.6 + index / 10,
            }
        )
        recoveries.append(
            {
                "cycle_name": name,
                "recovery_valid": True,
                "experiment_id": experiment,
                "pre_water_temperature_setpoint": 50.0,
                "recovery_electricity_kwh": 0.2 + index / 100,
                "recovery_water_heat_kwh": 0.7 + index / 100,
                "recovery_electricity_coverage": 1.0,
                "recovery_water_heat_coverage": 1.0,
            }
        )
    return {
        "preparation": pd.DataFrame(rows),
        "preparation_network_cycle_names": pd.Series(
            [row["cycle_name"] for row in rows], dtype="object"
        ),
        "tickets": pd.DataFrame(tickets),
        "recovery": pd.DataFrame(recoveries),
    }


def _points(loader: MemoryLoader) -> pd.DataFrame:
    record = loader.records["cycle_heldout"]
    return pd.DataFrame(
        {
            "cycle_name": ["cycle_heldout"],
            "experiment_id": ["heldout"],
            "valid": [True],
            "candidate_end": [record["defrost_preparation_start"]],
            "failure_reason": [""],
        }
    )


def test_candidates_start_at_one_minute_and_do_not_bridge_raw_gaps() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    frame = loader.frames["cycle_heldout"]
    loader.frames["cycle_heldout"] = frame.loc[
        ~frame["timestamp"].between(
            loader.start + pd.Timedelta(seconds=75), loader.start + pd.Timedelta(seconds=85)
        )
    ]

    curve, _ = build_v266_table(_points(loader), loader, _sources(loader))

    assert curve.iloc[0]["candidate_time"] == loader.start + pd.Timedelta(minutes=1)
    assert curve.iloc[-1]["candidate_time"] == loader.start + pd.Timedelta(seconds=150)
    assert curve.iloc[-1]["gap_seconds_total"] == 12.0
    assert curve.iloc[-1]["heating_electricity_kwh"] < 2.0 * 150 / 3600
    assert not curve["endpoint_extrapolated"].any()


def test_identification_curves_always_abstain() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    curve, audit = build_v266_table(_points(loader), loader, _sources(loader))

    assert curve["recommended_time"].isna().all()
    assert not curve["hard_label_eligible"].any()
    assert curve["decision_status"].eq("abstain_v266_identification_only").all()
    assert curve["t_star_semantics"].eq("diagnostic_raw_argmin_not_label").all()
    assert len(audit) == len(loader.records)
    assert curve["relative_regret"].min() == pytest.approx(0)
    assert curve["near_optimal_1pct"].any()
    assert curve["cycle_cop"].notna().all()
    assert curve["actual_preparation_time"].notna().all()


def test_heldout_terminal_outcomes_cannot_change_its_curve() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    sources = _sources(loader)
    before, _ = build_v266_table(_points(loader), loader, sources)
    changed = {
        key: value.copy(deep=True) if isinstance(value, pd.DataFrame) else value.copy()
        for key, value in sources.items()
    }
    changed["preparation"].loc[
        lambda x: x.experiment_id.eq("heldout"),
        ["inclusive_energy_kwh", "preparation_signed_heat_kwh"],
    ] = 1e9
    changed["tickets"].loc[
        lambda x: x.experiment_id.eq("heldout"),
        ["rule_defrost_duration_minutes", "defrost_absorbed_heat_kwh"],
    ] = 1e9
    changed["recovery"].loc[
        lambda x: x.experiment_id.eq("heldout"),
        ["recovery_electricity_kwh", "recovery_water_heat_kwh"],
    ] = 1e9
    after, _ = build_v266_table(_points(loader), loader, changed)

    columns = [
        "ED",
        "predicted_defrost_duration_minutes",
        "QD",
        "Qprep",
        "ER",
        "QR",
        "lambda0",
        "J",
    ]
    assert np.allclose(before[columns], after[columns], rtol=0, atol=1e-12, equal_nan=True)
    assert before["raw_t_star"].equals(after["raw_t_star"])
    assert (
        before["training_experiment_ids"]
        .str.split(",")
        .apply(lambda ids: "heldout" not in ids)
        .all()
    )


def test_qprep_uses_all_included_events_not_the_smaller_ed_cohort() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    sources = _sources(loader)
    sources["preparation_network_cycle_names"] = pd.Series(
        ["cycle_heldout", "cycle_train_a", "cycle_train_b"]
    )

    curve, _ = build_v266_table(_points(loader), loader, sources)

    assert curve["Qprep"].eq(0.05).all()


def test_each_component_has_heldout_clean_training_provenance() -> None:
    import json

    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    curve, _ = build_v266_table(_points(loader), loader, _sources(loader))
    provenance = json.loads(curve.iloc[0]["component_provenance"])

    for component in ("ED", "duration", "QD", "Qprep", "recovery", "lambda0"):
        audit = provenance["components"][component]
        assert audit["training_event_count"] >= 3
        assert audit["training_experiment_count"] >= 3
        assert "heldout" not in audit["training_experiment_ids"]
    assert provenance["components"]["lambda0"]["anchor_cycle_count"] >= 3


def test_cycle_status_uses_eligible_boundaries_and_local_extrapolation() -> None:
    from frost_analysis.cost.identification import _classify_cycle

    base = pd.DataFrame(
        {
            "J": [1.02, 1.0, 1.5, 2.0],
            "optimization_eligible": [True, True, True, True],
            "measurement_eligible": [True, True, True, True],
            "component_extrapolated_duration": [False, False, False, True],
        }
    )
    assert _classify_cycle(base, 1, {1}) == "identified_curve"

    leading_invalid = base.iloc[:3].copy()
    leading_invalid["optimization_eligible"] = [False, True, True]
    assert _classify_cycle(leading_invalid, 1, {1}) == "left_boundary_limited"

    assert _classify_cycle(base.iloc[:3], 2, {2}) == "right_censored"
    trailing = base.copy()
    trailing["optimization_eligible"] = [True, True, True, False]
    trailing["measurement_eligible"] = [True, True, True, False]
    assert _classify_cycle(trailing, 2, {2}) == "measurement_limited"
    trailing["measurement_eligible"] = True
    assert _classify_cycle(trailing, 2, {2}) == "unidentifiable_component"


def test_raw_argmin_tie_selects_earliest_candidate(monkeypatch) -> None:
    import frost_analysis.cost.identification as identification

    loader = MemoryLoader()

    def constant_components(state, models):
        return {
            "ED": 0.0,
            "ER": 0.0,
            "Qprep": 1.0,
            "QD": 0.0,
            "QR": 1.0,
            "lambda0": 0.5,
            "predicted_defrost_duration_minutes": 4.0,
            "component_extrapolated_duration": False,
            "training_experiment_ids": "train_a,train_b,train_c",
            "component_provenance": "{}",
        }

    monkeypatch.setattr(identification, "_component_predictions", constant_components)
    curve, _ = identification.build_v266_table(_points(loader), loader, _sources(loader))

    assert curve.J.nunique() == 1
    assert curve.raw_t_star.iloc[0] == curve.candidate_time.min()


def test_one_percent_basin_stops_at_ineligible_row() -> None:
    from frost_analysis.cost.identification import _basin

    curve = pd.DataFrame(
        {
            "candidate_time": pd.date_range("2026-01-01", periods=3, freq="min"),
            "J": [1.0, 1.0, 1.005],
            "optimization_eligible": [True, False, True],
        }
    )

    start, end, width = _basin(curve, 0.01, 0)

    assert start == end == curve.candidate_time.iloc[0]
    assert width == 0.0


def test_single_candidate_cannot_produce_an_identified_argmin_or_basin() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    points = _points(loader)
    points["candidate_end"] = loader.start + pd.Timedelta(minutes=1)

    curve, audit = build_v266_table(points, loader, _sources(loader))

    assert curve["raw_t_star"].isna().all()
    assert curve["t_star"].isna().all()
    assert curve.filter(regex=r"^basin_").isna().all().all()
    assert audit.loc[audit.cycle_name.eq("cycle_heldout"), "cycle_status"].item() == (
        "unidentifiable_boundary"
    )


def test_missing_candidate_end_falls_back_to_actual_preparation() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    points = _points(loader)
    points["candidate_end"] = pd.NaT
    points["t_actual_preparation"] = loader.start + pd.Timedelta(seconds=150)

    curve, _ = build_v266_table(points, loader, _sources(loader))

    assert curve.candidate_time.max() == loader.start + pd.Timedelta(seconds=150)


def test_missing_valid_flag_is_not_truthy_or_admitted_to_curve() -> None:
    from frost_analysis.cost.identification import build_v266_table

    loader = MemoryLoader()
    points = _points(loader)
    points["valid"] = pd.NA
    points["failure_reason"] = pd.NA

    curve, audit = build_v266_table(points, loader, _sources(loader))

    assert curve.empty
    heldout = audit.loc[audit.cycle_name.eq("cycle_heldout")].iloc[0]
    assert heldout.cycle_status == "unidentifiable_boundary"
    assert heldout.failure_reason != "<NA>"


def test_candidate_prefix_is_invariant_to_later_in_domain_sensor_values() -> None:
    from frost_analysis.cost.identification import STATE_COLUMNS, build_v266_table

    loader = MemoryLoader()
    before, _ = build_v266_table(_points(loader), loader, _sources(loader))
    changed_loader = deepcopy(loader)
    cutoff = loader.start + pd.Timedelta(minutes=1)
    changed_loader.frames["cycle_heldout"].loc[
        lambda x: x.timestamp.gt(cutoff) & x.timestamp.le(loader.start + pd.Timedelta(seconds=150)),
        [
            "power_total",
            "heating_capacity",
            "water_in_temperature",
            "water_out_temperature",
            "coil_temperature",
            "evaporating_pressure",
        ],
    ] = 1e9
    after, _ = build_v266_table(_points(loader), changed_loader, _sources(changed_loader))

    prefix_columns = [
        "candidate_time",
        "heating_electricity_kwh",
        "unit_heating_kwh",
        "power_coverage",
        "unit_coverage",
        *STATE_COLUMNS,
        "ED",
        "predicted_defrost_duration_minutes",
        "QD",
        "Qprep",
        "ER",
        "QR",
        "lambda0",
        "L",
        "K",
        "J",
    ]
    pd.testing.assert_frame_equal(
        before.loc[before.candidate_time.le(cutoff), prefix_columns].reset_index(drop=True),
        after.loc[after.candidate_time.le(cutoff), prefix_columns].reset_index(drop=True),
    )


def test_production_evidence_cohort_counts_are_explicit() -> None:
    from frost_analysis.cost.identification import _default_sources

    sources = _default_sources()
    preparation = sources["preparation"]
    qprep = preparation.loc[
        preparation.status.eq("included")
        & pd.to_numeric(preparation.preparation_signed_heat_kwh, errors="coerce").notna()
    ]
    ed = qprep.loc[
        qprep.cycle_name.astype(str).isin(
            set(sources["preparation_network_cycle_names"].astype(str))
        )
    ]
    tickets = sources["tickets"].loc[lambda x: x.valid.fillna(False)]
    recovery = sources["recovery"].loc[
        lambda x: (
            x.recovery_valid.fillna(False)
            & pd.to_numeric(x.recovery_electricity_coverage, errors="coerce").ge(0.95)
            & pd.to_numeric(x.recovery_water_heat_coverage, errors="coerce").ge(0.95)
        )
    ]

    assert (len(ed), ed.experiment_id.nunique()) == (61, 16)
    assert (len(qprep), qprep.experiment_id.nunique()) == (62, 16)
    assert (len(tickets), tickets.experiment_id.nunique()) == (68, 15)
    assert (len(recovery), recovery.experiment_id.nunique()) == (68, 15)


def test_selected_dispatches_v266_and_preserves_audit(monkeypatch) -> None:
    import frost_analysis.cost.identification as identification
    from frost_analysis.cost.selected import build_cost_function_table

    expected = pd.DataFrame({"cycle_name": ["curve"], "valid": [True]})
    audit = pd.DataFrame({"cycle_name": ["audit"]})
    monkeypatch.setattr(
        identification, "build_v266_table", lambda points, loader: (expected, audit)
    )

    result = build_cost_function_table(pd.DataFrame(), pd.DataFrame(), object(), "v2.6.6")

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(result.attrs["cycle_audit"], audit)


def test_selected_rejects_duplicate_points_before_v266_dispatch(monkeypatch) -> None:
    import frost_analysis.cost.identification as identification
    from frost_analysis.cost.selected import build_cost_function_table

    monkeypatch.setattr(
        identification,
        "build_v266_table",
        lambda points, loader: (_ for _ in ()).throw(AssertionError("dispatched")),
    )
    duplicate = pd.DataFrame({"cycle_name": ["same", "same"]})

    with pytest.raises(ValueError, match="one row per cycle"):
        build_cost_function_table(pd.DataFrame(), duplicate, object(), "v2.6.6")


def test_v266_cli_validation_uses_frozen_curve_and_audit_contract() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[2] / "scripts/cost/build.py"
    spec = importlib.util.spec_from_file_location("cost_build", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require_valid = module._require_valid

    expected = [f"curve_{index}" for index in range(69)]
    other = [f"other_{index}" for index in range(32)]
    points = pd.DataFrame({"cycle_name": [*expected, *other], "valid": [True] * 69 + [False] * 32})
    table = pd.DataFrame({"cycle_name": expected, "valid": False})
    audit = pd.DataFrame(
        {
            "cycle_name": [*expected, *other],
            "eligible_candidate_count": [2] * 60 + [0] * 41,
        }
    )
    table.attrs["cycle_audit"] = audit

    _require_valid(table, "v2.6.6", points)

    drift_points = points.copy()
    drift_points.loc[drift_points.cycle_name.eq(expected[-1]), "valid"] = False
    drift_table = table.loc[table.cycle_name.ne(expected[-1])].copy()
    drift_table.attrs["cycle_audit"] = audit
    with pytest.raises(RuntimeError, match="69"):
        _require_valid(drift_table, "v2.6.6", drift_points)

    scoped_audit = audit.copy()
    scoped_audit.loc[scoped_audit.cycle_name.eq(expected[59]), "eligible_candidate_count"] = 0
    scoped_audit.loc[scoped_audit.cycle_name.eq(other[0]), "eligible_candidate_count"] = 2
    table.attrs["cycle_audit"] = scoped_audit
    with pytest.raises(RuntimeError, match="at least 60"):
        _require_valid(table, "v2.6.6", points)
