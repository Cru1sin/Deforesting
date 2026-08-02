from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from frost_analysis.config import EvidencePolicy, load_evidence_settings
from frost_analysis.evidence import build_evidence_bundle, resolve_analysis_reference
from frost_analysis.evidence_cycle import (
    build_channel_evidence,
    build_cycle_slices,
    expected_grid,
)
from frost_analysis.io import load_evidence_runs, optional_sha256, write_evidence_outputs


def _defaults(tmp_path: Path) -> Path:
    path = tmp_path / "defaults.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "channels_path": "channels.yaml",
                "analysis": {
                    "evidence": {
                        "min_segment_coverage": 0.8,
                        "min_segment_points": 12,
                        "min_pair_coverage": 0.8,
                        "min_valid_pairs": 30,
                        "min_valid_cycles": 3,
                        "horizons_minutes": [5, 10, 20],
                        "targets": ["heating_capacity", "cop"],
                        "auto_reference_window_minutes": 5,
                        "auto_reference_min_observed_fraction": 0.8,
                        "auto_reference_max_gap_seconds": 60,
                        "onset_window_seconds": 60,
                        "onset_mad_multiplier": 3.0,
                        "onset_persistence_seconds": 60,
                        "similarity_threshold": 0.85,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_evidence_settings_reads_defaults_only(tmp_path: Path) -> None:
    path = _defaults(tmp_path)
    settings = load_evidence_settings(path, allow_date_config=False)

    assert settings.channels_path == tmp_path / "channels.yaml"
    assert settings.policy.horizons_minutes == (5, 10, 20)
    assert settings.policy.targets == ("heating_capacity", "cop")
    assert settings.policy.auto_reference_window_minutes == 5


def test_batch_evidence_settings_rejects_date_facts(tmp_path: Path) -> None:
    path = _defaults(tmp_path)
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping["experiment_date"] = "2026-07-15"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="date-specific"):
        load_evidence_settings(path, allow_date_config=False)


def test_auto_reference_uses_only_real_observations_and_delays_evidence() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=42, freq="10s")
    values = np.arange(42, dtype=float)
    imputed = np.zeros(42, dtype=bool)
    imputed[3] = True
    values[3] = 999.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["frost_development"] * 42,
            "signal": values,
            "signal__imputed": imputed,
        }
    )

    reference = resolve_analysis_reference(
        frame,
        channel="signal",
        cycle_start=start,
        formal_frost_start=start,
        formal_frost_end=start + pd.Timedelta(minutes=7),
        configured_baseline_available=False,
        configured_residual=None,
        configured_baseline_mask=None,
        window_minutes=5,
        minimum_observed_fraction=0.8,
        maximum_gap_seconds=60,
        interval_seconds=10,
    )

    assert reference.source == "auto_cycle_initial_reference"
    assert reference.observed_fraction == pytest.approx(29 / 30)
    assert reference.valid_from == start + pd.Timedelta(minutes=5)
    assert reference.center == pytest.approx(np.median(np.delete(values[:30], 3)))
    assert reference.residual.loc[3] == pytest.approx(values[3] - reference.center)


def test_auto_reference_does_not_recenter_residual_twice() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=42, freq="10s")
    values = np.full(42, 10.0)
    values[30:] = 14.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["frost_development"] * 42,
            "signal": values,
            "signal__imputed": [False] * 42,
        }
    )

    reference = resolve_analysis_reference(
        frame,
        channel="signal",
        cycle_start=start,
        formal_frost_start=start,
        formal_frost_end=start + pd.Timedelta(minutes=7),
        configured_baseline_available=False,
        configured_residual=None,
        configured_baseline_mask=None,
        window_minutes=5,
        minimum_observed_fraction=0.8,
        maximum_gap_seconds=60,
        interval_seconds=10,
    )

    assert reference.residual.iloc[30] == pytest.approx(4.0)


def test_bundle_uses_channel_reference_and_keeps_future_change_raw(tmp_path: Path) -> None:
    settings = load_evidence_settings(_defaults(tmp_path), allow_date_config=False)
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    progress = np.linspace(0.0, 1.0, len(timestamps))
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "cycle_progress": progress,
            "signal": np.r_[np.tile([9.0, 10.0, 11.0], 10), np.full(42, 14.0)],
            "signal__imputed": False,
            "heating_capacity": np.r_[np.full(30, 5.0), np.linspace(5.0, 4.0, 42)],
            "heating_capacity__imputed": False,
            "cop": np.r_[np.full(30, 3.0), np.linspace(3.0, 2.5, 42)],
            "cop__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal": {
            "analysis_candidate": True,
            "kind": "continuous",
            "role": "sensor",
            "expected_frost_direction": "decrease",
        },
        "heating_capacity": {
            "analysis_candidate": False,
            "kind": "continuous",
            "role": "performance",
        },
        "cop": {
            "analysis_candidate": False,
            "kind": "derived",
            "role": "performance",
            "dependencies": ["heating_capacity", "power_total"],
        },
    }

    bundle = build_evidence_bundle(frame, summary, settings, channels)

    metric = bundle.feature_cycle_metrics.iloc[0]
    assert metric["reference_source"] == "auto_cycle_initial_reference"
    assert metric["reference_valid_from"] == start + pd.Timedelta(minutes=5)
    assert metric["signed_sensitivity"] == pytest.approx(4.0 / metric["reference_scale"])

    future = bundle.future_association
    q_change = future.loc[
        future["target"].eq("heating_capacity")
        & future["target_type"].eq("future_change")
        & future["horizon_minutes"].eq(10)
    ].iloc[0]
    assert q_change["target_reference_source"] == "not_required"
    assert q_change["effect_metric"] == "within_cycle_spearman"
    assert set(future["feature"]) == {"signal"}
    assert set(bundle.feature_profile["feature"]) == {"signal"}


def test_auto_reference_uses_complete_expected_grid() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=30, freq="10s").delete(7)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": "frost_development",
            "signal": 1.0,
            "signal__imputed": False,
        }
    )

    reference = resolve_analysis_reference(
        frame,
        channel="signal",
        cycle_start=start,
        formal_frost_start=start,
        formal_frost_end=start + pd.Timedelta(minutes=6),
        configured_baseline_available=False,
        configured_residual=None,
        configured_baseline_mask=None,
        window_minutes=5,
        minimum_observed_fraction=0.8,
        maximum_gap_seconds=60,
        interval_seconds=10,
    )

    assert reference.source == "auto_cycle_initial_reference"
    assert reference.observed_fraction == pytest.approx(29 / 30)


def test_configured_reference_is_centered_once_and_wins() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=42, freq="10s")
    raw = np.r_[np.full(6, 10.0), np.full(30, 11.0), np.full(6, 15.0)]
    configured_residual = raw - 10.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": ["recovery"] * 6 + ["frost_development"] * 36,
            "signal": raw,
            "signal__imputed": False,
            "signal__baseline_residual": configured_residual,
        }
    )
    baseline_mask = frame["timestamp"].lt(start + pd.Timedelta(minutes=1))

    reference = resolve_analysis_reference(
        frame,
        channel="signal",
        cycle_start=start,
        formal_frost_start=start + pd.Timedelta(minutes=1),
        formal_frost_end=start + pd.Timedelta(minutes=7),
        configured_baseline_available=True,
        configured_residual=frame["signal__baseline_residual"],
        configured_baseline_mask=baseline_mask,
        window_minutes=5,
        minimum_observed_fraction=0.8,
        maximum_gap_seconds=60,
        interval_seconds=10,
    )

    assert reference.source == "configured_baseline"
    assert reference.center == pytest.approx(0.0)
    assert reference.residual.iloc[-1] == pytest.approx(5.0)
    assert reference.valid_from == start + pd.Timedelta(minutes=1)


def test_configured_reference_valid_from_uses_baseline_end_on_sparse_baseline() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=42, freq="10s")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cycle_stage": "frost_development",
            "signal": 10.0,
            "signal__imputed": False,
            "signal__baseline_residual": 0.0,
        }
    )
    baseline_mask = frame["timestamp"].eq(start + pd.Timedelta(seconds=50))

    reference = resolve_analysis_reference(
        frame,
        channel="signal",
        cycle_start=start,
        formal_frost_start=start,
        formal_frost_end=start + pd.Timedelta(minutes=7),
        configured_baseline_available=True,
        configured_residual=frame["signal__baseline_residual"],
        configured_baseline_mask=baseline_mask,
        configured_baseline_end=start + pd.Timedelta(minutes=1, seconds=3),
        window_minutes=5,
        minimum_observed_fraction=0.8,
        maximum_gap_seconds=60,
        interval_seconds=10,
    )

    assert reference.valid_from == start + pd.Timedelta(minutes=1, seconds=10)


def test_future_change_does_not_require_target_reference() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=150, freq="10s")
    target = np.linspace(5.0, 3.0, len(timestamps))
    target[:30] = np.nan
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": np.linspace(1.0, 4.0, len(timestamps)),
            "signal__imputed": False,
            "heating_capacity": target,
            "heating_capacity__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=25)],
        }
    )
    channels = {
        "signal": {
            "analysis_candidate": True,
            "role": "sensor",
            "expected_frost_direction": "decrease",
        },
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
    }
    policy = EvidencePolicy(min_valid_pairs=5, min_segment_points=5, min_valid_cycles=1)
    bundle = build_evidence_bundle(frame, summary, policy, channels)
    future = bundle.future_association
    change = future.loc[
        future["target_type"].eq("future_change") & future["horizon_minutes"].eq(5)
    ].iloc[0]
    level = future.loc[
        future["target_type"].eq("future_level") & future["horizon_minutes"].eq(5)
    ].iloc[0]
    assert change["target_reference_source"] == "not_required"
    assert change["exclusion_reason"] != "target_reference_unavailable"
    assert level["target_reference_source"] == "unavailable"


def test_evidence_writer_emits_five_tables_and_manifest(tmp_path: Path) -> None:
    settings = load_evidence_settings(_defaults(tmp_path), allow_date_config=False)
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": np.linspace(1.0, 2.0, len(timestamps)),
            "signal__imputed": False,
            "heating_capacity": np.linspace(5.0, 4.0, len(timestamps)),
            "heating_capacity__imputed": False,
            "cop": np.linspace(3.0, 2.5, len(timestamps)),
            "cop__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal": {
            "analysis_candidate": True,
            "role": "sensor",
            "expected_frost_direction": "decrease",
        },
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
        "cop": {"analysis_candidate": False, "role": "performance"},
    }
    bundle = build_evidence_bundle(frame, summary, settings, channels)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "evidence"
    write_evidence_outputs(bundle, output, run_dir, settings=settings, overwrite=False)

    expected = {
        "cycle_eligibility.csv",
        "feature_cycle_metrics.csv",
        "future_association.csv",
        "feature_profile.csv",
        "feature_pair_similarity.csv",
        "evidence_manifest.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    manifest = yaml.safe_load((output / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_row_counts"]["feature_profile.csv"] == 1


def test_expected_grid_contains_only_complete_stage_buckets() -> None:
    start = pd.Timestamp("2026-07-15 00:00:03")
    end = pd.Timestamp("2026-07-15 00:05:03")

    grid = expected_grid(start, end, 10)

    assert grid[0] == pd.Timestamp("2026-07-15 00:00:10")
    assert grid[-1] == pd.Timestamp("2026-07-15 00:04:50")


def test_bundle_rejects_nonpositive_grid_interval() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": np.linspace(1.0, 2.0, len(timestamps)),
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {"signal": {"analysis_candidate": True, "role": "sensor"}}

    with pytest.raises(ValueError, match="grid_interval_seconds"):
        build_evidence_bundle(
            frame,
            summary,
            EvidencePolicy(horizons_minutes=(5,), min_segment_points=2),
            channels,
            grid_interval_seconds=0,
        )


def test_trend_metrics_survive_unavailable_reference(tmp_path: Path) -> None:
    settings = EvidencePolicy(
        min_segment_coverage=0.5,
        min_segment_points=5,
        min_pair_coverage=0.5,
        min_valid_pairs=5,
        min_valid_cycles=1,
        horizons_minutes=(5,),
    )
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    signal = np.linspace(1.0, 8.0, len(timestamps))
    signal[:30] = np.nan
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": signal,
            "signal__imputed": False,
            "heating_capacity": np.linspace(5.0, 4.0, len(timestamps)),
            "heating_capacity__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal": {"analysis_candidate": True, "role": "sensor"},
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
    }

    bundle = build_evidence_bundle(frame, summary, settings, channels)
    metric = bundle.feature_cycle_metrics.iloc[0]

    assert metric["reference_source"] == "unavailable"
    assert np.isfinite(metric["global_spearman"])
    assert np.isfinite(metric["late_slope_per_min"])
    assert metric["metric_status"] == "available"
    assert metric["metric_exclusion_reason"] == ""


def test_past_slope_future_variant_does_not_require_feature_reference() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=150, freq="10s")
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": ["frost_development"] * 144 + ["defrost"] * 6,
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": np.r_[np.full(25, np.nan), np.linspace(1.0, 5.0, 125)],
            "signal__imputed": False,
            "heating_capacity": np.linspace(5.0, 3.0, len(timestamps)),
            "heating_capacity__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=24)],
        }
    )
    channels = {
        "signal": {"analysis_candidate": True, "role": "sensor"},
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
    }
    policy = EvidencePolicy(
        min_segment_coverage=0.5,
        min_segment_points=5,
        min_pair_coverage=0.5,
        min_valid_pairs=5,
        min_valid_cycles=1,
        horizons_minutes=(5,),
    )

    bundle = build_evidence_bundle(frame, summary, policy, channels)
    row = bundle.future_association.loc[
        bundle.future_association["feature_variant"].eq("past_slope_5min")
        & bundle.future_association["target_type"].eq("future_change")
    ].iloc[0]

    assert row["feature_reference_source"] == "not_required"
    assert row["metric_status"] == "available"


def test_target_onset_excludes_process_imputed_values() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    target = np.r_[np.arange(30, dtype=float), np.full(42, 100.0)]
    imputed = np.r_[np.zeros(30, dtype=bool), np.ones(42, dtype=bool)]
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "heating_capacity": target,
            "heating_capacity__imputed": imputed,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
            "baseline_status": ["unavailable"],
        }
    )
    cycle = build_cycle_slices(frame, summary, 10)[0]
    policy = EvidencePolicy(
        min_segment_coverage=0.5,
        min_segment_points=5,
        min_pair_coverage=0.5,
        min_valid_pairs=5,
        min_valid_cycles=1,
        horizons_minutes=(5,),
        onset_window_seconds=10,
        onset_persistence_seconds=20,
    )

    cache = build_channel_evidence(
        cycle,
        "heating_capacity",
        policy,
        target=True,
        interval_seconds=10,
    )

    assert cache.target_valid.iloc[30:].eq(False).all()
    assert np.isnan(cache.onset_elapsed_minutes)


def test_cycle_status_allows_only_known_open_end_incomplete_reason() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=12, freq="10s")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 36,
            "experiment_date": ["2026-07-15"] * 36,
            "cycle_id": ["valid"] * 12 + ["open"] * 12 + ["gap"] * 12,
            "cycle_stage": ["frost_development"] * 36,
            "timestamp": list(timestamps) * 3,
            "cycle_status": ["valid"] * 12 + ["incomplete"] * 12 + ["incomplete"] * 12,
            "cycle_status_reason": [""] * 12
            + ["defrost_end_not_observed"] * 12
            + ["defrost_state_gap"] * 12,
            "signal": 1.0,
            "signal__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 3,
            "experiment_date": ["2026-07-15"] * 3,
            "cycle_id": ["valid", "open", "gap"],
            "cycle_status": ["valid", "incomplete", "incomplete"],
            "cycle_status_reason": ["", "defrost_end_not_observed", "defrost_state_gap"],
            "stable_heating_start": [start] * 3,
            "defrost_start": [start + pd.Timedelta(minutes=2)] * 3,
        }
    )

    cycles = build_cycle_slices(frame, summary, 10)

    statuses = {cycle.key[2]: (cycle.eligible, cycle.eligibility_status) for cycle in cycles}
    assert statuses["valid"] == (True, "eligible")
    assert statuses["open"] == (True, "eligible_exploratory")
    assert statuses["gap"] == (False, "excluded")
    gap_cache = build_channel_evidence(
        next(cycle for cycle in cycles if cycle.key[2] == "gap"),
        "signal",
        EvidencePolicy(horizons_minutes=(5,)),
        target=False,
        interval_seconds=10,
    )
    assert gap_cache.reference.source == "unavailable"
    assert gap_cache.onset_elapsed_minutes != gap_cache.onset_elapsed_minutes


def test_load_evidence_runs_returns_contract_and_rejects_duplicate_path(tmp_path: Path) -> None:
    channels_path = tmp_path / "channels.yaml"
    channels_path.write_text("signal: {}\n", encoding="utf-8")
    registry_hash = optional_sha256(channels_path)
    assert registry_hash is not None

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=12, freq="10s")
    processed = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "cycle_status_reason": "",
            "timestamp": timestamps,
            "cycle_progress": np.linspace(0.0, 0.9, len(timestamps)),
            "cycle_elapsed_seconds": np.arange(len(timestamps)) * 10.0,
            "signal__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "cycle_status_reason": [""],
            "baseline_status": ["unavailable"],
            "baseline_failure_reason": [""],
            "baseline_reference_type": ["cycle_local_early_stable_proxy"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=2)],
        }
    )
    processed.to_parquet(run_dir / "processed_data.parquet", index=False)
    summary.to_csv(run_dir / "cycle_summary.csv", index=False)
    manifest = {
        "experiment_id": "exp_test",
        "experiment_date": "2026-07-15",
        "config_provenance": {"channels_sha256": registry_hash},
        "resolved_config": {
            "process": {
                "resample_interval_seconds": 10,
                "baseline": {"window_minutes": 5},
            }
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_evidence_runs([run_dir], registry_hash=registry_hash)

    assert loaded.grid_interval_seconds == 10
    assert loaded.run_contracts[0].baseline_reference_type == "cycle_local_early_stable_proxy"
    with pytest.raises(ValueError, match="duplicate resolved run"):
        load_evidence_runs([run_dir, run_dir], registry_hash=registry_hash)
