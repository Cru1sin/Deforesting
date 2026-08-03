from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from frost_analysis.config import EvidencePolicy, load_evidence_settings
from frost_analysis.config import is_iso_date as config_is_iso_date
from frost_analysis.evidence import (
    FEATURE_PAIR_SIMILARITY_COLUMNS,
    FEATURE_PROFILE_COLUMNS,
    build_evidence_bundle,
    resolve_analysis_reference,
)
from frost_analysis.evidence_cycle import (
    CycleChannelEvidence,
    CycleSlice,
    ResolvedReference,
    build_channel_evidence,
    build_cycle_slices,
    expected_grid,
)
from frost_analysis.evidence_summary import aggregate_feature_profiles, compute_pair_similarity
from frost_analysis.io import is_iso_date as io_is_iso_date
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

    bundle = build_evidence_bundle(frame, summary, settings, channels, grid_interval_seconds=10)

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
    bundle = build_evidence_bundle(frame, summary, policy, channels, grid_interval_seconds=10)
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
    bundle = build_evidence_bundle(frame, summary, settings, channels, grid_interval_seconds=10)
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


def test_build_evidence_bundle_requires_grid_interval() -> None:
    parameter = inspect.signature(build_evidence_bundle).parameters["grid_interval_seconds"]

    assert parameter.default is inspect.Parameter.empty


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


def test_segment_observed_fraction_excludes_process_imputation() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=72, freq="10s")
    imputed = np.zeros(len(timestamps), dtype=bool)
    imputed[1] = True
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": "frost_development",
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": np.linspace(1.0, 2.0, len(timestamps)),
            "signal__imputed": imputed,
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
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal": {"analysis_candidate": True, "role": "sensor"},
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
    }

    bundle = build_evidence_bundle(
        frame,
        summary,
        EvidencePolicy(min_segment_points=5, horizons_minutes=(5,)),
        channels,
        grid_interval_seconds=10,
    )
    metric = bundle.feature_cycle_metrics.iloc[0]

    assert metric["early_observed_fraction"] == pytest.approx(17 / 18)
    assert np.isfinite(metric["early_slope_per_min"])


def test_past_slope_requires_no_remaining_nan_in_fixed_window() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    timestamps = pd.date_range(start, periods=150, freq="10s")
    signal = np.linspace(1.0, 8.0, len(timestamps))
    signal[100] = np.nan
    frame = pd.DataFrame(
        {
            "experiment_id": "exp_test",
            "experiment_date": "2026-07-15",
            "cycle_id": "cycle_001",
            "cycle_stage": ["frost_development"] * 144 + ["defrost"] * 6,
            "cycle_status": "valid",
            "timestamp": timestamps,
            "signal": signal,
            "signal__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=24)],
        }
    )
    cycle = build_cycle_slices(frame, summary, 10)[0]
    cache = build_channel_evidence(
        cycle,
        "signal",
        EvidencePolicy(horizons_minutes=(5,)),
        target=False,
        interval_seconds=10,
    )

    assert cache.past_slope_5min is not None
    assert pd.isna(cache.past_slope_5min.iloc[100])
    assert pd.isna(cache.past_slope_5min.iloc[129])
    assert np.isfinite(cache.past_slope_5min.iloc[130])


def test_missing_grid_row_does_not_reduce_future_anchor_denominator() -> None:
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
            "signal": np.linspace(1.0, 4.0, len(timestamps)),
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
            "baseline_status": ["unavailable"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=12)],
        }
    )
    channels = {
        "signal": {"analysis_candidate": True, "role": "sensor"},
        "heating_capacity": {"analysis_candidate": False, "role": "performance"},
    }
    policy = EvidencePolicy(
        min_segment_points=5,
        min_valid_pairs=1,
        min_pair_coverage=0.5,
        horizons_minutes=(5,),
    )
    complete = build_evidence_bundle(frame, summary, policy, channels, grid_interval_seconds=10)
    missing = build_evidence_bundle(
        frame.drop(index=40), summary, policy, channels, grid_interval_seconds=10
    )

    selector = (
        (complete.future_association["feature_variant"] == "residual_level")
        & (complete.future_association["target"] == "heating_capacity")
        & (complete.future_association["target_type"] == "future_change")
        & (complete.future_association["horizon_minutes"] == 5)
    )
    complete_row = complete.future_association.loc[selector].iloc[0]
    missing_row = missing.future_association.loc[selector].iloc[0]

    assert missing_row["expected_anchor_count"] == complete_row["expected_anchor_count"]
    assert missing_row["valid_pairs"] < complete_row["valid_pairs"]
    assert missing_row["pair_coverage"] < complete_row["pair_coverage"]


def test_summary_only_and_stage_boundary_mismatch_are_excluded() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 3,
            "experiment_date": ["2026-07-15"] * 3,
            "cycle_id": ["bad_stage"] * 3,
            "cycle_stage": ["frost_development"] * 3,
            "cycle_status": ["valid"] * 3,
            "timestamp": [
                start - pd.Timedelta(seconds=10),
                start,
                start + pd.Timedelta(minutes=1, seconds=10),
            ],
            "signal": [1.0, 2.0, 3.0],
            "signal__imputed": False,
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "experiment_date": ["2026-07-15", "2026-07-15"],
            "cycle_id": ["bad_stage", "summary_only"],
            "cycle_status": ["valid", "valid"],
            "stable_heating_start": [start, start],
            "defrost_start": [start + pd.Timedelta(minutes=1), start + pd.Timedelta(minutes=1)],
        }
    )
    channels = {"signal": {"analysis_candidate": True, "role": "sensor"}}

    bundle = build_evidence_bundle(
        frame,
        summary,
        EvidencePolicy(min_segment_points=2, horizons_minutes=(5,)),
        channels,
        grid_interval_seconds=10,
    )
    eligibility = bundle.cycle_eligibility.set_index("cycle_id")
    metrics = bundle.feature_cycle_metrics.set_index("cycle_id")

    assert eligibility.loc["bad_stage", "exclusion_reason"] == "frost_stage_boundary_mismatch"
    assert eligibility.loc["summary_only", "exclusion_reason"] == "processed_cycle_unavailable"
    assert metrics.loc["bad_stage", "metric_status"] == "unavailable"
    assert metrics.loc["summary_only", "metric_status"] == "unavailable"


def test_build_cycle_slices_normalizes_mixed_cycle_key_types() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test", "exp_test"],
            "experiment_date": ["2026-07-15", "2026-07-15"],
            "cycle_id": [1, "1"],
            "cycle_stage": ["frost_development", "frost_development"],
            "timestamp": [start, start + pd.Timedelta(seconds=10)],
            "signal": [1.0, 2.0],
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["1"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=1)],
        }
    )

    cycles = build_cycle_slices(frame, summary, 10)

    assert len(cycles) == 1
    assert cycles[0].key == ("exp_test", "2026-07-15", "1")
    assert len(cycles[0].frost) == 2


def test_in_stage_non_grid_timestamp_excludes_cycle() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 3,
            "experiment_date": ["2026-07-15"] * 3,
            "cycle_id": ["cycle_001"] * 3,
            "cycle_stage": ["frost_development"] * 3,
            "cycle_status": ["valid"] * 3,
            "timestamp": [
                start,
                start + pd.Timedelta(seconds=5),
                start + pd.Timedelta(seconds=10),
            ],
            "signal": [1.0, 2.0, 3.0],
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(minutes=1)],
        }
    )
    channels = {"signal": {"analysis_candidate": True, "role": "sensor"}}
    policy = EvidencePolicy(
        min_segment_points=2,
        min_valid_pairs=1,
        min_valid_cycles=1,
        horizons_minutes=(5,),
        primary_horizon_minutes=5,
        targets=("heating_capacity",),
        primary_target="heating_capacity",
        lead_target="heating_capacity",
    )

    cycles = build_cycle_slices(frame, summary, interval_seconds=10)
    assert cycles[0].eligible is False
    assert cycles[0].exclusion_reason == "frost_stage_grid_mismatch"

    bundle = build_evidence_bundle(
        frame,
        summary,
        policy,
        channels,
        grid_interval_seconds=10,
    )

    eligibility = bundle.cycle_eligibility.iloc[0]
    metric = bundle.feature_cycle_metrics.iloc[0]
    assert eligibility["eligibility_status"] == "excluded"
    assert eligibility["exclusion_reason"] == "frost_stage_grid_mismatch"
    assert metric["metric_status"] == "unavailable"


def test_empty_frost_grid_excludes_all_evidence_paths() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    frame = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_stage": ["frost_development"],
            "cycle_status": ["valid"],
            "timestamp": [start],
            "signal_a": [1.0],
            "signal_b": [2.0],
            "heating_capacity": [3.0],
        }
    )
    summary = pd.DataFrame(
        {
            "experiment_id": ["exp_test"],
            "experiment_date": ["2026-07-15"],
            "cycle_id": ["cycle_001"],
            "cycle_status": ["valid"],
            "stable_heating_start": [start],
            "defrost_start": [start + pd.Timedelta(seconds=5)],
        }
    )
    channels = {
        "signal_a": {"analysis_candidate": True, "role": "sensor"},
        "signal_b": {"analysis_candidate": True, "role": "sensor"},
    }
    policy = EvidencePolicy(
        min_segment_points=2,
        min_valid_pairs=1,
        min_valid_cycles=1,
        horizons_minutes=(5,),
        primary_horizon_minutes=5,
        targets=("heating_capacity",),
        primary_target="heating_capacity",
        lead_target="heating_capacity",
    )

    cycles = build_cycle_slices(frame, summary, interval_seconds=10)
    assert cycles[0].eligible is False
    assert cycles[0].exclusion_reason == "no_complete_frost_grid_bucket"
    assert cycles[0].grid_coverage == 0.0

    bundle = build_evidence_bundle(
        frame,
        summary,
        policy,
        channels,
        grid_interval_seconds=10,
    )
    eligibility = bundle.cycle_eligibility
    metrics = bundle.feature_cycle_metrics
    future = bundle.future_association
    pair = bundle.feature_pair_similarity

    assert eligibility["eligible_feature_count"].eq(0).all()
    assert eligibility["exclusion_reason"].eq("no_complete_frost_grid_bucket").all()
    assert metrics["metric_status"].eq("unavailable").all()
    assert metrics["metric_exclusion_reason"].eq(
        "no_complete_frost_grid_bucket"
    ).all()
    assert future["metric_status"].eq("unavailable").all()
    assert future["exclusion_reason"].eq("no_complete_frost_grid_bucket").all()
    assert pair["evaluated_cycle_count"].eq(0).all()
    assert pair["valid_cycle_count"].eq(0).all()
    assert pair["similarity_status"].eq("no_valid_evidence").all()


def test_profile_trend_median_is_date_balanced() -> None:
    metrics = pd.DataFrame(
        {
            "experiment_id": ["exp_test"] * 4,
            "experiment_date": ["2026-07-16"] * 3 + ["2026-07-17"],
            "cycle_id": ["cycle_001", "cycle_002", "cycle_003", "cycle_004"],
            "feature": ["signal"] * 4,
            "cycle_eligible": [True] * 4,
            "reference_source": ["configured_baseline"] * 4,
            "global_spearman": [1.0, 1.0, 1.0, -1.0],
            "signed_sensitivity": [1.0] * 4,
            "onset_elapsed_minutes": [1.0] * 4,
        }
    )
    future = pd.DataFrame(
        columns=[
            "experiment_id",
            "experiment_date",
            "cycle_id",
            "feature",
            "feature_variant",
            "target",
            "target_type",
            "horizon_minutes",
            "effect",
            "lead_time_minutes",
        ]
    )
    policy = EvidencePolicy(min_valid_cycles=1)
    channels = {"signal": {"role": "sensor", "expected_frost_direction": "increase"}}

    profile = aggregate_feature_profiles(
        metrics,
        future,
        ["signal"],
        channels,
        policy,
        FEATURE_PROFILE_COLUMNS,
    )

    row = profile.iloc[0]
    assert row["trend_valid_cycle_count"] == 4
    assert row["trend_valid_date_count"] == 2
    assert row["global_spearman_median"] == pytest.approx(0.0)


def test_pair_coverage_audit_separates_evaluated_and_valid_cycles() -> None:
    grid = pd.date_range("2026-07-16 00:00:00", periods=40, freq="10s")
    reference = ResolvedReference(
        residual=pd.Series(np.nan, index=grid),
        source="unavailable",
        center=np.nan,
        scale=np.nan,
        observed_fraction=0.0,
        valid_from=grid[0],
        exclusion_reason="test",
    )

    def cache(values: list[float | None]) -> CycleChannelEvidence:
        slopes = pd.Series(values, index=grid, dtype=float)
        return CycleChannelEvidence(
            values=pd.Series(np.nan, index=grid),
            imputed=pd.Series(False, index=grid),
            target_valid=pd.Series(False, index=grid),
            reference=reference,
            analysis_residual=pd.Series(np.nan, index=grid),
            onset_elapsed_minutes=np.nan,
            onset_progress=np.nan,
            past_slope_5min=slopes,
        )

    def cycle(cycle_id: str, experiment_date: str) -> CycleSlice:
        return CycleSlice(
            key=("exp_test", experiment_date, cycle_id),
            frame=pd.DataFrame(),
            frost=pd.DataFrame(),
            grid=grid,
            summary=pd.Series(dtype=object),
            start=grid[0],
            end=grid[-1] + pd.Timedelta(seconds=10),
            grid_coverage=1.0,
            cycle_status="valid",
            cycle_status_reason=None,
            eligible=True,
            exclusion_reason=None,
        )

    full = [None] * 30 + list(np.arange(10, dtype=float))
    partial = [None] * 30 + [1.0, 2.0, 3.0, 4.0] + [None] * 6
    cycles = [cycle("cycle_001", "2026-07-16"), cycle("cycle_002", "2026-07-17")]
    caches = {
        (cycles[0].key, "signal_a"): cache(full),
        (cycles[0].key, "signal_b"): cache(full),
        (cycles[1].key, "signal_a"): cache(partial),
        (cycles[1].key, "signal_b"): cache(partial),
    }
    policy = EvidencePolicy(
        min_pair_coverage=0.8,
        min_valid_pairs=3,
        min_valid_cycles=1,
    )
    channels = {"signal_a": {}, "signal_b": {}}

    pairs = compute_pair_similarity(
        cycles,
        ["signal_a", "signal_b"],
        caches,
        channels,
        policy,
        FEATURE_PAIR_SIMILARITY_COLUMNS,
        interval_seconds=10,
    )

    row = pairs.iloc[0]
    assert row["evaluated_cycle_count"] == 2
    assert row["valid_cycle_count"] == 1
    assert row["valid_date_count"] == 1
    assert row["pair_coverage_median"] == pytest.approx(0.7)


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

    bundle = build_evidence_bundle(frame, summary, settings, channels, grid_interval_seconds=10)
    metric = bundle.feature_cycle_metrics.iloc[0]

    assert metric["reference_source"] == "unavailable"
    assert metric["reference_exclusion_reason"] == "reference_observed_coverage"
    assert np.isfinite(metric["global_spearman"])
    assert np.isfinite(metric["late_slope_per_min"])
    assert metric["metric_status"] == "available"
    assert metric["metric_exclusion_reason"] == ""
    assert bundle.cycle_eligibility.iloc[0]["eligible_feature_count"] == 1


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

    bundle = build_evidence_bundle(frame, summary, policy, channels, grid_interval_seconds=10)
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


def test_iso_date_validation_has_one_shared_implementation() -> None:
    assert io_is_iso_date is config_is_iso_date
    assert io_is_iso_date("2026-07-15")
    assert not io_is_iso_date("2026-7-15")


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
