from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from frost_analysis.config import EvidencePolicy, load_evidence_settings
from frost_analysis.evidence import build_evidence_bundle, resolve_analysis_reference
from frost_analysis.io import write_evidence_outputs


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
    assert settings.horizons_minutes == (5, 10, 20)
    assert settings.targets == ("heating_capacity", "cop")
    assert settings.auto_reference_window_minutes == 5


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
