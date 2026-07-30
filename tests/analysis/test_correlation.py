from __future__ import annotations

import pandas as pd

from frost_analysis.analysis.correlation import build_correlation_results
from frost_analysis.data.registry import FeatureSpec


def test_correlation_results_have_no_weighted_rank() -> None:
    rows: list[dict[str, object]] = []
    cycles: list[dict[str, object]] = []
    for cycle_number in range(1, 4):
        cycle_id = f"cycle_{cycle_number:03d}"
        start = pd.Timestamp("2026-07-15") + pd.Timedelta(days=cycle_number)
        for point in range(8):
            rows.append(
                {
                    "timestamp": start + pd.Timedelta(minutes=point),
                    "cycle_id": cycle_id,
                    "cycle_status": "valid",
                    "cycle_stage": "frost_development",
                    "cycle_progress": point / 7,
                    "cycle_gap_contaminated": False,
                    "signal": float(point),
                    "signal__baseline_offset": float(point),
                    "heating_capacity": 10.0 - point * 0.1,
                    "power_total": 3.0,
                    "cop": 3.3 - point * 0.03,
                }
            )
        cycles.append(
            {
                "cycle_id": cycle_id,
                "cycle_status": "valid",
                "heating_start": start,
                "defrost_start": start + pd.Timedelta(minutes=7),
                "defrost_end": start + pd.Timedelta(minutes=8),
                "clean_end": start,
                "max_sensor_gap_seconds": 1.0,
            }
        )
    spec = FeatureSpec(
        feature_id="signal",
        canonical_name="signal",
        raw_source="signal",
        meaning_zh="测试信号",
        physical_family="evaporator_response",
        source_type="measured",
        unit="unknown",
        formula="",
        data_role="X",
        availability="current_history",
        deployment_status="confirmed",
        confidence="high",
        primary_or_validation="primary",
        analysis_enabled=True,
        notes="",
    )
    result = build_correlation_results(
        pd.DataFrame(rows),
        pd.DataFrame(cycles),
        {"signal": spec},
        methods=["spearman"],
        lags_minutes=[0, 5],
        targets=["cop"],
        minimum_cycles=3,
    )
    assert len(result) == 1
    assert "candidate_score" not in result
    assert "rank" not in result
    assert result.iloc[0]["valid_cycle_count"] == 3
