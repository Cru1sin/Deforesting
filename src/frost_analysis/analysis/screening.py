"""Channel-level evidence checks for the candidate-channel phase.

The module deliberately reports evidence components instead of a weighted rank.
The physical channel is the unit of interpretation; rolling windows are only
calculation details and never become independent observations.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def screen_candidate_channels(
    frame: pd.DataFrame,
    cycles: pd.DataFrame,
    registry: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return one evidence row per Registry channel, without a composite score."""
    channels = registry.copy()
    if "analysis_enabled" in channels:
        channels = channels.loc[channels["analysis_enabled"].fillna(False).astype(bool)]
    roles = (
        channels["data_role"] if "data_role" in channels else pd.Series("X", index=channels.index)
    )
    channels = channels.loc[roles.eq("X")].copy()
    eligible = _eligible_rows(frame, cycles)
    rows: list[dict[str, object]] = []
    for item in channels.itertuples(index=False):
        name = str(getattr(item, "canonical_name", getattr(item, "feature_id", "")))
        signal_name = _signal_name(eligible, name)
        if not signal_name:
            rows.append(_unavailable_row(item, "insufficient_coverage"))
            continue
        signal = pd.to_numeric(eligible[signal_name], errors="coerce")
        trend = _trend_evidence(eligible, signal_name)
        context = _context_evidence(
            eligible,
            signal,
            minimum_cycles=int(config.get("minimum_valid_cycles", 2)),
        )
        lagged = _lagged_evidence(eligible, signal_name, config)
        reset = _reset_evidence(frame, cycles, signal_name, config)
        valid_cycles = int(cast(Any, trend["valid_cycle_count"]))
        status = _status(
            valid_cycles=valid_cycles,
            minimum_cycles=int(cast(Any, config.get("minimum_valid_cycles", 2))),
            reset_pairs=int(cast(Any, reset["reset_pair_count"])),
            context_confounded=bool(context["context_confounded"]),
            trend_available=bool(trend["trend_available"]),
            coverage=float(signal.notna().mean()),
            deployment_status=str(getattr(item, "deployment_status", "pending")),
            primary_or_validation=str(getattr(item, "primary_or_validation", "primary")),
        )
        rows.append(
            {
                "feature_id": str(getattr(item, "feature_id", name)),
                "canonical_name": name,
                "meaning_zh": str(getattr(item, "meaning_zh", "")),
                "physical_family": str(getattr(item, "physical_family", "unclassified")),
                "source_type": str(getattr(item, "source_type", "")),
                "unit": str(getattr(item, "unit", "unknown")),
                "primary_or_validation": str(getattr(item, "primary_or_validation", "primary")),
                "deployment_status": str(getattr(item, "deployment_status", "pending")),
                "signal_column": signal_name,
                "observed_count": int(signal.notna().sum()),
                "coverage": float(signal.notna().mean()),
                "valid_cycle_count": valid_cycles,
                "trend_direction": trend["trend_direction"],
                "trend_median_spearman": trend["trend_median_spearman"],
                "trend_direction_consistency": trend["trend_direction_consistency"],
                "early_late_effect": trend["early_late_effect"],
                "context_max_abs_spearman": context["context_max_abs_spearman"],
                "context_dominant_field": context["context_dominant_field"],
                "context_valid_cycle_count": context["context_valid_cycle_count"],
                "context_confounded": context["context_confounded"],
                "lag_valid_cycle_count": lagged["lag_valid_cycle_count"],
                "lag_q_5m": lagged["lag_q_5m"],
                "lag_q_10m": lagged["lag_q_10m"],
                "lag_power_5m": lagged["lag_power_5m"],
                "lag_power_10m": lagged["lag_power_10m"],
                "lag_cop_5m": lagged["lag_cop_5m"],
                "lag_cop_10m": lagged["lag_cop_10m"],
                "reset_pair_count": int(cast(Any, reset["reset_pair_count"])),
                "reset_direction_consistency": reset["reset_direction_consistency"],
                "reset_median_effect": reset["reset_median_effect"],
                "candidate_status": status,
                "risk": _risk(str(getattr(item, "physical_family", "")), context, reset),
                "next_validation": _next_validation(
                    status, int(cast(Any, reset["reset_pair_count"]))
                ),
            }
        )
    columns = _evidence_columns()
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["candidate_status", "physical_family", "canonical_name"], kind="stable")
        .reset_index(drop=True)
    )


def _eligible_rows(frame: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    quality = (
        result["cycle_quality"]
        if "cycle_quality" in result
        else pd.Series(False, index=result.index)
    )
    stage = result["stage"] if "stage" in result else pd.Series("", index=result.index)
    mask = quality.eq("complete")
    mask &= stage.isin(["stable_clean", "frost_development"])
    if "is_heating" in result:
        mask &= result["is_heating"].fillna(False).astype(bool)
    if "cycle_gap_contaminated" in result:
        mask &= ~result["cycle_gap_contaminated"].fillna(False).astype(bool)
    if "analysis_bin_available" in result:
        mask &= result["analysis_bin_available"].fillna(False).astype(bool)
    complete_ids = set(
        cycles.loc[
            (
                cycles["quality_flag"]
                if "quality_flag" in cycles
                else pd.Series(False, index=cycles.index)
            ).eq("complete"),
            "cycle_id",
        ].astype(str)
    )
    if "cycle_id" in result:
        mask &= result["cycle_id"].astype(str).isin(complete_ids)
    return result.loc[mask].sort_values(["cycle_id", "sensor_time"], kind="stable")


def _signal_name(frame: pd.DataFrame, name: str) -> str:
    offset = f"{name}__baseline_offset"
    if offset in frame and pd.to_numeric(frame[offset], errors="coerce").notna().any():
        return offset
    if name in frame and pd.to_numeric(frame[name], errors="coerce").notna().any():
        return name
    return ""


def _trend_evidence(frame: pd.DataFrame, signal_name: str) -> dict[str, object]:
    values: list[float] = []
    effects: list[float] = []
    for _, group in frame.groupby("cycle_id", sort=False):
        current = pd.to_numeric(group[signal_name], errors="coerce")
        phase_source = (
            group["cycle_phase"] if "cycle_phase" in group else pd.Series(np.nan, index=group.index)
        )
        phase = pd.to_numeric(phase_source, errors="coerce")
        valid = current.notna() & phase.notna()
        if int(valid.sum()) < 3 or current.loc[valid].nunique() < 2:
            continue
        rho = _rho(phase.loc[valid], current.loc[valid])
        if np.isfinite(rho):
            values.append(rho)
        early = current.loc[valid & phase.le(0.2)].dropna()
        late = current.loc[valid & phase.ge(0.8)].dropna()
        if len(early) >= 2 and len(late) >= 2:
            scale = max(_robust_scale(pd.concat([early, late])), 1e-12)
            effects.append(float((late.median() - early.median()) / scale))
    if not values:
        return {
            "valid_cycle_count": 0,
            "trend_direction": "undetermined",
            "trend_median_spearman": np.nan,
            "trend_direction_consistency": 0.0,
            "early_late_effect": np.nan,
            "trend_available": False,
        }
    signs = np.sign(values)
    positive = float(np.mean(signs > 0))
    negative = float(np.mean(signs < 0))
    return {
        "valid_cycle_count": len(values),
        "trend_direction": "positive" if positive >= negative else "negative",
        "trend_median_spearman": float(np.median(values)),
        "trend_direction_consistency": max(positive, negative),
        "early_late_effect": float(np.median(effects)) if effects else np.nan,
        "trend_available": True,
    }


def _context_evidence(
    frame: pd.DataFrame, signal: pd.Series, *, minimum_cycles: int
) -> dict[str, object]:
    context_fields = [
        "ambient_temperature",
        "water_in_temperature",
        "water_flow",
        "compressor_frequency_setpoint",
        "water_temperature_setpoint",
        "evaporating_temperature_setpoint",
        "target_subcooling",
        "compressor_frequency",
    ]
    values: list[tuple[str, float, int]] = []
    for name in context_fields:
        if name not in frame:
            continue
        cycle_values: list[float] = []
        for _, group in frame.groupby("cycle_id", sort=False):
            indices = group.index
            rho = _rho(
                signal.loc[indices],
                pd.to_numeric(group[name], errors="coerce"),
            )
            if np.isfinite(rho):
                cycle_values.append(abs(rho))
        if len(cycle_values) >= minimum_cycles:
            values.append((name, float(np.median(cycle_values)), len(cycle_values)))
    values.sort(key=lambda item: item[1], reverse=True)
    dominant = values[0] if values else ("", np.nan, 0)
    return {
        "context_max_abs_spearman": dominant[1],
        "context_dominant_field": dominant[0],
        "context_valid_cycle_count": dominant[2],
        "context_confounded": bool(np.isfinite(dominant[1]) and dominant[1] >= 0.8),
    }


def _lagged_evidence(
    frame: pd.DataFrame, signal_name: str, config: dict[str, Any]
) -> dict[str, object]:
    result: dict[str, object] = {}
    horizons = tuple(int(value) for value in config.get("lead_horizons_minutes", [5, 10]))
    minimum_cycles = int(config.get("minimum_valid_cycles", 2))
    lag_cycle_counts: list[int] = []
    for target, prefix in (("heating_capacity", "q"), ("power_total", "power"), ("cop", "cop")):
        for minutes in horizons:
            value, cycle_count = _future_rho(
                frame,
                signal_name,
                target,
                minutes,
                minimum_cycles=minimum_cycles,
                minimum_coverage=float(config.get("minimum_coverage", 0.7)),
            )
            result[f"lag_{prefix}_{minutes}m"] = value
            lag_cycle_counts.append(cycle_count)
    # The table keeps the stable two-horizon contract; extra configured horizons
    # remain internal and are intentionally not expanded into more CSV columns.
    for prefix in ("q", "power", "cop"):
        for minutes in (5, 10):
            result.setdefault(f"lag_{prefix}_{minutes}m", np.nan)
    result["lag_valid_cycle_count"] = max(lag_cycle_counts, default=0)
    return {
        key: result.get(key, np.nan)
        for key in (*_lag_columns(), "lag_valid_cycle_count")
    }


def _future_rho(
    frame: pd.DataFrame,
    signal_name: str,
    target: str,
    minutes: int,
    *,
    minimum_cycles: int = 2,
    minimum_coverage: float = 0.7,
) -> tuple[float, int]:
    if target not in frame or "sensor_time" not in frame:
        return np.nan, 0
    cycle_rhos: list[float] = []
    for _, group in frame.groupby("cycle_id", sort=False):
        available = pd.Series(True, index=group.index)
        if "analysis_bin_available" in group:
            available &= group["analysis_bin_available"].fillna(False).astype(bool)
        if "analysis_bin_coverage" in group:
            coverage = pd.to_numeric(group["analysis_bin_coverage"], errors="coerce")
            available &= coverage.ge(minimum_coverage)
        group = group.loc[available]
        source = group[["sensor_time", signal_name]].dropna().sort_values("sensor_time")
        future = group[["sensor_time", target]].dropna().sort_values("sensor_time")
        if source.empty or future.empty:
            continue
        requested = source[["sensor_time"]].copy()
        requested["future_time"] = requested["sensor_time"] + pd.Timedelta(minutes=minutes)
        matched = pd.merge_asof(
            requested.sort_values("future_time"),
            future.rename(columns={"sensor_time": "future_time", target: "future_value"}),
            on="future_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=15),
        )
        paired = pd.DataFrame(
            {"source": source[signal_name].to_numpy(), "target": matched["future_value"].to_numpy()}
        ).dropna()
        rho = _rho(paired["source"], paired["target"])
        if np.isfinite(rho):
            cycle_rhos.append(rho)
    if len(cycle_rhos) < minimum_cycles:
        return np.nan, len(cycle_rhos)
    return float(np.median(cycle_rhos)), len(cycle_rhos)


def _reset_evidence(
    frame: pd.DataFrame, cycles: pd.DataFrame, signal_name: str, config: dict[str, Any]
) -> dict[str, object]:
    quality = (
        cycles["quality_flag"] if "quality_flag" in cycles else pd.Series(False, index=cycles.index)
    )
    eligible = cycles.loc[quality.eq("complete")].sort_values("heating_start")
    window = pd.Timedelta(seconds=float(config.get("reset_window_seconds", 300)))
    effects: list[float] = []
    directions: list[bool] = []
    direction = str(_trend_evidence(_eligible_rows(frame, cycles), signal_name)["trend_direction"])
    for index in range(len(eligible) - 1):
        current, following = eligible.iloc[index], eligible.iloc[index + 1]
        if pd.isna(following.get("clean_end")):
            continue
        before = frame.loc[
            frame["cycle_id"].eq(current["cycle_id"])
            & frame["sensor_time"].between(
                pd.Timestamp(current["defrost_start"]) - window,
                pd.Timestamp(current["defrost_start"]),
                inclusive="left",
            ),
            signal_name,
        ]
        after = frame.loc[
            frame["cycle_id"].eq(following["cycle_id"])
            & frame["sensor_time"].between(
                pd.Timestamp(following["clean_end"]),
                pd.Timestamp(following["clean_end"]) + window,
                inclusive="left",
            ),
            signal_name,
        ]
        before = pd.to_numeric(before, errors="coerce").dropna()
        after = pd.to_numeric(after, errors="coerce").dropna()
        if len(before) < 2 or len(after) < 2:
            continue
        difference = float(before.median() - after.median())
        effects.append(abs(difference) / max(_robust_scale(pd.concat([before, after])), 1e-12))
        directions.append(difference > 0 if direction == "positive" else difference < 0)
    return {
        "reset_pair_count": len(effects),
        "reset_direction_consistency": float(np.mean(directions)) if directions else 0.0,
        "reset_median_effect": float(np.median(effects)) if effects else np.nan,
    }


def _status(
    *,
    valid_cycles: int,
    minimum_cycles: int,
    reset_pairs: int,
    context_confounded: bool,
    trend_available: bool,
    coverage: float,
    deployment_status: str,
    primary_or_validation: str,
) -> str:
    if coverage <= 0 or not trend_available:
        return "insufficient_coverage"
    if valid_cycles < minimum_cycles:
        return "pending_more_cycles"
    if context_confounded:
        return "confounded_by_context"
    if reset_pairs == 0:
        return "partial_evidence"
    if valid_cycles < 3 or deployment_status == "pending":
        return "pending_more_cycles"
    if primary_or_validation == "validation":
        return "partial_evidence"
    return "supported_candidate"


def _risk(family: str, context: dict[str, object], reset: dict[str, object]) -> str:
    risks: list[str] = []
    if bool(context["context_confounded"]):
        risks.append(f"context:{context['context_dominant_field']}")
    if int(cast(Any, reset["reset_pair_count"])) == 0:
        risks.append("no_reset_pair")
    if family in {"actuator_response", "control_targets"}:
        risks.append("control_response_is_not_direct_frost_mass")
    return ";".join(risks) or "none_identified"


def _next_validation(status: str, reset_pairs: int) -> str:
    if status == "insufficient_coverage":
        return "补充无缺口完整循环并确认字段可用"
    if status == "confounded_by_context":
        return "增加跨环境/负荷条件并建立条件无霜基准"
    if reset_pairs == 0:
        return "补充相邻除霜循环，验证除霜后复位"
    return "跨日期留一验证并与图像时间对齐"


def _unavailable_row(item: Any, reason: str) -> dict[str, object]:
    row = {
        "feature_id": str(getattr(item, "feature_id", "")),
        "canonical_name": str(getattr(item, "canonical_name", "")),
        "meaning_zh": str(getattr(item, "meaning_zh", "")),
        "physical_family": str(getattr(item, "physical_family", "unclassified")),
        "source_type": str(getattr(item, "source_type", "")),
        "unit": str(getattr(item, "unit", "unknown")),
        "primary_or_validation": str(getattr(item, "primary_or_validation", "primary")),
        "deployment_status": str(getattr(item, "deployment_status", "pending")),
        "signal_column": "",
        "observed_count": 0,
        "coverage": 0.0,
        "valid_cycle_count": 0,
        "trend_direction": "undetermined",
        "trend_median_spearman": np.nan,
        "trend_direction_consistency": 0.0,
        "early_late_effect": np.nan,
        "context_max_abs_spearman": np.nan,
        "context_dominant_field": "",
        "context_confounded": False,
        "reset_pair_count": 0,
        "reset_direction_consistency": 0.0,
        "reset_median_effect": np.nan,
        "candidate_status": reason,
        "risk": reason,
        "next_validation": "确认原始字段存在并补足连续数据",
    }
    row.update({key: np.nan for key in _lag_columns()})
    return row


def _robust_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    median = float(numeric.median())
    mad = float((numeric - median).abs().median() * 1.4826)
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return max(mad, std, 1e-12)


def _rho(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 3 or paired.nunique().min() < 2:
        return np.nan
    return float(spearmanr(paired["left"], paired["right"]).statistic)


def _lag_columns() -> tuple[str, ...]:
    return ("lag_q_5m", "lag_q_10m", "lag_power_5m", "lag_power_10m", "lag_cop_5m", "lag_cop_10m")


def _evidence_columns() -> list[str]:
    return [
        "feature_id",
        "canonical_name",
        "meaning_zh",
        "physical_family",
        "source_type",
        "unit",
        "primary_or_validation",
        "deployment_status",
        "signal_column",
        "observed_count",
        "coverage",
        "valid_cycle_count",
        "trend_direction",
        "trend_median_spearman",
        "trend_direction_consistency",
        "early_late_effect",
        "context_max_abs_spearman",
        "context_dominant_field",
        "context_valid_cycle_count",
        "context_confounded",
        *_lag_columns(),
        "lag_valid_cycle_count",
        "reset_pair_count",
        "reset_direction_consistency",
        "reset_median_effect",
        "candidate_status",
        "risk",
        "next_validation",
    ]
