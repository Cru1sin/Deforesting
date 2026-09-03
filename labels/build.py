"""Build cycle-safe RGB labels from canonical V1 cost curves."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from dataloader import DatasetLoader
from dataloader.images import RGB_CAMERA_ORDER

COST_REQUIRED_COLUMNS = (
    "cycle_name",
    "candidate_time",
    "relative_regret",
    "optimization_eligible",
    "is_censored",
    "label_eligible",
    "variant",
)
CAMERA_GROUPS: dict[str, tuple[str, ...]] = {
    "top": ("top",),
    "top_close": ("top_close",),
    "left": ("left",),
    "left_close": ("left_close",),
    "front": ("front",),
    "extreme": ("extreme",),
    "top_pair": ("top", "top_close"),
    "left_pair": ("left", "left_close"),
    "all": RGB_CAMERA_ORDER,
}


class _CurveSupport(NamedTuple):
    ordered: pd.DataFrame
    regret: pd.Series
    eligible: pd.Series
    eligible_run: pd.Series
    interpolatable: pd.Series
    optimum_index: int | None


def validate_cost(cost: pd.DataFrame) -> None:
    """Allow hard labels only from the canonical label-eligible cost curve."""
    missing = [column for column in COST_REQUIRED_COLUMNS if column not in cost]
    if missing:
        raise ValueError(f"cost is missing required columns: {', '.join(missing)}")
    if not cost["label_eligible"].fillna(False).eq(True).all():
        raise ValueError("label_eligible must be True for every row")
    if cost["variant"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("named cost variant cannot produce hard labels")


def validate_policy(policy: pd.DataFrame) -> None:
    """Require the shared selector contract used by policy boundary labels."""
    required = {
        "cycle_name",
        "candidate_time",
        "selected",
        "selected_time",
        "selection_policy",
        "selection_status",
    }
    missing = sorted(required.difference(policy.columns))
    if missing:
        raise ValueError(f"policy is missing required columns: {', '.join(missing)}")
    if policy["selection_policy"].astype(str).ne("ch_pareto_knee").any():
        raise ValueError("policy labels require ch_pareto_knee")


def threshold_suffix(threshold: float) -> str:
    """Format a regret threshold as an exact readable column suffix."""
    whole, dot, fraction = f"{threshold * 100:.12g}".partition(".")
    return f"{whole.zfill(2)}{f'p{fraction}' if dot else ''}pct"


def _curve_support(
    curve: pd.DataFrame,
) -> _CurveSupport:
    ordered = (
        curve.assign(
            candidate_time=pd.to_datetime(
                curve["candidate_time"], errors="coerce", format="mixed"
            )
        )
        .sort_values("candidate_time", kind="stable")
        .reset_index(drop=True)
    )
    regret = pd.to_numeric(ordered["relative_regret"], errors="coerce")
    eligible = regret.map(np.isfinite) & ordered["optimization_eligible"].fillna(False)
    run = eligible.ne(eligible.shift(fill_value=False)).cumsum()
    support = eligible & eligible.groupby(run).transform("sum").ge(2)
    optimum = int(regret.loc[eligible].idxmin()) if eligible.any() else None
    return _CurveSupport(ordered, regret, eligible, run, support, optimum)


def curve_label_exclusion_reason(curve: pd.DataFrame) -> str | None:
    """Return why a candidate curve cannot support hard image labels."""
    prepared = _curve_support(curve)
    if not prepared.eligible.any():
        return "no_eligible_candidates"
    if prepared.optimum_index is None or not prepared.interpolatable.iloc[
        prepared.optimum_index
    ]:
        return "t_star_not_in_interpolatable_run"
    return None


def assign_image_cost_states(
    image_times: pd.Series | pd.DatetimeIndex,
    curve: pd.DataFrame,
    *,
    regret_threshold: float,
) -> pd.DataFrame:
    """Interpolate regret within eligible runs and assign image states."""
    prepared = _curve_support(curve)
    times = pd.Series(pd.to_datetime(image_times, errors="coerce", format="mixed")).reset_index(
        drop=True
    )
    regret = pd.Series(np.nan, index=times.index, dtype=float)
    state = pd.Series(pd.NA, index=times.index, dtype="string")
    if (
        prepared.optimum_index is not None
        and prepared.interpolatable.iloc[prepared.optimum_index]
    ):
        for _, positions in prepared.ordered.loc[prepared.interpolatable].groupby(
            prepared.eligible_run.loc[prepared.interpolatable], sort=False
        ):
            inside = times.between(
                positions["candidate_time"].iloc[0],
                positions["candidate_time"].iloc[-1],
            )
            regret.loc[inside] = np.interp(
                times.loc[inside].astype("int64"),
                positions["candidate_time"].astype("int64"),
                prepared.regret.loc[positions.index],
            )
        labeled = regret.notna()
        optimum_time = prepared.ordered.loc[prepared.optimum_index, "candidate_time"]
        state.loc[labeled & times.lt(optimum_time)] = "pre_optimal"
        state.loc[labeled & times.ge(optimum_time)] = "post_optimal"
        state.loc[labeled & regret.le(regret_threshold)] = "near_optimal"
    return pd.DataFrame(
        {
            "image_time": times,
            "relative_regret": regret,
            "cost_state": state,
            "three_class_state": state,
            "binary_state": state.mask(state.eq("near_optimal"), pd.NA),
        }
    )


def complete_catalog_cycle_names(catalog: pd.DataFrame) -> list[str]:
    """Return valid cycles with all observed labeling boundaries."""
    required = (
        "cycle_name",
        "status",
        "stable_heating_start",
        "defrost_start",
        "defrost_end",
    )
    missing = [column for column in required if column not in catalog]
    if missing:
        raise ValueError(f"catalog is missing complete-cycle fields: {', '.join(missing)}")
    boundaries = catalog[list(required[2:])].apply(
        pd.to_datetime, errors="coerce", format="mixed"
    )
    complete = catalog.loc[catalog["status"].eq("valid") & boundaries.notna().all(axis=1)]
    return sorted(complete["cycle_name"].astype(str).unique())


def complete_observed_cycle_names(catalog: pd.DataFrame, cost: pd.DataFrame) -> list[str]:
    """Return complete catalog cycles with uncensored current curves."""
    complete = complete_catalog_cycle_names(catalog)
    scoped = cost.loc[cost["cycle_name"].isin(complete)]
    censored = (
        scoped["is_censored"].fillna(True).astype(bool).groupby(scoped["cycle_name"]).any()
    )
    return sorted(censored.index[~censored].astype(str).tolist())


def build_labels(  # noqa: C901 - explicit cycle labeling and audit flow.
    dataset_root: Path,
    cost: pd.DataFrame,
    thresholds: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three formal V1 label tables."""
    suffixes = [threshold_suffix(float(threshold)) for threshold in thresholds]
    if len(suffixes) != len(set(suffixes)):
        raise ValueError("threshold suffix collision")
    loader = DatasetLoader(dataset_root)
    catalog = loader.list_cycles()
    metadata = loader.load_image_metadata()

    complete_cycles = complete_catalog_cycle_names(catalog)
    valid_cycles = complete_observed_cycle_names(catalog, cost)
    metadata = metadata.loc[
        metadata["cycle_name"].isin(valid_cycles)
        & metadata["cycle_stage"].eq("frost_development")
    ].merge(catalog, on="cycle_name", how="left", validate="many_to_one")
    metadata["image_path"] = (
        "images/"
        + metadata["cycle_name"].astype(str)
        + "/"
        + metadata["camera_role"].astype(str)
        + "/"
        + metadata["file_name"].astype(str)
    )
    metadata["local_available"] = metadata["image_path"].map(
        lambda value: (dataset_root / value).is_file()
    )

    current_cost_cycles = set(cost["cycle_name"].astype(str))
    valid_cycle_set = set(valid_cycles)
    audit: list[dict[str, object]] = [
        {
            "cycle_name": cycle_name,
            "included": False,
            "reason": (
                "no_current_curve"
                if cycle_name not in current_cost_cycles
                else "censored_curve"
            ),
            "labeled_image_count": 0,
        }
        for cycle_name in complete_cycles
        if cycle_name not in valid_cycle_set
    ]
    image_groups = {
        str(cycle_name): images for cycle_name, images in metadata.groupby("cycle_name", sort=True)
    }
    labeled: list[pd.DataFrame] = []
    for cycle_name in valid_cycles:
        curve = cost.loc[cost["cycle_name"].eq(cycle_name)]
        images = image_groups.get(cycle_name)
        base = (
            None
            if images is None
            else assign_image_cost_states(
                images["image_time"], curve, regret_threshold=float(thresholds[0])
            )
        )
        reason = curve_label_exclusion_reason(curve)
        if reason is None:
            reason = (
                "no_interpolatable_image_times"
                if base is None or not base["relative_regret"].notna().any()
                else "labeled"
            )
        audit.append(
            {
                "cycle_name": cycle_name,
                "included": reason == "labeled",
                "reason": reason,
                "labeled_image_count": (
                    0 if base is None else int(base["relative_regret"].notna().sum())
                ),
            }
        )
        if images is None or base is None or reason != "labeled":
            continue
        result = images.reset_index(drop=True).copy()
        result["relative_regret"] = base["relative_regret"]
        for threshold in thresholds:
            states = assign_image_cost_states(
                images["image_time"], curve, regret_threshold=float(threshold)
            )
            suffix = threshold_suffix(float(threshold))
            result[f"cost_state_{suffix}"] = states["three_class_state"]
            result[f"three_class_state_{suffix}"] = states["three_class_state"]
            result[f"binary_state_{suffix}"] = states["binary_state"]
        labeled.append(result)

    if not labeled:
        raise RuntimeError("no supported RGB labels")
    labels = pd.concat(labeled, ignore_index=True)
    balance_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        state_column = f"cost_state_{threshold_suffix(float(threshold))}"
        for group_name, roles in CAMERA_GROUPS.items():
            selected = labels.loc[labels["camera_role"].isin(roles)]
            for state, rows in selected.groupby(state_column, observed=True):
                balance_rows.append(
                    {
                        "regret_threshold": threshold,
                        "camera_group": group_name,
                        "cost_state": state,
                        "image_count": len(rows),
                        "cycle_count": rows["cycle_name"].nunique(),
                        "local_image_count": int(rows["local_available"].sum()),
                    }
                )

    audit.sort(key=lambda row: str(row["cycle_name"]))
    return labels, pd.DataFrame(balance_rows), pd.DataFrame(audit)


def build_policy_labels(
    dataset_root: Path, policy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label frost-development images before or after each selected Pareto knee."""
    validate_policy(policy)
    loader = DatasetLoader(dataset_root)
    catalog = loader.list_cycles()
    metadata = loader.load_image_metadata()
    selected_times: dict[str, pd.Timestamp] = {}
    audit: list[dict[str, object]] = []
    for cycle_name, curve in policy.groupby("cycle_name", sort=True):
        selected = curve.loc[curve["selected"].fillna(False)]
        stored = pd.to_datetime(curve["selected_time"], errors="coerce", format="mixed").dropna()
        if selected.empty and stored.empty:
            audit.append(
                {
                    "cycle_name": cycle_name,
                    "included": False,
                    "reason": "policy_abstained",
                    "labeled_image_count": 0,
                }
            )
            continue
        if len(selected) != 1 or stored.nunique() != 1:
            raise ValueError(f"{cycle_name}: invalid CH Pareto selection")
        selected_time = pd.Timestamp(selected["candidate_time"].iloc[0])
        if selected_time != pd.Timestamp(stored.iloc[0]):
            raise ValueError(f"{cycle_name}: invalid CH Pareto selection")
        selected_times[str(cycle_name)] = selected_time

    rows = metadata.loc[
        metadata["cycle_name"].astype(str).isin(selected_times)
        & metadata["cycle_stage"].eq("frost_development")
    ].merge(catalog, on="cycle_name", how="left", validate="many_to_one")
    rows["image_time"] = pd.to_datetime(rows["image_time"], errors="coerce", format="mixed")
    rows["selected_time"] = rows["cycle_name"].astype(str).map(selected_times)
    rows = rows.loc[rows["image_time"].notna()].copy()
    rows["policy_state"] = np.where(
        rows["image_time"].lt(rows["selected_time"]), "pre_optimal", "post_optimal"
    )
    rows["label_source"] = "ch_pareto_knee"
    rows["image_path"] = (
        "images/"
        + rows["cycle_name"].astype(str)
        + "/"
        + rows["camera_role"].astype(str)
        + "/"
        + rows["file_name"].astype(str)
    )
    rows["local_available"] = rows["image_path"].map(
        lambda value: (dataset_root / value).is_file()
    )
    counts = rows.groupby("cycle_name").size()
    for cycle_name in selected_times:
        count = int(counts.get(cycle_name, 0))
        audit.append(
            {
                "cycle_name": cycle_name,
                "included": count > 0,
                "reason": "labeled" if count else "no_frost_development_images",
                "labeled_image_count": count,
            }
        )
    balance = []
    for group_name, roles in CAMERA_GROUPS.items():
        scoped = rows.loc[rows["camera_role"].isin(roles)]
        for state, values in scoped.groupby("policy_state", observed=True):
            balance.append(
                {
                    "label_source": "ch_pareto_knee",
                    "camera_group": group_name,
                    "policy_state": state,
                    "image_count": len(values),
                    "cycle_count": values["cycle_name"].nunique(),
                    "local_image_count": int(values["local_available"].sum()),
                }
            )
    if rows.empty:
        raise RuntimeError("no selected Pareto RGB labels")
    return (
        rows.reset_index(drop=True),
        pd.DataFrame(balance),
        pd.DataFrame(audit).sort_values("cycle_name").reset_index(drop=True),
    )
