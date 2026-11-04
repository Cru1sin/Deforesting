"""Measured sensor/RGB inputs and experiment-isolated effective COP references."""

import numpy as np
import pandas as pd

from dataset_tools import DatasetLoader
from image_models.control_context import stopping_inputs

RGB = [f"dinov2_{i:03d}" for i in range(384)]

FREQUENCY = stopping_inputs("headroom_fraction", "changes")["RGB_response"]

CHANGE = [f"history_change_{i:03d}" for i in range(len(RGB))]

PATCH_COLUMNS = [f"dinov2_patchmean_{index:03d}" for index in range(384)]

GRID_COLUMNS = [f"dinov2_grid2_{region}_{index:03d}" for region in range(4) for index in range(384)]

NUMERIC_INPUTS = [
    *FREQUENCY,
    "control_setpoint_c",
    "control_outlet_c",
    "elapsed_cycle_minutes",
    "frequency_used_margin_fraction",
]

TEMPERATURE = ["control_setpoint_c", "control_outlet_c"]

TEMPERATURE_RECIPES = {
    "GRU32_frequency": [],
    "GRU32_setpoint": TEMPERATURE[:1],
    "GRU32_setpoint_outlet": TEMPERATURE,
}

VISUAL_POOLING = {
    "zero": RGB,
    "cls": RGB,
    "patch_mean": PATCH_COLUMNS,
    "cls_patch_mean": [*RGB, *PATCH_COLUMNS],
    "cls_repeat": [*RGB, *[f"dinov2_repeat_{i:03d}" for i in range(384)]],
    "patch_grid2": GRID_COLUMNS,
    "patch_mean_repeat4": [*PATCH_COLUMNS, *[f"dinov2_patchrepeat_{i:04d}" for i in range(1152)]],
}

REPEAT_SOURCE = {"cls_repeat": RGB, "patch_mean_repeat4": PATCH_COLUMNS}


def prepare_inputs(rows, numeric_columns):
    """Derive the same past-only operating inputs for training and frozen replay."""

    rows = rows.copy()
    if "frequency_used_margin_fraction" in numeric_columns:
        rows = causal_compensation_progress(rows)
    if "elapsed_cycle_minutes" in numeric_columns:
        rows["elapsed_cycle_minutes"] = elapsed_cycle_minutes(rows)
    if "temperature_error_c" in numeric_columns:
        rows["temperature_error_c"] = rows.control_setpoint_c - rows.control_outlet_c
    return rows


def load_data(dataset, camera_role="front"):
    """Screen the full candidate catalog, then load the published camera inputs."""
    from image_models.cycle_selection import screen_published_cycles
    from image_models.labels import stopping_cdf_targets

    loader = DatasetLoader(dataset)
    screening = screen_published_cycles(loader.dataset_root, camera_role)
    rows, reference = loader.load_training_data(objective="cdf_ramp", camera_role=camera_role)
    selected = set(screening.loc[screening.selected, "cycle_name"])
    missing = selected - set(rows.cycle_name)
    if missing:
        raise ValueError(
            "Selected cycles have unpublished inputs; publish Dataset training-data first"
        )
    rows = rows.loc[rows.cycle_name.isin(selected)].copy()
    reference = reference.loc[reference.cycle_name.isin(selected)].copy()
    rows = prepare_inputs(rows, NUMERIC_INPUTS)
    rows = stopping_cdf_targets(rows)
    return rows, reference, screening


def temperature_columns(args):
    return (
        ["control_setpoint_c", "temperature_error_c"]
        if getattr(args, "temperature_representation", "absolute") == "tracking_error"
        else TEMPERATURE
    )


def policy_numeric_columns(args, extra=None):
    columns = [*FREQUENCY, *(temperature_columns(args) if extra is None else extra)]
    if getattr(args, "elapsed_context", False):
        columns.append("elapsed_cycle_minutes")
    if getattr(args, "compensation_progress", False):
        columns.append("frequency_used_margin_fraction")
    omitted = getattr(args, "drop_inputs", None) or []
    return [column for column in columns if column not in omitted]


def policy_visual_columns(args):
    history = getattr(args, "visual_history", "none")
    if history == "anchor_only":
        return CHANGE
    if history != "none":
        return [*RGB, *CHANGE]
    return VISUAL_POOLING[getattr(args, "visual_pooling", "cls")]


def prepare_ablation_inputs(args):

    loader = DatasetLoader(args.dataset)
    # Fail before fitting if the publication lacks evidence required by final evaluation.
    loader.load_screening_inputs(getattr(args, "camera_role", "front"))
    rows, reference = loader.load_training_data(
        objective=args.loss_name,
        visual_columns=REPEAT_SOURCE.get(args.visual_pooling, VISUAL_POOLING[args.visual_pooling]),
        camera_role=getattr(args, "camera_role", "front"),
    )
    if getattr(args, "cycle_cohort", None):
        names = set(pd.read_csv(args.cycle_cohort).cycle_name)
        if not names.issubset(set(rows.cycle_name)):
            raise ValueError("Requested cycle cohort contains unpublished camera inputs")
        rows = rows.loc[rows.cycle_name.isin(names)].copy()
        reference = reference.loc[reference.cycle_name.isin(names)].copy()
    rows = prepare_inputs(rows, policy_numeric_columns(args))
    if args.loss_name == "cdf_ramp":
        from image_models.labels import stopping_cdf_targets

        rows = stopping_cdf_targets(rows)
        if getattr(args, "cdf_link", "cumulative") == "instantaneous":
            rows["stopping_cdf_instantaneous"] = True
    if args.visual_pooling == "zero":
        rows[RGB] = 0.0
    if (
        args.candidate_set != "exact"
        or args.late_preference != "none"
        or getattr(args, "set_aggregation", "marginal") != "marginal"
    ):
        from image_models.labels import stopping_candidates

        rows = stopping_candidates(
            rows,
            args.candidate_set,
            args.late_preference,
            getattr(args, "set_aggregation", "marginal"),
        )
    if args.visual_pooling in REPEAT_SOURCE:
        source = REPEAT_SOURCE[args.visual_pooling]
        extra = VISUAL_POOLING[args.visual_pooling][len(source) :]
        repeated = pd.concat([rows[source]] * (len(extra) // len(source)), axis=1).set_axis(
            extra, axis=1
        )
        rows = pd.concat([rows, repeated], axis=1)
    if args.visual_history != "none":
        rows = causal_visual_history(rows, args.history_minutes, mode=args.visual_history)
    if args.recipe == "nested_temperature":
        if rows.loc[rows.input_valid, TEMPERATURE].isna().any().any():
            raise ValueError("Publish complete causal temperature inputs before this recipe")
    return rows, reference


def reliable_rgb_cohort(dataset, *, source=None, include_rb=False, require_complete_peak=False):
    """Return cycles valid now, before COP filtering, and for front RGB."""
    loader = dataset if isinstance(dataset, DatasetLoader) else DatasetLoader(dataset)
    cycles = loader.list_valid_cycles(include_rb=include_rb)
    if require_complete_peak:
        complete = cycles.cop_peak_quality.map(
            lambda x: isinstance(x, dict) and x.get("complete") is True
        )
        cycles = cycles.loc[complete]
    front_valid = cycles.get("front_rgb_valid", pd.Series(index=cycles.index, dtype=object))
    if front_valid.isna().any():
        metadata = loader.load_image_metadata()
        available = set(metadata.loc[metadata.camera_role.eq("front"), "cycle_name"])
        audited = cycles.get("rgb_frost_status", pd.Series(index=cycles.index, dtype=object)).eq(
            "valid"
        )
        front_valid = front_valid.where(
            front_valid.notna(), cycles.cycle_name.isin(available) & audited
        )
    cohort = (
        cycles.loc[
            front_valid.eq(True),
            [
                "cycle_name",
                "experiment_id",
            ],
        ]
        .sort_values(["experiment_id", "cycle_name"], kind="stable")
        .reset_index(drop=True)
    )
    if source is None:
        return cohort
    return cohort.merge(source, on=["cycle_name", "experiment_id"], how="inner")


def causal_visual_history(rows, minutes, *, columns=RGB, mode="delta"):
    """Backward image lookup within each cycle; repeat the first frame for warmup.

    No rows are dropped for insufficient history, so early errors cannot disappear
    through an artificial warmup gate. Missing current inputs remain missing.
    """
    result = rows.copy()
    delta = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    source_times = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
    padded = np.zeros(len(rows), dtype=bool)
    missing = np.zeros(len(rows), dtype=bool)
    for _, positions in rows.groupby("cycle_name", sort=False).indices.items():
        cycle = rows.iloc[positions]
        visual = (
            np.isfinite(cycle[columns]).all(axis=1)
            & cycle.image_time.notna()
            & cycle.image_time.le(cycle.candidate_defrost_time)
        )
        history = cycle.loc[visual].sort_values("image_time").drop_duplicates("image_time")
        current_positions = np.asarray(positions)[cycle.input_valid.to_numpy(bool)]
        if history.empty or not len(current_positions):
            continue
        current = rows.iloc[current_positions]
        requested = (current.candidate_defrost_time - pd.Timedelta(minutes=minutes)).to_numpy(
            "datetime64[ns]"
        )
        found = (
            np.searchsorted(history.image_time.to_numpy("datetime64[ns]"), requested, side="right")
            - 1
        )
        if mode in ("anchor", "anchor_only"):
            found[:] = 0  # First observable RGB, not an assumed frost-free teacher.
        past = history.iloc[np.maximum(found, 0)]
        past_times = past.image_time.to_numpy("datetime64[ns]")
        stale = (found >= 0) & (requested - past_times > np.timedelta64(30, "s"))
        if mode in ("anchor", "anchor_only"):
            stale[:] = False  # The intentionally fixed reference does not expire.
        changes = current[columns].to_numpy() - past[columns].to_numpy()
        changes[stale] = 0.0  # Missing lag contributes no invented visual change.
        past_times[stale] = current.image_time.to_numpy("datetime64[ns]")[stale]
        delta[current_positions] = changes
        source_times[current_positions] = past_times
        padded[current_positions] = found < 0
        missing[current_positions] = stale
    changes = pd.DataFrame(
        delta, index=rows.index, columns=[f"history_change_{i:03d}" for i in range(len(columns))]
    )
    result = pd.concat([result, changes], axis=1)
    result["history_image_time"] = source_times
    result["history_padded"] = padded
    result["history_missing"] = missing
    result["history_age_minutes"] = (
        result.candidate_defrost_time - result.history_image_time
    ).dt.total_seconds() / 60
    if mode == "current":
        # Equal-width control: additional parameters, but no additional information.
        result[changes.columns] = rows[columns].to_numpy()
    elif mode not in ("delta", "anchor", "anchor_only"):
        raise ValueError(f"Unknown visual history mode: {mode}")
    return result


def elapsed_cycle_minutes(rows):
    """Causal age from the recorded start, without future duration normalization."""
    return (
        pd.to_datetime(rows.candidate_defrost_time)
        - pd.to_datetime(rows.cycle_start, format="mixed")
    ).dt.total_seconds() / 60


def causal_compensation_progress(rows):
    """Used margin since the observed frequency trough, under the fixed 100 Hz cap."""
    result = rows.copy()
    result["frequency_used_margin_fraction"] = np.nan
    for _, g in result.sort_values("candidate_defrost_time", kind="stable").groupby(
        "cycle_name", sort=False
    ):
        headroom = g.frequency_headroom_fraction.where(g.input_valid)
        result.loc[g.index, "frequency_used_margin_fraction"] = headroom.cummax().ffill() - headroom
    return result
