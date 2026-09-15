"""Learn one fold-specific event representation from actual defrost outcomes."""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import BSpline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from defrost_event_models.ridge_models import (
    OUTCOME_TARGETS,
    select_events_complete_for_all_outcomes,
)

RGB = [f"dinov2_{index:03d}" for index in range(384)]
ACCOUNTING = [
    "online_pre_defrost_heat_kwh", "online_pre_defrost_electricity_kwh",
    "online_pre_defrost_compressor_electricity_kwh", "online_pointwise_valid",
    "rgb_missing", "rgb_age_seconds",
]
TARGETS = list(OUTCOME_TARGETS.values())
TIME_COLUMN = "minutes_since_heating_start"
EVENT_HEADS = {
    "multimodal_latent": ("fixed", 0.),
    "multimodal_time_linear": ("linear", 0.),
    "multimodal_time_linear_z16": ("linear", 0.),
    "multimodal_time_linear_z64": ("linear", 0.),
    "multimodal_time_linear_z32_curvature": ("linear", 0.),
    "multimodal_time_linear_z64_curvature": ("linear", 0.),
    "multimodal_time_varying": ("varying", 0.),
    "multimodal_time_varying_regularized": ("varying", .01),
}
LATENT_WIDTHS = {
    "multimodal_time_linear": 32,
    "multimodal_time_linear_z16": 16,
    "multimodal_time_linear_z64": 64,
    "multimodal_time_linear_z32_curvature": 32,
    "multimodal_time_linear_z64_curvature": 64,
}
CURVATURE_HEADS = {
    "multimodal_time_linear_z32_curvature",
    "multimodal_time_linear_z64_curvature",
}
SPLINE_KNOTS = np.array([0., 0., 0., .5, 1., 1., 1.])


class HeatingTime(NamedTuple):
    normalized: np.ndarray
    below_training_range: np.ndarray
    above_training_range: np.ndarray


def normalized_heating_time(
    minutes: pd.Series | np.ndarray, training_min: float, training_max: float
) -> HeatingTime:
    """Scale by the training maximum; retain both sides of time extrapolation."""
    values = np.asarray(minutes, dtype=float)
    return HeatingTime(
        values / training_max,
        values < training_min,
        values > training_max,
    )


def quadratic_spline_basis(normalized_time: np.ndarray) -> np.ndarray:
    """Four quadratic B-spline functions with polynomial endpoint extrapolation."""
    return np.asarray(
        BSpline(SPLINE_KNOTS, np.eye(4), 2, extrapolate=True)(normalized_time),
        dtype=float,
    )


class OutcomeRepresentation(nn.Module):
    """Sensor Sin encoder + optional frozen-RGB projection → z → four outcomes."""

    def __init__(
        self, sensor_width: int, accounting_width: int, *, use_rgb: bool,
        event_head: str = "multimodal_latent",
    ) -> None:
        super().__init__()
        self.sensor_width = sensor_width
        self.accounting_width = accounting_width
        self.use_rgb = use_rgb
        self.event_head = event_head
        self.latent_width = LATENT_WIDTHS.get(event_head, 32)
        self.sensor_first = nn.Linear(sensor_width, 60)
        self.sensor_second = nn.Linear(60, 60)
        self.sensor_output = nn.Linear(60, 32)
        self.dropout = nn.Dropout(.2)
        self.visual = nn.Linear(384, 32) if use_rgb else None
        self.fusion = nn.Linear(
            32 + accounting_width + (32 if use_rgb else 0), self.latent_width
        )
        kind, _ = EVENT_HEADS[event_head]
        self.head = nn.Linear(self.latent_width + (kind == "linear"), len(TARGETS))
        if kind == "varying":
            common = torch.cat([self.head.weight, self.head.bias[:, None]], dim=1)
            self.coefficients = nn.Parameter(common.detach().repeat(4, 1, 1))
            del self.head

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        stop = self.sensor_width
        sensor = torch.sin(self.sensor_first(values[:, :stop]))
        sensor = self.sensor_output(self.dropout(torch.sin(self.sensor_second(sensor))))
        pieces = [sensor]
        pieces.append(values[:, stop:stop + self.accounting_width])
        if self.use_rgb:
            assert self.visual is not None
            pieces.append(self.visual(values[:, stop + self.accounting_width:]))
        return torch.relu(self.fusion(torch.cat(pieces, dim=1)))

    def decode(self, latent: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        kind, _ = EVENT_HEADS[self.event_head]
        if kind == "fixed":
            return self.head(latent)
        if time is None:
            raise ValueError(f"{self.event_head} requires heating time")
        if kind == "linear":
            return self.head(torch.cat([latent, time], dim=1))
        augmented = torch.cat([latent, torch.ones_like(latent[:, :1])], dim=1)
        return torch.einsum("nk,koi,ni->no", time, self.coefficients, augmented)

    def forward(
        self, values: torch.Tensor, time: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.decode(self.encode(values), time)

    def coefficient_loss(self) -> torch.Tensor:
        if EVENT_HEADS[self.event_head][0] != "varying":
            return next(self.parameters()).new_zeros(())
        return (self.coefficients[1:] - self.coefficients[:-1]).square().mean()


def outcome_event_rows(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Select the one causal row at each actually observed preparation start."""
    valid = select_events_complete_for_all_outcomes(events)
    if "cycle_name" not in valid:
        valid = valid.assign(cycle_name=valid["event_id"].astype(str))
    preparation = (
        "observed_defrost_preparation_start"
        if "observed_defrost_preparation_start" in rows
        else "defrost_preparation_start"
    )
    selected = rows.loc[
        pd.to_datetime(rows["candidate_defrost_time"]).eq(pd.to_datetime(rows[preparation]))
    ].copy()
    fields = ["cycle_name", "event_id", *TARGETS]
    return selected.merge(valid[fields], on="cycle_name", how="inner", validate="one_to_one")


def _feature_columns(rows: pd.DataFrame, use_rgb: bool) -> tuple[list[str], int, int]:
    sensor = sorted(name for name in rows if name.startswith("stat_"))
    accounting = [name for name in ACCOUNTING if name in rows]
    return [*sensor, *accounting, *(RGB if use_rgb else [])], len(sensor), len(accounting)


def outcome_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Split the exact checkpoint inputs into non-overlapping diagnostic groups."""
    quality_suffixes = ("_missing", "_valid_count", "_age_seconds")
    groups = {name: [] for name in ("rgb", "sensor", "ledger", "quality")}
    for column in columns:
        if column.startswith("dinov2_"):
            group = "rgb"
        elif column.startswith("online_pre_defrost_"):
            group = "ledger"
        elif column in {"rgb_missing", "rgb_age_seconds", "online_pointwise_valid"} or (
            column.startswith("stat_") and column.endswith(quality_suffixes)
        ):
            group = "quality"
        elif column.startswith("stat_"):
            group = "sensor"
        else:
            raise ValueError(f"unclassified outcome input: {column}")
        groups[group].append(column)
    return groups


def temporal_triplet_indices(
    rows: pd.DataFrame, seconds: int = 10
) -> list[np.ndarray]:
    """Return exact equally spaced triplets, grouped by cycle for equal weighting."""
    work = rows.reset_index(drop=True).copy()
    work["candidate_defrost_time"] = pd.to_datetime(work["candidate_defrost_time"])
    eligible = work["is_teacher_candidate"].fillna(False).astype(bool)
    if "sensor_timestamp" in work:
        eligible &= pd.to_datetime(work["sensor_timestamp"], errors="coerce").notna()
    status = [
        column for column in work
        if column == "rgb_missing"
        or column.endswith("_missing")
        or column.endswith("_measurement_valid")
        or column.endswith("_uses_measurement_reconstruction")
    ]
    result = []
    for _, cycle in work.loc[eligible].groupby("cycle_name", sort=False):
        cycle = cycle.sort_values("candidate_defrost_time", kind="stable")
        indices = cycle.index.to_numpy()
        times = cycle["candidate_defrost_time"].astype("int64").to_numpy()
        if len(indices) < 3:
            continue
        step = seconds * 1_000_000_000
        lookup = dict(zip(times, indices, strict=True))
        centers = np.asarray([index for index, timestamp in zip(indices, times, strict=True)
                              if timestamp - step in lookup and timestamp + step in lookup])
        if not len(centers):
            continue
        triples = np.column_stack((
            [lookup[int(work.loc[index, "candidate_defrost_time"].value) - step]
             for index in centers],
            centers,
            [lookup[int(work.loc[index, "candidate_defrost_time"].value) + step]
             for index in centers],
        ))
        if status:
            states = work[status].astype("string").to_numpy()
            valid = np.all(states[triples[:, 0]] == states[triples[:, 1]], axis=1)
            valid &= np.all(states[triples[:, 1]] == states[triples[:, 2]], axis=1)
            triples = triples[valid]
        if len(triples):
            result.append(triples)
    return result


def latent_curvature_loss(
    latent: torch.Tensor, triplets: list[np.ndarray]
) -> torch.Tensor:
    """Mean squared second difference per dimension, with cycles weighted equally."""
    if not triplets:
        return latent.new_zeros(())
    losses = []
    for group in triplets:
        index = torch.as_tensor(group, dtype=torch.long, device=latent.device)
        second = latent[index[:, 2]] - 2 * latent[index[:, 1]] + latent[index[:, 0]]
        losses.append(second.square().mean())
    return torch.stack(losses).mean()


def scheduled_latent_curvature_loss(
    model: OutcomeRepresentation, values: torch.Tensor, triplets: list[np.ndarray]
) -> torch.Tensor:
    """Encode only the capped triplet rows used by this epoch."""
    if not triplets:
        return values.new_zeros(())
    losses = []
    for group in triplets:
        index = torch.as_tensor(group.reshape(-1), dtype=torch.long, device=values.device)
        latent = model.encode(values[index]).reshape(len(group), 3, -1)
        second = latent[:, 2] - 2 * latent[:, 1] + latent[:, 0]
        losses.append(second.square().mean())
    return torch.stack(losses).mean()


def fit_latent_scale(latents: pd.DataFrame, cycles: pd.Series | None = None) -> dict:
    """Fit an evaluation-only latent scale, optionally weighting cycles equally."""
    columns = sorted(name for name in latents if name.startswith("z_"))
    values = latents[columns].to_numpy(dtype=float)
    if cycles is None:
        mean = np.nanmean(values, axis=0)
        second = np.nanmean(np.square(values), axis=0)
    else:
        frame = pd.DataFrame(values, columns=columns).assign(_cycle=cycles.to_numpy())
        mean = frame.groupby("_cycle", sort=False)[columns].mean().mean().to_numpy()
        second = (
            frame.assign(**{name: np.square(frame[name]) for name in columns})
            .groupby("_cycle", sort=False)[columns].mean().mean().to_numpy()
        )
    scale = np.sqrt(np.maximum(second - np.square(mean), 0.))
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "zero_variance_dimensions": int(np.sum(scale <= 1e-12))}


def trajectory_roughness(
    rows: pd.DataFrame,
    latents: pd.DataFrame,
    predictions: pd.DataFrame,
    latent_scale: dict,
    target_scale: np.ndarray,
    deltas_seconds: tuple[int, ...] = (10, 60, 300),
) -> pd.DataFrame:
    """Compute per-cycle latent and outcome variation on exact candidate triplets."""
    z_columns = sorted(name for name in latents if name.startswith("z_"))
    target_indices = [index for index, target in enumerate(TARGETS)
                      if f"predicted_{target}" in predictions]
    y_columns = [f"predicted_{TARGETS[index]}" for index in target_indices]
    work = rows.drop(columns=y_columns, errors="ignore").merge(
        latents[["row_id", *z_columns]], on="row_id", validate="one_to_one"
    )
    work = work.merge(
        predictions[["row_id", *y_columns]], on="row_id", validate="one_to_one"
    )
    scale = np.asarray(latent_scale["scale"], dtype=float)
    supported = np.isfinite(scale) & (scale > 1e-12)
    target_scale = np.asarray(target_scale, dtype=float)[target_indices]
    records = []
    for (experiment, cycle_name), cycle in work.groupby(
        ["experiment_id", "cycle_name"], sort=False
    ):
        cycle = cycle.reset_index(drop=True)
        z = cycle[z_columns].to_numpy(dtype=float)
        y = cycle[y_columns].to_numpy(dtype=float) / target_scale
        for seconds in deltas_seconds:
            groups = temporal_triplet_indices(cycle, seconds)
            if not groups:
                continue
            triplets = np.concatenate(groups)
            z_second = z[triplets[:, 2]] - 2 * z[triplets[:, 1]] + z[triplets[:, 0]]
            y_second = y[triplets[:, 2]] - 2 * y[triplets[:, 1]] + y[triplets[:, 0]]
            z_rms = np.sqrt(np.mean(np.square(z_second[:, supported] / scale[supported]), axis=1))
            y_rms = np.sqrt(np.mean(np.square(y_second), axis=1))
            z_first = np.sqrt(np.mean(
                np.square((z[triplets[:, 2]][:, supported]
                           - z[triplets[:, 1]][:, supported]) / scale[supported]), axis=1
            ))
            y_first = np.sqrt(np.mean(
                np.square(y[triplets[:, 2]] - y[triplets[:, 1]]), axis=1
            ))
            records.append({
                "experiment_id": str(experiment), "cycle_name": cycle_name,
                "delta_seconds": seconds, "triplets": len(triplets),
                "latent_second_difference_rms_median": float(np.median(z_rms)),
                "outcome_second_difference_rms_median": float(np.median(y_rms)),
                "latent_first_difference_rms_median": float(np.median(z_first)),
                "outcome_first_difference_rms_median": float(np.median(y_first)),
                "zero_variance_dimensions": int(np.sum(~supported)),
            })
    return pd.DataFrame(records)


def group_balanced_squared_distance(
    query: np.ndarray, reference: np.ndarray, groups: list[list[int]]
) -> np.ndarray:
    """Average within-group mean-square distances so width does not set group weight."""
    active = [index for index in groups if index]
    return np.mean([
        np.mean(np.square(reference[:, index] - query[index]), axis=1)
        for index in active
    ], axis=0)


def event_nearest_neighbor_diagnostics(
    train: pd.DataFrame, test: pd.DataFrame, checkpoint: dict, k: int = 3
) -> pd.DataFrame:
    """Compare held-out event outcomes with nearby training events in input and z space."""
    columns = checkpoint["feature_columns"]
    preprocessor = checkpoint["preprocessor"]
    train_x = preprocessor.transform(train[columns])
    test_x = preprocessor.transform(test[columns])
    grouped_columns = outcome_feature_groups(columns)
    groups = [[columns.index(name) for name in names] for names in grouped_columns.values()]
    time_mean = float(train[TIME_COLUMN].mean())
    time_scale = float(train[TIME_COLUMN].std(ddof=0))
    if not np.isfinite(time_scale) or time_scale <= 1e-12:
        time_scale = 1.
    train_input = np.column_stack((
        train_x, (train[TIME_COLUMN].to_numpy(dtype=float) - time_mean) / time_scale
    ))
    test_input = np.column_stack((
        test_x, (test[TIME_COLUMN].to_numpy(dtype=float) - time_mean) / time_scale
    ))
    groups.append([train_input.shape[1] - 1])
    train_z = encode_outcome_rows(train, checkpoint).filter(like="z_").to_numpy(dtype=float)
    test_z = encode_outcome_rows(test, checkpoint).filter(like="z_").to_numpy(dtype=float)
    latent_scale = fit_latent_scale(pd.DataFrame(
        train_z, columns=[f"z_{index:02d}" for index in range(train_z.shape[1])]
    ))
    z_scale = np.asarray(latent_scale["scale"], dtype=float)
    supported = np.isfinite(z_scale) & (z_scale > 1e-12)
    train_y = checkpoint["target_scaler"].transform(train[TARGETS])
    test_y = checkpoint["target_scaler"].transform(test[TARGETS])
    records = []
    neighbor_count = min(k, len(train))
    for index in range(len(test)):
        distances = {
            "input": group_balanced_squared_distance(test_input[index], train_input, groups),
            "latent": np.mean(
                np.square((train_z[:, supported] - test_z[index, supported])
                          / z_scale[supported]), axis=1
            ),
        }
        for space, values in distances.items():
            neighbors = np.argsort(values, kind="stable")[:neighbor_count]
            discrepancies = np.median(
                np.abs(train_y[neighbors] - test_y[index]), axis=0
            )
            for target, discrepancy in zip(TARGETS, discrepancies, strict=True):
                records.append({
                    "event_id": test.iloc[index]["event_id"],
                    "cycle_name": test.iloc[index]["cycle_name"],
                    "experiment_id": str(test.iloc[index]["experiment_id"]),
                    "space": space, "neighbors": neighbor_count,
                    "target": target,
                    "standardized_outcome_discrepancy_median": float(discrepancy),
                })
    return pd.DataFrame(records)


def _triplet_schedule(
    groups: list[np.ndarray], epochs: int, seed: int, per_cycle: int = 32
) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(group)) for group in groups]
    schedule = []
    for epoch in range(epochs):
        selected = []
        for group, order in zip(groups, permutations, strict=True):
            count = min(per_cycle, len(order))
            positions = np.arange(epoch * count, (epoch + 1) * count) % len(order)
            selected.append(group[order[positions]])
        schedule.append(selected)
    return schedule


def _fit(
    train: pd.DataFrame, validation: pd.DataFrame | None, *, use_rgb: bool,
    event_head: str, epochs: int, patience: int, seed: int,
    temporal: pd.DataFrame | None = None,
) -> tuple[OutcomeRepresentation, Any, Any, pd.DataFrame, int, list[str]]:
    columns, sensor_width, accounting_width = _feature_columns(train, use_rgb)
    preprocessor = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    ).fit(train[columns])
    target_scaler = StandardScaler().fit(train[TARGETS])
    x = torch.tensor(preprocessor.transform(train[columns]), dtype=torch.float32)
    y = torch.tensor(target_scaler.transform(train[TARGETS]), dtype=torch.float32)
    if validation is not None:
        vx = torch.tensor(preprocessor.transform(validation[columns]), dtype=torch.float32)
        vy = torch.tensor(target_scaler.transform(validation[TARGETS]), dtype=torch.float32)
    time_min = float(train[TIME_COLUMN].min()) if event_head != "multimodal_latent" else np.nan
    time_max = float(train[TIME_COLUMN].max()) if event_head != "multimodal_latent" else np.nan

    def times(rows: pd.DataFrame) -> torch.Tensor | None:
        if event_head == "multimodal_latent":
            return None
        values = normalized_heating_time(rows[TIME_COLUMN], time_min, time_max).normalized
        if EVENT_HEADS[event_head][0] == "varying":
            values = quadratic_spline_basis(values)
        else:
            values = values[:, None]
        return torch.tensor(values, dtype=torch.float32)

    xt = times(train)
    vt = times(validation) if validation is not None else None
    torch.manual_seed(seed)
    model = OutcomeRepresentation(
        sensor_width, accounting_width, use_rgb=use_rgb, event_head=event_head
    )
    model.time_min_minutes = time_min
    model.time_scale_minutes = time_max
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    temporal_x = None
    schedule: list[list[np.ndarray]] = [[] for _ in range(epochs)]
    if event_head in CURVATURE_HEADS and temporal is not None:
        temporal = temporal.reset_index(drop=True)
        triplets = temporal_triplet_indices(temporal)
        if triplets:
            temporal_x = torch.tensor(
                preprocessor.transform(temporal[columns]), dtype=torch.float32
            )
            schedule = _triplet_schedule(triplets, epochs, seed)
    model.eval()
    with torch.no_grad():
        initial_event = float((model(x, xt) - y).square().mean())
        initial_curvature = (
            float(scheduled_latent_curvature_loss(model, temporal_x, schedule[0]))
            if temporal_x is not None else 0.
        )
    curvature_weight = (
        .1 * initial_event / initial_curvature
        if np.isfinite(initial_curvature) and initial_curvature >= 1e-12 else 0.
    )
    model.curvature_weight = curvature_weight
    history, best_loss, best_epoch, stale, best_state = [], np.inf, 1, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        event_loss = (model(x, xt) - y).square().mean()
        coefficient_loss = model.coefficient_loss()
        model.eval()
        curvature_loss = (
            scheduled_latent_curvature_loss(model, temporal_x, schedule[epoch - 1])
            if temporal_x is not None else event_loss.new_zeros(())
        )
        loss = (
            event_loss + EVENT_HEADS[event_head][1] * coefficient_loss
            + curvature_weight * curvature_loss
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = (
                float((model(vx, vt) - vy).square().mean())
                if validation is not None else np.nan
            )
            latent_variance = float(model.encode(x).var(dim=0).mean())
            head_weight_norm = float(
                model.coefficients.norm()
                if EVENT_HEADS[event_head][0] == "varying" else model.head.weight.norm()
            )
        history.extend((
            {"epoch": epoch, "split": "train", "loss": float(loss.detach()),
             "event_loss": float(event_loss.detach()),
             "coefficient_loss": float(coefficient_loss.detach()),
             "curvature_loss": float(curvature_loss.detach()),
             "weighted_curvature_loss": curvature_weight * float(curvature_loss.detach()),
             "latent_variance": latent_variance,
             "head_weight_norm": head_weight_norm},
            {"epoch": epoch, "split": "validation", "loss": validation_loss,
             "event_loss": validation_loss,
             "coefficient_loss": float(coefficient_loss.detach()),
             "curvature_loss": float(curvature_loss.detach()),
             "weighted_curvature_loss": curvature_weight * float(curvature_loss.detach()),
             "latent_variance": latent_variance,
             "head_weight_norm": head_weight_norm},
        ))
        if validation is not None:
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, preprocessor, target_scaler, pd.DataFrame(history), best_epoch, columns


def _checkpoint(
    fitted, *, use_rgb: bool, event_head: str, selected_epoch: int,
    training_events: int,
) -> dict[str, Any]:
    model, preprocessor, target_scaler, _, _, columns = fitted
    return {
        "model_state_dict": model.state_dict(), "preprocessor": preprocessor,
        "target_scaler": target_scaler, "feature_columns": columns,
        "sensor_width": model.sensor_width, "accounting_width": model.accounting_width,
        "use_rgb": use_rgb, "selected_epoch": selected_epoch, "event_head": event_head,
        "training_events": training_events, "latent_width": model.latent_width,
        "curvature_weight": getattr(model, "curvature_weight", 0.),
        "time_min_minutes": model.time_min_minutes
        if event_head != "multimodal_latent" else None,
        "time_scale_minutes": model.time_scale_minutes
        if event_head != "multimodal_latent" else None,
        "spline_knots": SPLINE_KNOTS.tolist()
        if EVENT_HEADS[event_head][0] == "varying" else None,
        "coefficient_weight": EVENT_HEADS[event_head][1],
    }


def fit_outcome_representation(
    inner_train: pd.DataFrame, inner_validation: pd.DataFrame, outer_train: pd.DataFrame, *,
    use_rgb: bool, event_head: str = "multimodal_latent", maximum_epochs: int = 200,
    patience: int = 25, seed: int = 0, inner_temporal: pd.DataFrame | None = None,
    outer_temporal: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Choose an epoch on one held-out experiment, then refit all outer training events."""
    inner = _fit(
        inner_train, inner_validation, use_rgb=use_rgb, event_head=event_head,
        epochs=maximum_epochs, patience=patience, seed=seed, temporal=inner_temporal,
    )
    outer = _fit(
        outer_train, None, use_rgb=use_rgb, event_head=event_head,
        epochs=inner[4], patience=patience, seed=seed, temporal=outer_temporal,
    )
    return {
        "checkpoint": _checkpoint(
            outer, use_rgb=use_rgb, event_head=event_head,
            selected_epoch=inner[4], training_events=len(outer_train),
        ),
        "inner_checkpoint": _checkpoint(
            inner, use_rgb=use_rgb, event_head=event_head,
            selected_epoch=inner[4], training_events=len(inner_train),
        ),
        "losses": inner[3],
    }


def fit_outcome_fixed(
    train: pd.DataFrame, *, use_rgb: bool, event_head: str, epochs: int,
    seed: int = 0, temporal: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one final outcome model for a fold-selected fixed epoch budget."""
    fitted = _fit(
        train, None, use_rgb=use_rgb, event_head=event_head, epochs=epochs,
        patience=epochs, seed=seed, temporal=temporal,
    )
    return (
        _checkpoint(
            fitted, use_rgb=use_rgb, event_head=event_head,
            selected_epoch=epochs, training_events=len(train),
        ),
        fitted[3],
    )


def _model(checkpoint: dict[str, Any]) -> OutcomeRepresentation:
    model = OutcomeRepresentation(
        checkpoint["sensor_width"], checkpoint["accounting_width"],
        use_rgb=checkpoint["use_rgb"],
        event_head=checkpoint.get("event_head", "multimodal_latent"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def _time_tensor(
    rows: pd.DataFrame, checkpoint: dict[str, Any]
) -> tuple[torch.Tensor | None, HeatingTime | None]:
    event_head = checkpoint.get("event_head", "multimodal_latent")
    if event_head == "multimodal_latent":
        return None, None
    values = normalized_heating_time(
        rows[TIME_COLUMN], checkpoint["time_min_minutes"], checkpoint["time_scale_minutes"]
    )
    inputs = (
        quadratic_spline_basis(values.normalized)
        if EVENT_HEADS[event_head][0] == "varying"
        else values.normalized[:, None]
    )
    return torch.tensor(inputs, dtype=torch.float32), values


def encode_outcome_rows(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    """Apply one frozen fold encoder without changing its coordinate system."""
    model = _model(checkpoint)
    values = checkpoint["preprocessor"].transform(rows[checkpoint["feature_columns"]])
    with torch.no_grad():
        latent = model.encode(torch.tensor(values, dtype=torch.float32)).numpy()
    result = rows[["row_id"]].reset_index(drop=True)
    result[[f"z_{index:02d}" for index in range(latent.shape[1])]] = latent
    if checkpoint.get("event_head", "multimodal_latent") != "multimodal_latent":
        _, heating = _time_tensor(rows, checkpoint)
        assert heating is not None
        result[TIME_COLUMN] = rows[TIME_COLUMN].to_numpy()
        result["normalized_heating_time"] = heating.normalized
        result["time_below_training_range"] = heating.below_training_range
        result["time_above_training_range"] = heating.above_training_range
    return result


def predict_outcomes_from_latent(
    latents: pd.DataFrame, checkpoint: dict[str, Any]
) -> pd.DataFrame:
    """Apply the saved linear event head to its own frozen latent coordinates."""
    columns = sorted(name for name in latents if name.startswith("z_"))
    values = torch.tensor(latents[columns].to_numpy(), dtype=torch.float32)
    model = _model(checkpoint)
    time, _ = _time_tensor(latents, checkpoint)
    with torch.no_grad():
        scaled = model.decode(values, time).numpy()
    predictions = checkpoint["target_scaler"].inverse_transform(scaled)
    metadata = [
        name for name in (
            TIME_COLUMN, "normalized_heating_time", "time_below_training_range",
            "time_above_training_range",
        ) if name in latents
    ]
    result = latents[["row_id", *metadata]].reset_index(drop=True)
    result[[f"predicted_{name}" for name in TARGETS]] = predictions
    return result


def predict_outcomes(rows: pd.DataFrame, checkpoint: dict[str, Any]) -> pd.DataFrame:
    """Predict the four actual-event outcomes in their original units."""
    model = _model(checkpoint)
    values = checkpoint["preprocessor"].transform(rows[checkpoint["feature_columns"]])
    with torch.no_grad():
        time, _ = _time_tensor(rows, checkpoint)
        scaled = model(torch.tensor(values, dtype=torch.float32), time).numpy()
    predictions = checkpoint["target_scaler"].inverse_transform(scaled)
    result = rows[["event_id", "cycle_name", "experiment_id", *TARGETS]].reset_index(drop=True)
    result[[f"predicted_{name}" for name in TARGETS]] = predictions
    for outcome, target in OUTCOME_TARGETS.items():
        unit = "minutes" if outcome == "event_duration" else "kwh"
        source = f"defrost_{outcome}_{unit}"
        if source in rows:
            result[f"ridge_predicted_{target}"] = rows[source].to_numpy()
    return result


def local_pathway_sensitivity(
    rows: pd.DataFrame, checkpoint: dict[str, Any],
    deltas_seconds: tuple[int, ...] = (10, 60, 300),
) -> pd.DataFrame:
    """Summarize one-group-at-a-time frozen-model perturbations within each cycle."""
    work = rows.loc[rows["is_teacher_candidate"].fillna(False)].reset_index(drop=True).copy()
    work["candidate_defrost_time"] = pd.to_datetime(work["candidate_defrost_time"])
    columns = checkpoint["feature_columns"]
    groups = outcome_feature_groups(columns)
    positions = {name: [columns.index(column) for column in names]
                 for name, names in groups.items()}
    values = torch.tensor(
        checkpoint["preprocessor"].transform(work[columns]), dtype=torch.float32
    )
    time, _ = _time_tensor(work, checkpoint)
    if time is None or EVENT_HEADS[checkpoint["event_head"]][0] != "linear":
        raise ValueError("local pathway sensitivity requires a time-linear event head")
    model = _model(checkpoint)
    target_scale = np.asarray(checkpoint["target_scaler"].scale_, dtype=float)
    records = []
    for cycle_name, cycle in work.groupby("cycle_name", sort=False):
        cycle_positions = cycle.index.to_numpy()
        timestamps = cycle["candidate_defrost_time"].astype("int64").to_numpy()
        lookup = dict(zip(timestamps, cycle_positions, strict=True))
        for seconds in deltas_seconds:
            base = np.asarray([
                position for position, timestamp in zip(cycle_positions, timestamps, strict=True)
                if timestamp + seconds * 1_000_000_000 in lookup
            ], dtype=int)
            if not len(base):
                continue
            following = np.asarray([
                lookup[int(work.loc[position, "candidate_defrost_time"].value)
                       + seconds * 1_000_000_000]
                for position in base
            ], dtype=int)
            base_x, next_x = values[base], values[following]
            base_time, next_time = time[base], time[following]
            with torch.no_grad():
                base_z = model.encode(base_x)
                next_z = model.encode(next_x)
                base_y = model.decode(base_z, base_time)
                actual_y = model.decode(next_z, next_time) - base_y
                pathway_y, pathway_z, input_change = {}, {}, {}
                for name, index in positions.items():
                    changed = base_x.clone()
                    changed[:, index] = next_x[:, index]
                    changed_z = model.encode(changed)
                    pathway_z[name] = changed_z - base_z
                    pathway_y[name] = model.decode(changed_z, base_time) - base_y
                    input_change[name] = (next_x[:, index] - base_x[:, index]).square().mean(
                        dim=1
                    ).sqrt()
                pathway_z["time"] = torch.zeros_like(base_z)
                pathway_y["time"] = model.decode(base_z, next_time) - base_y
                input_change["time"] = (next_time - base_time).square().mean(dim=1).sqrt()
                pathway_y["actual"] = actual_y
                pathway_z["actual"] = next_z - base_z
                pathway_y["interaction"] = actual_y - sum(
                    pathway_y[name] for name in (*positions, "time")
                )
                pathway_z["interaction"] = pathway_z["actual"] - sum(
                    pathway_z[name] for name in (*positions, "time")
                )
            for pathway, delta_y in pathway_y.items():
                delta = delta_y.numpy()
                latent_delta = pathway_z[pathway].square().mean(dim=1).sqrt().numpy()
                changed_input = input_change.get(pathway)
                changed_input_values = (
                    changed_input.numpy() if changed_input is not None
                    else np.full(len(base), np.nan)
                )
                for target_index, target in enumerate(TARGETS):
                    records.append({
                        "cycle_name": cycle_name,
                        "experiment_id": str(cycle.experiment_id.iloc[0]),
                        "delta_seconds": seconds,
                        "pathway": pathway,
                        "target": target,
                        "pairs": len(base),
                        "standardized_rms": float(np.sqrt(np.mean(delta[:, target_index] ** 2))),
                        "original_rms": float(
                            np.sqrt(np.mean(
                                (delta[:, target_index] * target_scale[target_index]) ** 2
                            ))
                        ),
                        "latent_delta_rms_median": float(np.median(latent_delta)),
                        "input_delta_rms_median": float(np.nanmedian(changed_input_values))
                        if np.isfinite(changed_input_values).any() else np.nan,
                    })
    return pd.DataFrame(records)
