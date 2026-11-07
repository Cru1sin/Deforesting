"""Front RGB feature loading and phase bins for native frost trajectories."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dataset_tools import DatasetLoader
from dataset_tools.cycle_metadata import image_root
from image_models.dinov2_features import KEYS, POOL_COLUMNS, merge_cached_features
from output_layout import PROJECT


def frost_interval(rows, preparation_start):
    """Stop at preparation; recovery is outside frost accumulation, partial stays explicit."""
    observed = rows.loc[
        rows.cycle_stage.isin(["defrost_preparation", "defrost"]), "image_time"
    ].min()
    stops = pd.to_datetime(pd.Series([preparation_start, observed])).dropna()
    if len(stops):
        rows = rows.loc[rows.image_time.lt(stops.min())]
    return rows.loc[~rows.cycle_stage.eq("recovery")]


def load_front_features(dataset, feature_caches, *, pooling="cls", cycles=None, image_list=None):
    """All catalog cycles, from stable heating to defrost; no quality/COP screening."""
    loader = DatasetLoader(dataset)
    images = image_root(loader.dataset_root)
    metadata = loader.load_image_metadata()
    metadata = metadata.loc[metadata.camera_role.eq("front")].copy()
    if image_list is not None:
        keys = pd.read_csv(image_list, usecols=KEYS).drop_duplicates()
        metadata = metadata.merge(keys, on=KEYS, how="inner", validate="one_to_one")
    metadata["image_time"] = pd.to_datetime(metadata.image_time)
    columns = POOL_COLUMNS[pooling]
    catalog = loader.list_cycles(include_rb=True)
    if cycles is not None:
        catalog = catalog.loc[catalog.cycle_name.isin(cycles)]
    parts = []
    for cycle in catalog.itertuples():
        start = pd.to_datetime(cycle.stable_heating_start)
        if pd.isna(start):
            start = pd.to_datetime(cycle.heating_start)
        end = pd.to_datetime(cycle.defrost_start)
        if pd.isna(end):
            end = pd.to_datetime(cycle.end_time)
        rows = metadata.loc[
            metadata.cycle_name.eq(cycle.cycle_name)
            & metadata.image_time.ge(start)
            & metadata.image_time.lt(end)
        ]
        rows = frost_interval(rows, cycle.defrost_preparation_start)
        if rows.empty:
            continue
        paths = [
            Path(p) / f"{cycle.cycle_name}.parquet" if Path(p).is_dir() else Path(p)
            for p in feature_caches
        ]
        rows = rows.merge(
            merge_cached_features(rows, paths, columns), on=KEYS, validate="one_to_one"
        )
        rows["experiment_id"] = cycle.experiment_id
        rows["local_image_available"] = rows.file_name.map(
            lambda name: (images / cycle.cycle_name / "front" / name).is_file()
        )
        parts.append(rows)
    if not parts:
        raise ValueError("no front images within heating boundaries")
    frames = pd.concat(parts, ignore_index=True).sort_values(
        ["cycle_name", "image_time", "file_name"]
    )
    values = frames[columns].to_numpy(float)
    frames["embedding_available"] = np.isfinite(values).all(axis=1) & (
        np.linalg.norm(values, axis=1) > 0
    )
    return frames.reset_index(drop=True)


def phase_bins(phase):
    """Nearest tenth, with midpoint ties assigned upward; one bin per image."""
    return np.clip(np.floor(np.asarray(phase) * 10 + 0.5).astype(int), 0, 10)


def match_states(points, loader, feature_caches, *, max_age_seconds=120):
    """Use the latest pre-action front frame; cached features may outlive local images."""
    from image_models.dinov2_features import DINO_COLUMNS, merge_cached_features

    rows = []
    for point in points.to_dict("records"):
        row = {
            **point,
            "file_name": "",
            "image_time": pd.NaT,
            "image_age_seconds": np.nan,
            "local_image_available": False,
            "match_status": "no_selected_point",
            **dict.fromkeys(DINO_COLUMNS, np.nan),
        }
        time = pd.to_datetime(point["selected_time"])
        metadata = loader.load_image_metadata(point["cycle_name"])
        metadata = metadata.loc[metadata.camera_role.eq("front")].copy()
        metadata["image_time"] = pd.to_datetime(metadata.image_time)
        if pd.notna(time):
            past = metadata.loc[metadata.image_time.le(time)].sort_values(
                ["image_time", "file_name"]
            )
            row["match_status"] = "no_recent_pre_action_image"
            if len(past) and (time - past.image_time.iloc[-1]).total_seconds() <= max_age_seconds:
                image = past.iloc[[-1]]
                name = point["cycle_name"]
                paths = [p / f"{name}.parquet" if p.is_dir() else p for p in feature_caches]
                values = merge_cached_features(image, paths).iloc[0]
                row.update(values.to_dict())
                row.update(
                    image_time=image.image_time.iloc[0],
                    image_age_seconds=(time - image.image_time.iloc[0]).total_seconds(),
                    local_image_available=(
                        loader.dataset_root / "images" / name / "front" / values.file_name
                    ).is_file(),
                    match_status="matched"
                    if values[DINO_COLUMNS].notna().all()
                    else "embedding_missing",
                )
        rows.append(row)
    return pd.DataFrame(rows)


def collect_states(
    root: Path,
    *,
    seed=0,
    dataset=(PROJECT / "dataset"),
    point="cv",
    cohort="regular",
    decisions=None,
    feature_caches=(),
    model="B0",
    max_age_seconds=120,
    cycle_list=None,
    reference_states=None,
):
    from dataset_tools import DatasetLoader

    loader = DatasetLoader(dataset)
    from image_models.experiments import common_cycle_metrics
    from image_models.inputs import reliable_rgb_cohort

    cycles = loader.list_cycles()
    eligible = reliable_rgb_cohort(
        loader, include_rb=cohort != "regular", require_complete_peak=True
    )
    cycles = cycles.loc[cycles.cycle_name.isin(eligible.cycle_name)]
    if point == "cv":
        metrics = pd.read_csv(root / "cycle_metrics.csv")
        common = common_cycle_metrics(metrics.loc[metrics.seed.eq(seed)])
        cycles = cycles.loc[cycles.cycle_name.isin(common.cycle_name)]
    if cycle_list is not None:
        cycles = cycles.loc[cycles.cycle_name.isin(pd.read_csv(cycle_list).cycle_name)]
    regular = set(loader.list_cycles(include_rb=False).cycle_name)
    if cohort != "all":
        cycles = cycles.loc[cycles.cycle_name.isin(regular).eq(cohort == "regular")]
    if point == "cv":
        metrics = pd.read_csv(root / "cycle_metrics.csv")
        points = metrics.loc[
            metrics.version.eq(model) & metrics.seed.eq(seed),
            ["cycle_name", "experiment_id", "selected_time"],
        ].copy()
    elif point == "observed":
        points = cycles[["cycle_name", "experiment_id", "defrost_start"]].rename(
            columns={"defrost_start": "selected_time"}
        )
    else:
        if decisions is None:
            raise ValueError("--decisions is required for optimal or replayed RB points")
        table = pd.read_csv(decisions / "cycle_comparison.csv")
        points = table[
            ["cycle_name", "experiment_id", "t_star" if point == "optimal" else "t_RB"]
        ].rename(columns={"t_star": "selected_time", "t_RB": "selected_time"})
    points = points.loc[points.cycle_name.isin(cycles.cycle_name)].copy()
    points["selected_time"] = pd.to_datetime(points.selected_time, format="mixed")
    if feature_caches:
        result = match_states(points, loader, feature_caches, max_age_seconds=max_age_seconds)
    elif point == "cv":
        # Preserve the original model's exact input snapshot for legacy runs.
        rows = []
        for record in points.itertuples(index=False):
            table = pd.read_parquet(root / "base" / f"{record.cycle_name}.parquet")
            match = table.loc[pd.to_datetime(table.candidate_defrost_time).eq(record.selected_time)]
            if len(match) != 1:
                raise ValueError(
                    f"expected one RGB state at {record.cycle_name} {record.selected_time}"
                )
            rows.append(
                {**record._asdict(), **match.filter(regex=r"^dinov2_\d{3}$").iloc[0].to_dict()}
            )
        result = pd.DataFrame(rows)
    else:
        raise ValueError("--feature-cache is required for these points")
    if result.empty:
        raise ValueError("no cycles in the selected cohort")
    result["point_source"], result["cohort"] = point, cohort
    result["cycle_id"] = result.cycle_name.str.rsplit("_", n=1).str[-1]
    if "local_image_available" in result:
        result.loc[
            ~result.local_image_available, result.filter(regex=r"^dinov2_\d{3}$").columns
        ] = np.nan
        result.loc[~result.local_image_available, "match_status"] = "local_image_missing"
    reference = pd.read_csv(reference_states) if reference_states is not None else None
    return state_coordinates(result, reference)


def state_coordinates(result, reference=None):
    """Keep within-cohort consistency; optionally use a fixed optimal-state reference."""
    result = result.copy()
    features = result.filter(regex=r"^dinov2_\d{3}$").to_numpy(float)
    norms = np.linalg.norm(features, axis=1)
    valid = np.isfinite(features).all(axis=1) & (norms > 0)
    result["embedding_available"], result["embedding_norm"] = valid, norms
    result[["cosine_to_centroid", "cosine_to_reference", "pca1", "pca2"]] = np.nan
    if not valid.any():
        return result
    unit = features[valid] / norms[valid, None]
    center = unit.mean(axis=0)
    center /= np.linalg.norm(center)
    result.loc[valid, "cosine_to_centroid"] = unit @ center
    fit = unit
    if reference is not None:
        fit = reference.filter(regex=r"^dinov2_\d{3}$").to_numpy(float)
        length = np.linalg.norm(fit, axis=1)
        keep = np.isfinite(fit).all(axis=1) & (length > 0)
        fit = fit[keep] / length[keep, None]
        if not len(fit):
            raise ValueError("reference has no valid embeddings")
    center = fit.mean(axis=0)
    center /= np.linalg.norm(center)
    result.loc[valid, "cosine_to_reference"] = unit @ center
    coordinates = (
        PCA(n_components=2, random_state=0).fit(fit).transform(unit)
        if len(fit) >= 2
        else np.zeros((len(unit), 2))
    )
    result.loc[valid, ["pca1", "pca2"]] = coordinates
    return result


def match_trajectory_frames(rows, frames, *, metric="visual_frost_score", max_age_seconds=120):
    """Shared pre-action matching for plots and numerical trajectory audits."""
    frames = frames.copy()
    frames["image_time"] = pd.to_datetime(frames.image_time)
    fields = ["cycle_name", "image_time", "file_name", metric] + [
        c
        for c in ("early_reference", "is_beyond_reference", "reference_cycle", "reference_index")
        if c in frames
    ]
    return pd.merge_asof(
        rows.sort_values("candidate_defrost_time"),
        frames[list(dict.fromkeys(fields))].sort_values("image_time"),
        left_on="candidate_defrost_time",
        right_on="image_time",
        by="cycle_name",
        direction="backward",
        tolerance=pd.Timedelta(seconds=max_age_seconds),
    )


def trajectory_scores(
    cycles: pd.DataFrame,
    candidates: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    max_age_seconds: int = 120,
) -> pd.DataFrame:
    """Match pre-action frames carrying either the historical distance or saved offline score."""
    frames = frames.copy()
    frames["image_time"] = pd.to_datetime(frames.image_time)
    metric = "visual_frost_score" if "visual_frost_score" in frames else "visual_distance"
    if metric == "visual_frost_score":
        frames.loc[frames.is_beyond_reference.eq(True), metric] = 1.0
    if metric == "visual_distance":
        feature_cols = frames.filter(regex=r"^dinov2_\d{3}$").columns.tolist()
        vectors = frames[feature_cols].to_numpy(float)
        lengths = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & (lengths > 0)
        frames["visual_distance"] = np.nan
        frames["early_reference"] = False
        starts = cycles.set_index("cycle_name").stable_heating_start.map(pd.Timestamp)
        elapsed = (frames.image_time - frames.cycle_name.map(starts)).dt.total_seconds()
        early = (
            frames.loc[valid & elapsed.between(0, 120)]
            .sort_values("image_time")
            .drop_duplicates("cycle_name")
        )
        if early.empty:
            raise ValueError("no local cached front frame within 120 s of stable heating start")
        frames.loc[early.index, "early_reference"] = True
        reference = frames.loc[early.index, feature_cols].to_numpy(float)
        reference /= np.linalg.norm(reference, axis=1)[:, None]
        center = reference.mean(axis=0)
        center /= np.linalg.norm(center)
        frames.loc[valid, "visual_distance"] = 1 - (vectors[valid] / lengths[valid, None]) @ center
    rows = candidates.merge(cycles, on="cycle_name", how="inner", validate="many_to_one")
    rows["candidate_defrost_time"] = pd.to_datetime(rows.candidate_defrost_time)
    for field in (
        "stable_heating_start",
        "t_star",
        "t_RB",
        "cycle_cop_basin_1pct_start",
        "cycle_cop_basin_1pct_end",
    ):
        rows[field] = pd.to_datetime(rows[field])
    rows["in_basin_grid"] = (
        rows.on_action_grid.fillna(False)
        & rows.cycle_cop_eligible.fillna(False)
        & rows.cycle_cop.ge(0.99 * rows.COP_eff_selected)
        & rows.candidate_defrost_time.between(
            rows.cycle_cop_basin_1pct_start, rows.cycle_cop_basin_1pct_end
        )
    )
    rows = match_trajectory_frames(rows, frames, metric=metric, max_age_seconds=max_age_seconds)
    if metric == "visual_distance":
        rows["reference_count"] = len(early)
    elif "frost_end" in rows:
        outside = rows.candidate_defrost_time.ge(pd.to_datetime(rows.frost_end))
        rows.loc[outside, metric] = np.nan
        rows.loc[outside, "is_beyond_reference"] = False
    rows["image_age_seconds"] = (rows.candidate_defrost_time - rows.image_time).dt.total_seconds()
    rows = rows.sort_values(["cycle_name", "candidate_defrost_time"]).reset_index(drop=True)
    if metric == "visual_distance":
        rows.attrs["early_reference"] = frames.loc[
            early.index, ["cycle_name", "file_name", "image_time", "visual_distance"]
        ].copy()
    return rows


def trajectory_snapshots(curve):
    """Use the displayed basin endpoints and available in-basin grid extrema."""
    band = curve.loc[curve.in_basin_grid]
    metric = "visual_frost_score" if "visual_frost_score" in curve else "visual_distance"
    valid = band.dropna(subset=[metric])
    bounds = [
        curve.loc[curve.candidate_defrost_time.eq(curve[key].iloc[0])]
        for key in ("cycle_cop_basin_1pct_start", "cycle_cop_basin_1pct_end")
    ]
    return [part.iloc[0] if len(part) else None for part in bounds] + [
        valid.loc[valid[metric].idxmax()] if len(valid) else None,
        valid.loc[valid[metric].idxmin()] if len(valid) else None,
    ]


def sample_phase_images(pool, *, seed=0):
    """The grid and its downloads use exactly the same independently sampled cells."""
    rng = np.random.default_rng(seed)
    if pool.groupby("phase_bin").cycle_name.nunique().reindex(range(11), fill_value=0).min() < 6:
        raise ValueError("each phase bin needs images from at least six cycles")
    return pd.concat(
        [
            (
                pool.loc[pool.phase_bin.eq(b)]
                .sample(frac=1, random_state=rng)
                .drop_duplicates("cycle_name")
                .head(6)
                .assign(sample_row=range(1, 7))
            )
            for b in range(11)
        ],
        ignore_index=True,
    )
