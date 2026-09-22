"""Compare frozen RGB states at supported optima, replayed/observed RB or CV decisions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


def match_states(points, loader, feature_caches, *, max_age_seconds=120):
    """Use the latest pre-action front frame; cached features may outlive local images."""
    from image_models.dinov2_features import DINO_COLUMNS, merge_cached_features

    rows = []
    for point in points.to_dict("records"):
        row = {**point, "file_name": "", "image_time": pd.NaT,
               "image_age_seconds": np.nan, "local_image_available": False,
               "match_status": "no_selected_point", **dict.fromkeys(DINO_COLUMNS, np.nan)}
        time = pd.to_datetime(point["selected_time"])
        metadata = loader.load_image_metadata(point["cycle_name"])
        metadata = metadata.loc[metadata.camera_role.eq("front")].copy()
        metadata["image_time"] = pd.to_datetime(metadata.image_time)
        if pd.notna(time):
            past = metadata.loc[metadata.image_time.le(time)].sort_values(["image_time", "file_name"])
            row["match_status"] = "no_recent_pre_action_image"
            if len(past) and (time - past.image_time.iloc[-1]).total_seconds() <= max_age_seconds:
                image = past.iloc[[-1]]
                name = point["cycle_name"]
                paths = [p / f"{name}.parquet" if p.is_dir() else p for p in feature_caches]
                values = merge_cached_features(image, paths).iloc[0]
                row.update(values.to_dict())
                row.update(image_time=image.image_time.iloc[0],
                           image_age_seconds=(time - image.image_time.iloc[0]).total_seconds(),
                           local_image_available=(loader.dataset_root / "images" / name / "front" / values.file_name).is_file(),
                           match_status="matched" if values[DINO_COLUMNS].notna().all() else "embedding_missing")
        rows.append(row)
    return pd.DataFrame(rows)


def collect_states(root: Path, *, seed=0, dataset=Path("dataset"), point="cv", cohort="regular",
                   decisions=None, feature_caches=(), model="B0", max_age_seconds=120,
                   cycle_list=None, reference_states=None):
    from dataset_tools import DatasetLoader

    loader = DatasetLoader(dataset)
    from image_models.inputs import reliable_rgb_cohort
    from image_models.core import common_cycle_metrics

    cycles = loader.list_cycles()
    eligible = reliable_rgb_cohort(loader, include_rb=cohort != "regular", require_complete_peak=True)
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
        points = metrics.loc[metrics.version.eq(model) & metrics.seed.eq(seed),
                             ["cycle_name", "experiment_id", "selected_time"]].copy()
    elif point == "observed":
        points = cycles[["cycle_name", "experiment_id", "defrost_start"]].rename(columns={"defrost_start": "selected_time"})
    else:
        if decisions is None:
            raise ValueError("--decisions is required for optimal or replayed RB points")
        table = pd.read_csv(decisions / "cycle_comparison.csv")
        points = table[["cycle_name", "experiment_id", "t_star" if point == "optimal" else "t_RB"]].rename(
            columns={"t_star": "selected_time", "t_RB": "selected_time"})
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
                raise ValueError(f"expected one RGB state at {record.cycle_name} {record.selected_time}")
            rows.append({**record._asdict(), **match.filter(regex=r"^dinov2_\d{3}$").iloc[0].to_dict()})
        result = pd.DataFrame(rows)
    else:
        raise ValueError("--feature-cache is required for these points")
    if result.empty:
        raise ValueError("no cycles in the selected cohort")
    result["point_source"], result["cohort"] = point, cohort
    result["cycle_id"] = result.cycle_name.str.rsplit("_", n=1).str[-1]
    if "local_image_available" in result:
        result.loc[~result.local_image_available, result.filter(regex=r"^dinov2_\d{3}$").columns] = np.nan
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
    coordinates = PCA(n_components=2, random_state=0).fit(fit).transform(unit) if len(fit) >= 2 else np.zeros((len(unit), 2))
    result.loc[valid, ["pca1", "pca2"]] = coordinates
    return result


def render(root: Path, output: Path, *, seed: int = 0, **options) -> pd.DataFrame:
    states = collect_states(root, seed=seed, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(output.with_suffix(".csv"), index=False)
    all_states = states
    states = states.loc[states.embedding_available].copy()
    if states.empty:
        raise ValueError("no available embeddings; missing-point audit saved to CSV")
    title = {"cv": f"CV model: {options.get('model', 'B0')}", "optimal": "Supported COP optimum", "rb": "Replayed RB",
             "observed": "Observed defrost"}[options.get("point", "cv")]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (scatter, consistency) = plt.subplots(1, 2, figsize=(14, max(7, len(states) * .19)), constrained_layout=True)
    scatter.scatter(states.pca1, states.pca2, s=35, alpha=.8, color="#4779a4")
    for row in states.itertuples():
        scatter.annotate(str(row.cycle_id), (row.pca1, row.pca2), xytext=(3, 3),
                         textcoords="offset points", fontsize=7)
    if len(states) >= 3:
        covariance = np.cov(states[["pca1", "pca2"]].to_numpy().T)
        values, vectors = np.linalg.eigh(covariance)
        order = values.argsort()[::-1]
        values, vectors = values[order], vectors[:, order]
        angle = np.degrees(np.arctan2(*vectors[:, 0][::-1]))
        scatter.add_patch(Ellipse(states[["pca1", "pca2"]].mean().to_numpy(), 2 * 1.96 * np.sqrt(values[0]), 2 * 1.96 * np.sqrt(values[1]),
                                  angle=angle, facecolor="#5b8ff9", alpha=0.12, edgecolor="#3867d6", lw=1.5))
    scatter.axhline(0, color="0.7", lw=0.7); scatter.axvline(0, color="0.7", lw=0.7)
    scatter.set(xlabel="PCA 1 (unit-normalized DINOv2)", ylabel="PCA 2", title=f"Frozen RGB state: {title}")

    metric = "cosine_to_reference" if options.get("reference_states") else "cosine_to_centroid"
    ordered = states.sort_values(metric)
    values = ordered[metric].to_numpy()
    y = np.arange(len(ordered))
    consistency.scatter(values, y, s=24, color="#c77e57")
    consistency.set(yticks=y, yticklabels=ordered.cycle_id.astype(str),
                    xlabel="Cosine to fixed optimal centroid" if options.get("reference_states")
                    else "Cosine to cohort centroid", ylabel="Cycle ID (least similar first)",
                    title="Identify inconsistent cycles", xlim=(max(-1, values.min() - .02), 1.005))
    consistency.tick_params(axis="y", labelsize=8)
    consistency.invert_yaxis()
    consistency.axvline(np.median(values), color="#777777", lw=1,
                       label=f"median = {np.median(values):.3f}")
    consistency.legend(frameon=True, loc="lower right")
    figure.suptitle(f"{title} | {options.get('cohort', 'regular')} cycles | "
                   f"{len(states)}/{len(all_states)} available states", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return all_states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/image_models/core_comparison_ridge106"))
    parser.add_argument("--output", type=Path, default=Path("output/RGB结霜状态/only_rgb_frost_state.png"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--point", choices=("optimal", "rb", "observed", "cv"), default="cv")
    parser.add_argument("--cohort", choices=("regular", "rb", "all"), default="regular")
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--feature-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--model", default="B0", help="model name in cycle_metrics.csv")
    parser.add_argument("--cycle-list", type=Path, help="CSV with cycle_name for a fixed paired cohort")
    parser.add_argument("--reference-states", type=Path, help="optimal-state CSV for fixed centroid and PCA")
    parser.add_argument("--max-age-seconds", type=int, default=120)
    args = parser.parse_args()
    states = render(args.root, args.output, seed=args.seed, dataset=args.dataset,
                    point=args.point, cohort=args.cohort, decisions=args.decisions,
                    feature_caches=args.feature_cache, model=args.model,
                    max_age_seconds=args.max_age_seconds, cycle_list=args.cycle_list,
                    reference_states=args.reference_states)
    valid = states.loc[states.embedding_available, "cosine_to_centroid"]
    print(f"{len(valid)}/{len(states)} states; median cosine={valid.median():.6f}; "
          f"IQR={valid.quantile(.25):.6f}–{valid.quantile(.75):.6f}")
