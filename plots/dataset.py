"""Explicit Dataset figure generation and publication of figure assets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from dataset_tools.cycle_metadata import image_root
from output_layout import PROJECT


def render_publication_asset(
    dataset_dir: Path,
    record: Mapping[str, Any],
    *,
    output_path: Path | None = None,
    recovery_boundaries: Path | None = None,
) -> None:
    """Render one Dataset physical-state publication."""
    from dataset_tools.cycle_metadata import read_catalog
    from dataset_tools.dataset_paths import read_json
    from dataset_tools.local_images import (
        _sensor_coverage_intervals,
        image_coverage_intervals,
        rgb_overall_intervals,
    )
    from plots.cycle import render_cycle_publication

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    if recovery_boundaries is not None:
        from dataset_tools.builder.detect_cycles import recovery_stages

        boundaries = pd.read_csv(recovery_boundaries)
        selected = boundaries.loc[
            boundaries.cycle_name.eq(cycle_name) & boundaries.recovery_rule.eq("frequency-setpoint")
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one recovery boundary for {cycle_name}")
        row = selected.iloc[0]
        boundary = pd.to_datetime(row.stable_heating_start)
        record = {
            **record,
            "recovery_status": row.recovery_status,
            "boundaries": {
                **record["boundaries"],
                "stable_heating_start": boundary,
                **{key: row[key] for key in ("heating_start", "heating_origin") if key in row},
            },
        }
        frame["cycle_stage"] = recovery_stages(frame, record["boundaries"])
    registry = read_json(dataset_dir / "channel_registry.json")
    if not isinstance(registry, dict):
        raise ValueError("channel_registry.json must contain an object")
    metadata = pd.read_parquet(image_root(dataset_dir) / "image_metadata.parquet")
    catalog = read_catalog(dataset_dir)
    roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
    scoped = metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)]
    intervals = image_coverage_intervals(frame, scoped, registry)
    render_cycle_publication(
        frame,
        record,
        output_path or dataset_dir / str(assets["publication"]),
        sensor_intervals=_sensor_coverage_intervals(frame, registry),
        rgb_intervals=rgb_overall_intervals(frame, intervals, roles),
    )


def _render_rgb_panel(
    dataset_dir: Path,
    record: Mapping[str, Any],
    metadata: pd.DataFrame,
    registry: Mapping[str, Any],
    camera_roles: tuple[str, ...],
    frame: pd.DataFrame | None = None,
    images: pd.DataFrame | None = None,
    intervals: (Mapping[str, Mapping[str, list[tuple[pd.Timestamp, pd.Timestamp]]]] | None) = None,
) -> None:
    from dataset_tools.local_images import scan_cycle_images
    from plots.cycle import render_rgb_panel

    cycle_name = str(record["cycle_name"])
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    if frame is None:
        frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    if images is None:
        images = scan_cycle_images(dataset_dir, cycle_name, metadata)
    if intervals is None:
        from dataset_tools.local_images import image_coverage_intervals

        expected = metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)]
        intervals = image_coverage_intervals(frame, expected, registry)
    render_rgb_panel(
        record,
        frame,
        images,
        intervals,
        camera_roles,
        dataset_dir / str(assets["rgb_panel"]),
    )


def render_effective_cop_assets(
    dataset_dir: Path, decisions_path: Path, *, cycle_name: str | None = None, n_jobs: int = 6
) -> None:
    """Publish missing cycle COP figures from saved full candidates, before admission."""
    from joblib import Parallel, delayed, parallel_config

    from dataset_tools import DatasetLoader
    from dataset_tools.cycle_metadata import read_catalog, write_catalog
    from dataset_tools.dataset_paths import read_json
    from image_models.cycle_selection import screen_cv_cycles
    from plots.defrost_decision import load_native_frost_frames, render_current_decision_cycle

    dataset_dir, decisions_path = Path(dataset_dir), Path(decisions_path)
    loader = DatasetLoader(dataset_dir)
    decisions = pd.read_csv(decisions_path, low_memory=False)
    if not decisions.preparation_heat.eq("zero").all():
        raise ValueError("Dataset effective COP uses zero preparation heat")
    screen_cv_cycles(decisions, loader.list_cycles(), exclude_rb=False)
    for column in decisions:
        if column.endswith(("_time", "_start", "_end")) or column in {"t_star", "t_RB"}:
            decisions[column] = pd.to_datetime(decisions[column], format="mixed")
    catalog = read_catalog(dataset_dir)
    records = {r["cycle_name"]: r for r in catalog["cycles"]}
    if cycle_name is not None and cycle_name not in set(decisions.cycle_name):
        raise KeyError(f"No saved COP candidates: {cycle_name}")
    targets = {}
    for name in decisions.cycle_name.unique():
        if cycle_name is not None and name != cycle_name:
            continue
        relative = str(Path(records[name]["assets"]["csv"]).with_name(f"{name}_effective_cop.png"))
        if cycle_name is not None or not (dataset_dir / relative).is_file():
            targets[name] = relative
    if not targets:
        return
    settings = read_json(decisions_path.parent / "run_settings.json")
    frames = load_native_frost_frames()
    frost = dict(tuple(frames.groupby("cycle_name")))
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel(verbose=5)(
            delayed(render_current_decision_cycle)(
                curve,
                loader,
                dataset_dir / targets[name],
                frost_frames=frost.get(name, frames.iloc[:0]),
            )
            for name, curve in decisions.loc[decisions.cycle_name.isin(targets)].groupby(
                "cycle_name"
            )
        )
    for name, relative in targets.items():
        records[name]["assets"]["effective_cop_without_trunc"] = relative
        records[name]["effective_cop_without_trunc"] = dict(
            status=str(decisions.loc[decisions.cycle_name.eq(name), "cycle_status"].iloc[0]),
            preparation_heat="zero",
            effective_heat_rule="without_trunc",
            source_table=str(decisions_path.resolve()),
            selection_method="refrigerant_effective_cop_maximum",
            electricity_accounting=settings["electricity_accounting"],
        )
    write_catalog(dataset_dir, catalog)
    print(f"[Dataset COP figures] published {len(targets)} cycles", flush=True)


def render_dataset(
    dataset_dir: Path,
    cycle_name: str | None,
    *,
    publication: bool = True,
    panel: bool = True,
    fetch_cloud_images: bool = False,
    cleanup_downloaded_images: bool = False,
    n_jobs: int = 6,
    recovery_boundaries: Path | None = None,
) -> Path:
    """Render one cycle, or every Catalog cycle when no name is given."""
    from dataset_tools.cycle_metadata import read_catalog
    from dataset_tools.dataset_paths import read_json

    catalog = read_catalog(dataset_dir)
    if cycle_name is None:
        records = [record for record in catalog["cycles"] if isinstance(record, Mapping)]
        if n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        if fetch_cloud_images and panel:
            from dataset_tools.cloud_images import materialize_image_members
            from dataset_tools.local_images import scan_cycle_images

            registry = read_json(dataset_dir / "channel_registry.json")
            if not isinstance(registry, dict):
                raise ValueError("channel_registry.json must contain an object")
            metadata = pd.read_parquet(image_root(dataset_dir) / "image_metadata.parquet")
            contexts = {
                str(record["cycle_name"]): _rgb_panel_inputs(dataset_dir, record, catalog, metadata)
                for record in records
            }
            requests = {cycle: context[3] for cycle, context in contexts.items()}
            by_name = {str(record["cycle_name"]): record for record in records}
            print(f"[render] {len(records)} cycles with {n_jobs} workers", flush=True)
            if publication:
                for record in records:
                    render_publication_asset(
                        dataset_dir, record, recovery_boundaries=recovery_boundaries
                    )
            with materialize_image_members(
                dataset_dir,
                requests,
                cleanup_downloaded=cleanup_downloaded_images,
                n_jobs=n_jobs,
            ) as completed_cycles:
                for completed, name in enumerate(completed_cycles, start=1):
                    roles, frame, _, _ = contexts[name]
                    images = scan_cycle_images(dataset_dir, name, metadata)
                    _render_rgb_panel(
                        dataset_dir,
                        by_name[name],
                        metadata,
                        registry,
                        roles,
                        frame=frame,
                        images=images,
                    )
                    print(f"[render:done] {completed}/{len(records)} {name}", flush=True)
            return dataset_dir
        from joblib import parallel_config
        from sklearn.utils.parallel import Parallel, delayed

        print(f"[render] {len(records)} cycles with {n_jobs} workers", flush=True)
        with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
            results = Parallel(return_as="generator_unordered")(
                delayed(_render_cycle_job)(
                    dataset_dir,
                    str(record["cycle_name"]),
                    publication,
                    panel,
                    fetch_cloud_images,
                    cleanup_downloaded_images,
                    recovery_boundaries,
                )
                for record in records
            )
            for completed, name in enumerate(results, start=1):
                print(f"[render:done] {completed}/{len(records)} {name}", flush=True)
        return dataset_dir
    if not any(
        isinstance(record, Mapping) and str(record.get("cycle_name")) == cycle_name
        for record in catalog["cycles"]
    ):
        raise KeyError(f"unknown cycle: {cycle_name}")
    record = next(
        record
        for record in catalog["cycles"]
        if isinstance(record, Mapping) and str(record.get("cycle_name")) == cycle_name
    )
    if publication:
        render_publication_asset(dataset_dir, record, recovery_boundaries=recovery_boundaries)
    if panel:
        from dataset_tools.local_images import scan_cycle_images

        registry = read_json(dataset_dir / "channel_registry.json")
        if not isinstance(registry, dict):
            raise ValueError("channel_registry.json must contain an object")
        metadata = pd.read_parquet(image_root(dataset_dir) / "image_metadata.parquet")
        roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
        images = scan_cycle_images(dataset_dir, cycle_name, metadata)
        if fetch_cloud_images:
            from dataset_tools.cloud_images import materialize_cycle_image_members

            roles, frame, images, missing = _rgb_panel_inputs(
                dataset_dir, record, catalog, metadata
            )

            with materialize_cycle_image_members(
                dataset_dir,
                cycle_name,
                missing,
                fetch_cloud=True,
                cleanup_downloaded=cleanup_downloaded_images,
            ) as cycle_dir:
                images = scan_cycle_images(dataset_dir, cycle_name, metadata, cycle_dir=cycle_dir)
        _render_rgb_panel(
            dataset_dir,
            record,
            metadata,
            registry,
            roles,
            frame=frame if fetch_cloud_images else None,
            images=images,
        )
    return dataset_dir


def _rgb_panel_inputs(
    dataset_dir: Path,
    record: Mapping[str, Any],
    catalog: Mapping[str, Any],
    metadata: pd.DataFrame,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame, list[str]]:
    from dataset_tools.local_images import scan_cycle_images
    from plots.cycle import build_rgb_panel_targets, select_rgb_panel_cells

    cycle_name = str(record["cycle_name"])
    roles = _experiment_camera_roles(catalog, metadata, str(record["experiment_id"]))
    assets = record.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError(f"cycle assets are missing: {cycle_name}")
    frame = pd.read_parquet(dataset_dir / str(assets["parquet"]))
    expected = metadata.loc[metadata["cycle_name"].astype(str).eq(cycle_name)].copy()
    expected["path"] = [
        Path(str(role)) / str(name)
        for role, name in zip(expected["camera_role"], expected["file_name"], strict=True)
    ]
    images = scan_cycle_images(dataset_dir, cycle_name, metadata)
    cells = select_rgb_panel_cells(
        expected,
        build_rgb_panel_targets(record, frame),
        list(roles),
    )
    local = set(
        zip(images["camera_role"].astype(str), images["file_name"].astype(str), strict=True)
    )
    missing = sorted(
        {
            path.as_posix()
            for role, paths in cells.items()
            for path in paths
            if path is not None and (role, path.name) not in local
        }
    )
    return roles, frame, images, missing


def _render_cycle_job(
    dataset_dir: Path,
    cycle_name: str,
    publication: bool,
    panel: bool,
    fetch_cloud_images: bool,
    cleanup_downloaded_images: bool,
    recovery_boundaries: Path | None = None,
) -> str:
    print(f"[render:start] {cycle_name}", flush=True)
    render_dataset(
        dataset_dir,
        cycle_name,
        publication=publication,
        panel=panel,
        fetch_cloud_images=fetch_cloud_images,
        cleanup_downloaded_images=cleanup_downloaded_images,
        recovery_boundaries=recovery_boundaries,
    )
    return cycle_name


def _experiment_camera_roles(
    catalog: Mapping[str, Any], image_metadata: pd.DataFrame, experiment_id: str
) -> tuple[str, ...]:
    cycle_names = {
        str(record["cycle_name"])
        for record in catalog["cycles"]
        if isinstance(record, Mapping) and str(record.get("experiment_id")) == experiment_id
    }
    scoped = image_metadata.loc[image_metadata["cycle_name"].astype(str).isin(cycle_names)]
    return tuple(sorted(scoped["camera_role"].dropna().astype(str).unique()))


def refresh_effective_decision_figures(dataset_dir: Path, sources: Mapping[str, Path]) -> None:
    """Replace only existing COP image assets and their Dataset copies; keep decision metadata."""
    from datetime import datetime, timezone
    from filecmp import cmp
    from shutil import copy2

    from dataset_tools.cycle_metadata import read_catalog

    dataset_dir = Path(dataset_dir)
    catalog = {r["cycle_name"]: r for r in read_catalog(dataset_dir)["cycles"]}
    archive = (PROJECT / "output/99_历史归档/dataset_effective_cop") / datetime.now(
        timezone.utc
    ).strftime("figure_refresh_%Y%m%d_%H%M%S_%f")
    targets = []
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        assets = catalog[name]["assets"]
        if "effective_cop_without_trunc" not in assets:
            continue
        copies = [dataset_dir / assets["effective_cop_without_trunc"]]
        for folder in ["training_data", "RB失败循环/早除霜", "RB失败循环/迟除霜"]:
            duplicate = dataset_dir / folder / f"{name}_effective_COP.png"
            if duplicate.is_file():
                copies.append(duplicate)
        targets.extend((source, p) for p in copies)
    for source, target in targets:
        if target.is_file() and cmp(source, target, shallow=False):
            continue
        if target.is_file():
            backup = archive / target.relative_to(dataset_dir)
            backup.parent.mkdir(parents=True, exist_ok=True)
            copy2(target, backup)
        copy2(source, target)
    print(
        f"[Dataset COP figures] {len(sources)} cycles; {len(targets)} image assets/copies",
        flush=True,
    )


def publish_effective_decision_assets(dataset_dir: Path, run: Path, figures: Path) -> None:
    """Publish the selected zero-preparation COP definition and archive prior figures."""
    import json
    from shutil import copy2

    from dataset_tools.cycle_metadata import read_catalog, write_catalog

    catalog = read_catalog(dataset_dir)
    settings_path = run / "run_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    without_trunc = settings.get("effective_heat_rule") == "without_trunc"
    summary = pd.read_csv(run / "cycle_comparison.csv").set_index("cycle_name")
    if not summary.preparation_heat.eq("zero").all():
        raise ValueError("Dataset effective COP uses zero preparation heat")
    source_dir = figures / "cost_function_effective_cop_cycle"
    sources = {
        name: source_dir / f"cycle_{int(name.rsplit('_', 1)[-1]):03d}_publication.png"
        for name in summary.index
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    archive = (PROJECT / "output/99_历史归档/dataset_effective_cop") / run.name
    archive.mkdir(parents=True, exist_ok=True)
    if not (archive / "cycle_catalog.json").exists():
        copy2(dataset_dir / "cycle_catalog.json", archive / "cycle_catalog.json")
    for record in catalog["cycles"]:
        name = record["cycle_name"]
        if name not in sources:
            continue
        assets = record["assets"]
        row = summary.loc[name]
        decision_key = "effective_cop_without_trunc" if without_trunc else "effective_cop"
        for key in (
            ("effective_cop", decision_key)
            if without_trunc
            else (
                "pareto_decision",
                "effective_cop_include_decision",
                "effective_cop_zero_decision",
                decision_key,
            )
        ):
            old = dataset_dir / assets[key] if key in assets else None
            if old is not None and old.is_file():
                if not (archive / old.name).exists():
                    copy2(old, archive / old.name)
                old.unlink()
            assets.pop(key, None)
            record.pop(key, None)
        relative = str(Path(assets["csv"]).with_name(f"{name}_effective_cop.png"))
        copy2(sources[name], dataset_dir / relative)
        assets[decision_key] = relative
        record[decision_key] = {
            "status": str(row.cycle_status),
            "preparation_heat": str(row.preparation_heat),
            "effective_heat_rule": settings.get(
                "effective_heat_rule", "outlet_at_least_recovery_temperature"
            ),
            "source_table": str((run / "candidate_decisions.csv").resolve()),
            "selection_method": "refrigerant_effective_cop_maximum",
            "electricity_accounting": settings.get(
                "electricity_accounting", "stable_heating_plus_future_recovery"
            ),
        }
    write_catalog(dataset_dir, catalog)
