"""Incremental, teacher-independent camera DINOv2 cache; reuse ZIP range extraction."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset_tools.cloud_images import _plan_image_members, _read_zip_member, _read_zip_range
from dataset_tools.cycle_metadata import image_root
from dataset_tools.load_dataset import DatasetLoader
from dataset_tools.local_images import RGB_CAMERA_ORDER
from output_layout import PROJECT

KEYS = ["cycle_name", "camera_role", "file_name"]
DINO_COLUMNS = [f"dinov2_{index:03d}" for index in range(384)]
from image_models.inputs import GRID_COLUMNS, PATCH_COLUMNS

POOL_COLUMNS = {"cls": DINO_COLUMNS, "patch_mean": PATCH_COLUMNS, "patch_grid2": GRID_COLUMNS}


def merge_cached_features(
    rows: pd.DataFrame, paths: list[Path], columns=DINO_COLUMNS
) -> pd.DataFrame:
    """First compatible cache wins; source labels never enter the RGB cache."""
    if rows.empty:
        return rows[KEYS].reindex(columns=KEYS + columns)
    cached = []
    for path in paths:
        if path.is_file():
            features = pd.read_parquet(
                path, filters=[("camera_role", "in", rows.camera_role.unique().tolist())]
            )
            if set(columns).issubset(features.columns):
                cached.append(features[KEYS + columns].dropna(subset=columns))
    if not cached:
        return rows[KEYS].reindex(columns=KEYS + columns)
    features = pd.concat(cached, ignore_index=True).drop_duplicates(KEYS)
    return rows[KEYS].merge(features, on=KEYS, how="left", validate="one_to_one")


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _load_backbone(device: torch.device) -> torch.nn.Module:
    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not hub.is_dir():
        raise FileNotFoundError(f"local DINOv2 torch hub is missing: {hub}")
    model = torch.hub.load(str(hub), "dinov2_vits14", source="local", pretrained=True)
    return model.eval().to(device)


def _embed(
    payloads: list[bytes],
    model: torch.nn.Module,
    device: torch.device,
    pooling="cls",
    image_map=map,
) -> np.ndarray:
    convert = _transform()
    tensors = []
    positions = []
    width = len(POOL_COLUMNS[pooling])
    result = np.full((len(payloads), width), np.nan, dtype=np.float32)

    def image_tensor(payload):
        try:
            with Image.open(io.BytesIO(payload)) as image:
                return convert(image.convert("RGB"))
        except OSError:
            return None

    for index, tensor in enumerate(image_map(image_tensor, payloads)):
        if tensor is not None:
            tensors.append(tensor)
            positions.append(index)
    if not tensors:
        return result
    with torch.inference_mode():
        inputs = torch.stack(tensors).to(device)
        if pooling == "cls":
            values = model(inputs)
        else:
            tokens = model.forward_features(inputs)["x_norm_patchtokens"]
            # Fixed 224px crop / 14px patches: row-major TL, TR, BL, BR.
            values = (
                tokens.reshape(len(inputs), 2, 8, 2, 8, 384).mean(dim=(2, 4)).flatten(1)
                if pooling == "patch_grid2"
                else tokens.mean(dim=1)
            )
        values = values.detach().cpu().numpy()
    if values.shape[1] != width:
        raise ValueError(f"expected {width} DINOv2 values, got {values.shape[1]}")
    result[positions] = values
    return result


def _group_remote_jobs(jobs: list[tuple], max_range_bytes: int = 8 * 1024**2) -> list[list[tuple]]:
    """Group archive-adjacent members into bounded contiguous range reads."""
    groups: list[list[tuple]] = []
    for job in sorted(jobs, key=lambda item: item[4][0]):
        offset, compressed, _, _, filename_size, _ = job[4]
        end = min(job[3], offset + 30 + filename_size + 65_535 + compressed)
        if not groups or end - groups[-1][0][4][0] > max_range_bytes:
            groups.append([job])
        else:
            groups[-1].append(job)
    return groups


def _row_indices(rows: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Return row indices in the same order as image payload names."""
    lookup = rows.reset_index().set_index("file_name")["index"]
    return lookup.loc[names].to_numpy()


def _complete_cycle(
    rows: pd.DataFrame,
    *,
    dataset: Path,
    model: torch.nn.Module,
    device: torch.device,
    n_jobs: int,
    batch_size: int,
    local_only: bool = False,
    pooling: str = "cls",
) -> tuple[pd.DataFrame, dict[str, object]]:
    cycle = str(rows["cycle_name"].iloc[0])
    if rows.cycle_name.nunique() != 1 or rows.camera_role.nunique() != 1:
        raise ValueError("Complete one cycle and camera at a time")
    camera = str(rows.camera_role.iloc[0])
    columns = POOL_COLUMNS[pooling]
    missing = rows[columns].isna().all(axis=1)
    incomplete = rows[columns].isna().any(axis=1) & ~missing
    if incomplete.any():
        raise ValueError(f"partial DINOv2 vectors in cache: {cycle}")
    wanted = rows.loc[missing, "file_name"].astype(str).tolist()
    print(f"[features] {cycle}/{camera}: {len(wanted)} missing of {len(rows)} frames", flush=True)
    jobs, transferred, started = (
        ([], [0], time.monotonic())
        if local_only
        else _plan_image_members(dataset, cycle, [f"{camera}/{name}" for name in wanted], None, 0)
    )
    local_root = image_root(dataset) / cycle / camera
    remote_jobs = {
        Path(job[-1]).name: job for job in jobs if not (local_root / Path(job[-1]).name).is_file()
    }
    result = rows.copy()
    generated = 0
    local_names = [name for name in wanted if (local_root / name).is_file()]
    unavailable = set(wanted).difference(local_names).difference(remote_jobs)
    if unavailable and not local_only:
        raise FileNotFoundError(
            f"{len(unavailable)} image(s) unavailable locally and in cloud ZIP for {cycle}"
        )

    def assign(names: list[str], payloads: list[bytes]) -> None:
        nonlocal generated
        values = _embed(payloads, model, device, pooling, pool.map)
        indices = _row_indices(result.loc[missing], names)
        result.loc[indices, columns] = values
        valid = np.isfinite(values).all(axis=1)
        generated += int(valid.sum())
        for name, ok in zip(names, valid):
            if not ok:
                print(f"[features] unreadable image left blank: {cycle}/{name}", flush=True)

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for start in range(0, len(local_names), batch_size):
            names = local_names[start : start + batch_size]
            assign(names, list(pool.map(Path.read_bytes, [local_root / name for name in names])))
    groups = _group_remote_jobs(list(remote_jobs.values()))
    range_bytes = 0

    def read_group(group: list[tuple]) -> tuple[int, bytes]:
        offset = group[0][4][0]
        end = max(min(job[3], job[4][0] + 30 + job[4][4] + 65_535 + job[4][1]) for job in group)
        payload = _read_zip_range(group[0][1], offset, end - offset, remote=group[0][2])
        return offset, payload

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        for start in range(0, len(groups), n_jobs):
            chunk = groups[start : start + n_jobs]
            for group, (offset, payload) in zip(chunk, pool.map(read_group, chunk), strict=True):
                range_bytes += len(payload)
                transferred[0] += len(payload)
                for batch_start in range(0, len(group), batch_size):
                    batch = group[batch_start : batch_start + batch_size]
                    names = [Path(job[-1]).name for job in batch]
                    images = [
                        _read_zip_member(
                            job[1],
                            job[4],
                            archive_size=job[3],
                            remote=job[2],
                            payload=memoryview(payload)[job[4][0] - offset :],
                        )
                        for job in batch
                    ]
                    assign(names, images)
    return result, {
        "cycle_name": cycle,
        "camera_role": camera,
        "frame_count": len(result),
        "cached_count": int((~missing).sum()),
        "generated_count": generated,
        "local_source_count": len(local_names),
        "cloud_source_count": len(remote_jobs),
        "unavailable_count": int(result[columns].isna().any(axis=1).sum()),
        "range_read_count": len(groups),
        "range_bytes": range_bytes,
        "transferred_bytes": transferred[0],
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=(PROJECT / "dataset"))
    parser.add_argument("--output", type=Path, help="defaults to Dataset/image_DINO_feature")
    parser.add_argument("--source-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--camera-roles", nargs="+", choices=RGB_CAMERA_ORDER, default=["front"])
    parser.add_argument(
        "--image-list", type=Path, help="CSV of cycle_name, camera_role, file_name to extract"
    )
    parser.add_argument(
        "--include-rb", action="store_true", help="同时提取 RB 循环；准备主模型输入时启用"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="never fetch cloud images; leave unavailable vectors blank",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--max-new-images", type=int)
    parser.add_argument("--pooling", choices=list(POOL_COLUMNS), default="cls")
    parser.add_argument(
        "--admission-settings",
        type=Path,
        help="screen full saved candidates before image filtering",
    )
    args = parser.parse_args()
    from output_layout import prepare_output

    prepare_output(args, training=False)
    columns = POOL_COLUMNS[args.pooling]
    suffix = "_" + args.pooling if args.pooling != "cls" else ""
    if args.camera_roles != ["front"]:
        suffix += "_" + "_".join(dict.fromkeys(args.camera_roles))
    torch.set_num_threads(1)
    loader = DatasetLoader(args.dataset)
    admitted = None
    if args.admission_settings is not None:
        from image_models.cycle_selection import screen_admission

        screening = screen_admission(args.admission_settings)
        admitted = set(screening.loc[screening.selected, "cycle_name"])
    metadata = loader.load_image_metadata()
    regular = loader.list_cycles(include_rb=args.include_rb).cycle_name
    metadata = metadata.loc[
        metadata.camera_role.isin(args.camera_roles) & metadata.cycle_name.isin(regular)
    ].copy()
    if admitted is not None:
        metadata = metadata.loc[metadata.cycle_name.isin(admitted)]
    if args.image_list:
        metadata = metadata.merge(
            pd.read_csv(args.image_list)[KEYS].drop_duplicates(),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    output = args.output or args.dataset / "image_DINO_feature"
    output.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output / f"index{suffix}.parquet", index=False)
    config = {
        "backbone": "dinov2_vits14",
        "pooling": args.pooling,
        "camera_roles": args.camera_roles,
        "features": len(columns),
        "transform": "RGB; bicubic Resize256; CenterCrop224; ImageNet normalization",
        "dataset": str(args.dataset.resolve()),
        "schema": loader.manifest["dataset_schema_version"],
        "metadata_source": str((image_root(args.dataset) / "image_metadata.parquet").resolve()),
        "metadata_mtime_ns": (image_root(args.dataset) / "image_metadata.parquet")
        .stat()
        .st_mtime_ns,
        "image_count": len(metadata),
        "source_caches": list(map(str, args.source_cache)),
        "local_only": args.local_only,
        "include_rb": args.include_rb,
        "image_list": str(args.image_list) if args.image_list else None,
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or None,
    }
    if args.admission_settings is not None:
        config["admission_settings"] = str(args.admission_settings.resolve())
    (output / f"config{suffix}.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.cycles:
        metadata = metadata.loc[metadata.cycle_name.isin(args.cycles)]
    model = None
    progress_path = output / f"progress{suffix}.csv"
    progress = pd.read_csv(progress_path).to_dict("records") if progress_path.exists() else []
    generated_total = 0
    groups = metadata.groupby(["cycle_name", "camera_role"], sort=True)
    for number, ((cycle, camera), index) in enumerate(groups, 1):
        destination = output / f"{cycle}.parquet"
        paths = [
            destination,
            *[p / f"{cycle}.parquet" if p.is_dir() else p for p in args.source_cache],
        ]
        rows = merge_cached_features(index, paths, columns)
        missing = rows[columns].isna().all(axis=1)
        if args.max_new_images is not None:
            remaining = max(0, args.max_new_images - generated_total)
            rows = rows.loc[~missing | rows.index.isin(rows.index[missing][:remaining])]
        rows = rows.reset_index(drop=True)
        if rows.empty:
            continue
        needed = int(rows[columns].isna().all(axis=1).sum())
        if needed:
            if model is None:
                model = _load_backbone(torch.device(args.device))
            rows, audit = _complete_cycle(
                rows,
                dataset=args.dataset,
                model=model,
                device=torch.device(args.device),
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
                local_only=args.local_only,
                pooling=args.pooling,
            )
        else:
            audit = {
                "cycle_name": cycle,
                "camera_role": camera,
                "frame_count": len(rows),
                "generated_count": 0,
            }
        generated_total += int(audit["generated_count"])
        temporary = destination.with_suffix(".tmp.parquet")
        updated = rows[KEYS + columns].set_index(KEYS)
        if destination.exists():
            updated = updated.combine_first(pd.read_parquet(destination).set_index(KEYS))
        updated.reset_index().to_parquet(temporary, index=False)
        temporary.replace(destination)
        audit["indexed_count"] = len(index)
        audit["complete"] = len(rows) == len(index) and rows[columns].notna().all().all()
        progress = [
            p
            for p in progress
            if (p["cycle_name"], p.get("camera_role", "front")) != (cycle, camera)
        ] + [audit]
        pd.DataFrame(progress).to_csv(progress_path, index=False)
        print(
            f"[cache] {number}/{len(groups)} {cycle}/{camera}: "
            f"{len(rows)}/{len(index)}, generated={audit['generated_count']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
