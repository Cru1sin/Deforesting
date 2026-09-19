"""Incremental, teacher-independent front DINOv2 cache; reuse ZIP range extraction."""

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
from dataset_tools.load_dataset import DatasetLoader

KEYS = ["cycle_name", "camera_role", "file_name"]
DINO_COLUMNS = [f"dinov2_{index:03d}" for index in range(384)]


def merge_cached_features(rows: pd.DataFrame, paths: list[Path]) -> pd.DataFrame:
    """First compatible cache wins; source labels never enter the RGB cache."""
    cached = [
        pd.read_parquet(p, columns=KEYS + DINO_COLUMNS, filters=[("camera_role", "==", "front")])
        for p in paths
        if p.is_file()
    ]
    if not cached:
        return rows[KEYS].reindex(columns=KEYS + DINO_COLUMNS)
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


def _embed(payloads: list[bytes], model: torch.nn.Module, device: torch.device) -> np.ndarray:
    convert = _transform()
    tensors = []
    for payload in payloads:
        with Image.open(io.BytesIO(payload)) as image:
            tensors.append(convert(image.convert("RGB")))
    with torch.inference_mode():
        values = model(torch.stack(tensors).to(device)).detach().cpu().numpy()
    if values.shape[1] != len(DINO_COLUMNS):
        raise ValueError(f"expected 384 DINOv2 values, got {values.shape[1]}")
    return values.astype(np.float32)


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
) -> tuple[pd.DataFrame, dict[str, object]]:
    cycle = str(rows["cycle_name"].iloc[0])
    missing = rows[DINO_COLUMNS].isna().all(axis=1)
    incomplete = rows[DINO_COLUMNS].isna().any(axis=1) & ~missing
    if incomplete.any():
        raise ValueError(f"partial DINOv2 vectors in cache: {cycle}")
    wanted = rows.loc[missing, "file_name"].astype(str).tolist()
    print(f"[features] {cycle}: {len(wanted)} missing of {len(rows)} frames", flush=True)
    jobs, transferred, started = _plan_image_members(dataset, cycle, wanted, None, 0)
    remote_jobs = {Path(job[-1]).name: job for job in jobs}
    local_root = dataset / "images" / cycle / "front"
    result = rows.copy()
    generated = 0
    local_names = [name for name in wanted if (local_root / name).is_file()]
    unavailable = set(wanted).difference(local_names).difference(remote_jobs)
    if unavailable:
        raise FileNotFoundError(
            f"{len(unavailable)} image(s) unavailable locally and in cloud ZIP for {cycle}"
        )

    def assign(names: list[str], payloads: list[bytes]) -> None:
        nonlocal generated
        values = _embed(payloads, model, device)
        indices = _row_indices(result.loc[missing], names)
        result.loc[indices, DINO_COLUMNS] = values
        generated += len(values)

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
        "frame_count": len(result),
        "cached_count": int((~missing).sum()),
        "generated_count": generated,
        "local_source_count": len(local_names),
        "cloud_source_count": generated - len(local_names),
        "range_read_count": len(groups),
        "range_bytes": range_bytes,
        "transferred_bytes": transferred[0],
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--cycles", nargs="*")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--max-new-images", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    loader = DatasetLoader(args.dataset)
    metadata = loader.load_image_metadata()
    metadata = metadata.loc[metadata.camera_role.eq("front")].copy()
    output = args.output
    (output / "cycles").mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output / "index.parquet", index=False)
    config = {
        "backbone": "dinov2_vits14",
        "features": 384,
        "transform": "RGB; bicubic Resize256; CenterCrop224; ImageNet normalization",
        "dataset": str(args.dataset.resolve()),
        "schema": loader.manifest["dataset_schema_version"],
        "metadata_source": str((args.dataset / "image_metadata.parquet").resolve()),
        "metadata_mtime_ns": (args.dataset / "image_metadata.parquet").stat().st_mtime_ns,
        "image_count": len(metadata),
        "source_caches": list(map(str, args.source_cache)),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.cycles:
        metadata = metadata.loc[metadata.cycle_name.isin(args.cycles)]
    model = None
    progress_path = output / "progress.csv"
    progress = pd.read_csv(progress_path).to_dict("records") if progress_path.exists() else []
    generated_total = 0
    for number, (cycle, index) in enumerate(metadata.groupby("cycle_name", sort=True), 1):
        destination = output / "cycles" / f"{cycle}.parquet"
        paths = [
            destination,
            *[p / f"{cycle}.parquet" if p.is_dir() else p for p in args.source_cache],
        ]
        rows = merge_cached_features(index, paths)
        missing = rows[DINO_COLUMNS].isna().all(axis=1)
        if args.max_new_images is not None:
            remaining = max(0, args.max_new_images - generated_total)
            rows = rows.loc[~missing | rows.index.isin(rows.index[missing][:remaining])]
        rows = rows.reset_index(drop=True)
        if rows.empty:
            continue
        needed = int(rows[DINO_COLUMNS].isna().all(axis=1).sum())
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
            )
        else:
            audit = {"cycle_name": cycle, "frame_count": len(rows), "generated_count": 0}
        generated_total += needed
        temporary = destination.with_suffix(".tmp.parquet")
        rows[KEYS + DINO_COLUMNS].to_parquet(temporary, index=False)
        temporary.replace(destination)
        audit["indexed_count"] = len(index)
        audit["complete"] = len(rows) == len(index)
        progress = [p for p in progress if p["cycle_name"] != cycle] + [audit]
        pd.DataFrame(progress).to_csv(progress_path, index=False)
        print(
            f"[cache] {number}/{metadata.cycle_name.nunique()} {cycle}: "
            f"{len(rows)}/{len(index)}, new={needed}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
