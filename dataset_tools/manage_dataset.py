"""Build and maintain a self-contained Dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .cloud_upload import upload_cycle_images, upload_date_data
from .dataset_maintenance import (
    add_dataset,
    aggregate_original,
    edit_dataset,
    refresh_dataset,
    remove_dataset,
    render_dataset,
    replace_dataset,
    review_cycle,
)
from .dts_downloader import DEFAULT_WORKERS, download_shared_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    for name in ("add", "replace"):
        command = actions.add_parser(name, help=f"{name} raw experiment data")
        command.add_argument("input_dir", type=Path)
        command.add_argument("--dataset", type=Path, default=Path("dataset"))

    aggregate = actions.add_parser("aggregate-original", help="re-aggregate original cycles")
    aggregate.add_argument("--dataset", type=Path, default=Path("dataset"))
    aggregate.add_argument("--seconds", type=int, default=10)

    remove = actions.add_parser("remove", help="remove an experiment date")
    remove.add_argument("date")
    remove.add_argument("--dataset", type=Path, default=Path("dataset"))

    refresh = actions.add_parser("refresh", help="refresh derived Dataset assets")
    refresh.add_argument("mode", choices=["roles", "images", "figures", "all"])
    refresh.add_argument("--dataset", type=Path, default=Path("dataset"))

    review = actions.add_parser("review-cycle", help="record a cycle review")
    review.add_argument("cycle")
    review.add_argument("--dataset", type=Path, default=Path("dataset"))
    review.add_argument("--status", required=True, choices=["valid", "invalid"])
    review.add_argument("--reason")
    review.add_argument("--rgb-frost", choices=["valid", "invalid", "not_applicable"])
    review.add_argument("--rgb-defrost", choices=["valid", "invalid", "not_applicable"])

    edit = actions.add_parser("edit", help="edit Dataset processing settings")
    edit.add_argument("--dataset", type=Path, default=Path("dataset"))
    edit.add_argument("--baseline-seconds", type=int)
    recovery = edit.add_mutually_exclusive_group()
    recovery.add_argument("--recovery-seconds", type=int)
    recovery.add_argument("--recovery-end-by", choices=["ts-minus"])
    edit.add_argument("--defrost-preparation", action="store_true")
    edit.add_argument("--skip-rgb-panels", action="store_true")

    render = actions.add_parser("render", help="render cycle figures")
    render.add_argument("cycle", nargs="?")
    render.add_argument("--dataset", type=Path, default=Path("dataset"))
    render.add_argument("--publication", action="store_true")
    render.add_argument("--recovery-boundaries", type=Path)
    render.add_argument("--panel", action="store_true")
    render.add_argument("--fetch-cloud-images", action="store_true")
    render.add_argument("--cleanup-downloaded-images", action="store_true")
    render.add_argument(
        "--n-jobs", type=int, default=6, help="outer workers for rendering and decisions"
    )

    upload = actions.add_parser("upload-data", help="package raw dates and upload to OneDrive")
    upload.add_argument("dates", nargs="+")
    upload.add_argument("--data", type=Path, default=Path("data"))

    images = actions.add_parser("upload-images", help="package cycle images and upload to OneDrive")
    images.add_argument("cycles", nargs="+")
    images.add_argument("--dataset", type=Path, default=Path("dataset"))
    images.add_argument("--cleanup-uploaded-images", action="store_true")

    download = actions.add_parser("dts-download", help="download shared Midea DTS archives")
    download.add_argument("url", help="DTS download page or Outlook Safe Links URL")
    download.add_argument("--password")
    download.add_argument("--start-no", type=int, default=1)
    download.add_argument("--end-no", type=int)
    download.add_argument("--workers", type=int, choices=(1, 2), default=DEFAULT_WORKERS)
    download.add_argument("--retry-delay", type=float, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "add":
        print(add_dataset(args.input_dir, args.dataset))
    elif args.action == "replace":
        print(replace_dataset(args.input_dir, args.dataset))
    elif args.action == "aggregate-original":
        print(aggregate_original(args.dataset, seconds=args.seconds))
    elif args.action == "remove":
        print(remove_dataset(args.dataset, args.date))
    elif args.action == "refresh":
        print(refresh_dataset(args.dataset, args.mode))
    elif args.action == "review-cycle":
        review_cycle(
            args.dataset,
            args.cycle,
            status=args.status,
            reason=args.reason,
            rgb_frost=args.rgb_frost,
            rgb_defrost=args.rgb_defrost,
        )
        print(args.dataset)
    elif args.action == "edit":
        print(
            edit_dataset(
                args.dataset,
                baseline_seconds=args.baseline_seconds,
                recovery_seconds=args.recovery_seconds,
                recovery_end_by=args.recovery_end_by,
                defrost_preparation=args.defrost_preparation,
                render_rgb_panels=not args.skip_rgb_panels,
            )
        )
    elif args.action == "upload-data":
        if "all" in args.dates and args.dates != ["all"]:
            parser.error("'all' cannot be combined with explicit dates")
        print(upload_date_data(args.data, None if args.dates == ["all"] else args.dates))
    elif args.action == "upload-images":
        if "all" in args.cycles and args.cycles != ["all"]:
            parser.error("'all' cannot be combined with explicit cycles")
        print(
            upload_cycle_images(
                args.dataset,
                None if args.cycles == ["all"] else args.cycles,
                args.cleanup_uploaded_images,
            )
        )
    elif args.action == "dts-download":
        print(
            download_shared_data(
                args.url,
                data_dir=Path("/Volumes/Proust/DefrostingSensor/data"),
                password=args.password,
                start_no=args.start_no,
                end_no=args.end_no,
                workers=args.workers,
                retry_delay=args.retry_delay,
            )
        )
    else:
        selected = args.publication or args.panel
        print(
            render_dataset(
                args.dataset,
                args.cycle,
                publication=args.publication or not selected,
                panel=args.panel or not selected,
                fetch_cloud_images=args.fetch_cloud_images,
                cleanup_downloaded_images=args.cleanup_downloaded_images,
                n_jobs=args.n_jobs,
                recovery_boundaries=args.recovery_boundaries,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
