"""Build and maintain a self-contained Dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
    render.add_argument("--panel", action="store_true")
    render.add_argument("--fetch-cloud-images", action="store_true")
    render.add_argument("--cleanup-downloaded-images", action="store_true")
    render.add_argument(
        "--n-jobs", type=int, default=10, help="maximum concurrent OneDrive requests"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    args = build_parser().parse_args(argv)

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
    else:
        print(
            render_dataset(
                args.dataset,
                args.cycle,
                publication=args.publication or not args.panel,
                panel=args.panel or not args.publication,
                fetch_cloud_images=args.fetch_cloud_images,
                cleanup_downloaded_images=args.cleanup_downloaded_images,
                n_jobs=args.n_jobs,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
