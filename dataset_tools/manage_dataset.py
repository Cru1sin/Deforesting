"""数据预处理和阶段划分。

输入：原始运行数据及通道定义；输出：Dataset 循环、阶段与处理后的测量表。
先根据运行状态识别循环，划分恢复、稳定制热、准备及除霜阶段；
再构建 1 s 时间网格，按通道聚合，对连续测量的内部缺失量线性插值，
并平滑供热功率、蒸发压力 Pe 和盘管温度 T3（60 s 过去窗口，阶段处重置）。
循环边界来自运行状态，不从平滑曲线反推；原始值及插值标记保留用于核验。

首次导入：add 原始数据目录 --dataset temp/test/import_demo/dataset
已有数据：smooth 更新处理列；其余子命令用于替换、复核、设置和图表维护。
这些命令会修改指定 Dataset；模型输入发布见 dataset_tools.training_data。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dataset_tools.dataset_maintenance import (
    add_dataset,
    edit_dataset,
    refresh_dataset,
    remove_dataset,
    replace_dataset,
    review_cycle,
)
from dataset_tools.measurements import (
    add_missing_channel,
    aggregate_original,
    refresh_processed_measurements,
)
from output_layout import PROJECT
from plots.dataset import render_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    actions = parser.add_subparsers(dest="action", required=True)

    for name in ("add", "replace"):
        description = (
            "导入一个实验：识别循环、1 s 处理、复制照片并生成循环图。"
            "已有同名实验跳过；新增通道时会统一旧循环表的列。"
            if name == "add"
            else "替换一个实验并更新后续循环编号、照片和图表。"
        )
        command = actions.add_parser(
            name,
            help=description,
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog="通道定义：dataset_tools/builder/channel_mapping.py。首次试用请指定新的 temp/test/import_demo/dataset。",
        )
        command.add_argument("input_dir", type=Path, help="原始实验日期目录，例如 data/0819")
        command.add_argument(
            "--dataset", type=Path, default=PROJECT / "dataset", help="写入的 Dataset 目录"
        )

    aggregate = actions.add_parser(
        "aggregate-original",
        help="从原始循环重建正式 1 s 测量表",
        description="原地重写指定 Dataset 的 1 s 测量表、catalog 和 registry。此命令执行重建，不是预览。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    aggregate.add_argument(
        "--dataset", type=Path, default=PROJECT / "dataset", help="重建的 Dataset 目录"
    )
    aggregate.add_argument("--n-jobs", type=int, default=6, help="按循环并行的进程数")

    channel = actions.add_parser(
        "add-missing-channel", help="add an empty channel without rebuilding"
    )
    channel.add_argument("name")
    channel.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    channel.add_argument("--cycles", nargs="+", help="Update only the named cycle annotations")

    remove = actions.add_parser("remove", help="remove an experiment date")
    remove.add_argument("date")
    remove.add_argument("--dataset", type=Path, default=PROJECT / "dataset")

    refresh = actions.add_parser("refresh", help="refresh derived Dataset assets")
    refresh.add_argument("mode", choices=["roles", "images", "figures", "all"])
    refresh.add_argument("--dataset", type=Path, default=PROJECT / "dataset")

    review = actions.add_parser("review-cycle", help="record a cycle review")
    review.add_argument("cycle")
    review.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    review.add_argument("--status", required=True, choices=["valid", "invalid"])
    review.add_argument("--reason")
    review.add_argument("--rgb-frost", choices=["valid", "invalid", "not_applicable"])
    review.add_argument("--rgb-defrost", choices=["valid", "invalid", "not_applicable"])

    edit = actions.add_parser("edit", help="edit Dataset processing settings")
    edit.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    edit.add_argument("--baseline-seconds", type=int)
    recovery = edit.add_mutually_exclusive_group()
    recovery.add_argument("--recovery-seconds", type=int)
    recovery.add_argument("--recovery-end-by", choices=["frequency-setpoint", "frequency-actual"])
    edit.add_argument("--defrost-preparation", action="store_true")
    edit.add_argument("--skip-rgb-panels", action="store_true")
    edit.add_argument("--skip-figures", action="store_true")

    render = actions.add_parser("render", help="render cycle figures")
    render.add_argument("cycle", nargs="?")
    render.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    render.add_argument("--publication", action="store_true")
    render.add_argument(
        "--effective-cop",
        action="store_true",
        help="Publish missing COP figures for all saved candidate cycles",
    )
    render.add_argument(
        "--decisions", type=Path, help="完整候选 COP 表，用于绘制循环结果"
    )
    render.add_argument("--recovery-boundaries", type=Path)
    render.add_argument("--panel", action="store_true")
    render.add_argument("--fetch-cloud-images", action="store_true")
    render.add_argument("--cleanup-downloaded-images", action="store_true")
    render.add_argument(
        "--n-jobs", type=int, default=6, help="outer workers for rendering and decisions"
    )

    smooth = actions.add_parser(
        "smooth",
        help="更新插值及三通道 60 s 平滑列",
        description="在指定 Dataset 内更新连续插值列、控制量保持列、供热功率/Pe/T3 平滑列及处理说明。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    smooth.add_argument("--dataset", type=Path, default=PROJECT / "dataset")
    smooth.add_argument("--n-jobs", type=int, default=6)

    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"Dataset ({args.action}): {args.dataset.resolve()}", flush=True)
    if args.action == "smooth":
        print(refresh_processed_measurements(args.dataset, n_jobs=args.n_jobs))
        return 0
    if args.action == "add":
        print(add_dataset(args.input_dir, args.dataset))
    elif args.action == "replace":
        print(replace_dataset(args.input_dir, args.dataset))
    elif args.action == "aggregate-original":
        print(aggregate_original(args.dataset, seconds=1, publish=True, n_jobs=args.n_jobs))
    elif args.action == "add-missing-channel":
        print(add_missing_channel(args.dataset, args.name, cycles=args.cycles))
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
                render_figures=not args.skip_figures,
            )
        )
    else:
        if args.effective_cop:
            if args.decisions is None:
                parser.error("--effective-cop requires --decisions")
            from plots.dataset import render_effective_cop_assets

            render_effective_cop_assets(
                args.dataset, args.decisions, cycle_name=args.cycle, n_jobs=args.n_jobs
            )
            return 0
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
