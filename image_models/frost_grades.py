"""霜层等级分析：读取已保存轨迹，计算近优覆盖率和平均归一化 COP。

默认读取论文采用的原生 Harmony + Slingshot 全局轨迹及离线序列估计。
先按完整候选 COP 与 Dataset 重新筛选循环，再匹配霜层等级；无需原图。
结果保存到新的临时目录；--plot 可生成不含照片的统计图。
"""

import argparse
from datetime import datetime
from pathlib import Path

from output_layout import PROJECT, analysis_output

FROST_RESULTS = PROJECT / "output/03_数据与物理依据/离线连续霜态与COP_16级"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT / "dataset", help="Dataset 目录")
    parser.add_argument(
        "--progression",
        type=Path,
        default=FROST_RESULTS / "progression.csv",
        help="已保存的霜层轨迹；同目录包含 run_settings.json",
    )
    parser.add_argument("--experiment-end", default="exp_20260910", help="数据集的最后一个实验")
    parser.add_argument("--output", type=Path, help="新的结果目录，省略时自动创建")
    parser.add_argument("--plot", action="store_true", help="同时绘制覆盖率和平均归一化 COP")
    args = parser.parse_args(argv)
    output = args.output or PROJECT / "temp/test" / datetime.now().strftime(
        "frost_grades_%Y%m%d_%H%M%S_%f"
    )
    output = analysis_output(output)
    print(f"Dataset: {args.dataset.resolve()}\nTrajectory: {args.progression.resolve()}")
    print(f"Experiments through: {args.experiment_end}\nOutput: {output.resolve()}", flush=True)
    if output.exists():
        raise FileExistsError(f"结果目录已存在，请使用新目录：{output}")

    from plots.frost_comparison import render_frost_overlap

    for metric in ("coverage", "observed-cop"):
        render_frost_overlap(
            args.dataset,
            args.progression,
            output,
            dataset=args.dataset,
            experiment_end=args.experiment_end,
            grade_scale="vertices",
            summary_metric=metric,
            mean_only=True,
            render_cohorts=None if args.plot else (),
        )
    print(f"Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
