"""除霜电耗联合预测：输入 → 三输入 Ridge → 准备＋除霜电耗。

输入：Dataset 中压缩机频率、环境温度、盘管温度，以及实测事件电耗。
模型：标准化三项输入 z，拟合 E_prep+D = β0 + βᵀz；目标单位为 kWh。
采用按实验隔离的嵌套留一验证，选择 Ridge 正则强度并保存各折及全量模型。
输出：candidate_model_parameters.json、事件清单、逐事件预测与验证汇总。
按实际训练事件数和采样秒数自动命名；同名结果直接更新。
默认保存至 dataset/除霜预测模型/；COP 阶段直接读取模型参数。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dataset_tools import DatasetLoader
from defrost_event_models.event_electricity_ridge import (
    fit_preparation_defrost_model,
    prepare_training_events,
)
from defrost_event_models.training_data import (
    build_defrost_event_training_table,
)
from output_layout import PROJECT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT / "dataset", help="含事件测量的 Dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="本次模型目录；省略时在 <dataset>/除霜预测模型 下按实际事件数和采样秒数命名",
    )
    parser.add_argument("--n-jobs", type=int, default=6, help="按实验并行拟合的进程数")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="render held-out event calibration figures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    loader = DatasetLoader(args.dataset)
    boundaries = loader.configure_recovery({"recovery_rule": "dataset"})
    # 事件实测值提供监督：E_prep+D = E_prep + E_D（kWh）。
    events = prepare_training_events(build_defrost_event_training_table(loader))
    count = int(events.selected_for_training.sum())
    seconds = loader.registry["resample_interval_seconds"]
    run = (
        args.output or args.dataset / "除霜预测模型" / f"三输入Ridge-两阶段-{count}-{seconds}"
    ).resolve()
    print(
        f"Dataset: {args.dataset.resolve()}\nTraining events: {count}\nOutput (replace same-name files): {run.resolve()}",
        flush=True,
    )
    fit_preparation_defrost_model(loader, events, run, args.n_jobs, args.figures)
    boundaries.to_csv(run / "recovery_boundaries.csv", index=False)
    (run / "settings.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
