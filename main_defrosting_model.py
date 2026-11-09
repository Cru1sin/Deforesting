"""基于图像与运行状态序列的除霜时机预测。

输入：Dataset 已发布的候选 COP、监督标签、7 项运行特征和 384 维冻结 DINO 特征；无需原图。
主流程：循环筛选 → 按实验划分 → 特征输入和拼接 → 时序模型 → 除霜概率预测。
train：训练 LSTM（宽度 25、FiLM、固定 60 轮），输出权重和独立测试集评价。
evaluate：读取指定冻结权重，输出逐循环触发时刻、COP、RB 对比及近优命中。
直接运行默认执行冻结演示；Dataset、权重、实验与本次输出目录均有预设。
两种命令均只写 --output；训练不重新计算电耗模型、候选 COP 或 RB 规则。
预设训练使用已有标签：连续近优区间内构造线性增加的 CDF 目标，循环等权训练；
预测累计停止概率首次达到 0.5 时触发。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from dataset_tools import DatasetLoader
from image_models.evaluation import evaluate_external
from image_models.experiments import record_training
from image_models.inputs import NUMERIC_INPUTS, RGB, load_data
from image_models.training import fit_model, summarize
from output_layout import PROJECT, analysis_output, experiment_root, model_output


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    tasks = parser.add_subparsers(dest="task", required=True)
    train = tasks.add_parser(
        "train",
        help="训练模型并评价独立测试集",
        description="读取已有输入，固定轮数训练并评价独立测试集。默认正面 LSTM25 + FiLM，60 轮。输出训练记录、权重与测试指标。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate = tasks.add_parser(
        "evaluate",
        help="快速验收：回放已有权重",
        description="读取指定权重和实验，输出逐循环结果及汇总；不训练、不下载、不绘图。输入 Dataset 和权重只读。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for task in (train, evaluate):
        task.add_argument(
            "--dataset",
            type=Path,
            default=PROJECT / "dataset",
            help="包含 training_data 和 image_DINO_feature 的 Dataset",
        )
        output_help = (
            "新的 temp/test/<名称> 或 output/90_复现材料/analysis/<名称> 目录"
            if task is evaluate
            else "output/90_复现材料/01_视觉模型训练/<名称> 或 temp/test/<名称>；内含 training、checkpoints、evaluation"
        )
        task.add_argument("--output", type=Path, help=output_help + "；省略时使用预设目录")
        task.add_argument(
            "--camera-role",
            choices=("front", "left", "left_close", "top", "top_close", "extreme"),
            default="front" if task is train else None,
            help="训练默认正面；评价从权重自动读取，指定时核对一致性",
        )
    network = train.add_argument_group("模型与训练：默认即可复现主模型")
    network.add_argument("--model", choices=("LSTM", "GRU", "MLP"), default="LSTM", help="网络类型")
    network.add_argument("--hidden-width", type=int, default=25, help="隐藏层宽度")
    network.add_argument(
        "--film", action=argparse.BooleanOptionalAction, default=True, help="运行条件调节"
    )
    network.add_argument(
        "--epochs", type=int, default=60, help="Fixed before testing; no test-based early stopping"
    )
    network.add_argument("--learning-rate", type=float, default=0.003, help="学习率")
    network.add_argument("--weight-decay", type=float, default=0.0001, help="权重衰减")
    network.add_argument("--visual-dropout", type=float, default=0.2, help="视觉输入 dropout")
    network.add_argument("--output-dropout", type=float, default=0.2, help="输出 dropout")
    network.add_argument("--seed", type=int, default=42, help="随机种子")
    network.add_argument(
        "--standard-through",
        default="exp_20260910",
        help="数据集的最后一个实验；之后的实验仅用于独立测试",
    )
    evaluate.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "output/02_模型与消融/部署模型_六机位/正面_部署模型.pkl",
        help="已保存的 .pkl 权重，默认交付的正面模型",
    )
    evaluate.add_argument(
        "--experiments", nargs="+", default=["exp_20260920"], help="未参与训练的实验 ID"
    )
    evaluate.add_argument("--n-jobs", type=int, default=6, help="多权重评价进程数；单权重演示用 1")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv or ["evaluate"])
    if args.output is None:
        args.output = (
            PROJECT / "output/90_复现材料/01_视觉模型训练/主模型"
            if args.task == "train"
            else PROJECT
            / "temp/test"
            / ("frozen_model_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
    if args.task == "train":
        if (
            args.epochs < 1
            or args.hidden_width < 1
            or args.learning_rate <= 0
            or args.weight_decay < 0
        ):
            parser.error(
                "Epochs, width and learning rate must be positive; weight decay must be nonnegative"
            )
        if not all(0 <= p < 1 for p in (args.visual_dropout, args.output_dropout)):
            parser.error("Dropout must be in [0, 1)")
    elif args.n_jobs < 1:
        parser.error("n-jobs must be positive")
    return args


def main(argv=None):
    args = get_args(argv)
    args.output = (
        model_output(args.output)
        if args.task == "train"
        else analysis_output(args.output.resolve(), replay=True)
    )
    print(f"Dataset: {args.dataset.resolve()}\nOutput: {args.output}", flush=True)
    if args.task == "evaluate":
        evaluate_external(
            None,
            args.dataset,
            args.experiments,
            args.output,
            checkpoint=args.checkpoint,
            camera_role=args.camera_role,
            n_jobs=args.n_jobs,
        )
        return

    # 1. 完整候选先按 screen_cv_cycles 筛选，再检查 RGB/DINO≥70%、插值≤30%。
    # 当前正面开发期得到 101 循环；数量由数据决定，不复用固定名单。
    rows, reference, screening = load_data(args.dataset, args.camera_role)

    # 2. Keep every later experiment out of fitting and parameter selection.
    standard = set(rows.loc[rows.experiment_id.le(args.standard_through), "cycle_name"])
    external = set(rows.loc[rows.experiment_id.gt(args.standard_through), "cycle_name"])
    split = DatasetLoader.split_deployment_data(rows, standard, external)

    # 3. State the actual network and inputs directly; names never encode width.
    config = dict(
        architecture=args.model,
        hidden_width=args.hidden_width,
        film=args.film,
        numeric_columns=NUMERIC_INPUTS,
        visual_columns=RGB,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_config=dict(
            visual_dropout=args.visual_dropout,
            output_dropout=args.output_dropout,
            visual_time_shared=True,
            output_time_shared=True,
        ),
    )
    if args.model == "MLP":
        config["mlp_layers"] = 2
    else:
        config["recurrent_layers"] = 1
    output = record_training(args, config, split, screening)

    # 4. 固定轮数训练，并用共享预测、CDF 触发和评分函数回放后期测试。
    checkpoint = fit_model(
        split,
        reference,
        config,
        epochs=args.epochs,
        seed=args.seed,
        output=output,
        name=args.model,
        camera_role=args.camera_role,
    )

    # 5. 导出检查点中已保存的测试指标。
    summary = summarize(checkpoint, experiment_root(output) / "evaluation" / "external")
    print(f"Checkpoint: {checkpoint}")
    print(
        summary[
            ["cohort_cycles", "mean_cycle_cop", "gain_vs_rb_pct", "hit_rate_all_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
