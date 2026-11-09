"""Prepare source files and a self-contained, front-camera reproduction archive."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed, parallel_config

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dataset_tools import DatasetLoader  # noqa: E402
from image_models.cycle_selection import screen_published_cycles  # noqa: E402
from image_models.dinov2_features import POOL_COLUMNS  # noqa: E402
from image_models.frost_grades import FROST_RESULTS  # noqa: E402

MODULES = (
    "dataset_tools",
    "defrost_decision",
    "defrost_event_models",
    "heat_smoothing",
    "image_models",
    "plots",
    "tools",
    "tests",
)
NETWORKS = ("网络比较_LSTM25_FiLM", "网络比较_GRU32_FiLM", "网络比较_MLP64x64_FiLM")
MODEL_ROOT = Path("output/90_复现材料/01_视觉模型训练")
RIDGE = Path("dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑")
RESULTS = (
    "01_核心结果/除霜电耗预测_287事件",
    "01_核心结果/决策收益_101循环",
    "01_核心结果/稳定制热时长_101循环",
    "01_核心结果/制热与电耗_101循环",
    "01_核心结果/六机位比较",
    "02_模型与消融/网络模型比较_三种子",
    "02_模型与消融/模型组件与学习曲线",
    "03_数据与物理依据/RB提升空间与触发时机",
    "03_数据与物理依据/离线连续霜态与COP_16级",
    "03_数据与物理依据/历史队列构成_101循环",
)


def export_features(dataset, cycle, destination):
    """Keep front-camera CLS and patch-mean features, without raw photographs."""
    columns = [
        "cycle_name",
        "camera_role",
        "file_name",
        *POOL_COLUMNS["cls"],
    ]
    source = dataset / "image_DINO_feature" / f"{cycle}.parquet"
    if set(POOL_COLUMNS["patch_mean"]).issubset(pq.read_schema(source).names):
        columns.extend(POOL_COLUMNS["patch_mean"])
    frame = pd.read_parquet(source, columns=columns, filters=[("camera_role", "==", "front")])
    target = destination / f"{cycle}.parquet"
    frame.to_parquet(target, index=False, compression="zstd")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "dist")
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new packaging directory: {output}")
    dataset = PROJECT / "dataset"
    loader = DatasetLoader(dataset)
    screening = screen_published_cycles(dataset, "front")
    selected = screening.loc[screening.selected]
    cycles = sorted(selected.cycle_name)
    cohort = loader.training_cohort("front")
    if set(cohort.cycle_name) != set(cycles):
        raise ValueError("Published front inputs differ from the authoritative screening")

    source_root = output / "github"
    source_files = list(PROJECT.glob("*.py"))
    source_files += [
        PROJECT / name
        for name in (
            "README.md",
            "DATASET_FORMAT.md",
            "RB_RULES_FROZEN.md",
            ".gitignore",
            ".python-version",
            "pyproject.toml",
            "uv.lock",
            "dataset/README.md",
            "docs/README.md",
            "docs/paper.tex",
            "docs/technical_report.tex",
            "docs/figures/README.md",
            "docs/cycle_cop_first_principles_zh.md",
        )
    ]
    for directory in MODULES:
        source_files += [
            path
            for path in (PROJECT / directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md"}
        ]
    for path in sorted(set(source_files)):
        target = source_root / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    with ZipFile(output / "source.zip", "w", ZIP_DEFLATED) as bundle:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                bundle.write(path, str(path.relative_to(source_root)))

    assets = set()
    for record in loader.catalog["cycles"]:
        assets.update(
            PROJECT / "dataset" / record["assets"][key]
            for key in ("csv", "parquet", "original_csv")
        )
    assets.update(
        dataset / name
        for name in (
            "dataset_manifest.json",
            "cycle_catalog.json",
            "channel_registry.json",
            "priors/rb_reference.csv",
        )
    )
    training = dataset / "training_data"
    assets.update(training / f"{cycle}.parquet" for cycle in cycles)
    assets.update(
        training / name
        for name in (
            "candidate_decisions.parquet",
            "reference.parquet",
            "quality_metrics.csv",
            "action_validity.parquet",
            "cohort.csv",
            "cycle_screening.csv",
            "settings.json",
        )
    )
    assets.update((PROJECT / RIDGE).iterdir())
    for name in NETWORKS:
        assets.update(
            path
            for path in (PROJECT / MODEL_ROOT / name).rglob("*")
            if path.is_file() and path.suffix in {".pkl", ".csv", ".json"}
        )
    case_root = PROJECT / MODEL_ROOT / "六机位_正面_LSTM"
    fold = "front_exp_20260727_seed42"
    assets.update(
        [
            case_root / "training/settings.json",
            case_root / f"checkpoints/{fold}/{fold}.pkl",
        ]
    )
    assets.add(PROJECT / "output/02_模型与消融/部署模型_六机位/正面_部署模型.pkl")
    for directory in RESULTS:
        assets.update(
            path
            for path in (PROJECT / "output" / directory).rglob("*")
            if path.is_file() and path.suffix in {".csv", ".json"}
        )
    frost_settings = json.loads((FROST_RESULTS / "run_settings.json").read_text())
    geometry = PROJECT / frost_settings["geometry_source"]
    assets.update(
        path for path in geometry.iterdir() if path.suffix in {".csv", ".json", ".joblib"}
    )
    for document in ("paper.tex", "technical_report.tex"):
        figures = re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", (PROJECT / "docs" / document).read_text()
        )
        assets.update(PROJECT / "docs/figures" / name for name in figures)
    assets.update(PROJECT / "docs" / name for name in ("paper.pdf", "technical_report.pdf"))
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Packaging inputs are missing: {missing}")

    features = output / "features"
    features.mkdir(parents=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        feature_files = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(export_features)(dataset, cycle, features) for cycle in cycles
        )
    manifest = loader.manifest
    manifest.pop("images_root", None)
    metadata = {Path("dataset/dataset_manifest.json"): manifest}
    for name in NETWORKS:
        relative = MODEL_ROOT / name / "training/settings.json"
        settings = json.loads((PROJECT / relative).read_text())
        settings["camera_role"] = "front"
        metadata[relative] = settings
    archive = output / "reproduction.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for index, path in enumerate(sorted(assets), start=1):
            relative = path.relative_to(PROJECT)
            if relative in metadata:
                bundle.writestr(
                    str(relative), json.dumps(metadata[relative], ensure_ascii=False, indent=2)
                )
            else:
                bundle.write(path, str(relative))
            if index % 100 == 0:
                print(f"[archive] {index}/{len(assets)} files", flush=True)
        for path in feature_files:
            bundle.write(path, f"dataset/image_DINO_feature/{path.name}")
    shutil.rmtree(features)
    training_count = int(selected.experiment_id.le("exp_20260910").sum())
    size = archive.stat().st_size / 1024**2
    (output / "README.md").write_text(
        "# GitHub 交付文件\n\n"
        "| 位置 | 上传位置 | 内容 |\n| --- | --- | --- |\n"
        "| `github/` 中的文件 | 仓库 main，保留相对路径 | 源码、依赖、测试、论文与说明源码 |\n"
        "| `source.zip` | GitHub Release 附件 | 与 github/ 一致的源码 |\n"
        "| `reproduction.zip` | 单独交付，不上传 GitHub | "
        "数据、特征、权重、霜层轨迹、论文结果与 PDF |\n\n"
        f"压缩包 {size:.1f} MiB。正面视觉输入为 {training_count} 个数据集循环和 "
        f"{len(cycles) - training_count} 个独立测试循环。完整数值目录保留 "
        f"{len(loader.catalog['cycles'])} 个循环，用于全候选筛选、数值校验和 Ridge 重拟合。\n\n"
        "下载源码后，将压缩包解压到项目根目录，保留 `dataset/`、`output/`、`docs/` 路径。"
        "按技术报告安装环境，运行演示、四阶段命令与霜层等级分析。\n\n"
        "权重包含 LSTM、GRU、MLP 各 87 个留出模型、正面部署模型和案例留出模型。"
        "多机位材料保留论文数值结果，模型输入交付正面机位。"
        "`docs/figures/` 包含两份文档引用的全部插图；全部循环保留 CLS 特征，"
        "101 个数据集循环另含霜层分析使用的图像块平均特征。\n"
        "霜层分析读取已保存的 Harmony + Slingshot 轨迹和离线序列估计，运行 "
        "`uv run --no-sync python -m image_models.frost_grades` 即可重新计算等级覆盖率与 COP。\n\n"
        "原始照片不在包内；文档中的示例图片仅用于结果说明。\n",
        encoding="utf-8",
    )
    print(f"Source: {source_root}\nArchive: {archive}\nSize: {size:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
