"""Explicit relocation of historical results; never imported by the model workflow."""

import argparse
import json
import re
import shutil
from pathlib import Path

from output_layout import ARCHIVE, OUTPUT, PROJECT, REPRODUCTION

RESULT_FOLDERS = {
    "analysis/camera_LSTM25_FiLM_20261025": "01_核心结果/六机位比较",
    "analysis/LSTM_FiLM决策收益_绝对与百分比_历史101循环_20261016": "01_核心结果/决策收益_101循环",
    "analysis/Ridge论文效果图_287事件_统一平滑_总电耗": "01_核心结果/除霜电耗预测_287事件",
    "analysis/LSTM_FiLM稳定制热时长增加_历史101循环_20261016": "01_核心结果/稳定制热时长_101循环",
    "analysis/LSTM_FiLM除霜频率下降_历史101循环_20261016": "01_核心结果/除霜频率_101循环",
    "analysis/LSTM_FiLM制热延长与除霜耗电_历史101循环_20261027": "01_核心结果/制热与电耗_101循环",
    "analysis/典型模型性能比较_三种子_20261012": "02_模型与消融/网络模型比较_三种子",
    "analysis/LSTM25_FiLM_Fig7_记忆验证更新_20261013": "02_模型与消融/模型组件与学习曲线",
    "analysis/LSTM记忆与显式历史_匹配LOEO_20261013": "02_模型与消融/递归记忆与频率历史",
    "analysis/LSTM运行时间与记忆_匹配LOEO_20261015": "02_模型与消融/运行时间与记忆",
    "analysis/LSTM_FiLM_unseen_conditions_20261016": "02_模型与消融/跨工况评价",
    "analysis/LSTM_FiLM_unseen_conditions_memory_20261016": "02_模型与消融/跨工况记忆对照",
    "reports/data_overview_20261012": "03_数据与物理依据/历史队列构成_101循环",
    "analysis/RB规则说明": "03_数据与物理依据/RB触发规则",
    "analysis/RB提升空间与触发早晚_20261012": "03_数据与物理依据/RB提升空间与触发时机",
    "analysis/COP四种成本对照_10s固定主配置101循环_20261011": "03_数据与物理依据/四种成本定义对照",
    "analysis/Step3_真实循环示例_20261013": "03_数据与物理依据/真实循环与启动过程",
    "analysis/heating_capacity_double_curve": "03_数据与物理依据/逐循环制热量",
    "analysis/除霜前后配对UA": "03_数据与物理依据/除霜前后换热能力",
    "analysis/恢复电耗_本循环实测基线对照_20261024": "03_数据与物理依据/恢复电耗记账",
    "analysis/霜层等级分析/霜态构建_论文核心单图_20261012": "03_数据与物理依据/视觉霜态构建_Harmony与MST",
    "analysis/霜层等级分析/16级视觉霜态_近优覆盖与平均COP_20261015": "03_数据与物理依据/视觉霜态与COP_16级",
    "analysis/霜层等级分析/16级离线连续路径_覆盖与平均COP_20261015": "03_数据与物理依据/离线连续霜态与COP_16级",
    "reports/COP监督视觉除霜_三阶段": "04_汇报材料/三阶段方法汇报",
    "reports/COP监督视觉除霜_中期答辩": "04_汇报材料/中期答辩",
    "reports/COP监督视觉除霜_模板修订": "04_汇报材料/模板修订版",
    "reports/循环COP驱动除霜_论证版": "04_汇报材料/循环COP与视觉除霜",
    "reports/霜态方法PPT_实际数据替换图": "04_汇报材料/霜态方法配图",
}


FIGURE_NAMES = {
    "camera_core_grouped": "01_共同集与跨工况比较",
    "camera_policy_comparison": "02_六机位决策表现",
    "camera_paired_effects": "03_机位配对差异",
    "camera_coverage": "04_机位样本覆盖",
    "camera_test_diagnostics": "05_测试诊断",
    "camera_frost_images_groups16_square_resized": "06_六机位结霜照片",
    "Fig2a_ranked_cycle_gains": "逐循环COP收益",
    "Fig2b_ranked_stable_heating_increase": "逐循环稳定制热时长",
    "Fig2b_ranked_frequency_reduction": "逐循环除霜频率变化",
    "event_calibration_preparation_defrost": "准备与除霜总电耗预测",
    "LSTM25_FiLM_Fig7_abcd": "模型组件与学习曲线",
    "unseen_conditions": "跨工况评价",
    "data_overview_refined": "数据组成与筛选",
    "Fig2e_heating_electricity_relationship": "制热时长与电耗关系",
    "RB_improvement_headroom": "RB改善空间",
    "RB_trigger_timing_compact": "RB触发时机",
    "repair_ua_event_example": "除霜前后换热实例",
    "repair_ua_feasibility": "除霜前后换热比较",
    "cluster_count_tradeoff": "聚类数量选择",
    "cluster_representative_images": "各霜态代表照片",
    "cycle_selection": "循环筛选过程",
    "harmony_corrected_space": "Harmony校正后特征空间",
    "kmeans_clusters_compact": "视觉特征聚类",
    "mst_center_polyline_compact": "MST霜态轨迹",
    "mst_center_polyline_comparison_compact": "MST轨迹与主曲线对照",
    "pca_experiment_offsets": "PCA实验间分布",
    "pca_retained_variance": "PCA保留信息量",
}


CAMERA_NAMES = {
    "front": "正面",
    "left": "左侧",
    "left_close": "左侧近景",
    "top": "顶部",
    "top_close": "顶部近景",
    "extreme": "极端视角",
}


EXPERIMENT_NAMES = {
    **{
        f"camera_LSTM25_FiLM_20261025_{key}": f"六机位_{label}_LSTM"
        for key, label in CAMERA_NAMES.items()
    },
    **{
        f"四模型统一早停_{model}_FiLM_20261011": f"网络比较_{model}_FiLM"
        for model in ("LSTM25", "GRU32", "MLP64x64")
    },
    **{
        f"LSTM25_标准组件与CDF消融_20261013_LSTM组件_{name}": f"组件消融_{name}"
        for name in ("仅视觉", "无CDF", "无CDF且无LSTM历史", "无FiLM", "无视觉")
    },
    **{
        f"LSTM记忆与显式历史_匹配LOEO_20261013_{name}": f"记忆对照_{name}"
        for name in ("B_递归记忆_无频率历史", "C_清零记忆_完整输入", "D_清零记忆_无频率历史")
    },
    "LSTM25_FiLM_去运行时间_20261015": "运行时间消融_保留记忆",
    "LSTM25_FiLM_去运行时间_清零记忆_20261015": "运行时间消融_清零记忆",
}


def relocation_plan():
    """Put final results first and move exploration out of the daily reading path."""
    plan = {
        OUTPUT / source: OUTPUT / target
        for source, target in RESULT_FOLDERS.items()
        if (OUTPUT / source).is_dir()
    }
    figure_folders = list(plan.items()) + [
        (OUTPUT / p, OUTPUT / p) for p in RESULT_FOLDERS.values() if (OUTPUT / p).is_dir()
    ]
    for source, target in figure_folders:
        for file in source.iterdir():
            if file.suffix not in {".png", ".pdf", ".svg", ".tiff"}:
                continue
            for old, new in FIGURE_NAMES.items():
                if file.stem == old or file.stem.startswith(old + "_"):
                    plan[file] = target / (new + file.stem[len(old) :] + file.suffix)
                    break
    for category in ("analysis", "reports"):
        for source in (OUTPUT / category).glob("*"):
            if source.is_dir() and source not in plan:
                plan[source] = (
                    OUTPUT
                    / ARCHIVE
                    / ("探索与旧版图表" if category == "analysis" else "旧汇报与阅读材料")
                    / source.name
                )
    for source in (OUTPUT / "reports").glob("*"):
        if source.suffix == ".pptx" and not source.name.startswith("~$"):
            plan[source] = OUTPUT / "04_汇报材料/学术报告" / source.name
        elif source.suffix == ".zip":
            plan[source] = OUTPUT / "04_汇报材料/霜态方法配图" / source.name
    for name in ("training_checkpoint", "shared"):
        source = OUTPUT / name
        if source.is_dir():
            plan[source] = OUTPUT / REPRODUCTION / name
    if (OUTPUT / "archive").is_dir():
        plan[OUTPUT / "archive"] = OUTPUT / ARCHIVE
    if (OUTPUT / "index.html").is_file():
        plan[OUTPUT / "index.html"] = OUTPUT / ARCHIVE / "旧导航/index.html"
    for index in (OUTPUT / "training_checkpoint").glob("*/index.html"):
        plan[index] = OUTPUT / ARCHIVE / "旧导航/模型" / (index.parent.name + ".html")
    return plan


def reproduction_plan():
    """Keep the inputs of published results, grouped by the task they reproduce."""
    base = OUTPUT / REPRODUCTION
    shared = base / "shared"
    plan = {base / "training_checkpoint": base / "01_视觉模型训练"}
    selected = {
        "camera_LSTM_FiLM_20261024": "02_循环筛选/六机位质量筛选",
        "cycle_admission_rgb70_interpolation30_20261009": "02_循环筛选/历史101循环_质量筛选",
        "cycle_admission_rgb70_interpolation30_cdf_20261009": "02_循环筛选/历史101循环_CDF资格",
        "训练循环筛选/统一队列": "02_循环筛选/统一队列规则与名单",
        "defrost_decisions/effective_cop_external_0928_1010_20261016": "03_COP候选/新增工况候选",
        "defrost_decisions/all_event_three_input_without_trunc_20261008": "03_COP候选/全事件无截断候选",
        "defrost_decisions/effective_cop_offline_linear_60s_20261009": "03_COP候选/离线线性60秒候选",
    }
    for source, target in selected.items():
        plan[shared / source] = base / target
    # Move the remainder after extracting the selected inputs. No files are deleted.
    plan[shared] = OUTPUT / ARCHIVE / "淘汰共享材料"
    for name in (
        "八输入Ridge-三阶段-231-10",
        "八输入Ridge-三阶段-256-10",
        "三输入Ridge-两阶段-279-10",
        "三输入Ridge-两阶段-287-1-离线口径",
        "三输入Ridge-恢复阶段-310-1",
    ):
        plan[OUTPUT / ARCHIVE / "ridge_models" / name] = base / "04_历史成本模型" / name
    frost = OUTPUT / ARCHIVE / "探索与旧版图表/霜层等级分析"
    for source, name in {
        "16级连续性约束评估_20261015": "16级连续性评估",
        "MST主干簇内细分_观测末端_20261012/每阶段2000图_末端q0.99": "16级霜态划分",
        "MST折线聚类数分析_20261012/MST折线_7中心": "MST七中心轨迹",
        "Slingshot曲线改进_20261012/patch_mean_main101_df6_uniform_aggregate": "Slingshot均匀采样轨迹",
        "Slingshot曲线改进_20261012/patch_mean_main101_residual0003_aggregate": "Slingshot原生Harmony轨迹",
        "Slingshot曲线改进_20261012/cls_main101": "原始帧名单与CLS轨迹",
    }.items():
        plan[frost / source] = base / "05_霜态轨迹" / name
    plan[OUTPUT / ARCHIVE / "淘汰共享材料/frosting/images"] = base / "05_霜态轨迹/结霜代表照片"
    evidence = OUTPUT / ARCHIVE / "探索与旧版图表/全事件除霜成本_三输入Ridge_20261008"
    for name in (
        "training_events.csv",
        "event_admission.csv",
        "event_predictions.csv",
        "event_metrics.csv",
        "program_metrics.csv",
    ):
        plan[evidence / name] = base / "04_历史成本模型/三输入Ridge-两阶段-279-10" / name
    models = base / "01_视觉模型训练"
    for old, new in EXPERIMENT_NAMES.items():
        plan[models / old] = models / new
    for source, target in {
        "淘汰共享材料/defrost_decisions/legacy_candidates/effective_cop_without_trunc_20260924": "03_COP候选/历史统一队列_无截断候选",
        "experiments/LSTM16_训练侧快速消融_20261012": "06_模型选择依据/LSTM16宽度筛选",
        "experiments/LSTM_FiLM分阶段优化_20261011": "06_模型选择依据/LSTM宽度与训练参数筛选",
        "experiments/LSTM25_快速组件消融_20261013": "06_模型选择依据/LSTM25早期组件对照",
        "淘汰共享材料/defrost_decisions/rb_sampling_time_audit": "02_循环筛选/RB触发时间核验",
    }.items():
        plan[OUTPUT / ARCHIVE / source] = base / target
    return {source: target for source, target in plan.items() if source.exists()}


def relocate(plan=None, *, rebuild_index=False):
    plan = relocation_plan() if plan is None else plan
    if not plan:
        print("Output directories already follow the nested layout.")
        return
    for source, target in plan.items():
        if target.exists():
            raise FileExistsError(f"Relocation would overwrite {target}")
    # Rewrite path-bearing text, never weights or numerical values. Longest source wins.
    replacements = {
        str(s.relative_to(PROJECT)): str(t.relative_to(PROJECT)) for s, t in plan.items()
    }
    for source, target in plan.items():
        if source.parent.parent.name == "training_checkpoint" and list(source.glob("*.pkl")):
            for model in source.glob("*.pkl"):
                replacements[f"output/{source.parent.name}/{model.name}"] = str(
                    (target / model.name).relative_to(PROJECT)
                )
    replacements.update(
        {
            json.dumps(s)[1:-1]: json.dumps(t)[1:-1]
            for s, t in list(replacements.items())
            if json.dumps(s)[1:-1] != s
        }
    )
    prefixes = {str(s.parent.relative_to(PROJECT)) + "/" for s in plan}
    prefixes.update(json.dumps(s)[1:-1] for s in list(prefixes))
    prefixes = {p for p in prefixes if not any(p != q and p.startswith(q) for q in prefixes)}
    pattern = re.compile(
        "(?:"
        + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
        + r")(?=$|[/\s\"\x27`<>),])"
    )
    roots = [p for p in PROJECT.iterdir() if p.is_file() and p.suffix in {".py", ".md", ".toml"}]
    for name in [
        "image_models",
        "dataset_tools",
        "defrost_decision",
        "defrost_event_models",
        "heat_smoothing",
        "plots",
        "paper_workflow",
        "scripts",
        "tests",
        "docs",
        "output",
        "temp",
        "dataset",
    ]:
        folder = PROJECT / name
        if folder.exists():
            roots.extend(
                p
                for p in folder.rglob("*")
                if p.suffix
                in {".py", ".json", ".md", ".txt", ".sh", ".yaml", ".yml", ".toml", ".csv", ".html"}
                and p.is_file()
                and not p.is_symlink()
            )
    roots.append(PROJECT / "dataset/cycle_catalog.json")
    print(
        f"Checking references in {len(roots)} text files before moving {len(plan)} directories.",
        flush=True,
    )
    changed = 0
    for path in roots:
        if path == Path(__file__).resolve() or not path.exists():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if not any(prefix in text for prefix in prefixes):
            continue
        updated = pattern.sub(lambda m: replacements[m.group()], text)
        if updated != text:
            path.write_text(updated)
            changed += 1
    print(f"Updated {changed} reference files; moving directories.", flush=True)
    pending = dict(plan)
    while pending:
        for source, target in list(pending.items()):
            if any(s != source and target.is_relative_to(t) for s, t in pending.items()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            del pending[source]
            for child, destination in list(pending.items()):
                if child.is_relative_to(source):
                    del pending[child]
                    pending[target / child.relative_to(source)] = destination
            break
        else:
            raise ValueError("Cyclic directory move plan")
    for folder in (OUTPUT / REPRODUCTION / "01_视觉模型训练").glob("*"):
        if folder.is_dir() and not list(folder.iterdir()):
            folder.rmdir()
    if rebuild_index:
        from tools.result_index import write_output_index

        write_output_index()
    print(f"Moved {len(plan)} directories; updated references in {changed} text files.")


def archive(paths):
    """Archive explicitly chosen experiments or Ridge models with their references."""
    destinations = {
        OUTPUT / REPRODUCTION / "training_checkpoint": OUTPUT / ARCHIVE / "experiments",
        OUTPUT / REPRODUCTION / "01_视觉模型训练": OUTPUT / ARCHIVE / "experiments",
        PROJECT / "dataset/除霜预测模型": OUTPUT / ARCHIVE / "ridge_models",
    }
    plan = {}
    for path in paths:
        source = Path(path).resolve()
        if source.parent not in destinations or not source.is_dir():
            raise ValueError(f"Expected an existing experiment or Ridge model: {path}")
        plan[source] = destinations[source.parent] / source.name
    relocate(plan)


def publish_accessible_files():
    """Put the six deployment models and current lists within three folder steps."""
    camera_names = {
        "front": "正面",
        "left": "左侧",
        "left_close": "左侧近景",
        "top": "顶部",
        "top_close": "顶部近景",
        "extreme": "极端视角",
    }
    copies = {}
    comparison = OUTPUT / "02_模型与消融/运行时间与记忆"
    for group in ("全部101循环", "排除循环11和155"):
        copies[comparison / group / "四组对照.csv"] = comparison / f"四组对照_{group}.csv"
    for camera, label in camera_names.items():
        folder = (
            OUTPUT
            / REPRODUCTION
            / "01_视觉模型训练"
            / EXPERIMENT_NAMES[f"camera_LSTM25_FiLM_20261025_{camera}"]
            / "checkpoints"
            / f"{camera}_deployment_seed42"
        )
        target = OUTPUT / "02_模型与消融/部署模型_六机位"
        copies[folder / f"{camera}_deployment_seed42.pkl"] = target / f"{label}_部署模型.pkl"
        copies[folder / "cycles.csv"] = target / f"{label}_训练与测试循环.csv"
        suffix = "" if camera == "front" else "__" + camera
        target = OUTPUT / "03_数据与物理依据/当前循环名单与筛选"
        copies[PROJECT / f"dataset/training_data/cycle_screening{suffix}.csv"] = (
            target / f"{label}_循环筛选结果.csv"
        )
        copies[PROJECT / f"dataset/training_data/settings{suffix}.json"] = (
            target / f"{label}_筛选配置.json"
        )
    copies[PROJECT / "dataset/training_data/cohort.csv"] = (
        OUTPUT / "03_数据与物理依据/当前循环名单与筛选/全部机位_循环名单与评价资格.csv"
    )
    ridge = PROJECT / "dataset/除霜预测模型/三输入Ridge-两阶段-287-1-统一平滑"
    target = OUTPUT / "01_核心结果/除霜电耗预测_287事件"
    for source, name in [
        ("candidate_model_parameters.json", "Ridge模型参数.json"),
        ("defrost_events.csv", "287事件训练清单.csv"),
        ("event_admission.csv", "事件筛选结果.csv"),
    ]:
        copies[ridge / source] = target / name
    for run in (OUTPUT / REPRODUCTION / "01_视觉模型训练").glob("*"):
        for source, label in (
            ("settings.json", "训练配置.json"),
            ("frozen_configurations.csv", "冻结训练配置.csv"),
            ("summary.csv", "训练汇总.csv"),
        ):
            copies[run / "training" / source] = run / label
    for source, destination in copies.items():
        if not source.is_file():
            continue
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise FileExistsError(f"Published copy differs: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in ("analysis", "reports"):
        folder = OUTPUT / name
        if folder.is_dir() and all(p.name == ".DS_Store" for p in folder.iterdir()):
            (folder / ".DS_Store").unlink(missing_ok=True)
            folder.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--migrate", action="store_true", help="Apply the existing flat-output relocation"
    )
    parser.add_argument(
        "--archive",
        nargs="+",
        type=Path,
        help="Archive named experiment or Ridge model directories",
    )
    parser.add_argument(
        "--organize-reproduction",
        action="store_true",
        help="Group retained reproduction inputs by purpose",
    )
    args = parser.parse_args()
    if args.organize_reproduction:
        relocate(reproduction_plan())
        publish_accessible_files()
    elif args.archive:
        archive(args.archive)
    elif args.migrate:
        relocate()
        publish_accessible_files()
    else:
        for source, target in relocation_plan().items():
            print(f"{source.relative_to(PROJECT)} -> {target.relative_to(PROJECT)}")
