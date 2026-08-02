"""The one small orchestration entry point for a complete run."""

from __future__ import annotations

from pathlib import Path

from .analysis import analyze
from .channels import load_channels
from .config import load_config
from .io import remove_manifest_for_overwrite, write_run_outputs
from .prepare import prepare
from .process import process
from .validation import validate_analysis, validate_prepared, validate_processed


def run_pipeline(config_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    """Run Prepare, Process, and Analyze in their documented order."""

    # 1. 读取某一天实验的路径、时间阈值和分析阈值
    config = load_config(config_path)

    remove_manifest_for_overwrite(output_dir, config.input_dir, overwrite=overwrite)

    # 2. 读取所有通道的原始字段名、单位、类型和处理策略
    channels = load_channels(config.channels_path)

    # 3. 原始文件 → Prepared：
    #    解析、单位换算、质量标记、循环切分、图片初步匹配
    prepared, initial_summary, prepare_summary = prepare(config, channels)

    # 4. 检查 Prepared 是否满足结构合同
    validate_prepared(prepared, initial_summary)

    # 5. Prepared → Processed：
    #    10 秒网格、coverage、bounded fill、派生量、baseline、动态特征
    processed, final_summary = process(prepared, initial_summary, config, channels)

    # 6. 检查 Processed 是否满足结构与科学合同
    validate_processed(processed, final_summary)

    # 7. Processed → 候选通道证据
    evidence = analyze(processed, final_summary, config, channels)

    # 8. 检查证据表字段、计数和 decision 是否有效
    validate_analysis(evidence)

    # 9. 写出正式结果和 manifest
    write_run_outputs(
        prepared,
        processed,
        final_summary,
        evidence,
        prepare_summary,
        config,
        config_path,
        output_dir,
        config.input_dir,
        overwrite=overwrite,
    )
    return output_dir
