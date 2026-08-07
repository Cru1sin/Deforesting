# Cycle Dataset v3 数据合同

Cycle Dataset 是自包含的科学数据发布层。构建链直接从日期原始目录执行
`prepare → validate_prepared → process → validate_processed → Dataset`；Dataset
操作不读取 Raw、YAML 或任何旧版本 Dataset。

## 目录

```text
dataset/
├── dataset_manifest.json
├── cycle_catalog.json
├── image_metadata.parquet
├── channel_registry.json
├── README.md
├── cycles/
│   ├── frost_cycle_000001.parquet
│   ├── frost_cycle_000001.csv
│   ├── frost_cycle_000001.png
│   └── frost_cycle_000001_rgb_coverage.png
├── cycles_original/
│   └── frost_cycle_000001.csv
└── images/
    └── frost_cycle_000001/
        └── <source_camera_id>__<current_role>/
            └── <original-basename>
```

`cycles/` 保存 10 秒 Processed 科学数据；`cycles_original/` 保存按 cycle 切分的
Prepared 分辨率标准化数据。Original CSV 保留真实 timestamp、标准通道、identity、
构建状态和 cycle stage，但不保存逐通道质量审计列、图片字段、baseline、residual、
dynamic features 或 10 秒聚合列。追加日期发现新标准通道时，历史 Original CSV 会补充
该列并写入空值，保持跨日期 schema 一致。

## Manifest 与 Catalog

`dataset_manifest.json` 只记录 Dataset 身份和实验来源：

```json
{
  "dataset_schema_version": 3,
  "dataset_id": "frost_cycle_dataset",
  "experiments": [
    {
      "experiment_id": "exp_20260714",
      "experiment_date": "2026-07-14",
      "camera_roles": {}
    }
  ]
}
```

`cycle_catalog.json` 保存每个 cycle 的身份、构建状态、当前人工状态、边界、数据摘要、
图片摘要和固定资产路径。`pipeline_status` 是上游事实；
`status` 是当前唯一 Dataset 使用状态，只能通过 `review-cycle` 修改。Evidence 只按
`status` 过滤。

## 图片与 coverage

图片的稳定逻辑身份由 cycle UID、source camera ID 和来源相对路径生成；图片文件名保持
原始 basename，不按 cycle 重新编号。当前角色只体现在父目录后缀：

```text
camera01__unassigned_01 → camera01__front
```

`image_metadata.parquet` 保留图片匹配、来源、frame index、文件大小和 stage 信息，但不
保存当前角色目录、最终路径或图片内容 SHA。Loader、refresh 和 coverage 通过扫描当前
父目录并连接 metadata 获取实际存在的图片；不会做图片内容哈希、闭包校验或 orphan 失败。

RGB coverage 先按 `max_image_gap_seconds` 合并 available/missing 时间区间，再由同一组
区间同时生成 Catalog 摘要和 coverage PNG。Sensor coverage 独立使用 Processed 10 秒网格。

## CLI

```bash
python -m frost_analysis dataset add data/0714
python -m frost_analysis dataset add data/0715
python -m frost_analysis dataset rebuild data/0714 data/0715
python -m frost_analysis dataset validate --dataset dataset
python -m frost_analysis dataset refresh --dataset dataset
python -m frost_analysis dataset review-cycle frost_cycle_000001 \
  --status valid --reason manual_review_confirmed
python -m frost_analysis dataset edit --dataset dataset --baseline-seconds 60
python -m frost_analysis dataset edit --dataset dataset --recovery-seconds 180
python -m frost_analysis dataset render --dataset dataset frost_cycle_000001 \
  --publication --coverage
```

`dataset add` 对相同实验 identity 直接 no-op；原始数据或配置变化由用户显式执行
`rebuild`。新日期必须晚于 Dataset 最后日期。`rebuild` 从 Raw 重新生成，不迁移旧
status、人工边界、baseline 或 camera role。

科学 edit 的当前规则保存在 `channel_registry.json`。后续 `add` 会对新 cycle 应用同一
baseline/recovery 规则，避免一个 Dataset 混用不同的管理设定。`--recovery-seconds` 与
`--recovery-end-by ts-minus` 互斥。Recovery edit 会同步更新 Original、Processed、
cycle coordinates、stage-partitioned dynamic features、图片 metadata stage 和相关图形。

## 写入与校验

Dataset 直接写入目标目录，不持久化 staging、hardlink、rollback 或事务状态。`add`、
`rebuild`、`edit` 和 `refresh` 失败后，按当前文件状态修正问题并重新运行相应操作；需要
从零恢复时使用 `dataset rebuild`。

科学构建阶段由 `validate_prepared()` 和 `validate_processed()` 检查 Prepared/Processed
数据。显式 `dataset validate` 才读取已发布 Dataset，检查非空数据、schema、时间顺序、
cycle identity、row count 和 Original 的质量列合同。Dataset 不保存或核对资产 SHA，
图片也不做内容哈希、闭包或 orphan 校验。

Dataset 下游统一使用：

```python
loader.load_cycle("frost_cycle_000001")
loader.load_cycle_original("frost_cycle_000001")
loader.load_cycle_images("frost_cycle_000001")
```
