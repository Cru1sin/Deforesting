# Cycle Dataset v2 数据合同

Cycle Dataset 是正式日期 run 的自包含发布层。它只读取已经通过科学 validator 的
Prepared、Processed 和 Summary 产物；它不重新切分 cycle、重采样、匹配图片、计算
baseline、派生量或质量结论。

## 目录

```text
dataset/
├── cycle_index.parquet
├── image_metadata.parquet
├── dataset_manifest.json
├── README.md
├── cycles/
│   ├── frost_cycle_000001.parquet
│   ├── frost_cycle_000001.csv
│   ├── frost_cycle_000001.png
│   └── frost_cycle_000001_rgb_coverage.png
└── images/
    └── frost_cycle_000001/
        └── unassigned_01/
            └── <image-id>.jpg
```

每个 Summary cycle 都会有上述四个 cycle 文件。没有 Processed 行的 cycle 使用保留来源
schema 的空 Parquet/CSV；它仍有 publication 和 RGB coverage 图，不伪造科学数值。

`cycle_uid` 为 `experiment_id + "::" + source_cycle_id`，`cycle_name` 按
`experiment_date → experiment_id → cycle_id`（cycle 数字自然排序）分配全局六位编号。
图片在每个 cycle 和初始 camera slot 内按 `image_time → source_relative_path` 稳定排序，
`image_id` 等于发布文件的 stem。

## 单一 assessment

Manifest 的每个 cycle 只有一个可人工修改的 `assessment`：

```json
{
  "status": "valid",
  "reasons": [],
  "note": null,
  "updated_at": "2026-08-03T16:00:00+08:00"
}
```

状态为 `valid`、`partial`、`incomplete` 或 `invalid`。初始值来自现有 Summary；人工审阅
直接修改同一对象。`dataset refresh-manifest` 只刷新文件事实和图片覆盖摘要，不覆盖四个
assessment 字段。`dataset review-cycle` 是该 JSON 编辑的轻量 CLI 入口。

## 图片角色和读取入口

图片当前的直接父目录名是 camera role 的唯一权威来源。Loader 不读取 YAML 角色映射；将
`images/<cycle>/<old-role>/` 重命名后，新的 `DatasetLoader.load_cycle_images()` 直接返回
新目录名。`image_metadata.parquet` 保存 source camera、初始 slot、匹配时间、offset、来源
相对路径、文件大小和 SHA-256，但不覆盖当前目录角色。

所有 Dataset 下游通过 `frost_analysis.dataset_loader.DatasetLoader` 读取：

```python
loader = DatasetLoader(Path("dataset"))
cycles = loader.list_cycles(statuses={"valid"})
frame = loader.load_cycle("frost_cycle_000001")
images = loader.load_cycle_images("frost_cycle_000001")
```

Publication、RGB coverage 和 Dataset analysis 只接收 Loader 提供的数据。分析结果写入
Dataset 外部的可再生输出目录，不回写 Dataset。

## 发布和验证

0714 从不存在的目录开始，0715 使用同一个 `add` 入口追加：

```bash
python -m frost_analysis dataset add \
  --run outputs/runs/0714 \
  --dataset outputs/datasets/frost_cycles_v2

python -m frost_analysis dataset add \
  --run outputs/runs/0715 \
  --dataset outputs/datasets/frost_cycles_v2

python -m frost_analysis dataset validate \
  --input outputs/datasets/frost_cycles_v2
```

Append 在 Dataset 同级 staging 中写入新增 cycle、图片和合并 metadata；旧文件不重写，
metadata 按 `cycle_index → image_metadata → manifest` 顺序替换。相同 source fingerprint
是 no-op；不同 fingerprint 拒绝。普通异常会恢复旧 metadata、删除本次移动文件，并再次
执行轻量结构检查。

公开 validator 会重新计算所有 cycle 资产和图片 SHA，检查四件套、路径、schema、cycle
引用、image metadata 闭包、assessment、时间和 orphan。进程被强制终止留下的临时文件由
validator 发现；恢复方式是从正式 run 重新 `add` 到新的 Dataset。
