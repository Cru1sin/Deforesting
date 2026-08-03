# Cycle Dataset 数据合同

Cycle Dataset 是正式日期 run 的只读发布层。它不重新执行 Prepare、Process 或 Analyze，
只读取已经通过科学 validator 的 `prepared_data.parquet`、`processed_data.parquet` 和
`cycle_summary.csv`。

## 目录和身份

```text
outputs/datasets/frost_cycles_v1/
├── cycles/frost_cycle_000001.parquet
├── images/frost_cycle_000001__front__000001__20260715T101010123.jpg
├── cycle_index.parquet
├── image_index.parquet
└── dataset_manifest.json
```

`dataset_id` 是最终目录名，必须匹配小写下划线格式。固定合同版本为 `1`，全局 cycle
编号宽度为 `6`。来源身份和展示身份分别为：

```text
cycle_uid  = experiment_id + "__" + cycle_id
cycle_name = frost_cycle_{dataset_cycle_index:06d}
```

Build 按 `experiment_date → experiment_id → cycle_id` 排序；`cycle_id` 中的数字按自然
顺序排序，因此 `cycle_2` 排在 `cycle_10` 前。Append 只在历史编号之后追加新编号。

## 文件和索引

每个发布 cycle 文件保留来源 Processed 的全部列、顺序、值和逻辑类型，并在末尾追加：

```text
dataset_id, dataset_schema_version, dataset_cycle_index, cycle_name, cycle_uid
```

图片只从 Prepared 的匹配记录导出，不从 Processed 反向发现。每个
`cycle_uid × camera_role` 内按 `image_time → source_relative_path` 排序，`frame_index`
从 `1` 开始；图片文件名去掉扩展名后的 stem 就是 `image_id`。Processed 中的图片路径
改写为 Dataset 内的相对 POSIX 路径。

`cycle_index.parquet` 包含所有 Summary cycle。只有 `processed_row_count > 0` 的 cycle
才发布文件；无 Processed 行的 cycle 使用 `dataset_exclusion_reason=no_processed_rows`。
`recommended_for_analysis` 只表示：

```python
published and cycle_status == "valid" and baseline_status == "available"
```

它不是通用训练资格。`image_index.parquet` 保存每张导出图片的来源相对路径、匹配时间、
文件大小和单一 SHA-256。

Manifest 保存来源 run 的内容组成指纹和两张索引的 SHA/行数。Dataset fingerprint 只包含
配置、Prepared、Processed、Summary 和匹配图片 inventory；Evidence、Git commit、manifest
SHA 和 source path 只作审计信息，不参与 Dataset 幂等判断。

## Build、Append 和验证

Build 要求目标目录不存在，先在同级 staging 中写出并完整验证，验证通过后才发布。Append
先做轻量历史结构检查，再把新 cycle、图片和合并 metadata 写入同级 staging；metadata 按
`cycle_index → image_index → manifest` 顺序原子替换。相同 experiment 的相同 fingerprint
是 no-op，不同 fingerprint 拒绝；普通异常会恢复旧 metadata 并删除本次新增文件。

```bash
python -m frost_analysis dataset build \
  --run outputs/runs/0715 \
  --run outputs/runs/0716 \
  --output outputs/datasets/frost_cycles_v1

python -m frost_analysis dataset append \
  --run outputs/runs/0720 \
  --dataset outputs/datasets/frost_cycles_v1

python -m frost_analysis dataset validate \
  --input outputs/datasets/frost_cycles_v1
```

公开 `dataset validate` 会重新计算全部 cycle 和图片 SHA，检查索引闭包、时间顺序、逻辑
schema 和 orphan 数据文件。Append 内部只重新检查本次新增文件；进程被强制终止留下的
orphan 文件由公开 validator 报告，恢复方式是从正式 source runs 重新 Build。
