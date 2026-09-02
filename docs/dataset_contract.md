# Cycle Dataset v3 数据合同

Cycle Dataset 是包含人工科研 review 的增量维护发布层。`dataset add` 从日期原始目录执行
`prepare → validate_prepared → process → validate_processed → Dataset`；其余 Dataset
操作只维护当前 Dataset。

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
│   └── frost_cycle_000001_rgb_panel.png
├── cycles_original/
│   └── frost_cycle_000001.csv
└── images/
    └── frost_cycle_000001/
        └── <camera_role>/
            └── <original-basename>
```

`cycles/` 保存经过正式 channel 合同筛选的 10 秒 Processed 科学数据；
`cycles_original/` 保存按 cycle 切分的原始传感器点位全集。Original CSV 是 Prepared
标准通道、identity、构建状态和 cycle stage 的超集，并额外保留全部带参数组前缀的
原始控制器列（例如 `p1__TL(28.1)'2_31'`）。重复 timestamp 观测不会被折叠。
双 SHT40 的四个原始观测另存为 `edf__sensor_1/2_temperature/humidity`。
Original 不保存逐通道质量审计列、图片字段、baseline、residual 或 10 秒聚合列。
追加日期发现新原始点位时，历史 Original CSV 会补充该列并写入空值，保持跨日期
schema 一致。

超过连续信号插值上限的传感器缺口，只有在缺口两侧同时观察到运行模式从制热变为
停机时才切分；缺口前片段保留为 `invalid`，缺口后片段保留为 `valid`。断联前后仍是
制热模式时不切分，只保留缺口质量事实。

## Manifest 与 Catalog

`dataset_manifest.json` 记录 Dataset 身份、唯一图片根目录和实验来源：

```json
{
  "dataset_schema_version": 3,
  "dataset_id": "frost_cycle_dataset",
  "images_root": "images",
  "experiments": [
    {
      "experiment_id": "exp_20260714",
      "experiment_date": "2026-07-14"
    }
  ]
}
```

`cycle_catalog.json` 保存每个 cycle 的身份、构建状态、当前人工状态、边界、数据摘要、
图片摘要和固定资产路径。`pipeline_status` 是上游事实；
`status` 是当前唯一 Dataset 使用状态，只能通过 `review-cycle` 修改。Evidence 只按
`status` 过滤。自动与人工状态都只允许 `valid` / `invalid`；开放边界、状态缺口等
结构信息保存在 reason 和 boundaries 中，不再形成第三种状态。

`images_root` 可以是相对 Dataset 的 `images`，也可以在图片整体移动后直接改成
OneDrive `images` 文件夹的绝对路径。目录内部仍固定为
`<cycle_name>/<camera_role>/<original_basename>`；此时本地 `dataset/images/` 可以不存在，
也无需改 `image_metadata.parquet`。

## 图片与 availability

图片文件名保持原始 basename，不按 cycle 重新编号；父文件夹名就是唯一 camera role：

```text
unverified_camera_05 → front_center
```

图片只按 `start_time <= image_time < end_time` 归属 cycle，不以 Sensor 时间戳匹配作为
准入条件。`image_metadata.parquet` 只保存 `cycle_name`、`camera_role`、`file_name`、
`frame_index`、`image_time` 和 `cycle_stage`；不提前保存 Sensor–RGB 对齐结果。
Loader 和 refresh 从 `images_root` 扫描当前父目录；不做图片内容哈希。

RGB availability 先按 `max_image_gap_seconds` 合并时间区间，再取当前实验全部 camera
roles 的交集。Publication 的 stage ribbon 下固定绘制 Sensor/RGB 两行 availability；
不再生成独立 coverage PNG。

## CLI

```bash
uv run python main_data.py add data/0714
uv run python main_data.py add data/0715
uv run python main_data.py aggregate-original --seconds 10
uv run python main_data.py aggregate-original --seconds 30
uv run python main_data.py edit --defrost-preparation
uv run python main_data.py remove 0722 --dataset dataset
uv run python main_data.py validate --dataset dataset
uv run python main_data.py refresh roles --dataset dataset
uv run python main_data.py refresh images --dataset dataset
uv run python main_data.py refresh figures --dataset dataset
uv run python main_data.py review-cycle frost_cycle_000001 \
  --status valid --reason manual_review_confirmed
uv run python main_data.py edit --dataset dataset --baseline-seconds 60
uv run python main_data.py edit --dataset dataset --recovery-seconds 180
uv run python main_data.py render --dataset dataset frost_cycle_000001 \
  --publication --panel
uv run python main_data.py render frost_cycle_000020 \
  --panel --fetch-cloud-images
```

`main_data.py render` 默认只使用本地 `dataset/images/<cycle_name>`，不会访问云端。只有显式
增加 `--fetch-cloud-images` 时，RGB panel 才检查 OneDrive 中精确命名的
`<cycle_name>.zip`。存在时使用 `rclone` 直接下载到 `dataset/images` 下的独立临时
目录，校验 ZIP 路径、解压并绘图，然后在 `finally` 中删除临时 ZIP 和解压目录；云端
对象从不被删除或修改。云端没有相应 ZIP 时，该 cycle 按无 RGB 图片处理。

Python 图像训练代码可复用 `dataset_images.materialize_cycle_images(...)` 上下文；用
`contextlib.ExitStack` 同时保持多个 cycle 上下文，即可在一批训练结束时统一清理。

`aggregate-original` 不读取 Data 或图片。Dataset 当前间隔（默认 10 秒）写回
`cycles/`；其他间隔写入 `cycles_<seconds>s/`。

`main_data.py add` 对相同实验 identity 直接 no-op，新日期必须晚于 Dataset 最后日期。
实验日期自动取自 XLS 参数文件名；EDF 日期不参与判断，多个 XLS 日期会直接报错。
`remove` 只删除匹配日期/experiment 的循环且不重编号其他循环；后续 `add` 从现有最大
cycle 编号加一。`refresh roles/images/all` 自动更新所有图片派生结果并保留人工状态；
`refresh figures` 只重画 publication 和 RGB panel。没有日常 rebuild 入口。

某日切分逻辑修正后，用原日期目录重新发布；循环增减和后续编号顺延自动完成：

```bash
uv run python main_data.py replace data/0729 --dataset dataset
```

科学 edit 的当前规则保存在 `channel_registry.json`。后续 `add` 会对新 cycle 应用同一
baseline/recovery 规则，避免一个 Dataset 混用不同的管理设定。`--recovery-seconds` 与
`--recovery-end-by ts-minus` 互斥。Recovery edit 会同步更新 Original、Processed、
cycle coordinates、图片 metadata stage 和相关图形。

## 写入与校验

Dataset 直接写入目标目录，不持久化 staging、hardlink、rollback 或事务状态。`add`、
`remove`、`edit` 和 `refresh` 失败后，按当前文件状态修正问题并重新运行相应操作。
真的需要从零恢复时，明确删除或另建 Dataset 目录，再逐日期执行 `add`。

科学构建阶段由 `validate_prepared()` 和 `validate_processed()` 检查 Prepared/Processed
数据。显式 `main_data.py validate` 才读取已发布 Dataset，检查非空数据、schema、时间顺序、
cycle identity、row count 和 Original 的质量列合同。Dataset 不保存或核对资产 SHA，
图片也不做内容哈希、闭包或 orphan 校验。

Dataset 下游统一使用：

```python
loader.load_cycle("frost_cycle_000001")
loader.load_cycle_original("frost_cycle_000001")
loader.load_cycle_images("frost_cycle_000001")
```
