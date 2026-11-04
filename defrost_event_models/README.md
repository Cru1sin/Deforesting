# 准备＋除霜电耗模型

[返回阶段 2 验收](../README.md#2-除霜电耗模型)

入口是 [defrosting_electricity_model.py](../defrosting_electricity_model.py)。输入压缩机频率、环境温度、盘管温度，以实测准备＋除霜电耗为监督，单位 kWh。

## 读代码的顺序

1. [training_data.py](training_data.py) `build_defrost_event_training_table`：按真实阶段边界汇总事件输入与实测电耗。
2. [event_electricity_ridge.py](event_electricity_ridge.py) `prepare_training_events`：筛选可用于电耗拟合的事件，资格独立于视觉循环筛选。
3. 同文件 `fit_preparation_defrost_model`：按实验隔离进行嵌套验证，选择正则强度，保存留出模型与全量模型。
4. 同文件 `load_defrost_event_models`：供 COP 读取冻结参数。

三项输入标准化为 z 后，模型为 `E_prep+D = β0 + βᵀz`。盘管温度读取 Dataset 的处理值，不重复平滑。

## 结果与使用

`validation_summary.csv` 看整体误差；`model_validation.csv` 比较 `E_prep_D_kwh` 与 `predicted_E_prep_D_kwh`；`event_admission.csv` 解释事件准入；`candidate_model_parameters.json` 供下一阶段计算。

省略 `--output` 时按实际入模数量和采样秒数命名为 `<dataset>/除霜预测模型/三输入Ridge-两阶段-<事件数>-<采样秒数>/`。同名结果更新，冻结历史目录保留。

COP 默认使用已发布冻结模型；新拟合结果必须通过 `--model-file` 显式接入。已有实验评价使用 `cross-fitted` 留出模型，新实验预测使用 `full-model`；两种用途不混报。
