# RB 参照与测量核验

[返回 COP](../README.md)

| 文件 | 职责 |
| --- | --- |
| [rule_based.py](rule_based.py) | 回放冻结 RB 触发规则；规则正文见 [RB_RULES_FROZEN.md](../../RB_RULES_FROZEN.md) |
| [audit.py](audit.py) | 对照实际运行记录，核验规则行为 |

主流程直接读取 [rb_reference.csv](../../dataset/priors/rb_reference.csv)。`t_RB` 是规则首次触发时间；`right_censored` 表示观测结束前未触发，不能用实机时刻补填。

只有新增数据或明确更新先验时才运行根目录 `calculate_rb_reference.py --publish-prior --output <本次目录>`。该操作更新先验；普通 COP 计算不回放或补算 RB。

RB 时刻与 RB COP 是不同资产：时刻可冻结，COP 随本次电耗模型及测量定义计算。

主流程的电量积分统一由 [training_data.py](../../defrost_event_models/training_data.py) 计算。
