# 论文与技术报告

[返回项目验收](../README.md)

| 文档 | 内容 | 本地 PDF |
| --- | --- | --- |
| [论文](paper.tex) | 引言、方法、实验设置、结果、讨论、结论；附录补充霜层分析与方法细节 | `paper.pdf` |
| [技术报告](technical_report.tex) | 快速演示、四阶段输入输出、霜层等级分析与代码索引 | `technical_report.pdf` |

先跑原型看技术报告首页；理解科学方法看论文。方法先定义有效供热 COP，再介绍电耗预测与视觉决策；实验设置先说明筛选条件与过程，再介绍最终样本。结果给出电耗验证、RB 改善空间、模型比较和运行分析，主要贡献统一归纳于结论。附录 A–C 分别为霜层等级识别、数据预处理和 RB 完整规则。两份文档独立阅读，代码用法无需对照论文查找。

编译需 XeLaTeX（TeX Live 或 MacTeX），在项目根目录执行：

```bash
latexmk -xelatex -outdir=temp/test/paper docs/paper.tex
latexmk -xelatex -outdir=temp/test/technical_document docs/technical_report.tex
```

编译结果分别位于上述输出目录。论文引用 [figures](figures/README.md) 中的 15 张 PNG 与 2 张矢量 PDF；RB 表由 LaTeX 排版。技术报告附循环 000043 的三张演示图及霜层方法、覆盖率和 COP 三图。编译不运行模型或处理实验数据；PDF 与图片在本地交付。

定义与数据格式：[循环 COP](cycle_cop_first_principles_zh.md) · [Dataset](../DATASET_FORMAT.md) · [冻结 RB](../RB_RULES_FROZEN.md)。
