# 论文与代码使用手册

[返回项目验收](../README.md)

| 文档 | 内容 | 本地 PDF |
| --- | --- | --- |
| [论文](paper.tex) | 引言、方法、实验设置、结果、讨论、结论；附录补充霜层分析与方法细节 | `paper.pdf` |
| [代码使用手册](technical_report.tex) | 快速演示、四阶段输入输出、霜层等级分析与代码索引 | `technical_report.pdf` |

首次运行看代码使用手册首页；按任务选择四阶段、报告案例或霜层统计。新增实验的完整操作放在附录，代码索引随后列出。研究方法和实验结果统一放在论文中。

编译需 XeLaTeX（TeX Live 或 MacTeX），在项目根目录执行：

```bash
latexmk -xelatex -outdir=temp/test/paper docs/paper.tex
latexmk -xelatex -outdir=temp/test/technical_document docs/technical_report.tex
```

编译结果分别位于上述输出目录。论文引用 [figures](figures/README.md) 中的 15 张 PNG 与 2 张矢量 PDF；RB 表由 LaTeX 排版。代码使用手册仅保留循环 000043 的三张既有案例附图。编译不运行模型或处理实验数据；PDF 与图片在本地交付。

定义与数据格式：[循环 COP](cycle_cop_first_principles_zh.md) · [Dataset](../DATASET_FORMAT.md) · [冻结 RB](../RB_RULES_FROZEN.md)。
