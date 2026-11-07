# 技术文档与科学定义

[返回项目验收](../README.md)

- [循环 COP 的物理定义](cycle_cop_first_principles_zh.md)：供热、电耗与循环边界。
- [中文技术文档（LaTeX）](technical_report.tex)：方法、验证结果、四阶段操作和代码索引。
- [Dataset 格式](../DATASET_FORMAT.md)：测量、阶段、资格和模型输入。
- [冻结 RB 规则](../RB_RULES_FROZEN.md)：触发规则及观测结束前未触发的处理。

按需要阅读技术文档：

| 目的 | 位置 |
| --- | --- |
| 先跑演示、核对结果 | 首页“快速演示” |
| 理解方法与验证 | 第 1–6 节 |
| 分别验收四个阶段 | 第 7 节：输入、命令、输出、判读 |
| 定位文件、函数与字段 | 第 8 节 |

编译需已安装 XeLaTeX 的 TeX Live 或 MacTeX，在项目根目录执行：

```bash
mkdir -p temp/test/technical_document
latexmk -xelatex -outdir=temp/test/technical_document docs/technical_report.tex
```

PDF 位于 `temp/test/technical_document/technical_report.pdf`。编译不读取实验数据。
