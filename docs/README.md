# 技术文档与科学定义

[返回项目验收](../README.md)

- [循环 COP 的物理定义](cycle_cop_first_principles_zh.md)：供热、电耗与循环边界。
- [中文技术文档（LaTeX）](technical_report.tex)：研究方法、四阶段用法、结果解读和代码阅读顺序。
- [Dataset 格式](../DATASET_FORMAT.md)：测量、阶段、资格和模型输入。
- [冻结 RB 规则](../RB_RULES_FROZEN.md)：触发规则及观测结束前未触发的处理。

操作从项目 README 开始；技术文档从研究问题逐步展开到代码。编译需已安装 XeLaTeX 的 TeX Live 或 MacTeX，在项目根目录执行：

```bash
mkdir -p temp/test/technical_document
latexmk -xelatex -outdir=temp/test/technical_document docs/technical_report.tex
```

PDF 位于 `temp/test/technical_document/technical_report.pdf`。编译不读取实验数据。
