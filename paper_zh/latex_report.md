# LaTeX 构建报告

## 基线

- 主稿来源：`paper_zh/main.tex`，中文 `ctexart`，不是 Nature 官方模板。
- 图：六张主图和一份59页逐循环补充图集均来自已提交的 Python 输出。

## 科学边界

- “最优”限定为固定经验除霜门票与离线参考下的策略条件最小值。
- Tier B 是右截尾敏感性，不并入主效果。
- pooled training 不称为同步 multi-view fusion。
- cycle11 OOD 尚未本地提取与预测，正文只写计划，不写结果。
- 单帧视觉集中命题在3实验开发性分析中未获支持。

## 待作者补充

- 作者、单位、资助、伦理/许可与利益冲突。
- 正式数据仓库、代码版本与持久标识符。
- 全文文献作者列表和逐条全文证据复核。

## 编译与视觉检查

- 引擎：XeLaTeX 3.141592653 / latexmk 4.83。
- 状态：成功，10页A4 PDF；引用和图表交叉引用均解析。
- 日志：无 undefined citation、undefined reference、overfull box 或 fatal error；仅保留macOS STSong CJK script的非致命字体提示。
- 视觉检查：10页全部渲染为PNG后检查；中文无缺字，六张图均未裁切，图注、页码和章节层级可读。
- 当前版本仍是本地链路demo；全量云端特征、学习曲线和cycle11 OOD完成后必须替换相关数字并重新编译。
