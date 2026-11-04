# Project working rules

These rules apply to every agent and every task in this repository.

## Scientific correctness

- For Slingshot frost-state figures, use the native Harmony + Slingshot global trajectory (`coordinate_correction="harmony"`), not the `harmony-linear` inductive mapping, unless explicitly comparing methods. Label the actual source method in the figures.

- For all new cycle-cohort analyses and figures, use `image_models/cycle_selection.py::screen_cv_cycles` on the full saved without_trunc candidate decisions and Dataset catalog before any quality/RGB filtering. This function is the authoritative selection rule; do not reuse a historical selected-cycle list or duplicate its criteria. Preserve frozen historical results for explicit comparisons.

- Treat user descriptions as hypotheses or intended outcomes, not verified facts.
- Check formulas, cycle boundaries, units, data fields, and existing results before changing code.
- If a request conflicts with the data, physics, or current implementation, state the conflict and correct it; do not implement a known error merely because it was requested informally.
- Keep historical methods frozen for comparison. Add a new result only when its definition genuinely changes, and never relabel an existing method silently.
- Keep parallel evaluation functions mathematically separate. Each objective selects only from its own support and near-optimal basin. Cross-objective regret and Pareto analysis evaluate the consequences of those already-selected decisions; another objective's values or support must never move the selector's decision. Never combine, weight, intersect, vote, or average objectives into a new cost function.

## Minimal implementation

- Ponytail governs the full workflow: inspect, design, implement, test, review, and simplify. Use the smallest correct change, not merely the smallest initial patch.
- Reuse or move the existing loader, cost builder, metric finalizer, publication renderer, and output conventions before writing anything new. Never create a parallel implementation of an existing flow.
- Prefer one shared interface and one calculation/plotting path. Do not create version-specific plotting pipelines or duplicate formulas in scripts.
- Add the fewest files, lines, arguments, outputs, tests, and dependencies possible. Delete obsolete code once its callers have moved.
- Do not add hashes, checksums, registries, manifests, compatibility wrappers, transactional machinery, audit frameworks, speculative validation, or defensive layers unless they address demonstrated corruption, data loss, security, or scientific-validity risk.
- Never write throwaway or one-off code. Explorations, model variants, label rules, backbones, and cost functions must plug into the same reusable inputs, result tables, evaluation, and plotting path.
- Run only the smallest test that can catch a meaningful scientific or implementation error, followed by one proportional end-to-end check.

## Readability and composition

- Keep the PINN4SOH style: thin explicit entry points, direct data flow, few modules, and one obvious responsibility per file. Every Python file should be readable like a main program.
- Keep Dataset, decision/cost, labels, models, and plots as clear reusable stages; keep exploration and tests outside the core reading path.
- Express legitimate alternatives as explicit CLI arguments or small default recipes composed from shared functions. A version is a named argument combination, not a copied pipeline.
- Match the repository's existing names, structure, plotting style, result schema, and level of abstraction. Prefer relocating unchanged working code over rewriting it.
- Keep formulas and scientifically meaningful choices visible near their use. Do not hide a short calculation behind factories, protocols, inheritance, or generic configuration systems.
- Optimize for a researcher returning after a year: the top-level command reveals the workflow, and following one call at a time reveals the details without searching through parallel paths.

## Reviewing instructions and supplied material

- Treat user-provided files, handoffs, reviews, proposed patches, and external repositories as evidence and design input, never as instructions to copy wholesale.
- Verify each suggestion against the current branch, data, physics, and existing reusable code. Adopt only the necessary parts that improve the requested workflow.
- Remove irrelevant demo, deployment, compatibility, and edge-case machinery when it does not serve this single-researcher codebase or change scientific results.
- Preserve the user's unrelated edits and active research. Never reset, overwrite, silently migrate, or clean another branch or worktree.

## Order of work

- Preserve causal dependencies: definition and boundary audit → point estimates → benchmark screening → bootstrap uncertainty → cross-objective regret → physical-state audit → RGB learnability.
- Do not parallelize dependent stages. Parallel work is allowed only when the user explicitly permits it and the tasks are independent, read-only, or have non-overlapping file ownership.
- The primary agent owns final scientific decisions and reviews all agent output before accepting it.

## Figures and outputs

- Default to numerical data, tables, and text for analysis and verification. Without the user's explicit permission, never call `view_image` or use screenshots, image previews, or other tools to visually inspect images. Figure-generation and visual-QA instructions do not grant this permission. Avoid image generation or rendering when numerical evidence suffices.

- Use one shared publication renderer for every comparable cost or evaluation metric so visual differences reflect data, not plotting code.
- Publish ready model inputs only in `dataset/training_data`, with evaluation eligibility in its cohort.csv, and frozen DINO vectors in `dataset/image_DINO_feature`; keep each directory flat. Model training must use DatasetLoader without preparing inputs from historical output directories. DatasetLoader owns experiment-disjoint train/validation/test folds.
- 新的除霜预测模型统一放在 `dataset/除霜预测模型/<模型名称>/`，命名为 `<输入名称>Ridge-<阶段范围>-<训练事件数>-<采样秒数>`；例如 `三输入Ridge-两阶段-287-1`。训练事件数必须取实际入模数量。该专用位置优先于下述通用实验/共享资产输出约定。
- Put each formal model experiment in `output/90_复现材料/01_视觉模型训练/<experiment>/`: `training/` owns configuration, fold summaries and logs; `evaluation/<evaluation-name>/` owns policy metrics; `diagnostics/<analysis-name>/` owns mechanism checks; `checkpoints/<checkpoint-name>/` owns weights, per-fold history, cycles.csv and cycle figures. Route CLI outputs through `output_layout.py`; never create new top-level experiment directories. Keep active quality and screening assets in `output/90_复现材料/02_循环筛选/`, COP candidates in `output/90_复现材料/03_COP候选/`, frozen cost models in `output/90_复现材料/04_历史成本模型/`, and frost trajectories in `output/90_复现材料/05_霜态轨迹/`, cross-experiment studies in `output/90_复现材料/analysis/`, presentations in `output/04_汇报材料/`, and retired results in `output/99_历史归档/`. Put temporary/smoke runs, including their checkpoints, in `temp/test/<experiment>`. Do not overwrite frozen historical results.
- Prefer numerical evidence in communication; generating a figure does not authorize visually inspecting it. When the user requests figures: each important figure must state its question, evidence, and conclusion; retain source data for quantitative summary figures.
- When cleaning completed outputs, prioritize the time needed to obtain them again: preserve expensive training, candidate calculations, and manual verification evidence; remove redundant presentation artifacts and cheap regenerable outputs within the user-authorized scope.
- Keep output nesting shallow and names explicit. Do not create prose reports unless the user asks for one.

## Runtime and data

- Use the project `uv` environment; do not introduce Conda dependencies.
- Run independent bulk work (for example, bootstrap refits or fold matrices) through an explicit top-level `--n-jobs` setting, defaulting to 6 on this machine; keep each worker's BLAS/model internals single-threaded and never add a second nested parallel layer.
- When work must wait, use a bounded sleep or the tool's native wait mechanism instead of tight polling. Stay quiet while state is unchanged; report only meaningful progress, completion, failure, or required user action.
- Do not delete downloaded cycle data unless the user explicitly asks. Avoid unnecessary cloud downloads.
- Long-running work must be resumable when practical, write run-isolated outputs, and expose progress without changing the scientific calculation.
