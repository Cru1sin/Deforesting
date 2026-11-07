"""COP support intervals, result summaries, and replay of saved candidate quantities."""

import json

import numpy as np
import pandas as pd


def five_minute_support_runs(times: pd.Series, selected: pd.Series) -> pd.Series:
    """Keep candidate runs spanning at least five continuous minutes."""
    result = pd.Series(False, index=selected.index)
    chosen = np.flatnonzero(selected.to_numpy(dtype=bool))
    if not chosen.size:
        return result
    parsed = pd.to_datetime(times, errors="coerce")
    breaks = np.flatnonzero(
        (np.diff(chosen) != 1)
        | (
            parsed.iloc[chosen[1:]].to_numpy() - parsed.iloc[chosen[:-1]].to_numpy()
            > np.timedelta64(90, "s")
        )
    )
    for left, right in zip(np.r_[0, breaks + 1], np.r_[breaks, len(chosen) - 1], strict=True):
        positions = chosen[left : right + 1]
        if parsed.iloc[positions[-1]] - parsed.iloc[positions[0]] >= pd.Timedelta(minutes=5):
            result.iloc[positions] = True
    return result


def summarize_decisions(decisions):
    rows = []
    for name, curve in decisions.groupby("cycle_name", sort=True):
        first = curve.iloc[0]
        optimum_time, rb_time = pd.to_datetime(
            [first.t_star, first.t_RB], format="mixed", errors="coerce"
        )
        selected = curve.loc[curve.is_selected]
        at_rb = curve.loc[curve.candidate_defrost_time.eq(rb_time) & curve.cycle_cop_eligible]
        best = selected.cycle_cop.iloc[0] if len(selected) else np.nan
        rb_cop = at_rb.cycle_cop.iloc[0] if len(at_rb) else np.nan
        rows.append(
            {
                "cycle_name": name,
                "experiment_id": first.experiment_id,
                "cycle_status": first.cycle_status,
                **{
                    key: first.get(key)
                    for key in (
                        "catalog_status",
                        "catalog_status_reason",
                        "pipeline_status",
                        "pipeline_status_reason",
                    )
                },
                "candidate_end_source": first.get("candidate_end_source", "actual_preparation"),
                "observation_end": first.get("observation_end", pd.NaT),
                "preparation_heat": first.preparation_heat,
                "cycle_start": first.cycle_start,
                "stable_heating_start": first.stable_heating_start,
                "heating_accounting_start_source": first.get(
                    "heating_accounting_start_source", "identified_recovery"
                ),
                "t_star": optimum_time,
                "t_RB": rb_time,
                "rb_status": first.rb_status,
                "COP_eff_selected": best,
                "COP_eff_RB": rb_cop,
                "COP_relative_change_pct": 100 * (best / rb_cop - 1) if rb_cop > 0 else np.nan,
                "time_difference_minutes": (optimum_time - rb_time).total_seconds() / 60,
                **{col: first[col] for col in curve if col.startswith("cycle_cop_basin_")},
            }
        )
    return pd.DataFrame(rows)


def replay_saved_run(
    source, run, dataset, n_jobs=6, publish=False, *, figures=False, dry_run=False
):
    """Reuse observed/predicted quantities and the shared selector and renderer."""
    from dataset_tools import DatasetLoader
    from defrost_decision.performance_objectives import calculate_cycle_cop, select_cycle_optimum
    from plots.dataset import publish_effective_decision_assets
    from plots.defrost_decision import render_current_decision_figures

    settings = json.loads((source / "run_settings.json").read_text())
    if dry_run:
        pd.read_csv(source / "candidate_decisions.csv", nrows=0)
        print(
            f"Reselect input: {source.resolve()}\nOutput: {run.resolve()}\nFigures: {figures}\nPublish Dataset: {publish}\nPreview only; no writes"
        )
        return 0
    if publish and not figures:
        raise ValueError("Dataset publication requires figures")
    if run.exists():
        raise FileExistsError(run)
    decisions = pd.read_csv(source / "candidate_decisions.csv", low_memory=False)
    decisions["candidate_defrost_time"] = pd.to_datetime(
        decisions.candidate_defrost_time, format="mixed"
    )
    selected_cycles = []
    for _, cycle in decisions.groupby("cycle_name"):
        ordered_candidates = cycle.sort_values("candidate_defrost_time").reset_index(drop=True)
        candidate_cop = calculate_cycle_cop(ordered_candidates, effective=True)
        selected_cycles.append(select_cycle_optimum(candidate_cop))
    decisions = pd.concat(selected_cycles, ignore_index=True)
    decisions.loc[decisions.groupby("cycle_name").is_selected.transform("any"), "cycle_status"] = (
        "identified_curve"
    )
    settings.update(
        reselected_from=str(source), extrapolation_policy="annotation_only", run_name=run.name
    )
    run.mkdir(parents=True)
    decisions.to_csv(run / "candidate_decisions.csv", index=False)
    summarize_decisions(decisions).to_csv(run / "cycle_comparison.csv", index=False)
    (run / "run_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    if figures:
        render_current_decision_figures(
            decisions, DatasetLoader(dataset), run / "figures", n_jobs=n_jobs
        )
    if publish:
        publish_effective_decision_assets(dataset, run, run / "figures")
    print(f"Reselected {decisions.cycle_name.nunique()} cycles: {run}", flush=True)
    return 0
