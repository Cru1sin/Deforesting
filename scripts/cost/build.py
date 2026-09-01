#!/usr/bin/env python3
"""Export candidate-level defrost cost functions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from frost_analysis.cost.selected import (
    build_cost_function_table,
    write_cost_function_csv,
)
from frost_analysis.dataset.loader import DatasetLoader

BASE_CURVES = Path("output/test/成本函数/其他/经验经济窗口/源数据/candidate_cost_curves.parquet")
OPTIMAL_POINTS = Path("output/test/成本函数/其他/经验经济窗口/源数据/cycle_optimal_points.csv")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite([numerator, denominator]).all():
        return float("inf")
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def _mse_metrics(values: pd.DataFrame) -> dict[str, float]:
    required = ["residual_kwh", "baseline_residual_kwh"]
    numeric = values[required].apply(pd.to_numeric, errors="coerce")
    if (
        values.empty
        or values["experiment_id"].isna().any()
        or not np.isfinite(numeric.to_numpy()).all()
    ):
        return {
            "ratio": float("inf"),
            "macro_ratio": float("inf"),
            "win_fraction": 0.0,
            "maximum_experiment_ratio": float("inf"),
        }
    squared = values["residual_kwh"].pow(2)
    baseline = values["baseline_residual_kwh"].pow(2)
    by_experiment = values.assign(squared=squared, baseline=baseline).groupby(
        "experiment_id"
    )[["squared", "baseline"]].mean()
    experiment_ratios = by_experiment.apply(
        lambda row: _safe_ratio(row["squared"], row["baseline"]), axis=1
    )
    return {
        "ratio": _safe_ratio(float(squared.mean()), float(baseline.mean())),
        "macro_ratio": _safe_ratio(
            float(by_experiment["squared"].mean()),
            float(by_experiment["baseline"].mean()),
        ),
        "win_fraction": float(experiment_ratios.lt(1).mean()),
        "maximum_experiment_ratio": float(experiment_ratios.max()),
    }


def _write_v267_artifacts(table: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "cycle_audit": "cost_function_v2.6.7_cycle_audit.csv",
        "ticket_loeo": "cost_function_v2.6.7_ticket_loeo.csv",
        "v1r": "cost_function_v1-r.csv",
        "v1r_audit": "cost_function_v2.6.7_v1r_audit.csv",
        "bootstrap_audit": "cost_function_v2.6.7_bootstrap_audit.csv",
    }
    for attribute, name in names.items():
        table.attrs[attribute].to_csv(output / name, index=False)


def _require_valid(  # noqa: C901
    table: pd.DataFrame, algorithm: str, points: pd.DataFrame | None = None
) -> None:
    if algorithm == "v2.6.7":
        if points is None:
            raise RuntimeError("v2.6.7 validation requires points")
        artifacts = table.attrs
        audit = artifacts.get("cycle_audit")
        loeo = artifacts.get("ticket_loeo")
        v1r = artifacts.get("v1r")
        comparison = artifacts.get("v1r_audit")
        bootstrap = artifacts.get("bootstrap_audit")
        failures: list[str] = []
        expected = set(points.loc[points["valid"].fillna(False), "cycle_name"].astype(str))
        emitted = set(table["cycle_name"].astype(str))
        if len(expected) != 69 or emitted != expected:
            failures.append("exactly 69 valid point cycles must be emitted")
        if (
            not isinstance(audit, pd.DataFrame)
            or len(audit) != 101
            or audit["cycle_name"].duplicated().any()
        ):
            failures.append("cycle audit must contain 101 unique cycles")
        elif (
            audit["eligible_candidate_count"].isna().any()
            or not np.isfinite(audit["eligible_candidate_count"].to_numpy()).all()
        ):
            failures.append("cycle audit eligibility counts must be complete and finite")
        elif (
            audit.loc[audit["cycle_name"].astype(str).isin(expected), "eligible_candidate_count"]
            .ge(2)
            .sum()
            < 60
        ):
            failures.append("fewer than 60/69 cycles have two joint-eligible candidates")
        if table["recommended_time"].notna().any() or table[
            "hard_label_eligible"
        ].fillna(False).any():
            failures.append("identification-only output contains a recommendation or hard label")
        if not table["decision_status"].eq("abstain_v267_identification_only").all():
            failures.append("decision status drift")
        if (
            table["endpoint_extrapolated"].fillna(False).any()
            or table["prediction_clipped"].fillna(False).any()
            or table["interpolated"].fillna(False).any()
        ):
            failures.append("interpolation, clipping, or endpoint extrapolation detected")
        if not isinstance(loeo, pd.DataFrame):
            failures.append("missing ticket LOEO audit")
        else:
            if loeo["supported"].isna().any():
                failures.append("ticket LOEO support flags must be complete")
            for target, values in loeo.loc[loeo["supported"].fillna(False)].groupby("target"):
                metrics = _mse_metrics(values)
                if len(values) < 40 or values["experiment_id"].nunique() < 12:
                    failures.append(f"{target} supported LOEO cohort too small")
                if metrics["ratio"] > 0.90 or metrics["macro_ratio"] > 0.90:
                    failures.append(f"{target} LOEO MSE ratio gate failed")
                if (
                    metrics["win_fraction"] < 0.70
                    or metrics["maximum_experiment_ratio"] > 4
                ):
                    failures.append(f"{target} experiment-level LOEO gate failed")
            if set(loeo["target"].unique()) != {"E_T", "Q_T"}:
                failures.append("both independent ticket targets are required")
            leaked = loeo.apply(
                lambda row: str(row["heldout_experiment_id"])
                in str(row["training_experiment_ids"]).split(","),
                axis=1,
            )
            if leaked.any():
                failures.append("held-out terminal outcomes leaked into a training fold")
        if not isinstance(v1r, pd.DataFrame) or not (
            v1r["oracle_only"].fillna(False).all()
            and ~v1r["available_at_candidate_time"].fillna(True).any()
        ):
            failures.append("V1-r oracle availability contract failed")
        if not isinstance(comparison, pd.DataFrame):
            failures.append("missing V1-r comparison audit")
        else:
            comparable = comparison.loc[comparison["comparable"].fillna(False)]
            v1r_metrics = [
                "oracle_regret_at_main_t_star",
                "within_cycle_spearman",
            ]
            if (
                comparison["comparable"].isna().any()
                or comparable[v1r_metrics].isna().any().any()
                or not np.isfinite(comparable[v1r_metrics].to_numpy()).all()
                or comparable["main_t_star_in_oracle_5pct_basin"].isna().any()
            ):
                failures.append("V1-r comparison metrics must be complete and finite")
            if len(comparable) < 40:
                failures.append("fewer than 40 V1-r comparable cycles")
            if comparable["main_t_star_in_oracle_5pct_basin"].mean() < 0.90:
                failures.append("V1-r connected 5% basin hit rate below 90%")
            if comparable["oracle_regret_at_main_t_star"].quantile(0.90) > 0.05:
                failures.append("V1-r p90 oracle regret exceeds 5%")
            if comparable["within_cycle_spearman"].median() < 0.80:
                failures.append("V1-r median within-cycle Spearman below 0.8")
            if not comparison["candidate_intersection_consistent"].fillna(False).all():
                failures.append("main/V1-r candidate intersection drift")
        if not isinstance(bootstrap, pd.DataFrame) or bootstrap.empty:
            failures.append("missing 200-repeat experiment bootstrap")
        else:
            bootstrap_metrics = [
                "repeat_count",
                "two_candidate_repeat_fraction",
                "argmin_in_original_5pct_basin_fraction",
            ]
            if (
                bootstrap[bootstrap_metrics].isna().any().any()
                or not np.isfinite(bootstrap[bootstrap_metrics].to_numpy()).all()
            ):
                failures.append("bootstrap metrics must be complete and finite")
            stable = bootstrap["two_candidate_repeat_fraction"].ge(0.80) & bootstrap[
                "argmin_in_original_5pct_basin_fraction"
            ].ge(0.75)
            if not bootstrap["repeat_count"].eq(200).all():
                failures.append("bootstrap repeat count is not 200")
            if stable.mean() < 0.75:
                failures.append("fewer than 75% identified cycles pass bootstrap stability")
            if bootstrap["argmin_in_original_5pct_basin_fraction"].median() < 0.80:
                failures.append("bootstrap median basin hit rate below 80%")
        hashes = {
            "cost_function_v1.csv": (
                "2f2c0812cb95a4606f9ea31c3a7ce06ddc11261d33385fa52d5bc3fdb85791c0"
            ),
            "cost_function_v2.5.csv": (
                "c1a0ce5da08723421589664b203cae0f39be0741e70558ce1775044f86b6d183"
            ),
            "cost_function_v2.6.5.csv": (
                "e0124f1d90bacbefa7f52cd529fa48172813d0e81dbb4ed7166abf3247acf590"
            ),
            "cost_function_v2.6.6.csv": (
                "dfd950eb877228470b02d86ce4c06ffe5d8a2d0fe61529f8039e4ff30c2c1ae5"
            ),
        }
        for name, expected_hash in hashes.items():
            path = Path("output/成本函数") / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                failures.append(f"historical hash drift: {name}")
        if failures:
            raise RuntimeError("v2.6.7 validation failed: " + "; ".join(failures))
        return
    if algorithm == "v2.6.6":
        if points is None:
            raise RuntimeError("v2.6.6 validation requires points")
        expected = set(points.loc[points["valid"].fillna(False), "cycle_name"].astype(str))
        emitted = set(table["cycle_name"].astype(str))
        if len(expected) != 69 or len(emitted) != 69:
            raise RuntimeError("v2.6.6 requires exactly 69 valid and emitted curve cycles")
        if emitted != expected:
            raise RuntimeError("v2.6.6 emitted curve cycle set does not match valid points")
        audit = table.attrs.get("cycle_audit")
        if not isinstance(audit, pd.DataFrame) or len(audit) != 101:
            raise RuntimeError("v2.6.6 cycle audit must contain exactly 101 rows")
        if audit["cycle_name"].duplicated().any():
            raise RuntimeError("v2.6.6 cycle audit cycle_name must be unique")
        audit_names = set(audit["cycle_name"].astype(str))
        if not expected <= audit_names:
            raise RuntimeError("v2.6.6 cycle audit must cover every emitted curve cycle")
        main_audit = audit.loc[audit["cycle_name"].astype(str).isin(expected)]
        if int(main_audit["eligible_candidate_count"].ge(2).sum()) < 60:
            raise RuntimeError("v2.6.6 requires at least 60 curves with two eligible candidates")
        return
    valid_by_cycle = table["valid"].fillna(False).groupby(table["cycle_name"]).any()
    failed = valid_by_cycle.index[~valid_by_cycle].astype(str).tolist()
    if failed:
        raise RuntimeError(f"{algorithm} produced failed cycles: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=(
            "v1",
            "v2",
            "v2.1",
            "v2.2",
            "v2.3",
            "v2.4",
            "v2.5",
            "v2.6",
            "v2.6.1",
            "v2.6.2",
            "v2.6.3",
            "v2.6.4",
            "v2.6.5",
            "v2.6.6",
            "v2.6.7",
            "v3",
        ),
        default=("v1", "v2"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/成本函数"))
    args = parser.parse_args()

    base = pd.read_parquet(BASE_CURVES)
    points = pd.read_csv(OPTIMAL_POINTS)
    loader = DatasetLoader(Path("dataset"))
    for algorithm in args.algorithm:
        table = build_cost_function_table(base, points, loader, algorithm)
        write_cost_function_csv(table, args.output, algorithm)
        if algorithm == "v2.6.7":
            _write_v267_artifacts(table, args.output)
        _require_valid(table, algorithm, points)
        if algorithm == "v2.6.6":
            table.attrs["cycle_audit"].to_csv(
                args.output / "cost_function_v2.6.6_cycle_audit.csv", index=False
            )


if __name__ == "__main__":
    main()
