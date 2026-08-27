#!/usr/bin/env python3
"""Audit per-cycle RGB transaction assets and write consolidated CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROLES = ("top", "top_close", "left", "left_close", "front", "extreme")
ALLOWED_MISSING = {
    "frost_cycle_000007": {"extreme"},
    "frost_cycle_000012": {"front", "extreme"},
}
CYCLE_COLUMNS = (
    "cycle_name",
    "panel_path",
    "panel_exists",
    "panel_readable",
    "manifest_exists",
    "manifest_valid",
    "views_expected",
    "views_available",
    "views_exported",
    "views_export_readable",
    "optimal_views_pass",
    "binary_exists",
    "binary_readable",
    "binary_rows",
    "three_exists",
    "three_readable",
    "three_rows",
    "passed",
)
EMPTY_VIEW_COLUMNS = (
    "cycle_name",
    "camera_role",
    "relative_path",
    "available",
    "exported",
    "manifest_exists",
    "manifest_readable",
    "manifest_row_present",
    "allowed_missing",
    "export_path",
    "export_exists",
    "export_readable",
    "view_pass",
)


def _read_catalog(dataset: Path) -> dict[str, dict[str, object]]:
    for name in ("catalog.yml", "catalog.yaml", "cycle_catalog.json"):
        path = dataset / name
        if path.is_file():
            payload = (
                yaml.safe_load(path.read_text(encoding="utf-8"))
                if path.suffix in {".yml", ".yaml"}
                else json.loads(path.read_text(encoding="utf-8"))
            )
            return {str(row["cycle_name"]): row for row in payload["cycles"]}
    raise FileNotFoundError(f"no cycle catalog found under {dataset}")


def _readable_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _parquet_status(path: Path) -> tuple[bool, bool, int]:
    if not path.is_file():
        return False, False, 0
    try:
        return True, True, len(pd.read_parquet(path))
    except (OSError, ValueError):
        return True, False, 0


def _as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _cycle_views(cycle: str, root: Path) -> tuple[pd.DataFrame, bool, bool]:
    path = root / "optimal" / cycle / "optimal_rgb_views_manifest.csv"
    if path.is_file():
        try:
            source = pd.read_csv(path)
            manifest_readable = True
        except (OSError, ValueError, pd.errors.ParserError):
            source = pd.DataFrame()
            manifest_readable = False
    else:
        source = pd.DataFrame()
        manifest_readable = False

    has_roles = manifest_readable and "camera_role" in source
    roles = source["camera_role"].astype(str) if has_roles else pd.Series(dtype=str)
    known = source.loc[roles.isin(ROLES)].copy() if has_roles else pd.DataFrame()
    known.index = roles.loc[known.index] if has_roles else known.index
    valid_roles = (
        has_roles
        and len(source) == len(ROLES)
        and roles.is_unique
        and set(roles) == set(ROLES)
        and "cycle_name" in source
        and source["cycle_name"].astype(str).eq(cycle).all()
    )
    rows = []
    for role in ROLES:
        present = role in known.index
        selected = known.loc[role] if present else pd.Series(dtype=object)
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[0]
        row = selected.to_dict() if present else {}
        row.update({"cycle_name": cycle, "camera_role": role})
        row["manifest_row_present"] = present
        rows.append(row)
    views = pd.DataFrame(rows)
    for column in ("available", "exported"):
        views[column] = _as_bool(views.get(column, pd.Series(False, index=views.index)))
    allowed = ALLOWED_MISSING.get(cycle, set())
    views["manifest_exists"] = path.is_file()
    views["manifest_readable"] = manifest_readable
    views["allowed_missing"] = [role in allowed for role in ROLES]
    relative_paths = views.get("relative_path", pd.Series("", index=views.index)).fillna("")
    export_paths = [root / "optimal" / str(value) for value in relative_paths]
    views["export_path"] = [
        str(path) if str(relative).strip() else ""
        for relative, path in zip(relative_paths, export_paths, strict=True)
    ]
    views["export_exists"] = [
        bool(str(relative).strip()) and path.is_file()
        for relative, path in zip(relative_paths, export_paths, strict=True)
    ]
    views["export_readable"] = [
        exists and _readable_image(path)
        for exists, path in zip(views["export_exists"], export_paths, strict=True)
    ]
    views["view_pass"] = (
        views["available"] & views["exported"] & views["export_readable"]
    ) | (
        ~views["available"] & views["allowed_missing"]
    )
    return views, path.is_file(), manifest_readable and valid_roles


def audit_rgb_cycle_assets(
    dataset: Path,
    labels_path: Path,
    transaction_root: Path,
    output: Path,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write cycle- and view-level audits for every labeled RGB cycle."""
    labels = pd.read_parquet(labels_path, columns=["cycle_name", "relative_regret"])
    cycles = sorted(
        labels.loc[labels["relative_regret"].notna(), "cycle_name"].astype(str).unique()
    )
    catalog = _read_catalog(dataset)
    audits: list[dict[str, object]] = []
    all_views: list[pd.DataFrame] = []
    for cycle in cycles:
        record = catalog.get(cycle, {})
        assets = record.get("assets", {})
        panel_value = assets.get("rgb_panel", "") if isinstance(assets, dict) else ""
        panel = dataset / str(panel_value) if panel_value else dataset / "__missing_panel__"
        views, manifest_exists, manifest_valid = _cycle_views(cycle, transaction_root)
        all_views.append(views)
        binary = _parquet_status(
            transaction_root / "features" / "binary" / "cycles" / f"{cycle}.parquet"
        )
        three = _parquet_status(
            transaction_root / "features" / "three" / "cycles" / f"{cycle}.parquet"
        )
        optimal_pass = manifest_valid and bool(views["view_pass"].all())
        row = {
            "cycle_name": cycle,
            "panel_path": str(panel_value),
            "panel_exists": panel.is_file(),
            "panel_readable": _readable_image(panel),
            "manifest_exists": manifest_exists,
            "manifest_valid": manifest_valid,
            "views_expected": len(ROLES),
            "views_available": int(views["available"].sum()),
            "views_exported": int(views["exported"].sum()),
            "views_export_readable": int(views["export_readable"].sum()),
            "optimal_views_pass": optimal_pass,
            "binary_exists": binary[0],
            "binary_readable": binary[1],
            "binary_rows": binary[2],
            "three_exists": three[0],
            "three_readable": three[1],
            "three_rows": three[2],
        }
        row["passed"] = bool(
            row["panel_readable"]
            and optimal_pass
            and binary[1]
            and binary[2] > 0
            and three[1]
            and three[2] > 0
        )
        audits.append(row)

    cycle_audit = pd.DataFrame(audits, columns=CYCLE_COLUMNS)
    view_manifest = (
        pd.concat(all_views, ignore_index=True)
        if all_views
        else pd.DataFrame(columns=EMPTY_VIEW_COLUMNS)
    )
    output.mkdir(parents=True, exist_ok=True)
    cycle_audit.to_csv(output / "cycle_asset_audit.csv", index=False)
    view_manifest.to_csv(output / "optimal_rgb_views_manifest.csv", index=False)
    if strict and cycle_audit.empty:
        raise RuntimeError("RGB cycle asset audit found no cycles with valid relative_regret")
    if strict and not cycle_audit["passed"].all():
        failed = cycle_audit.loc[~cycle_audit["passed"], "cycle_name"].tolist()
        raise RuntimeError(f"RGB cycle asset audit failed: {', '.join(failed)}")
    return cycle_audit, view_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("report/03_RGB标签与模型/成本标签/image_cost_labels.parquet"),
    )
    parser.add_argument("--transaction-root", type=Path, default=Path("outputs/RGB循环事务"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/RGB循环事务/asset_audit")
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    audit_rgb_cycle_assets(
        args.dataset, args.labels, args.transaction_root, args.output, strict=args.strict
    )


if __name__ == "__main__":
    main()
