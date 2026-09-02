"""Select RGB assets at the empirical cost optimum."""

from __future__ import annotations

import pandas as pd

from dataloader.images import RGB_CAMERA_ORDER, RGB_PANEL_MAX_OFFSET


def optimal_view_report(manifest: pd.DataFrame) -> str:
    """Summarize six-view selection and export completeness."""
    selected = int(manifest["available"].sum())
    exported = int(
        manifest.get("exported", pd.Series(False, index=manifest.index)).sum()
    )
    return f"""# 最优除霜点六机位导出

- 完整观测循环：{manifest["cycle_name"].nunique()}。
- 目标机位图像：{selected}/{len(manifest)}（最近图像须位于最优点前后 2 min 内）。
- 已导出图像：{exported}/{selected}。
- 每行清单对应一个循环—机位组合；未找到合格近邻图像时仍保留该行，便于审计缺失。
"""


def build_optimal_view_manifest(
    curves: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Return one nearest image row per cycle and expected camera role."""
    rows: list[dict[str, object]] = []
    for cycle_name, curve in curves.groupby("cycle_name", sort=True):
        ordered = curve.assign(
            candidate_time=pd.to_datetime(curve["candidate_time"], errors="coerce"),
            inverse_cop=pd.to_numeric(curve["inverse_cop"], errors="coerce"),
        ).dropna(subset=["candidate_time", "inverse_cop"])
        optimum = ordered.sort_values("candidate_time", kind="stable").loc[
            lambda values: values["inverse_cop"].eq(values["inverse_cop"].min())
        ].iloc[0]
        target = pd.Timestamp(optimum["candidate_time"])
        scoped = metadata.loc[
            metadata["cycle_name"].astype(str).eq(str(cycle_name))
        ].copy()
        scoped["image_time"] = pd.to_datetime(scoped["image_time"], errors="coerce")
        for role in RGB_CAMERA_ORDER:
            camera = scoped.loc[scoped["camera_role"].astype(str).eq(role)].dropna(
                subset=["image_time"]
            )
            selected = None
            offset = pd.NaT
            if not camera.empty:
                offsets = (camera["image_time"] - target).abs()
                nearest = offsets.idxmin()
                if offsets.loc[nearest] <= RGB_PANEL_MAX_OFFSET:
                    selected = camera.loc[nearest]
                    offset = offsets.loc[nearest]
            file_name = str(selected["file_name"]) if selected is not None else ""
            rows.append(
                {
                    "cycle_name": str(cycle_name),
                    "camera_role": role,
                    "target_time": target,
                    "inverse_cop": float(optimum["inverse_cop"]),
                    "cycle_cop": float(
                        optimum.get("cycle_cop", 1 / optimum["inverse_cop"])
                    ),
                    "relative_regret": float(
                        optimum["inverse_cop"] / ordered["inverse_cop"].min() - 1
                    ),
                    "image_time": (
                        selected["image_time"] if selected is not None else pd.NaT
                    ),
                    "offset_seconds": (
                        float(offset.total_seconds())
                        if selected is not None
                        else float("nan")
                    ),
                    "file_name": file_name,
                    "available": selected is not None,
                    "relative_path": (
                        f"{cycle_name}/{role}/{file_name}"
                        if selected is not None
                        else ""
                    ),
                    "source_relative_path": (
                        f"images/{cycle_name}/{role}/{file_name}"
                        if selected is not None
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)
