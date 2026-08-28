from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _module():
    path = Path("scripts/cost/plot.py")
    spec = importlib.util.spec_from_file_location("cost_function_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(algorithm: str) -> pd.DataFrame:
    stable = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        [
            {
                "cycle_name": cycle,
                "candidate_time": stable + pd.Timedelta(minutes=minute),
                "t_heating_stable": stable,
                "t_star": stable + pd.Timedelta(minutes=optimum),
                "water_reference_t_star": stable + pd.Timedelta(minutes=10),
                "t_RB": stable + pd.Timedelta(minutes=14),
                "rb_status": "triggered",
                "inverse_cop": cost,
                "relative_regret": 0.0 if minute == optimum else 0.1,
                "water_reference_inverse_cop": cost + 0.1,
                "water_reference_relative_regret": 0.0 if minute == 10 else 0.1,
                "optimization_eligible": True,
                "valid": True,
                "is_censored": False,
                "algorithm": algorithm,
            }
            for cycle, optimum, end in (("cycle_003", 10, 16), ("cycle_005", 12, 20))
            for minute, cost in ((optimum, 0.4), (end, 0.5))
        ]
    )


def test_cycle_points_accept_mixed_fractional_timestamp_formats() -> None:
    module = _module()
    table = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000058", "frost_cycle_000071"],
            "candidate_time": [
                "2026-01-01 00:10:00.000000000",
                "2026-01-01 00:20:00",
            ],
            "t_heating_stable": [
                "2026-01-01 00:00:00.000000000",
                "2026-01-01 00:00:00",
            ],
            "t_star": [
                "2026-01-01 00:08:00.000000000",
                "2026-01-01 00:18:00",
            ],
            "t_RB": [
                "2026-01-01 00:09:00.000000000",
                "2026-01-01 00:19:00",
            ],
            "rb_status": ["triggered", "triggered"],
        }
    )

    points = module._cycle_points(table)

    assert points["length_minutes"].tolist() == [10.0, 20.0]
    assert points["optimum_minutes"].tolist() == [8.0, 18.0]
    assert points["rb_minutes"].tolist() == [9.0, 19.0]


def test_cost_comparison_exports_three_grids_and_three_png_cycle_sets(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    v1_path, v2_path = tmp_path / "v1.csv", tmp_path / "v2.csv"
    for algorithm, path in (("v1", v1_path), ("v2", v2_path)):
        table = _table(algorithm)
        excluded = pd.concat(
            [
                table.iloc[:2].assign(cycle_name="cycle_007", valid=False),
                table.iloc[:2].assign(cycle_name="cycle_009", is_censored=True),
            ],
            ignore_index=True,
        )
        pd.concat([table, excluded], ignore_index=True).to_csv(path, index=False)
    seen: list[tuple[str, int, str, str, list[str]]] = []
    band_widths: list[list[float]] = []
    original_save = module._save_png

    def capture(fig: plt.Figure, path: Path) -> None:
        axis = fig.axes[0]
        seen.append(
            (
                path.name,
                len(fig.axes),
                axis.get_xlabel(),
                axis.get_ylabel(),
                [tick.get_text() for tick in axis.get_yticklabels()],
            )
        )
        band_widths.append([patch.get_width() for patch in axis.patches])
        original_save(fig, path)

    rendered: list[Path] = []
    decision_times: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    optimal_labels: dict[str, str] = {}

    def render(_frame, _record, _curve, _images, output, **_kwargs) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        rendered.append(output)
        if output.is_relative_to(tmp_path / "figures"):
            relative = output.relative_to(tmp_path / "figures").as_posix()
            decision_times[relative] = (
                pd.Timestamp(_images["optimal"]["target_time"]),
                pd.Timestamp(_images["rb"]["target_time"]),
            )
            optimal_labels[relative] = _kwargs["optimal_label"]

    monkeypatch.setattr(module, "_save_png", capture)
    monkeypatch.setattr(module, "render_decision_publication", render)
    loads: list[tuple[str, str]] = []

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            loads.append(("record", cycle_name))
            return {"cycle_name": cycle_name, "stable_heating_start": "2026-01-01"}

        @staticmethod
        def load_cycle(cycle_name: str) -> pd.DataFrame:
            loads.append(("frame", cycle_name))
            return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, freq="min")})

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            loads.append(("metadata", cycle_name))
            return pd.DataFrame(columns=["camera_role", "file_name", "image_time"])

        @staticmethod
        def load_cycle_images(cycle_name: str) -> pd.DataFrame:
            loads.append(("images", cycle_name))
            return pd.DataFrame(columns=["camera_role", "file_name", "path"])

    output = tmp_path / "figures"
    module.generate_cost_function_figures(
        {"v2": v2_path, "v1": v1_path}, Loader(), output
    )

    assert seen == [
        ("comparison_v1_RB.png", 1, "Minutes from stable heating start", "Cycle index", ["3", "5"]),
        ("comparison_v2_RB.png", 1, "Minutes from stable heating start", "Cycle index", ["3", "5"]),
        (
            "comparison_v1_v2_RB.png",
            1,
            "Minutes from stable heating start",
            "Cycle index",
            ["3", "5"],
        ),
    ]
    assert band_widths == [[16, 20], [16, 20], [16, 20]]
    assert {path.relative_to(output).as_posix() for path in rendered} == {
        f"{directory}/cycle_00{cycle}_publication.png"
        for directory in (
            "水侧制热量_cycle",
            "cost_function_v1_cycle",
            "cost_function_v2_cycle",
        )
        for cycle in (3, 5)
    }
    assert decision_times["cost_function_v2_cycle/cycle_005_publication.png"] == (
        pd.Timestamp("2026-01-01 00:12"),
        pd.Timestamp("2026-01-01 00:14"),
    )
    assert {
        optimal_labels[f"{directory}/cycle_003_publication.png"]
        for directory in (
            "水侧制热量_cycle",
            "cost_function_v1_cycle",
            "cost_function_v2_cycle",
        )
    } == {
        "Water-heat optimum",
        "Unit-heat V1 optimum",
        "Updated V2 optimum",
    }
    assert not list(output.rglob("*.svg"))
    assert not list(output.rglob("*.pdf"))
    assert sorted(loads) == sorted(
        (kind, cycle)
        for kind in ("record", "frame", "metadata", "images")
        for cycle in ("cycle_003", "cycle_005")
    )

    rendered.clear()
    v1_output = tmp_path / "v1_only"
    module.generate_cost_function_figures({"anything": v1_path}, Loader(), v1_output)
    assert {path.name for path in v1_output.glob("comparison*.png")} == {
        "comparison_v1_RB.png"
    }
    assert {path.parent.name for path in rendered} == {
        "水侧制热量_cycle",
        "cost_function_v1_cycle",
    }

    rendered.clear()
    v2_output = tmp_path / "v2_only"
    module.generate_cost_function_figures({"anything": v2_path}, Loader(), v2_output)
    assert {path.name for path in v2_output.glob("comparison*.png")} == {
        "comparison_v2_RB.png"
    }
    assert {path.parent.name for path in rendered} == {"cost_function_v2_cycle"}
