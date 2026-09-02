from __future__ import annotations

import importlib.util
import json
import warnings
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb


def _module():
    path = Path("plots/cost.py")
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
                "experiment_id": ("exp_20260101" if cycle == "cycle_003" else "exp_20260102"),
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


def test_standardized_public_entrypoint_preserves_existing_output_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    tables = {"v1": _table("v1"), "v2.5": _table("v2.5")}
    for algorithm, heat_basis in (("v1", "unit"), ("v2.5", "water")):
        tables[algorithm].attrs["heat_basis"] = heat_basis
    saved: list[Path] = []
    suites: list[tuple[tuple[str, ...], Path]] = []
    curves: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(module, "_load_result_tables", lambda *_args: tables)
    monkeypatch.setattr(module, "_comparison_figure", lambda *_args: plt.figure())
    monkeypatch.setattr(
        module, "_save_png", lambda figure, path: (saved.append(path), plt.close(figure))
    )
    monkeypatch.setattr(
        module,
        "_render_cycle_sets",
        lambda tables, _loader, _records, output: suites.append((tuple(tables), output)),
    )
    monkeypatch.setattr(
        module,
        "_render_cost_curve_comparisons",
        lambda tables, _loader, output: curves.append((tuple(tables), output)),
    )

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"cycle_name": cycle_name}

    output = tmp_path / "figures"
    module.generate_cost_function_figures([tmp_path / "v1", tmp_path / "v2.5"], Loader(), output)

    assert saved == [
        output / "comparison_v1_RB.png",
        output / "comparison_v2.5_RB.png",
        output / "comparison_v1_v2.5_RB.png",
    ]
    assert suites == [(("v1", "v2.5"), output)]
    assert curves == [
        (("v1",), output / "cost_curves" / "unit"),
        (("v2.5",), output / "cost_curves" / "water"),
    ]


def test_standardized_public_entrypoint_keeps_renewal_variant_svg_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    table = _table("renewal_water__trial")
    table.attrs["heat_basis"] = "water"
    saved: list[Path] = []
    monkeypatch.setattr(
        module, "_load_result_tables", lambda *_args: {"renewal_water__trial": table}
    )
    monkeypatch.setattr(module, "_comparison_figure", lambda *_args: plt.figure())
    monkeypatch.setattr(
        module, "_save_svg_png", lambda figure, path: (saved.append(path), plt.close(figure))
    )
    monkeypatch.setattr(module, "_render_cycle_sets", lambda *_args: None)
    monkeypatch.setattr(module, "_render_cost_curve_comparisons", lambda *_args: None)

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"cycle_name": cycle_name}

    output = tmp_path / "figures"
    module.generate_cost_function_figures([tmp_path / "renewal"], Loader(), output)

    assert saved == [output / "comparison_renewal_water__trial_RB.png"]


def test_standardized_public_entrypoint_preflights_existing_renewal_svg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    tables = {"v1": _table("v1"), "renewal_water__trial": _table("renewal_water__trial")}
    for table in tables.values():
        table.attrs["heat_basis"] = "water"
    output = tmp_path / "figures"
    existing = output / "comparison_renewal_water__trial_RB.svg"
    existing.parent.mkdir()
    existing.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(module, "_load_result_tables", lambda *_args: tables)
    monkeypatch.setattr(
        module,
        "_comparison_figure",
        lambda *_args: (_ for _ in ()).throw(AssertionError("rendered before preflight")),
    )

    class Loader:
        @staticmethod
        def get_cycle_record(cycle_name: str) -> dict[str, object]:
            return {"cycle_name": cycle_name}

    with pytest.raises(FileExistsError, match=str(existing)):
        module.generate_cost_function_figures(
            [tmp_path / "v1", tmp_path / "renewal"], Loader(), output
        )

    assert existing.read_text(encoding="utf-8") == "existing"
    assert not (output / "comparison_v1_RB.png").exists()


def test_cycle_sets_render_variants_and_v268_as_diagnostic_minima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    tables = {
        "v1__alpha": _table("v1__alpha").drop(
            columns=[
                "water_reference_t_star",
                "water_reference_inverse_cop",
                "water_reference_relative_regret",
            ]
        ),
        "v2.6.8": _table("v2.6.8"),
    }
    rendered: list[tuple[str, str]] = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, _curve, _images, output, **kwargs: rendered.append(
            (output.parent.name, kwargs["optimal_label"])
        ),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    records = {cycle: {"cycle_name": cycle} for cycle in tables["v1__alpha"]["cycle_name"].unique()}
    module._render_cycle_sets(tables, Loader(), records, tmp_path)

    assert set(rendered) == {
        ("cost_function_v1__alpha_cycle", "V1 (alpha) optimum"),
        ("cost_function_v2.6.8_cycle", "V2.6.8 diagnostic minimum"),
    }


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
            "cycle_start": [
                "2025-12-31 23:58:00.000000000",
                "2025-12-31 23:58:00",
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

    assert points["length_minutes"].tolist() == [12.0, 22.0]
    assert points["optimum_minutes"].tolist() == [10.0, 20.0]
    assert points["rb_minutes"].tolist() == [11.0, 21.0]


def test_cycle_points_preserves_unknown_support() -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        t_star_model_supported=np.nan,
    )

    points = module._cycle_points(table)

    assert points["optimum_supported"].isna().all()


def test_water_reference_curve_uses_its_own_selected_time() -> None:
    module = _module()
    table = _table("v1")

    curve = module._publication_curve(table, "water_reference")

    pd.testing.assert_series_equal(
        curve["t_star"], table["water_reference_t_star"], check_names=False
    )


def test_render_all_cost_curves_writes_curve_and_paginated_rgb_plates(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    algorithms = ("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6")
    tables = {
        algorithm: _table(algorithm).assign(
            cycle_start=lambda values: values["cycle_name"].map(
                {
                    "cycle_003": pd.Timestamp("2025-12-31 23:55:00"),
                    "cycle_005": pd.Timestamp("2025-12-31 23:50:00"),
                }
            )
        )
        for algorithm in algorithms
    }
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))

    class Loader:
        @staticmethod
        def load_image_metadata(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "camera_role": ["front", "front"],
                    "file_name": ["front.jpg", "front.jpg"],
                    "image_time": [
                        pd.Timestamp("2026-01-01 00:10:00"),
                        pd.Timestamp("2026-01-01 00:12:00"),
                    ],
                }
            )

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "camera_role": ["front"],
                    "file_name": ["front.jpg"],
                    "path": [image_path],
                }
            )

    rendered = []
    original_save = module._save_png

    def capture(figure: plt.Figure, path: Path) -> None:
        rendered.append(
            (
                path.relative_to(tmp_path).as_posix(),
                len(figure.axes),
                figure.axes[0].get_ylabel(),
                figure.axes[-1].get_xlabel(),
                figure.axes[0].get_title(loc="left"),
            )
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            original_save(figure, path)

    monkeypatch.setattr(module, "_save_png", capture)
    module._render_cost_curve_comparisons(tables, Loader(), tmp_path)

    assert rendered == [
        (
            "cycle_003_cost_curves.png",
            2,
            "Cost J = 1/COP",
            "Minutes from cycle start",
            "Cycle 3: cost-function variants",
        ),
        (
            "optimal_rgb/cycle_003_optimal_rgb_01.png",
            4,
            "",
            "",
            "V1 optimum\n15.0 min · image offset 0 s",
        ),
        (
            "optimal_rgb/cycle_003_optimal_rgb_02.png",
            4,
            "",
            "",
            "V2.3 optimum\n15.0 min · image offset 0 s",
        ),
        (
            "cycle_005_cost_curves.png",
            2,
            "Cost J = 1/COP",
            "Minutes from cycle start",
            "Cycle 5: cost-function variants",
        ),
        (
            "optimal_rgb/cycle_005_optimal_rgb_01.png",
            4,
            "",
            "",
            "V1 optimum\n22.0 min · image offset 0 s",
        ),
        (
            "optimal_rgb/cycle_005_optimal_rgb_02.png",
            4,
            "",
            "",
            "V2.3 optimum\n22.0 min · image offset 0 s",
        ),
    ]


def test_optimal_rgb_figures_paginate_four_methods_per_page() -> None:
    module = _module()
    algorithms = tuple(name for name in module.STYLES if name != "RB")
    images = {
        algorithm: {
            "available": False,
            "status": "physical_image_missing",
            "target_time": pd.Timestamp("2026-01-01 00:10:00"),
        }
        for algorithm in algorithms
    }

    pages = list(
        module._optimal_rgb_figures(
            images,
            algorithms,
            "cycle_003",
            pd.Timestamp("2026-01-01"),
        )
    )

    assert len(pages) == 3
    assert [sum(axis.get_visible() for axis in figure.axes) for figure in pages] == [
        4,
        4,
        3,
    ]
    for figure in pages:
        plt.close(figure)


def test_five_method_v267_family_uses_one_front_image_plate() -> None:
    module = _module()
    algorithms = ("v1", "v2.5", "v2.6.5", "v2.6.6", "v2.6.7")
    images = {algorithm: {"available": False, "status": "missing"} for algorithm in algorithms}

    pages = list(
        module._optimal_rgb_figures(
            images, algorithms, "cycle_003", pd.Timestamp("2026-01-01")
        )
    )

    assert len(pages) == 1
    assert sum(axis.get_visible() for axis in pages[0].axes) == 5
    plt.close(pages[0])

    v267_page = next(
        module._optimal_rgb_figures(
            {
                "v2.6.7": {
                    "available": False,
                    "status": "no_valid_optimal",
                    "target_status": "model_support_limited",
                }
            },
            ("v2.6.7",),
            "cycle_003",
            pd.Timestamp("2026-01-01"),
        )
    )
    assert "model support limited · no eligible diagnostic minimum" in v267_page.axes[
        0
    ].get_title(loc="left")
    plt.close(v267_page)


def test_v266_rgb_support_and_page_title_follow_cycle_status(monkeypatch) -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_status="measurement_limited",
        t_star_model_supported=True,
    )
    monkeypatch.setattr(
        module,
        "match_decision_rgb_images",
        lambda *_args: pd.DataFrame(
            {"target_type": ["optimal"], "available": [True], "offset_seconds": [0]}
        ),
    )

    matched = module._match_optimal_front_images(
        {"v2.6.6": table}, "cycle_003", pd.DataFrame(), pd.DataFrame()
    )
    page = next(
        module._optimal_rgb_figures(
            matched, ("v2.6.6",), "cycle_003", pd.Timestamp("2026-01-01")
        )
    )

    assert matched["v2.6.6"]["target_supported"] is False
    assert "measurement limited" in page.axes[0].get_title(loc="left")
    assert "selected/diagnostic cost-function times" in page._suptitle.get_text()
    plt.close(page)

    unknown = module._match_optimal_front_images(
        {"v2.6.6": table.assign(t_star_model_supported=np.nan)},
        "cycle_003",
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert unknown["v2.6.6"]["target_supported"] is None


def test_v266_overview_encodes_each_nonidentified_status() -> None:
    module = _module()
    rows = []
    for index, status in enumerate(
        ("measurement_limited", "component_extrapolated", "right_censored"), start=1
    ):
        rows.append(
            _table("v2.6.6")
            .iloc[:1]
            .assign(
                cycle_name=f"cycle_{index:03d}",
                cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
                cycle_status=status,
                t_star_model_supported=True,
            )
        )
    figure = module._comparison_figure(
        {"v2.6.6": pd.concat(rows, ignore_index=True)}, ("v2.6.6",)
    )

    labels = {collection.get_label() for collection in figure.axes[0].collections}
    assert {
        "V2.6.6 diagnostic minimum (measurement-limited)",
        "V2.6.6 diagnostic minimum (component-extrapolated)",
        "V2.6.6 diagnostic minimum (right-censored)",
    } <= labels
    plt.close(figure)


def test_v266_overview_rejects_unrecognized_status() -> None:
    module = _module()
    table = _table("v2.6.6").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="new_unmapped_status",
    )

    with pytest.raises(ValueError, match="unrecognized V2.6.6 cycle_status"):
        module._comparison_figure({"v2.6.6": table}, ("v2.6.6",))


def test_cost_curve_family_rejects_mismatched_cycle_sets(tmp_path: Path) -> None:
    module = _module()
    tables = {
        "v1": _table("v1").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
        "v2.6.6": _table("v2.6.6")
        .loc[lambda values: values["cycle_name"].eq("cycle_003")]
        .assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
    }

    with pytest.raises(ValueError, match="identical cycle sets"):
        module._render_cost_curve_comparisons(tables, object(), tmp_path)


def test_v266_publication_label_includes_cycle_status(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    table = _table("v2.6.6").assign(cycle_status="right_censored")
    labels = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda *_args, **kwargs: labels.append(kwargs["optimal_label"]),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    records = {cycle: {} for cycle in table["cycle_name"].unique()}
    module._render_cycle_sets({"v2.6.6": table}, Loader(), records, tmp_path)

    assert labels == [
        "V2.6.6 diagnostic identification minimum (right censored)",
        "V2.6.6 diagnostic identification minimum (right censored)",
    ]


def test_v267_nonidentified_publication_preserves_support_and_labels_cycle_limit(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_status="measurement_limited",
        measurement_eligible=True,
        model_supported=True,
        t_star_model_supported=True,
        heating_electricity_kwh=0.2,
        unit_heating_kwh=0.4,
        E_T_hat_kwh=0.1,
        Q_T_hat_kwh=0.2,
    )
    seen = []
    monkeypatch.setattr(module, "_decision_images", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "render_decision_publication",
        lambda _frame, _record, curve, _images, _output, **kwargs: seen.append(
            (
                curve["model_supported"].tolist(),
                kwargs["display_metric"],
                kwargs["minimum_label"],
                kwargs["minimum_support_label"],
            )
        ),
    )

    class Loader:
        load_cycle = load_image_metadata = load_cycle_images = staticmethod(
            lambda _cycle_name: pd.DataFrame()
        )

    records = {cycle: {} for cycle in table["cycle_name"].unique()}
    module._render_cycle_sets({"v2.6.7": table}, Loader(), records, tmp_path)

    assert seen == [
        (
            [True, True],
            module.V267_DISPLAY_METRIC,
            "Diagnostic/raw minimum",
            "measurement limited",
        )
    ] * 2


def test_all_cost_curves_use_distinct_colors_and_line_styles() -> None:
    module = _module()
    algorithms = ("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6")
    tables = {
        algorithm: _table(algorithm).assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
        for algorithm in algorithms
    }

    figure = module._cost_curve_figure(tables, "cycle_003")
    labels = set(map(str.upper, algorithms))
    lines = [line for line in figure.axes[0].lines if line.get_label() in labels]
    colors = [to_rgb(line.get_color()) for line in lines]
    distances = [
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) ** 0.5
        for left, right in combinations(colors, 2)
    ]

    assert min(distances) > 0.2
    assert len({line.get_linestyle() for line in lines}) >= 4
    plt.close(figure)


def test_cost_curve_selected_marker_uses_true_regret() -> None:
    module = _module()
    table = _table("v2.6.5").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00")
    )
    selected = table["cycle_name"].eq("cycle_003") & table["candidate_time"].eq(
        pd.Timestamp("2026-01-01 00:16:00")
    )
    table.loc[table["cycle_name"].eq("cycle_003"), "t_star"] = pd.Timestamp(
        "2026-01-01 00:16:00"
    )
    table.loc[selected, "relative_regret"] = 0.009208

    figure = module._cost_curve_figure({"v2.6.5": table}, "cycle_003")

    marker_y = float(figure.axes[1].collections[0].get_offsets()[0, 1])
    assert marker_y == pytest.approx(0.9208)
    plt.close(figure)


def test_comparison_marks_extrapolated_renewal_optima() -> None:
    module = _module()
    table = _table("renewal_water").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        t_star_model_supported=lambda values: values["cycle_name"].ne("cycle_003"),
    )

    figure = module._comparison_figure({"renewal_water": table}, ("renewal_water",))

    labels = {collection.get_label() for collection in figure.axes[0].collections}
    assert "Renewal-water optimum (extrapolated)" in labels
    plt.close(figure)


def test_cost_curve_rgb_fetches_only_missing_optimal_front_members(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    tables = {
        algorithm: _table(algorithm).assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00"))
        for algorithm in ("v1", "v2")
    }
    image_path = tmp_path / "front.jpg"
    plt.imsave(image_path, np.ones((4, 4, 3)))

    class Loader:
        dataset_root = tmp_path / "dataset"

        @staticmethod
        def load_image_metadata(cycle_name: str) -> pd.DataFrame:
            optimum = 10 if cycle_name == "cycle_003" else 12
            return pd.DataFrame(
                {
                    "camera_role": ["front"],
                    "file_name": [f"front_{optimum}.jpg"],
                    "image_time": [pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=optimum)],
                }
            )

        @staticmethod
        def load_cycle_images(_cycle_name: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["camera_role", "file_name", "path"])

    requested = []

    @contextmanager
    def materialize(_dataset, cycle_name, names, **options):
        requested.append((cycle_name, tuple(names), options))
        yield tmp_path / "range"

    def scan(_dataset, cycle_name, metadata, *, cycle_dir):
        return pd.DataFrame(
            {
                "camera_role": ["front"],
                "file_name": [metadata["file_name"].iloc[0]],
                "path": [image_path],
            }
        )

    monkeypatch.setattr(module, "materialize_cycle_image_members", materialize)
    monkeypatch.setattr(module, "scan_cycle_images", scan)
    monkeypatch.setattr(module, "_save_png", lambda figure, _path: plt.close(figure))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        module._render_cost_curve_comparisons(
            tables, Loader(), tmp_path, fetch_cloud=True, minimum_free_gib=5
        )

    assert requested == [
        ("cycle_003", ("front_10.jpg",), {"fetch_cloud": True, "minimum_free_gib": 5}),
        ("cycle_005", ("front_12.jpg",), {"fetch_cloud": True, "minimum_free_gib": 5}),
    ]


def test_v267_overview_rejects_unrecognized_status() -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="new_unmapped_status",
    )

    with pytest.raises(ValueError, match="unrecognized V2.6.7 cycle_status"):
        module._comparison_figure({"v2.6.7": table}, ("v2.6.7",))


def test_v267_cost_curve_draws_unsupported_extension_without_selecting_it() -> None:
    module = _module()
    table = _table("v2.6.7").loc[lambda values: values["cycle_name"].eq("cycle_003")].assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="model_support_limited",
        measurement_eligible=True,
        model_supported=False,
        optimization_eligible=False,
        valid=False,
        t_star=pd.NaT,
        heating_electricity_kwh=[0.2, 0.4],
        unit_heating_kwh=[0.4, 0.8],
        E_T_hat_kwh=[0.1, 0.1],
        Q_T_hat_kwh=[0.2, 0.2],
    )

    figure = module._cost_curve_figure({"v2.6.7": table}, "cycle_003")

    labels = [line.get_label() for line in figure.axes[0].lines]
    assert "V2.6.7 unsupported model extension, display only" in labels
    assert not figure.axes[0].collections
    assert "model support limited" in figure.axes[0].get_title(loc="left")
    plt.close(figure)


def test_v267_display_extension_preserves_unknown_model_support() -> None:
    module = _module()
    curve = pd.DataFrame(
        {
            "heating_electricity_kwh": [0.2, 0.2, 0.2],
            "unit_heating_kwh": [0.4, 0.4, 0.4],
            "E_T_hat_kwh": [0.1, 0.1, 0.1],
            "Q_T_hat_kwh": [0.2, 0.2, 0.2],
            "measurement_eligible": [True, True, True],
            "model_supported": pd.Series([True, False, np.nan], dtype=object),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        display = module._with_v267_display_extension(curve)[module.V267_DISPLAY_METRIC]

    assert display.iloc[:2].isna().tolist() == [True, False]
    assert display.iloc[1] == pytest.approx(0.5)
    assert pd.isna(display.iloc[2])


def test_v267_overview_marks_missing_minimum_off_the_data_axis() -> None:
    module = _module()
    table = _table("v2.6.7").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        cycle_status="model_support_limited",
        t_star_model_supported=False,
    )
    table.loc[table["cycle_name"].eq("cycle_003"), "t_star"] = pd.NaT

    figure = module._comparison_figure({"v2.6.7": table}, ("v2.6.7",))

    axis = figure.axes[0]
    marker = next(
        item
        for item in axis.collections
        if item.get_label() == "V2.6.7 diagnostic minimum (no diagnostic minimum)"
    )
    assert marker.get_offsets().tolist() == [[0.0, -0.04]]
    assert marker.get_offset_transform() == axis.get_xaxis_transform()
    plt.close(figure)


def test_comparison_rejects_mismatched_cycle_sets() -> None:
    module = _module()
    tables = {
        "v1": _table("v1").assign(cycle_start=pd.Timestamp("2025-12-31 23:55:00")),
        "v2.6.7": _table("v2.6.7")
        .loc[lambda values: values["cycle_name"].eq("cycle_003")]
        .assign(
            cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
            cycle_status="identified_curve",
        ),
    }

    with pytest.raises(ValueError, match="identical cycle sets"):
        module._comparison_figure(tables, ("v1", "v2.6.7"))


def test_v267_evidence_writes_separate_bootstrap_and_loeo_pngs(tmp_path: Path) -> None:
    module = _module()
    bootstrap = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000003", "frost_cycle_000006"],
            "experiment_id": ["exp_20260101", "exp_20260102"],
            "two_candidate_repeat_fraction": [0.9, 0.7],
            "argmin_in_original_5pct_basin_fraction": [0.8, 0.6],
        }
    )
    loeo = pd.DataFrame(
        {
            "cycle_name": ["frost_cycle_000003", "frost_cycle_000006"] * 2,
            "experiment_id": ["exp_20260101", "exp_20260102"] * 2,
            "target": ["E_T", "E_T", "Q_T", "Q_T"],
            "observed_kwh": [0.2, 0.4, 0.5, 0.8],
            "loeo_prediction_kwh": [0.21, 0.38, 0.52, 0.76],
            "training_mean_kwh": [0.3, 0.3, 0.65, 0.65],
            "supported": [True, False, True, False],
            "training_event_count": [10, 10, 8, 8],
            "training_experiment_count": [1, 1, 1, 1],
        }
    )

    module.generate_v267_evidence(bootstrap, loeo, tmp_path)

    assert {path.name for path in tmp_path.glob("*.png")} == {
        "bootstrap_stability_by_cycle.png",
        "ticket_E_T_loeo.png",
        "ticket_Q_T_loeo.png",
    }


def test_bootstrap_title_and_experiment_bars_follow_global_gate() -> None:
    module = _module()
    bootstrap = pd.DataFrame(
        {
            "cycle_name": [f"frost_cycle_{value:06d}" for value in range(1, 5)],
            "experiment_id": ["exp_20260101", "exp_20260101", "exp_20260102", "exp_20260102"],
            "two_candidate_repeat_fraction": [0.9, 0.9, 0.9, 0.9],
            "argmin_in_original_5pct_basin_fraction": [0.9, 0.7, 0.9, 0.9],
        }
    )

    figure = module._plot_bootstrap_stability(bootstrap)

    assert "passes the hard-label gate" in figure._suptitle.get_text()
    assert "3/4 stable (75.0%)" in figure._suptitle.get_text()
    assert "median basin hit 90.0%" in figure._suptitle.get_text()
    experiment_axis = figure.axes[1]
    assert "descriptive" in experiment_axis.get_title().lower()
    assert [to_rgb(bar.get_facecolor()) for bar in experiment_axis.patches[:2]] == [
        to_rgb("#C6C6CC"),
        to_rgb("#7884B4"),
    ]
    plt.close(figure)


def test_standard_run_adapter_uses_authoritative_selected_fields_and_keeps_no_minimum(
    tmp_path: Path,
) -> None:
    module = _module()
    start = pd.Timestamp("2026-01-01")
    rows = {
        "v1": pd.DataFrame(
            {
                "cycle_name": ["cycle_003"] * 2,
                "candidate_time": [
                    start + pd.Timedelta(minutes=10),
                    start + pd.Timedelta(minutes=12),
                ],
                "optimization_eligible": [True, True],
                "supported": [True, False],
                "is_optimum": [True, False],
                "relative_regret": [0.2, 0.0],
                "inverse_cop": [0.6, 0.5],
            }
        ),
        "v2.5": pd.DataFrame(
            {
                "cycle_name": ["cycle_003"] * 2,
                "candidate_time": [
                    start + pd.Timedelta(minutes=10),
                    start + pd.Timedelta(minutes=12),
                ],
                "optimization_eligible": [False, False],
                "supported": [False, False],
                "is_optimum": [False, False],
                "relative_regret": [np.nan, np.nan],
                "inverse_cop": [0.6, 0.5],
            }
        ),
        "v2.6.8": pd.DataFrame(
            {
                "cycle_name": ["cycle_003"] * 2,
                "candidate_time": [
                    start + pd.Timedelta(minutes=10),
                    start + pd.Timedelta(minutes=12),
                ],
                "optimization_eligible": [True, True],
                "model_supported": [False, True],
                "diagnostic_minimum": [start + pd.Timedelta(minutes=10)] * 2,
                "relative_regret": [0.2, 0.0],
                "inverse_cop": [0.6, 0.5],
            }
        ),
    }
    runs = []
    for base_cost, table in rows.items():
        run = tmp_path / base_cost
        run.mkdir()
        (run / "recipe.json").write_text(
            json.dumps({"base_cost": base_cost, "heat_basis": "water"}), encoding="utf-8"
        )
        table.to_csv(run / "cost.csv", index=False)
        runs.append(run)

    class Loader:
        @staticmethod
        def get_cycle_record(_: str) -> dict[str, object]:
            return {"boundaries": {"start_time": start}}

    tables = module._load_result_tables(runs, Loader())

    assert tables["v1"]["t_star"].eq(start + pd.Timedelta(minutes=10)).all()
    assert tables["v1"]["t_star_model_supported"].eq(True).all()
    assert tables["v2.5"]["t_star"].isna().all()
    assert tables["v2.5"]["t_star_model_supported"].isna().all()
    assert tables["v2.6.8"]["t_star"].eq(start + pd.Timedelta(minutes=10)).all()
    assert tables["v2.6.8"]["t_star_model_supported"].eq(False).all()
    assert tables["v1"]["t_RB"].isna().all()
    assert tables["v1"]["rb_status"].eq("unavailable").all()


def test_variants_reuse_their_base_style_independent_of_input_order() -> None:
    module = _module()

    assert module._style("v1__alpha")[:2] == module._style("v1")[:2]
    assert module._style("v2.5__beta")[:2] == module._style("v2.5")[:2]
    assert module._style("v1__alpha")[2] == "V1 (alpha) optimum"


def test_common_comparison_marks_any_unselected_cycle_off_the_data_axis() -> None:
    module = _module()
    table = _table("v2.6.8").assign(
        cycle_start=pd.Timestamp("2025-12-31 23:55:00"),
        t_star=pd.NaT,
        t_star_model_supported=pd.NA,
    )

    figure = module._comparison_figure({"v2.6.8": table}, ("v2.6.8",))

    marker = next(
        item
        for item in figure.axes[0].collections
        if item.get_label() == "V2.6.8 diagnostic minimum (no diagnostic minimum)"
    )
    assert marker.get_offsets().tolist() == [[0.0, -0.04], [1.0, -0.04]]
    assert marker.get_offset_transform() == figure.axes[0].get_xaxis_transform()
    plt.close(figure)
