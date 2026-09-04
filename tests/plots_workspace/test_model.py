from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from plots import image_models as model_plots


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": [
                "root-a/current",
                "root-a/current",
                "root-a/current",
                "root-a/current",
                "root-b/current",
            ],
            "image_feature": [
                "color_gradient",
                "embedding",
                "color_gradient",
                "color_gradient",
                "color_gradient",
            ],
            "classifier": [
                "logistic_regression",
                "mlp",
                "logistic_regression",
                "logistic_regression",
                "logistic_regression",
            ],
            "camera": ["front", "front", "novel_camera", "front", "front"],
            "input_feature": ["image_only", "image_only", "image_only", "time_only", "image_only"],
            "balanced_accuracy_mean": [0.80, 0.90, 0.85, 0.82, 0.81],
            "balanced_accuracy_std": [0.02, 0.03, 0.04, 0.01, 0.025],
            "macro_f1_mean": [0.79, 0.89, 0.84, 0.81, 0.80],
            "macro_f1_std": [0.03, 0.04, 0.05, 0.02, 0.035],
        }
    )


def test_figure_5_adapts_current_settings_as_mean_plus_or_minus_sd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, str]] = []
    original = plt.Axes.errorbar

    def capture(self: plt.Axes, x: object, y: object, *, yerr: object, **kwargs: object) -> object:
        calls.append(
            (np.asarray(y, dtype=float), np.asarray(yerr, dtype=float), str(kwargs["label"]))
        )
        return original(self, x, y, yerr=yerr, **kwargs)

    monkeypatch.setattr(plt.Axes, "errorbar", capture)
    monkeypatch.setattr(model_plots, "_export", lambda figure, _stem, _formats: plt.close(figure))
    model_plots.plot_model_figures(
        summary=_summary(),
        output=tmp_path / "figures",
        source_output=tmp_path / "source",
    )

    assert len(calls) == 8
    assert {label for _, _, label in calls} == {
        "embedding + mlp + image only",
        "color gradient + logistic regression + image only + run=root-a/current",
        "color gradient + logistic regression + image only + run=root-b/current",
        "color gradient + logistic regression + time only",
    }
    source = pd.read_csv(tmp_path / "source" / "figure_5_model_comparison.csv")
    assert set(source["model_setting"]) == {
        "embedding + mlp + image only",
        "color gradient + logistic regression + image only + run=root-a/current",
        "color gradient + logistic regression + image only + run=root-b/current",
        "color gradient + logistic regression + time only",
    }
    assert set(source["camera"].astype(str)) == {"front", "novel_camera"}


def test_figure_6_is_generated_only_when_both_inputs_are_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    concentration_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    monkeypatch.setattr(model_plots, "_plot_model_comparison", lambda *_args: None)
    monkeypatch.setattr(
        model_plots,
        "_plot_concentration",
        lambda optima, concentration, *_args: concentration_calls.append((optima, concentration)),
    )
    common = {
        "summary": _summary(),
        "output": tmp_path / "figures",
        "source_output": tmp_path / "source",
    }
    model_plots.plot_model_figures(**common)
    assert concentration_calls == []

    optima = pd.DataFrame({"cycle_name": ["a"]})
    concentration = pd.DataFrame({"camera_group": ["front"]})
    model_plots.plot_model_figures(**common, optima=optima, concentration=concentration)
    assert concentration_calls == [(optima, concentration)]
    with pytest.raises(ValueError, match="provided together"):
        model_plots.plot_model_figures(**common, optima=optima)
