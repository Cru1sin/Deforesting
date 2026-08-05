from __future__ import annotations

from pathlib import Path

import pytest

from frost_analysis.evidence import EvidenceBundle, build_evidence, write_evidence
from frost_analysis.evidence.contracts import (
    FEATURE_CYCLE_METRIC_COLUMNS,
    FUTURE_ASSOCIATION_COLUMNS,
)

from .conftest import settings


def test_new_evidence_public_api_is_defined() -> None:
    assert EvidenceBundle.__dataclass_params__.frozen is True
    assert callable(build_evidence)
    assert callable(write_evidence)


@pytest.mark.parametrize(
    "channel",
    [
        {
            "analysis_candidate": True,
            "expected_frost_direction": "increase",
        },
        {
            "analysis_candidate": True,
            "expected_frost_direction": "decrease",
            "role": "performance",
        },
    ],
)
def test_candidate_feature_rejects_target_or_performance_candidate(
    channel: dict[str, object],
) -> None:
    from frost_analysis.evidence.core import candidate_features

    with pytest.raises(ValueError, match="candidate"):
        candidate_features(
            {"channels": {"heating_capacity": channel}},
            settings(targets=("heating_capacity",), horizons=(1,)),
        )


def test_evidence_entry_points_accept_loader_contract_without_runtime_type_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest import frame_for, settings, write_dataset

    real_loader = write_dataset(
        tmp_path / "dataset", [("c1", "2026-07-01", "valid", frame_for())]
    )

    class LoaderContract:
        def __init__(self, loader: object) -> None:
            self._loader = loader

        def __getattr__(self, name: str) -> object:
            return getattr(self._loader, name)

    loader = LoaderContract(real_loader)
    evidence_settings = settings(targets=("heating_capacity",), horizons=(1,))
    bundle = build_evidence(loader, evidence_settings)
    monkeypatch.setattr(
        "frost_analysis.evidence.output.write_figures",
        lambda *_args: (),
    )

    output = write_evidence(
        bundle,
        tmp_path / "evidence",
        loader=loader,
        settings=evidence_settings,
    )

    assert output.is_dir()


def test_contract_columns_end_with_status_and_include_degradation_support() -> None:
    assert FEATURE_CYCLE_METRIC_COLUMNS[-2:] == ["metric_status", "exclusion_reason"]
    assert "degradation_support" in FUTURE_ASSOCIATION_COLUMNS
