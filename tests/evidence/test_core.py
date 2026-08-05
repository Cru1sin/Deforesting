from __future__ import annotations

from frost_analysis.evidence import EvidenceBundle, build_evidence, write_evidence


def test_new_evidence_public_api_is_defined() -> None:
    assert EvidenceBundle.__dataclass_params__.frozen is True
    assert callable(build_evidence)
    assert callable(write_evidence)
