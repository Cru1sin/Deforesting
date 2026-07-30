from __future__ import annotations

import pandas as pd

from frost_analysis.analysis import EVIDENCE_COLUMNS
from frost_analysis.validation import validate_analysis


def test_flat_analysis_contract_has_explicit_evidence_columns() -> None:
    assert "trend_cycle_count" in EVIDENCE_COLUMNS
    assert "valid_cycle_count" not in EVIDENCE_COLUMNS
    assert "weighted_score" not in EVIDENCE_COLUMNS
    validate_analysis(pd.DataFrame(columns=EVIDENCE_COLUMNS))
