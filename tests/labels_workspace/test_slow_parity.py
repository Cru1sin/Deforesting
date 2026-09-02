from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from labels import build

ROOT = Path("/Users/cruisin/Documents/DeforestingSensor")
DATASET = ROOT / "dataset"
OLD_COST = ROOT / "output/成本函数/cost_function_v1.csv"
OLD_LABELS = ROOT / "output/label/cost_function_v1_binary/image_cost_labels.parquet"
OLD_BALANCE = ROOT / "output/label/cost_function_v1_binary/label_balance.csv"


@pytest.mark.slow
def test_formal_v1_output_parity(tmp_path: Path) -> None:
    if not all(path.exists() for path in (DATASET, OLD_COST, OLD_LABELS, OLD_BALANCE)):
        pytest.skip("formal V1 Dataset/cost/labels are not available")
    expected = pd.read_parquet(OLD_LABELS)
    expected_balance = pd.read_csv(OLD_BALANCE)
    cost = pd.read_csv(OLD_COST).assign(label_eligible=True, variant=None)
    actual, actual_balance, _ = build.build_labels(
        DATASET, cost, (0.01, 0.02, 0.05, 0.10)
    )
    assert len(actual) == len(expected) == 89_282
    pd.testing.assert_frame_equal(
        actual.drop(columns="local_available"),
        expected.drop(columns=["local_available", "split"]),
        check_exact=True,
    )
    balance_keys = ["regret_threshold", "camera_group", "cost_state"]
    expected_balance = (
        expected_balance.drop(columns="split")
        .groupby(balance_keys, as_index=False)
        .sum()
        .sort_values(balance_keys, kind="stable")
        .reset_index(drop=True)
    )
    actual_balance = actual_balance.sort_values(
        balance_keys, kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_balance.drop(columns="local_image_count"),
        expected_balance.drop(columns="local_image_count"),
        check_exact=True,
    )
