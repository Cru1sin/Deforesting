from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from dataloader.alignment import match_nearest_one_to_one


def test_match_maximizes_cardinality_before_offset() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    left = pd.Series([start, start + pd.Timedelta(seconds=0.9)])
    right = pd.Series([start + pd.Timedelta(seconds=0.8), start + pd.Timedelta(seconds=1.7)])

    assert match_nearest_one_to_one(left, right, pd.Timedelta(seconds=0.9)) == [(0, 0), (1, 1)]


def test_match_minimizes_total_offset_after_cardinality() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    left = pd.Series([start, start + pd.Timedelta(seconds=1)])
    right = pd.Series([start + pd.Timedelta(seconds=0.9)])

    assert match_nearest_one_to_one(left, right, pd.Timedelta(seconds=1)) == [(1, 0)]


def test_match_uses_stable_candidate_order_for_exact_ties() -> None:
    timestamp = pd.Timestamp("2026-07-15 00:00:00")
    left = pd.Series([timestamp, timestamp + pd.Timedelta(seconds=1)])
    right = pd.Series([timestamp + pd.Timedelta(seconds=0.5)])

    assert match_nearest_one_to_one(left, right, pd.Timedelta(seconds=0.5)) == [(0, 0)]


def test_match_rejects_negative_tolerance() -> None:
    times = pd.Series([pd.Timestamp("2026-07-15")])

    with pytest.raises(ValueError):
        match_nearest_one_to_one(times, times, pd.Timedelta(seconds=-1))


def test_match_rejects_nat_tolerance() -> None:
    times = pd.Series([pd.Timestamp("2026-07-15")])

    with pytest.raises(ValueError):
        match_nearest_one_to_one(times, times, cast(pd.Timedelta, pd.NaT))


def test_match_zero_tolerance_accepts_only_exact_timestamps() -> None:
    timestamp = pd.Timestamp("2026-07-15 00:00:00")
    left = pd.Series([timestamp, timestamp + pd.Timedelta(seconds=1)])
    right = pd.Series([timestamp, timestamp + pd.Timedelta(seconds=1, nanoseconds=1)])

    assert match_nearest_one_to_one(left, right, pd.Timedelta(0)) == [(0, 0)]


def test_match_ignores_nat_preserves_positions_and_orders_by_time() -> None:
    start = pd.Timestamp("2026-07-15 00:00:00")
    left = pd.Series([pd.NaT, start + pd.Timedelta(seconds=2), start])
    right = pd.Series([start + pd.Timedelta(seconds=2), pd.NaT, start])

    assert match_nearest_one_to_one(left, right, pd.Timedelta(milliseconds=500)) == [
        (2, 2),
        (1, 0),
    ]
