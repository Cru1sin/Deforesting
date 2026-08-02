"""Ordered one-to-one matching for timestamp series."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class _Candidate:
    left_position: int
    right_position: int
    right_rank: int
    offset_ns: int
    creation_order: int


@dataclass(frozen=True)
class _State:
    count: int
    offset_ns: int
    creation_order: int
    candidate_id: int
    previous_state_id: int


def _is_better_state(candidate_id: int, current_id: int, states: list[_State]) -> bool:
    candidate = states[candidate_id]
    current = states[current_id]
    if candidate.count != current.count:
        return candidate.count > current.count
    if candidate.offset_ns != current.offset_ns:
        return candidate.offset_ns < current.offset_ns
    return candidate.creation_order < current.creation_order


class _PrefixBest:
    def __init__(self, state_ids: list[_State]) -> None:
        self._states = state_ids
        self._tree: list[int] = [0]

    def resize(self, size: int) -> None:
        self._tree = [0] * (size + 1)

    def best_before(self, right_rank: int) -> int:
        state_id = 0
        while right_rank:
            tree_state_id = self._tree[right_rank]
            if _is_better_state(tree_state_id, state_id, self._states):
                state_id = tree_state_id
            right_rank -= right_rank & -right_rank
        return state_id

    def update(self, right_rank: int, state_id: int) -> None:
        tree_index = right_rank + 1
        while tree_index < len(self._tree):
            if _is_better_state(state_id, self._tree[tree_index], self._states):
                self._tree[tree_index] = state_id
            tree_index += tree_index & -tree_index


def _valid_timestamp_positions(series: pd.Series) -> list[tuple[int, int]]:
    parsed = pd.to_datetime(series, errors="coerce")
    positions: list[tuple[int, int]] = []
    for position, value in enumerate(parsed.tolist()):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            continue
        positions.append((timestamp.value, position))
    positions.sort()
    return positions


def match_nearest_one_to_one(
    left_times: pd.Series, right_times: pd.Series, tolerance: pd.Timedelta
) -> list[tuple[int, int]]:
    """Match timestamp positions in order, maximizing count then minimizing offset."""
    if pd.isna(tolerance) or tolerance < pd.Timedelta(0):
        raise ValueError("tolerance must be non-negative and not NaT")

    left = _valid_timestamp_positions(left_times)
    right = _valid_timestamp_positions(right_times)
    if not left or not right:
        return []

    right_values = [timestamp for timestamp, _ in right]
    tolerance_ns = tolerance.value
    candidates: list[_Candidate] = []
    states = [_State(0, 0, -1, -1, -1)]
    prefix_best = _PrefixBest(states)
    prefix_best.resize(len(right))
    creation_order = 0

    for left_timestamp, left_position in left:
        first_right = bisect_left(right_values, left_timestamp - tolerance_ns)
        after_last_right = bisect_right(right_values, left_timestamp + tolerance_ns)
        pending: list[tuple[int, int]] = []
        for right_rank in range(first_right, after_last_right):
            previous_state_id = prefix_best.best_before(right_rank)
            previous = states[previous_state_id]
            candidate = _Candidate(
                left_position,
                right[right_rank][1],
                right_rank,
                abs(left_timestamp - right[right_rank][0]),
                creation_order,
            )
            candidate_id = len(candidates)
            candidates.append(candidate)
            state_id = len(states)
            states.append(
                _State(
                    previous.count + 1,
                    previous.offset_ns + candidate.offset_ns,
                    candidate.creation_order,
                    candidate_id,
                    previous_state_id,
                )
            )
            pending.append((candidate.right_rank, state_id))
            creation_order += 1

        for right_rank, candidate_id in pending:
            prefix_best.update(right_rank, candidate_id)

    state_id = prefix_best.best_before(len(right))
    candidate_ids: list[int] = []
    while state_id:
        state = states[state_id]
        candidate_ids.append(state.candidate_id)
        state_id = state.previous_state_id

    candidate_ids.reverse()
    return [
        (candidates[candidate_id].left_position, candidates[candidate_id].right_position)
        for candidate_id in candidate_ids
    ]
