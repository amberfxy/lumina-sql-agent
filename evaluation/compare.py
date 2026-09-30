"""Execution-accuracy comparison between gold and predicted result sets.

A prediction is correct when some injective mapping of its columns onto the gold
columns reproduces the gold rows (as a multiset, or as a list for ordered questions).
This ignores column names and order and tolerates extra predicted columns, while
numbers are compared after rounding to 2 decimal places.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Iterator, Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

_CENTS = Decimal("0.01")
_MAX_ASSIGNMENTS = 5000


def normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int | float | Decimal):
        try:
            return Decimal(str(value)).quantize(_CENTS, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            return str(value)
    if isinstance(value, dt.timedelta):
        return Decimal(str(value.total_seconds())).quantize(_CENTS, rounding=ROUND_HALF_UP)
    if isinstance(value, dt.date):  # also covers datetime
        return value.isoformat()
    return str(value).strip()


def to_matrix(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_value(row.get(column)) for column in columns) for row in rows]


def results_match(gold: list[tuple[Any, ...]], predicted: list[tuple[Any, ...]], ordered: bool) -> bool:
    if len(gold) != len(predicted):
        return False
    if not gold:
        return True

    gold_width = len(gold[0])
    if not predicted[0] or len(predicted[0]) < gold_width:
        return False

    gold_columns = list(zip(*gold, strict=True))
    predicted_columns = list(zip(*predicted, strict=True))

    candidates: list[list[int]] = []
    for gold_column in gold_columns:
        signature = Counter(gold_column)
        matches = [index for index, column in enumerate(predicted_columns) if Counter(column) == signature]
        if not matches:
            return False
        candidates.append(matches)

    expected = gold if ordered else Counter(gold)
    for assignment in _assignments(candidates):
        projected = [tuple(row[index] for index in assignment) for row in predicted]
        if (projected if ordered else Counter(projected)) == expected:
            return True
    return False


def _assignments(candidates: list[list[int]]) -> Iterator[tuple[int, ...]]:
    """Yield injective column assignments (bounded to avoid pathological blowups)."""
    produced = 0
    chosen: list[int] = []

    def backtrack(position: int) -> Iterator[tuple[int, ...]]:
        nonlocal produced
        if produced >= _MAX_ASSIGNMENTS:
            return
        if position == len(candidates):
            produced += 1
            yield tuple(chosen)
            return
        for index in candidates[position]:
            if index in chosen:
                continue
            chosen.append(index)
            yield from backtrack(position + 1)
            chosen.pop()

    yield from backtrack(0)
