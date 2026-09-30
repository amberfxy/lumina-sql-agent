from __future__ import annotations

import datetime as dt
from decimal import Decimal

from evaluation.compare import normalize_value, results_match, to_matrix


def m(rows):
    return [tuple(normalize_value(v) for v in row) for row in rows]


def test_numbers_compared_at_two_decimals():
    assert results_match(m([(Decimal("57.12345"),)]), m([(57.12,)]), ordered=False)
    assert not results_match(m([(57.12,)]), m([(57.2,)]), ordered=False)


def test_int_and_decimal_are_equal():
    assert results_match(m([(1, 10)]), m([(Decimal("1"), 10.0)]), ordered=False)


def test_unordered_ignores_row_order():
    assert results_match(m([("a", 1), ("b", 2)]), m([("b", 2), ("a", 1)]), ordered=False)


def test_ordered_requires_row_order():
    assert not results_match(m([("a", 1), ("b", 2)]), m([("b", 2), ("a", 1)]), ordered=True)


def test_column_order_and_extra_columns_tolerated():
    gold = m([("Seattle", 3), ("Austin", 5)])
    predicted = m([(5, "Austin", "TX"), (3, "Seattle", "WA")])
    assert results_match(gold, predicted, ordered=False)


def test_missing_column_fails():
    assert not results_match(m([("a", 1)]), m([("a",)]), ordered=False)


def test_row_count_mismatch_fails():
    assert not results_match(m([(1,), (2,)]), m([(1,)]), ordered=False)


def test_columns_must_stay_aligned_within_rows():
    gold = m([("a", 1), ("b", 2)])
    predicted = m([("a", 2), ("b", 1)])
    assert not results_match(gold, predicted, ordered=False)


def test_dates_and_timedeltas_normalized():
    assert normalize_value(dt.date(2024, 3, 1)) == "2024-03-01"
    assert normalize_value(dt.timedelta(hours=1)) == Decimal("3600.00")


def test_to_matrix_uses_column_order():
    rows = [{"b": 2, "a": 1}]
    assert to_matrix(rows, ["a", "b"]) == [(Decimal("1.00"), Decimal("2.00"))]
