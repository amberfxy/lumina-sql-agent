from __future__ import annotations

import pytest

from config import DatabaseBackend
from src.db_executor import DatabaseExecutionError, DatabaseExecutor


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM customers",
        "update orders set status = 'x'",
        "DROP TABLE customers",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
    ],
)
def test_mutations_blocked_before_touching_database(settings, query):
    executor = DatabaseExecutor(settings)
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute(query, backend=DatabaseBackend.POSTGRES)
    assert excinfo.value.raw_error == DatabaseExecutor._BLOCKED_ERROR
    assert executor._postgres_engine is None


def test_column_names_containing_keywords_are_not_blocked():
    assert not DatabaseExecutor._MUTATING_PATTERN.search("SELECT created_at, last_update FROM t")


def test_empty_query_rejected(settings):
    with pytest.raises(DatabaseExecutionError):
        DatabaseExecutor(settings).execute("   ", backend=DatabaseBackend.POSTGRES)


def test_dedupe_columns():
    assert DatabaseExecutor._dedupe_columns(["name", "name", "id", "name"]) == ["name", "name_2", "id", "name_3"]


def test_dynamodb_value_round_trip():
    value = {"s": "x", "n": 3, "f": 1.5, "b": True, "null": None, "l": [1, "a"], "m": {"k": 2}}
    serialized = DatabaseExecutor._serialize_dynamodb_value(value)
    assert DatabaseExecutor._deserialize_dynamodb_value(serialized) == value


def test_partiql_json_payload_is_unwrapped():
    assert DatabaseExecutor._normalize_partiql('{"statement": "SELECT * FROM t"}') == "SELECT * FROM t"
