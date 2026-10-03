from __future__ import annotations

import pytest

from config import DatabaseBackend
from src.db_executor import DatabaseExecutionError, DatabaseExecutor
from src.failures import FailureCategory


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM customers",
        "update orders set status = 'x'",
        "DROP TABLE customers",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
        "SET TRANSACTION READ WRITE; SELECT nextval('orders_order_id_seq')",
        "COPY (SELECT 1) TO PROGRAM 'id'",
    ],
)
def test_unsafe_queries_blocked_before_touching_database(settings, query):
    executor = DatabaseExecutor(settings)
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute(query, backend=DatabaseBackend.POSTGRES)
    assert excinfo.value.category == FailureCategory.UNSAFE_QUERY
    assert executor._postgres_engine is None


def test_empty_query_rejected(settings):
    with pytest.raises(DatabaseExecutionError) as excinfo:
        DatabaseExecutor(settings).execute("   ", backend=DatabaseBackend.POSTGRES)
    assert excinfo.value.category == FailureCategory.MALFORMED_OUTPUT


def test_unreachable_database_is_a_connection_error(settings):
    executor = DatabaseExecutor(
        settings.model_copy(
            update={"postgres_host": "127.0.0.1", "postgres_port": 1, "postgres_connect_timeout_seconds": 1}
        )
    )
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute("SELECT 1", backend=DatabaseBackend.POSTGRES)
    assert excinfo.value.category == FailureCategory.CONNECTION_ERROR
    executor.close()


def test_dedupe_columns():
    assert DatabaseExecutor._dedupe_columns(["name", "name", "id", "name"]) == ["name", "name_2", "id", "name_3"]


def test_dynamodb_value_round_trip():
    value = {"s": "x", "n": 3, "f": 1.5, "b": True, "null": None, "l": [1, "a"], "m": {"k": 2}}
    serialized = DatabaseExecutor._serialize_dynamodb_value(value)
    assert DatabaseExecutor._deserialize_dynamodb_value(serialized) == value


def test_partiql_json_payload_is_unwrapped():
    assert DatabaseExecutor._normalize_partiql('{"statement": "SELECT * FROM t"}') == "SELECT * FROM t"
