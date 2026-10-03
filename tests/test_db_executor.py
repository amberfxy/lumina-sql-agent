from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

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


class StubDynamoDB:
    """Serves `pages` in order, linking them with NextToken like ExecuteStatement does."""

    def __init__(self, pages: list[list[dict]] | None = None, error: Exception | None = None) -> None:
        self.pages = pages or [[]]
        self.error = error
        self.requests: list[dict] = []

    def execute_statement(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        index = int(request.get("NextToken", 0))
        response = {
            "Items": self.pages[index],
            "ConsumedCapacity": {"TableName": "orders", "CapacityUnits": 0.5},
        }
        if index + 1 < len(self.pages):
            response["NextToken"] = str(index + 1)
        return response


def dynamodb_executor(settings, client: StubDynamoDB, **overrides) -> DatabaseExecutor:
    executor = DatabaseExecutor(settings.model_copy(update=overrides))
    executor._dynamodb_client = client
    return executor


def items(start: int, count: int) -> list[dict]:
    return [{"pk": {"N": str(i)}} for i in range(start, start + count)]


def test_dynamodb_follows_next_token_until_matches_are_found(settings):
    client = StubDynamoDB(pages=[[], [], items(0, 2)])
    result = dynamodb_executor(settings, client).execute("SELECT pk FROM orders", backend=DatabaseBackend.DYNAMODB)

    assert result.rows == [{"pk": 0}, {"pk": 1}]
    assert result.metadata == {"row_limit": 500, "truncated": False, "pages": 3, "consumed_capacity_units": 1.5}
    assert [request.get("NextToken") for request in client.requests] == [None, "1", "2"]
    assert all(request["ReturnConsumedCapacity"] == "TOTAL" for request in client.requests)


def test_dynamodb_rows_are_capped_and_reported_as_truncated(settings):
    client = StubDynamoDB(pages=[items(0, 4), items(4, 4), items(8, 4)])
    result = dynamodb_executor(settings, client, max_result_rows=5).execute(
        "SELECT pk FROM orders", backend=DatabaseBackend.DYNAMODB
    )

    assert result.row_count == 5
    assert result.metadata["truncated"] is True
    assert len(client.requests) == 2


def test_dynamodb_page_budget_bounds_read_cost(settings):
    client = StubDynamoDB(pages=[[] for _ in range(50)])
    result = dynamodb_executor(settings, client, dynamodb_max_pages=3).execute(
        "SELECT pk FROM orders WHERE status = 'rare'", backend=DatabaseBackend.DYNAMODB
    )

    assert len(client.requests) == 3
    assert result.rows == []
    assert result.metadata["truncated"] is True


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM orders WHERE pk = 1",
        "UPDATE orders SET status = 'x' WHERE pk = 1",
        "SELECT pk FROM orders; DELETE FROM orders WHERE pk = 1",
    ],
)
def test_unsafe_partiql_never_reaches_dynamodb(settings, statement):
    client = StubDynamoDB()
    with pytest.raises(DatabaseExecutionError) as excinfo:
        dynamodb_executor(settings, client).execute(statement, backend=DatabaseBackend.DYNAMODB)
    assert excinfo.value.category == FailureCategory.UNSAFE_QUERY
    assert client.requests == []


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("ResourceNotFoundException", FailureCategory.UNKNOWN_TABLE),
        ("ValidationException", FailureCategory.SYNTAX_ERROR),
        ("ProvisionedThroughputExceededException", FailureCategory.RATE_LIMIT),
    ],
)
def test_dynamodb_errors_carry_their_failure_category(settings, code, category):
    error = ClientError({"Error": {"Code": code, "Message": "boom"}}, "ExecuteStatement")
    with pytest.raises(DatabaseExecutionError) as excinfo:
        dynamodb_executor(settings, StubDynamoDB(error=error)).execute(
            "SELECT pk FROM orders", backend=DatabaseBackend.DYNAMODB
        )
    assert excinfo.value.category == category
    assert json.loads(excinfo.value.raw_error)["code"] == code
