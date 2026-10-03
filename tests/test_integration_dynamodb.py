"""Integration tests for the PartiQL path against DynamoDB Local.

Runs when DYNAMODB_ENDPOINT_URL points at a reachable DynamoDB Local (Compose `test` profile, CI).
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from config import DatabaseBackend, Settings
from src.db_executor import DatabaseExecutionError, DatabaseExecutor
from src.failures import FailureCategory
from src.schema_manager import SchemaManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("DYNAMODB_ENDPOINT_URL"), reason="DYNAMODB_ENDPOINT_URL is not set"),
]

DDB = DatabaseBackend.DYNAMODB
ITEMS = 30


@pytest.fixture(scope="module")
def prefix() -> str:
    return f"it{uuid.uuid4().hex[:8]}_"


@pytest.fixture(scope="module")
def ddb_settings(prefix) -> Settings:
    return Settings(redis_url=None, dynamodb_table_prefix=prefix, max_result_rows=10)


@pytest.fixture(scope="module")
def executor(ddb_settings, prefix):
    executor = DatabaseExecutor(ddb_settings)
    client = executor.dynamodb_client
    deadline = time.monotonic() + 30
    while True:
        try:
            client.list_tables(Limit=1)
            break
        except (BotoCoreError, ClientError):
            if time.monotonic() > deadline:
                raise
            time.sleep(1)

    table = f"{prefix}orders"
    client.create_table(
        TableName=table,
        AttributeDefinitions=[{"AttributeName": "order_id", "AttributeType": "N"}],
        KeySchema=[{"AttributeName": "order_id", "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=table)
    for i in range(ITEMS):
        client.put_item(
            TableName=table,
            Item={
                "order_id": {"N": str(i)},
                "status": {"S": "shipped" if i % 3 == 0 else "pending"},
                "total": {"N": f"{i * 10}.5"},
            },
        )
    yield executor
    client.delete_table(TableName=table)
    executor.close()


def test_schema_extraction_lists_prefixed_tables(ddb_settings, executor, prefix):
    catalog = SchemaManager(settings=ddb_settings, db_executor=executor)._get_catalog(DDB)
    assert [table.name for table in catalog.tables] == [f"{prefix}orders"]
    assert catalog.tables[0].columns[0].is_primary_key


def test_filtered_select_returns_typed_rows(executor, prefix):
    result = executor.execute(
        f"SELECT order_id, total FROM \"{prefix}orders\" WHERE status = 'shipped' AND order_id < 7", backend=DDB
    )
    assert sorted(row["order_id"] for row in result.rows) == [0, 3, 6]
    assert {row["total"] for row in result.rows} == {0.5, 30.5, 60.5}
    assert result.metadata["truncated"] is False


def test_rows_are_capped_at_max_result_rows(executor, prefix):
    result = executor.execute(f'SELECT order_id FROM "{prefix}orders"', backend=DDB)
    assert result.row_count == 10
    assert result.metadata["truncated"] is True


def test_unknown_table_is_llm_correctable(executor, prefix):
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute(f'SELECT * FROM "{prefix}missing"', backend=DDB)
    assert excinfo.value.category == FailureCategory.UNKNOWN_TABLE


def test_invalid_partiql_is_a_syntax_error(executor, prefix):
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute(f'SELECT order_id FROM "{prefix}orders" WHERE', backend=DDB)
    assert excinfo.value.category == FailureCategory.SYNTAX_ERROR


@pytest.mark.parametrize(
    "template",
    [
        'DELETE FROM "{t}" WHERE order_id = 1',
        "UPDATE \"{t}\" SET status = 'x' WHERE order_id = 1",
        "INSERT INTO \"{t}\" VALUE {{'order_id': 999}}",
        'SELECT * FROM "{t}" WHERE order_id = 1; DELETE FROM "{t}" WHERE order_id = 1',
    ],
)
def test_writes_are_blocked_and_data_is_unchanged(executor, prefix, template):
    table = f"{prefix}orders"
    with pytest.raises(DatabaseExecutionError) as excinfo:
        executor.execute(template.format(t=table), backend=DDB)
    assert excinfo.value.category == FailureCategory.UNSAFE_QUERY

    count = executor.dynamodb_client.scan(TableName=table, Select="COUNT")["Count"]
    status = executor.dynamodb_client.get_item(TableName=table, Key={"order_id": {"N": "1"}})["Item"]["status"]
    assert count == ITEMS
    assert status == {"S": "pending"}
